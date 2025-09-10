import sys, os, time, platform, re
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from utilities.detectpi.box_smoothing import BoxSmoother
from utilities.detectpi.paths import find_latest_best, get_output_folder
from utilities.detectpi.arg_parser import parse_arguments
from utilities.detectpi.logger import DetectionDashboard
from utilities.detectpi.video_rotation import get_rotation_angle, rotate_frame

try:
    "imports picamera2 only if accessible by the system"
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

BASE_DIR = Path(__file__).resolve().parent

def open_source(src):
    """Open a video file or picamera2 stream"""
    if str(src).lower() == "picamera":
        if not HAS_PICAMERA:
            raise RuntimeError("Picamera2 not available on this system!")
        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (640, 480)}) # sets resolution
        cam.configure(config)
        cam.start()
        return cam, "picamera"
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {src}")
        return cap, "video"

def read_frame(source, source_type):
    if source_type == "picamera":
        return True, source.capture_array()
    else:
        return source.read()

def run_detection(model, src, dashboard, smoother):
    from utilities.detectpi.video_rotation import get_rotation_angle, rotate_frame

    # ---------- Prepare output ----------
    raw_source_name = Path(src).stem
    display_name = raw_source_name
    safe_source_name = re.sub(r"[^\w\-\.]", "_", raw_source_name)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # Determine source type
    source, source_type = open_source(src)
    out_path = get_output_folder(
        model.weights_path,
        "picamera" if source_type=="picamera" else "video",
        raw_source_name
    )
    out_file = out_path / f"{safe_source_name}_{timestamp}.mp4"

    # ---------- Setup rotation for videos ----------
    rotation_angle = 0
    if source_type == "video":
        rotation_angle = get_rotation_angle(src)

    # ---------- Open first frame to get dimensions ----------
    ret, frame = read_frame(source, source_type)
    if not ret or frame is None:
        dashboard.log(f"[ERROR] Could not read from {raw_source_name}")
        return

    if rotation_angle != 0:
        frame = rotate_frame(frame, rotation_angle)

    height, width = frame.shape[:2]
    fps = 20.0  # fallback for picamera
    total_duration_sec = 0
    if source_type == "video":
        total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = source.get(cv2.CAP_PROP_FPS) or 20.0
        total_duration_sec = total_frames / fps

    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    dashboard.register_writer(raw_source_name, writer, source, source_type, out_file)

    # ---------- Loop ----------
    frame_count, fps_smooth, prev_time = 0, 0, time.time()
    start_time = time.time()
    is_video = source_type == "video"

    try:
        while True:
            ret, frame = read_frame(source, source_type)
            if not ret:
                break

            if rotation_angle != 0:
                frame = rotate_frame(frame, rotation_angle)

            results = model.predict(frame, verbose=False, show=False, imgsz=640)
            draw_frame = results[0].plot() if results else frame

            # ----- Smooth boxes -----
            smoothed_boxes_list = []
            if results and hasattr(results[0], "obb") and results[0].obb is not None:
                boxes = results[0].obb.xywhr.cpu().numpy()
                classes = results[0].obb.cls.cpu().numpy()
                if frame_count % 3 == 0:
                    smoothed_boxes = smoother.smooth([
                        [cx, cy, w, h, float(angle), int(cls)]
                        for cx, cy, w, h, angle, cls in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], classes)
                    ])
                    smoothed_boxes_list.extend(smoothed_boxes)

            names = results[0].names if results else {}
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / (time.time() - prev_time + 1e-6))
            prev_time = time.time()
            frame_count += 1

            # ----- Count objects -----
            males = sum(1 for b in smoothed_boxes_list if names.get(b[5]) == "M")
            females = sum(1 for b in smoothed_boxes_list if names.get(b[5]) == "F")
            other_objects = sum(1 for b in smoothed_boxes_list if names.get(b[5]) not in ["M", "F"])

            # ----- Timer -----
            if is_video:
                elapsed_sec = frame_count / fps
                remaining_sec = max(0, total_duration_sec - elapsed_sec)
            else:
                elapsed_sec = time.time() - start_time
                remaining_sec = 0

            eh, em = divmod(int(elapsed_sec) // 60, 60)
            es = int(elapsed_sec) % 60
            rh, rm = divmod(int(remaining_sec) // 60, 60)
            rs = int(remaining_sec) % 60
            time_info = f"{eh:02d}:{es:02d}/{rh:02d}:{rm:02d}:{rs:02d}"

            # ----- Display logging -----
            if frame_count % 5 == 0:
                dashboard.update_line(
                    1,
                    f"[{display_name}] Frames:{frame_count} | FPS:{fps_smooth:.1f} | "
                    f"Males:{males} | Females:{females} | Objects:{other_objects} | Time:{time_info}"
                )

            writer.write(draw_frame)
            time.sleep(0.001)

    except KeyboardInterrupt:
        dashboard.log("[EXIT] Stop signal received. Terminating pipeline...")
    finally:
        dashboard.safe_release_writer(raw_source_name)


# ---- MAIN ----
if __name__ == "__main__":
    args = parse_arguments()

    # Always use main folder
    runs_dir = BASE_DIR / "runs/main"
    weights_path = find_latest_best(runs_dir)
    if not weights_path:
        print(f"[ERROR] No best.pt found in {runs_dir}")
        sys.exit(1)

    print(f"[INFO] Loading model from {weights_path}")
    model = YOLO(str(weights_path))
    model.weights_path = weights_path

    dashboard = DetectionDashboard(1)
    smoother = BoxSmoother(max_history=args.max_history, alpha=args.smooth, dist_thresh=args.dist_thresh)

    # ----- Report smoothing if lab mode -----
    if args.lab:
        dashboard.report_smoothing(args, {
            'smooth': '--smooth' in sys.argv,
            'dist_thresh': '--dist-thresh' in sys.argv,
            'max_history': '--max-history' in sys.argv
        })

    src = args.source

    # ----- Try running detection, handle picamera gracefully -----
    try:
        run_detection(model, src, dashboard, smoother)
    except RuntimeError as e:
        if "Picamera2 not available" in str(e):
            dashboard.log("[ERROR] Picamera2 not supported on this system. Please use a video file instead.")
        else:
            raise  # re-raise any other runtime errors

    dashboard.release_all_writers()
    dashboard.log("[EXIT] All detection threads safely terminated.")
