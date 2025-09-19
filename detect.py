import sys, os, time, platform, re
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from utilities.detectpi.paths import find_latest_best, get_output_folder
from utilities.detectpi.arg_parser import parse_arguments
from utilities.detectpi.logger import Dashboard
from utilities.detectpi.video_rotation import get_rotation_angle, rotate_frame

try:
    import picamera
    import picamera.array
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

BASE_DIR = Path(__file__).resolve().parent

def open_source(src):
    if str(src).lower() == "picamera":
        if not HAS_PICAMERA:
            raise RuntimeError("picamera not available on this system!")

        cam = picamera.PiCamera()
        cam.resolution = (640, 480)
        cam.framerate = 20
        raw_capture = picamera.array.PiRGBArray(cam, size=(640, 480))
        time.sleep(0.2)  # give camera time to warm up
        return (cam, raw_capture), "picamera"
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {src}")
        return cap, "video"

def read_frame(source, source_type):
    if source_type == "picamera":
        cam, raw_capture = source
        raw_capture.truncate(0)  # clear the stream
        cam.capture(raw_capture, format="bgr", use_video_port=True)
        frame = raw_capture.array
        return True, frame
    else:
        return source.read()

def run_detection(model, src, dashboard):
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

            # ----- Use current boxes -----
            current_boxes_list = []
            if results and hasattr(results[0], "obb") and results[0].obb is not None:
                boxes = results[0].obb.xywhr.cpu().numpy()
                classes = results[0].obb.cls.cpu().numpy()
                # Build [cx, cy, w, h, angle, cls] items directly from detections
                current_boxes_list = [
                    [cx, cy, w, h, float(angle), int(cls)]
                    for cx, cy, w, h, angle, cls in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], classes)
                ]

            names = results[0].names if results else {}
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / (time.time() - prev_time + 1e-6))
            prev_time = time.time()
            frame_count += 1

            # ----- Count objects from current detections -----
            males = sum(1 for b in current_boxes_list if names.get(b[5]) == "M")
            females = sum(1 for b in current_boxes_list if names.get(b[5]) == "F")
            other_objects = sum(1 for b in current_boxes_list if names.get(b[5]) not in ["M", "F"])

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
        if source_type == "picamera":
            cam, _ = source
            cam.close()

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

    dashboard = Dashboard(1)

    src = args.source

    # ----- Try running detection, handle picamera gracefully -----
    try:
        run_detection(model, src, dashboard)
    except RuntimeError as e:
        if "Picamera2 not available" in str(e):
            dashboard.log("[ERROR] Picamera2 not supported on this system. Please use a video file instead.")
        else:
            raise  # re-raise any other runtime errors

    dashboard.release_all_writers()
    dashboard.log("[EXIT] All detection threads safely terminated.")
