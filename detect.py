import sys, os, time, platform, re, threading, queue
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from utilities.detectpi_test.paths import find_latest_best, get_output_folder
from utilities.detectpi_test.arg_parser import parse_arguments
from utilities.detectpi_test.logger import Dashboard
from utilities.detectpi_test.video_rotation import get_rotation_angle, rotate_frame
from utilities.detectpi_test.temporal_aggregator import Aggregator

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

BASE_DIR = Path(__file__).resolve().parent

def open_source(src):
    """Open a video file or Picamera2 stream."""
    if str(src).lower() == "picamera":
        if not HAS_PICAMERA:
            raise RuntimeError("Picamera2 not available on this system!")
        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (640, 480)})
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

def run_detection(model, src, dashboard):
    # ---------- Prepare Output ----------
    raw_source_name = Path(src).stem
    display_name = raw_source_name
    safe_source_name = re.sub(r"[^\w\-\.]", "_", raw_source_name)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    source, source_type = open_source(src)
    out_path = get_output_folder(
        model.weights_path,
        "picamera" if source_type=="picamera" else "video",
        raw_source_name
    )
    out_file = out_path / f"{safe_source_name}_{timestamp}.mp4"
    results_folder = out_path / "video-in"  # temporal results folder

    # ---------- Setup Rotation ----------
    rotation_angle = get_rotation_angle(src) if source_type == "video" else 0

    ret, frame = read_frame(source, source_type)
    if not ret or frame is None:
        dashboard.log(f"[ERROR] Could not read from {raw_source_name}")
        return
    if rotation_angle != 0:
        frame = rotate_frame(frame, rotation_angle)

    height, width = frame.shape[:2]
    fps = 20.0
    total_duration_sec = 0
    if source_type == "video":
        total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = source.get(cv2.CAP_PROP_FPS) or 20.0
        total_duration_sec = total_frames / fps

    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    dashboard.register_writer(raw_source_name, writer, source, source_type, out_file)

    # ---------- Aggregator ----------
    aggregator = Aggregator(interval_sec=5, session_sec=10)  # adjust session_sec as needed

    # ---------- Frame Queue ----------
    frame_queue = queue.Queue(maxsize=5)
    stop_reader = threading.Event()
    frame_queue.put(frame)

    def capture_frames():
        while not stop_reader.is_set():
            ret, f = read_frame(source, source_type)
            if not ret:
                break
            if rotation_angle != 0:
                f = rotate_frame(f, rotation_angle)
            try:
                frame_queue.put(f, timeout=0.1)
            except queue.Full:
                pass
        stop_reader.set()

    reader_thread = threading.Thread(target=capture_frames, daemon=True)
    reader_thread.start()

    frame_count, fps_smooth, prev_time = 0, 0, time.time()
    start_time = time.time()
    is_video = source_type == "video"

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_reader.is_set() and frame_queue.empty():
                    break
                continue

            # ---------- Inference ----------
            results = model.predict(frame, verbose=False, show=False, imgsz=640)
            draw_frame = results[0].plot() if results else frame

            # ---------- Current Boxes ----------
            current_boxes_list = []
            if results and hasattr(results[0], "obb") and results[0].obb is not None:
                boxes = results[0].obb.xywhr.cpu().numpy()
                classes = results[0].obb.cls.cpu().numpy()
                current_boxes_list = [
                    [cx, cy, w, h, float(angle), int(cls)]
                    for cx, cy, w, h, angle, cls in zip(
                        boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4], classes
                    )
                ]

            names = results[0].names if results else {}
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / (time.time() - prev_time + 1e-6))
            prev_time = time.time()
            frame_count += 1

            # ---------- Count Objects ----------
            males = sum(1 for b in current_boxes_list if names.get(b[5]) == "M")
            females = sum(1 for b in current_boxes_list if names.get(b[5]) == "F")

            other_counts = {
                'Feeder': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Feeder'),
                'Main_Perch': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Main_Perch'),
                'Wooden_Perch': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Wooden_Perch'),
                'Sky_Perch': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Sky_Perch'),
                'Nesting_Box': sum(1 for b in current_boxes_list if names.get(b[5]) == 'Nesting_Box')
            }

            aggregator.push_frame_data(datetime.now(), males, females, other_counts)

            # ---------- Timer ----------
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

            # ---------- Logging ----------
            if frame_count % 5 == 0:
                dashboard.update_line(
                    1,
                    f"[{display_name}] Frames:{frame_count} | FPS:{fps_smooth:.1f} | "
                    f"Males:{males} | Females:{females} | Objects:{sum(other_counts.values())} | Time:{time_info}"
                )

            writer.write(draw_frame)
            if source_type == "picamera":
                time.sleep(0.001)

    except KeyboardInterrupt:
        dashboard.log("[EXIT] Stop signal received. Terminating pipeline...")
    finally:
        stop_reader.set()
        reader_thread.join()
        dashboard.safe_release_writer(raw_source_name)

        # ---------- Save Aggregator Results ----------
        # determine scores folder
        scores_root = BASE_DIR / "logs" / "scores"  # ~/YOLO/logs/scores
        safe_src_name = re.sub(r"[^\w\-\.]", "_", raw_source_name)
        timestamp_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        aggregator_folder = scores_root / safe_src_name / timestamp_str
        aggregator_folder.mkdir(parents=True, exist_ok=True)

        interval_file, session_file = aggregator.save_results(aggregator_folder)
        dashboard.log(f"[INFO] Interval results saved to {interval_file}")
        dashboard.log(f"[INFO] Session summary saved to {session_file}")


# ---- MAIN ----
if __name__ == "__main__":
    args = parse_arguments()

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

    try:
        run_detection(model, src, dashboard)
    except RuntimeError as e:
        if "Picamera2 not available" in str(e):
            dashboard.log("[ERROR] Picamera2 not supported on this system. Please use a video file instead.")
        else:
            raise

    dashboard.release_all_writers()
    dashboard.log("[EXIT] All detection threads safely terminated.")
