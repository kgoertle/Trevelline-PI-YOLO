from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[2]  

def find_latest_best(base_path):
    base_path = Path(base_path)
    if not base_path.exists():
        return None
    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dirs:
        return None

    def parse_ts(name):
        try:
            return datetime.strptime(name, "%m-%d-%Y_%H-%M-%S")
        except Exception:
            return datetime.min

    latest = max(dirs, key=lambda d: parse_ts(d.name))
    pt = latest / "weights" / "best.pt"
    return pt if pt.exists() else None


def get_output_folder(weights_path, source_type, source_name):
    train_folder = weights_path.parent.parent
    if source_type == "video":
        out_dir = BASE_DIR / "logs" / "recordings" / "video-in"
    elif source_type == "picamera":
        out_dir = BASE_DIR / "logs" / "recordings" / "picamera"
    else:
        out_dir = BASE_DIR / "logs" / "recordings" / source_name

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


