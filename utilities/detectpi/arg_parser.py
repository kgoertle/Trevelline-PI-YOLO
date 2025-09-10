import argparse
import sys
from pathlib import Path

def parse_arguments():
    """
    Parses command-line arguments for YOLO detection script (Pi version).
    Only one source allowed: 'picamera' or a video file path.
    """
    parser = argparse.ArgumentParser(description="Run YOLO detection using latest best.pt")
    parser.add_argument("--source", required=True, help="Input source: 'picamera' or path to video file")
    parser.add_argument("--lab", action="store_true", help="Print smoothing parameters for lab testing")
    parser.add_argument("--smooth", type=float, default=1.0, help="Smoothing alpha (0–1)")
    parser.add_argument("--dist-thresh", type=float, default=None, help="Distance threshold for smoothing")
    parser.add_argument("--max-history", type=int, default=0, help="Max history length for smoothing")
    
    args = parser.parse_args()

    # Validate source
    if args.source.lower() != "picamera":
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"[ERROR] Video file not found: {args.source}")
            sys.exit(1)

    return args
