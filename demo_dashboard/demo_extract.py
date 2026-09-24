"""
demo_extract.py - runs demo.py's iris pipeline on a video file (or a timed webcam clip) and writes the x,y coordinates
to CSV. MUST be run with demo.py's own .venv Python; the dashboard does that for you. Does not modify demo.py.

    <venv python> demo_extract.py --source clip.mp4 --out out.csv --demo-dir <MediaPipe-Iris>

stdout protocol (read by the dashboard):  "PROGRESS <0..1>" lines, then "DONE <csv>" or "ERROR <message>".
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from demo_track import COLUMNS, IrisTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, or a camera index such as 0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--demo-dir", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=0.0, help="webcam only: stop after this many seconds")
    ap.add_argument("--mirror", action="store_true", help="flip the frame left-right, like the demo window")
    ap.add_argument("--calibration", type=Path, default=None, help="gaze_calibration.json (optional)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    a = ap.parse_args()

    tracker = IrisTracker(a.demo_dir, a.calibration)
    import cv2
    is_cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source)) if is_cam else cv2.VideoCapture(a.source)
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        print("ERROR cannot open the video / camera", flush=True)
        sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = 0 if is_cam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0, i = time.monotonic(), 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        while True:
            elapsed = time.monotonic() - t0
            if is_cam and a.seconds and elapsed >= a.seconds:
                break
            ok, img = cap.read()
            if not ok:
                break
            if a.mirror:
                img = cv2.flip(img, 1)
            h, wd = img.shape[:2]
            row, _ = tracker.process(img)
            w.writerow(dict(row, frame=i, timestamp_s=round(elapsed if is_cam else i / fps, 3), frame_w=wd, frame_h=h))
            i += 1
            if i % 10 == 0:
                frac = (i / n_frames) if n_frames else (elapsed / a.seconds if a.seconds else 0.0)
                print(f"PROGRESS {min(frac, 1.0):.3f}", flush=True)
    cap.release()
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()
