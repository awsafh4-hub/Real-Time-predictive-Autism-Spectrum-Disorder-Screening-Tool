"""
demo_live.py - the live demo window: opens the webcam and shows demo.py's iris tracking (same drawing as demo.py),
and records every frame's x,y coordinates to a CSV. Press Esc (or close the window) to stop and save.

MUST be run with demo.py's own .venv Python; the dashboard does that for you. Does not modify demo.py.

    <venv python> demo_live.py --demo-dir <MediaPipe-Iris> --out session.csv [--seconds 30] [--calibration cal.json]

stdout protocol (read by the dashboard):  "REC <elapsed_s> <frames>" about twice a second, then "DONE <csv>" / "ERROR <msg>".
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from demo_track import COLUMNS, IrisTracker

WINDOW = "demo.py live  -  Esc = stop and save"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop by itself after this long (0 = only on Esc)")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the picture (demo.py always flips)")
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--ec-dir", type=Path, default=None, help="eye_contact.py folder: also detect eye contact (needs its calibration.json)")
    ap.add_argument("--tolerance", type=float, default=0.5, help="eye-contact tolerance (lower = stricter)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    a = ap.parse_args()

    # ORDER MATTERS: the eye-contact model (MediaPipe "tasks" API) must be created BEFORE demo.py's legacy face-mesh.
    # The old MediaPipe in demo.py's .venv sets a resource folder when the face-mesh starts, after which it mangles the
    # absolute Windows path of the tasks model ("Unable to open file at ...site-packages/C:\...").
    eye = None
    if a.ec_dir:
        from demo_track import EyeContactTracker
        eye = EyeContactTracker(a.ec_dir, a.tolerance)
    tracker = IrisTracker(a.demo_dir, a.calibration)
    d = tracker.demo
    cap = cv2.VideoCapture(a.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        print("ERROR cannot open the webcam (is another program using it?)", flush=True)
        sys.exit(2)
    fps_calc = d.CvFpsCalc(buffer_len=10)                     # demo.py's own FPS counter
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, a.width, a.height)               # open big enough to show an audience (default is tiny)
    cv2.moveWindow(WINDOW, 40, 30)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    frames, t0, last_report, shown = 0, time.monotonic(), 0.0, False
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        while True:
            fps = fps_calc.get()
            elapsed = time.monotonic() - t0
            if a.seconds and elapsed >= a.seconds:
                break
            ok, img = cap.read()
            if not ok:
                break
            if not a.no_mirror:
                img = cv2.flip(img, 1)                        # mirror display, exactly like demo.py
            h, wd = img.shape[:2]
            row, draw = tracker.process(img)
            ec_out = None
            if eye:
                ec_row, ec_out = eye.process(img, elapsed)        # the frame is untouched so far: nothing drawn on it yet
                row.update(ec_row)
            w.writerow(dict(row, frame=frames, timestamp_s=round(elapsed, 3), frame_w=wd, frame_h=h))
            frames += 1

            shown_img = img.copy()
            if draw:
                shown_img = d.draw_debug_image(shown_img, *draw)  # demo.py's own circles + x,y labels
            if ec_out:                                            # big status bar along the bottom
                if not ec_out["face"]:
                    label, col = "NO FACE", (130, 130, 130)
                elif ec_out["contact"]:
                    label, col = "EYE CONTACT", (0, 170, 0)
                else:
                    label, col = "LOOKING AWAY" + (f" - {ec_out['direction']}" if ec_out["direction"] else ""), (0, 0, 210)
                cv2.rectangle(shown_img, (0, h - 56), (wd, h), col, -1)
                cv2.putText(shown_img, label, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(shown_img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            rec = f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}  {frames} frames"
            (tw, _), _ = cv2.getTextSize(rec, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.circle(shown_img, (wd - tw - 40, 25), 9, (0, 0, 255), -1)      # right-aligned, never clipped
            cv2.putText(shown_img, rec, (wd - tw - 20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if not draw:
                cv2.putText(shown_img, "no face found", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, shown_img)
            if not shown:                                     # make sure the window is in front, not behind the browser
                try:
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
                shown = True

            if elapsed - last_report >= 0.5:
                print(f"REC {elapsed:.1f} {frames}", flush=True)
                last_report = elapsed
            if cv2.waitKey(1) == 27:                          # Esc
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # window closed with the X button
                break
    cap.release()
    if eye:
        eye.close()
    cv2.destroyAllWindows()
    if frames == 0:
        print("ERROR no frames were recorded", flush=True)
        sys.exit(3)
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()
