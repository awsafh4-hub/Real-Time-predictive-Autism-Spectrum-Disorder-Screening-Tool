#!/usr/bin/env python3
"""
eye_contact.py - portable eye-contact detector (webcam or video file).

Runs anywhere Python 3.9+ runs:   pip install -r requirements.txt

    python eye_contact.py                       # webcam 0; calibrates the first time (~45 s)
    python eye_contact.py --recalibrate         # new camera / new seat / new lighting
    python eye_contact.py --source clip.mp4 --log out.csv --no-window   # uses saved calibration

Keys while running:  c = recalibrate   q / Esc = quit

How it decides (all measured per user + camera during calibration, nothing is hard-coded):
  * iris position inside each eye, measured along the eye axis (so head roll does not matter)
    and divided by eye width;
  * head yaw/pitch from MediaPipe's face transform. During the HEAD calibration step you keep
    your eyes on the dot while moving your head; that fits how much your eyes counter-rotate,
    and it is subtracted, so turning your head while still looking at the camera stays "contact";
  * every direction (left / right / up / down) has its OWN scale, so a weak side (camera off
    centre, a habit of turning the head) can still reach the threshold;
  * blinks are ignored, the decision is smoothed and needs a few frames to flip.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # much faster webcam start on Windows
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# MediaPipe FaceLandmarker: 468 face points + 5 iris points per eye.
EYES = (
    dict(iris=[468, 469, 470, 471, 472], c1=33, c2=133, up=159, lo=145),
    dict(iris=[473, 474, 475, 476, 477], c1=362, c2=263, up=386, lo=374),
)
ANCHORS = ("CENTER", "LEFT", "RIGHT", "UP", "DOWN")
CAL_VERSION = 2
CLOCK0 = time.monotonic()   # ONE clock for the tracker's timestamps: calibration and live use must not restart it


# --------------------------------------------------------------------------- features

@dataclass
class Feat:
    ex: float   # iris offset along the eye axis, in eye widths (image-right positive)
    ey: float   # iris offset across the eye axis, in eye widths (image-down positive)
    hy: float   # head yaw, degrees (sign is arbitrary but consistent)
    hp: float   # head pitch, degrees
    ear: float  # eye openness: lid gap / eye width, mean of both eyes

    def vec(self):
        return [self.ex, self.ey, self.hy, self.hp, self.ear]


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        print(f"Downloading face model (~4 MB) -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class GazeTracker:
    """One frame in, one Feat (or None if no face) out. Uses MediaPipe's tracking mode."""

    def __init__(self, model_path: Path = MODEL_PATH):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def close(self):
        if self._lm is not None:
            self._lm.close()
            self._lm = None

    def process(self, bgr: np.ndarray, ts_ms: float):
        ts = max(int(ts_ms), self._last_ts + 1)          # VIDEO mode needs strictly increasing ms
        self._last_ts = ts
        h, w = bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])  # pixels: fixes aspect ratio

        exs, eys, ears = [], [], []
        for e in EYES:
            c1, c2 = pts[e["c1"]], pts[e["c2"]]
            axis = c2 - c1
            width = float(np.hypot(*axis))
            if width < 4.0:                                  # face too small / degenerate
                return None
            u = axis / width
            if u[0] < 0:                                     # same image direction for both eyes
                u = -u
            v = np.array([-u[1], u[0]])                      # perpendicular, image-down positive
            off = pts[e["iris"]].mean(axis=0) - (c1 + c2) / 2.0
            exs.append(float(off @ u) / width)
            eys.append(float(off @ v) / width)
            ears.append(float(np.hypot(*(pts[e["up"]] - pts[e["lo"]]))) / width)

        hy = hp = 0.0
        mats = res.facial_transformation_matrixes
        if len(mats):
            m = np.asarray(mats[0])[:3, :3]
            r = m / (np.linalg.norm(m[:, 0]) + 1e-9)
            hy = math.degrees(math.atan2(r[0, 2], r[2, 2]))
            hp = math.degrees(math.asin(float(np.clip(-r[1, 2], -1.0, 1.0))))
        return Feat(float(np.mean(exs)), float(np.mean(eys)), hy, hp, float(np.mean(ears)))


# --------------------------------------------------------------------------- calibration maths

def _robust_std(a) -> float:
    a = np.asarray(a, float)
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) + 1e-9


def _head_gain(eye_off, head_dev, warn, name):
    """Slope of iris offset vs head angle while the eyes stay on the lens (0 if unreliable)."""
    if len(eye_off) < 30 or np.std(head_dev) < 3.0:
        warn.append(f"head {name}: head barely moved during the HEAD step - head compensation off")
        return 0.0
    var = float(np.var(head_dev))
    slope = float(np.mean((head_dev - head_dev.mean()) * (eye_off - eye_off.mean())) / var)
    r2 = float(np.corrcoef(head_dev, eye_off)[0, 1] ** 2)
    if r2 < 0.25:
        warn.append(f"head {name}: no clear link between head angle and iris offset (r2={r2:.2f}) "
                    "- head compensation off")
        return 0.0
    return float(np.clip(slope, -0.05, 0.05))


def fit_calibration(samples: dict, head=None, use_head=True, per_side=True) -> dict:
    """samples[name] -> N x 5 array of Feat.vec() while looking at that target.
    head -> N x 5 array recorded while moving the head with the eyes held on the centre dot."""
    S = {k: np.asarray(v, float) for k, v in samples.items()}
    ctr = S["CENTER"]
    c_hy, c_hp = float(np.median(ctr[:, 2])), float(np.median(ctr[:, 3]))
    warn = []
    gx = gy = 0.0
    if use_head and head is not None:
        H = np.asarray(head, float)
        gx = _head_gain(H[:, 0], H[:, 2] - c_hy, warn, "left/right")
        gy = _head_gain(H[:, 1], H[:, 3] - c_hp, warn, "up/down")

    def comp(A):
        return np.c_[A[:, 0] - gx * (A[:, 2] - c_hy), A[:, 1] - gy * (A[:, 3] - c_hp)]

    C = {k: np.median(comp(v), axis=0) for k, v in S.items()}
    cx, cy = C["CENTER"]
    cc = comp(ctr)
    noise = (_robust_std(cc[:, 0]), _robust_std(cc[:, 1]))

    def axis(neg_name, pos_name, i, min_snr, what):
        dn, dp = C[neg_name][i] - C["CENTER"][i], C[pos_name][i] - C["CENTER"][i]
        if dn < 0 < dp:
            neg, pos, nl, pl = -dn, dp, neg_name, pos_name
        elif dp < 0 < dn:
            neg, pos, nl, pl = -dp, dn, pos_name, neg_name
        else:
            warn.append(f"{what}: {neg_name} and {pos_name} moved the iris the same way - recalibrate")
            neg = pos = max(abs(dn), abs(dp))
            nl, pl = neg_name, pos_name
        if per_side:
            floor = 0.5 * max(neg, pos)         # one noisy side must not become hypersensitive
            neg, pos = max(neg, floor), max(pos, floor)
        else:
            neg = pos = (neg + pos) / 2.0
        valid = max(neg, pos) >= min_snr * noise[i]
        if not valid:
            warn.append(f"{what}: eye movement too small vs noise - {what} ignored "
                        "(sit closer, better light, or make the dots further apart)")
        return dict(neg=float(neg), pos=float(pos), neg_label=nl, pos_label=pl, valid=bool(valid))

    ax_x = axis("LEFT", "RIGHT", 0, 4.0, "horizontal")
    ax_y = axis("UP", "DOWN", 1, 3.0, "vertical")
    return dict(
        version=CAL_VERSION, c_ex=float(cx), c_ey=float(cy), c_hy=c_hy, c_hp=c_hp,
        gx=gx, gy=gy, ear0=float(np.median(ctr[:, 4])),
        x=ax_x, y=ax_y, noise=[float(noise[0]), float(noise[1])], warnings=warn,
    )


def gaze_scores(cal: dict, ex, ey, hy, hp):
    """Normalised offsets: 1.0 == exactly on the calibrated target for that direction."""
    dx = (ex - cal["gx"] * (hy - cal["c_hy"])) - cal["c_ex"]
    dy = (ey - cal["gy"] * (hp - cal["c_hp"])) - cal["c_ey"]
    nx = dx / (cal["x"]["pos"] if dx >= 0 else cal["x"]["neg"])
    ny = dy / (cal["y"]["pos"] if dy >= 0 else cal["y"]["neg"]) if cal["y"]["valid"] else 0.0
    return float(nx), float(ny)


class ContactClassifier:
    """Smooths features, ignores blinks, and changes its answer only after the new answer has held
    for `hold_ms` (and at least `min_frames` frames) - a time, not a frame count, so it is just as
    quick on a 15 fps webcam as on a 60 fps one."""

    def __init__(self, cal, tolerance=0.6, hold_ms=60.0, alpha=0.6, head_max=(40.0, 35.0), min_frames=2):
        self.cal, self.tol, self.alpha, self.head_max = cal, tolerance, alpha, head_max
        self.hold_s, self.min_frames = hold_ms / 1000.0, min_frames
        self.blink_thr = 0.6 * cal["ear0"]
        self.ema = None
        self.state = False
        self._pending, self._count, self._t_first, self._misses = None, 0, 0.0, 0
        self.last = dict(nx=0.0, ny=0.0, dist=0.0, direction="")

    def _push(self, raw: bool, t: float):
        if raw == self.state:
            self._pending, self._count = None, 0
            return
        if self._pending != raw:
            self._pending, self._count, self._t_first = raw, 0, t
        self._count += 1
        if self._count >= self.min_frames and t - self._t_first >= self.hold_s:
            self.state, self._pending, self._count = raw, None, 0

    def update(self, f, t: float):
        """f: Feat or None (no face); t: time of this frame in seconds."""
        if f is None:
            self._misses += 1
            if self._misses >= 3:
                self._push(False, t)
            return dict(face=False, contact=self.state, blink=False, **self.last)
        if self._misses > 10:
            self.ema = None
        self._misses = 0
        blink = f.ear < self.blink_thr
        if not blink:
            v = np.array(f.vec()[:4])
            self.ema = v if self.ema is None else self.ema + self.alpha * (v - self.ema)
            ex, ey, hy, hp = self.ema
            nx, ny = gaze_scores(self.cal, ex, ey, hy, hp)
            dist = math.hypot(nx, ny)
            head_far = (abs(hy - self.cal["c_hy"]) > self.head_max[0]
                        or abs(hp - self.cal["c_hp"]) > self.head_max[1])
            self._push(dist <= self.tol and not head_far, t)
            direction = ""
            if not self.state:
                if head_far:
                    direction = "HEAD TURNED"
                elif abs(nx) >= abs(ny):
                    direction = self.cal["x"]["pos_label" if nx > 0 else "neg_label"]
                else:
                    direction = self.cal["y"]["pos_label" if ny > 0 else "neg_label"]
            self.last = dict(nx=nx, ny=ny, dist=dist, direction=direction)
        return dict(face=True, contact=self.state, blink=blink, **self.last)


def save_cal(cal, path: Path, meta: dict):
    path.write_text(json.dumps({**cal, "meta": meta}, indent=2), encoding="utf-8")


def load_cal(path: Path):
    try:
        cal = json.loads(path.read_text(encoding="utf-8"))
        return cal if cal.get("version") == CAL_VERSION else None
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- camera + UI

def open_capture(src, width, height):
    if isinstance(src, int) and sys.platform == "win32":
        # MSMF gave 30 fps where DSHOW gave 15 fps on the same webcam - half the delay. DSHOW is the fallback.
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(src, backend)
            if cap.isOpened():
                break
            cap.release()
    else:
        cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)              # keep at most one frame queued in the driver
    return cap


def _text(img, s, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def run_calibration(cap, tracker, mirror, win, use_head=True):
    """Interactive ~45 s calibration. Returns (cal, samples) or None if aborted."""
    READY, MIN_N = 2.0, 25
    plan = [
        ("CENTER", 3.5, (0.50, 0.50), "Look at the CENTRE dot"),
        ("LEFT",   3.5, (0.06, 0.50), "Look at the LEFT dot - move only your eyes"),
        ("RIGHT",  3.5, (0.94, 0.50), "Look at the RIGHT dot - move only your eyes"),
        ("UP",     3.5, (0.50, 0.08), "Look at the TOP dot - move only your eyes"),
        ("DOWN",   3.5, (0.50, 0.92), "Look at the BOTTOM dot - move only your eyes"),
    ]
    if use_head:
        plan.append(("HEAD", 10.0, (0.50, 0.50),
                     "Keep your EYES on the centre dot, slowly turn/tilt your HEAD left, right, up, down"))
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def frame_iter():
        while True:
            ok, fr = cap.read()
            if not ok:
                return
            yield cv2.flip(fr, 1) if mirror else fr

    frames = frame_iter()
    # wait for the user
    for fr in frames:
        _text(fr, "Calibration: sit where you will normally sit, head still.", (20, 40))
        _text(fr, "Press SPACE to start (Esc to cancel)", (20, 80), (0, 255, 255))
        cv2.imshow(win, fr)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            break
        if k in (27, ord("q")):
            return None
    samples = {}
    for name, dur, (fx, fy), msg in plan:
        buf, t0, aborted = [], time.monotonic(), False
        for fr in frames:
            h, w = fr.shape[:2]
            el = time.monotonic() - t0
            sampling = el >= READY + 1.0
            f = tracker.process(fr, (time.monotonic() - CLOCK0) * 1000)
            if sampling and f is not None:
                buf.append(f.vec())
            if el >= READY + dur and len(buf) >= MIN_N:
                break
            if el > READY + 3 * dur and len(buf) < MIN_N:
                print(f"Calibration failed: face not seen during {name}. Check light / position.")
                return None
            cv2.circle(fr, (int(fx * w), int(fy * h)), 16, (0, 255, 255), -1)
            _text(fr, msg, (20, 40), (0, 255, 255))
            _text(fr, ("get ready..." if not sampling else "hold...") +
                  f"  {name}  {max(0.0, READY + dur - el):.0f}s   face:{'yes' if f else 'NO'}", (20, 80))
            cv2.imshow(win, fr)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                aborted = True
                break
        if aborted:
            return None
        samples[name] = np.array(buf)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    head = samples.pop("HEAD", None)
    return fit_calibration(samples, head, use_head), samples


def draw_overlay(fr, out, fps, cal):
    h, w = fr.shape[:2]
    if not out["face"]:
        label, col = "NO FACE", (160, 160, 160)
    elif out["contact"]:
        label, col = "EYE CONTACT", (0, 200, 0)
    else:
        label, col = "LOOKING AWAY" + (f" - {out['direction']}" if out["direction"] else ""), (0, 0, 230)
    cv2.rectangle(fr, (0, 0), (w, 44), col, -1)
    _text(fr, label, (10, 30), (255, 255, 255), 0.9, 2)
    _text(fr, f"x:{out['nx']:+.2f} y:{out['ny']:+.2f}  (1.0 = your calibrated target)  {fps:.0f} fps",
          (10, h - 12), (255, 255, 255), 0.5, 1)
    return fr


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Portable eye-contact detector")
    ap.add_argument("--source", default="0", help="camera index or video file (default 0)")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--calib-file", type=Path, default=HERE / "calibration.json")
    ap.add_argument("--tolerance", type=float, default=0.6,
                    help="fraction of the way to your calibrated LEFT/RIGHT/UP/DOWN target at which "
                         "it flips to 'away' (lower = stricter). default 0.6")
    ap.add_argument("--hold-ms", type=float, default=60.0,
                    help="how long a new answer must hold before it is shown (lower = faster but flickers more)")
    ap.add_argument("--no-head-comp", action="store_true", help="skip the head-movement calibration step")
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=None)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--log", type=Path, help="write a CSV of every frame")
    ap.add_argument("--no-window", action="store_true", help="no display (needs saved calibration)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after this many seconds (0 = run until quit)")
    a = ap.parse_args(argv)

    src = int(a.source) if a.source.isdigit() else a.source
    is_cam = isinstance(src, int)
    mirror = is_cam if a.mirror is None else a.mirror
    cap = open_capture(src, a.width, a.height)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {a.source!r}")
    tracker = GazeTracker()
    win = "eye contact (c = recalibrate, q = quit)"
    if not a.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cal = None if a.recalibrate else load_cal(a.calib_file)
    if cal is not None and cal.get("meta", {}).get("mirror") not in (None, mirror):
        print("Note: saved calibration was made with the opposite mirror setting - recalibrate if results look flipped.")

    def calibrate():
        if a.no_window:
            sys.exit("Calibration needs a window. Run once without --no-window.")
        res = run_calibration(cap, tracker, mirror, win, use_head=not a.no_head_comp)
        if res is None:
            return None
        c, _ = res
        save_cal(c, a.calib_file, dict(mirror=mirror, source=str(a.source), saved=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"Calibration saved -> {a.calib_file}")
        for wmsg in c["warnings"]:
            print("  WARNING:", wmsg)
        return c

    if cal is None:
        if not is_cam:
            sys.exit(f"No calibration at {a.calib_file}. Calibrate once with the webcam first.")
        cal = calibrate()
        if cal is None:
            sys.exit("Calibration cancelled.")

    clf = ContactClassifier(cal, a.tolerance, a.hold_ms)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log_f = writer = None
    if a.log:
        log_f = open(a.log, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_f)
        writer.writerow(["t_s", "face", "contact", "direction", "nx", "ny", "dist", "ex", "ey", "head_yaw", "head_pitch"])
    t0 = time.monotonic()
    t_first = None
    frame_i, fps, t_prev = 0, 0.0, time.monotonic()
    try:
        while True:
            ok, fr = cap.read()
            if not ok or (a.max_seconds and time.monotonic() - t0 > a.max_seconds):
                break
            if mirror:
                fr = cv2.flip(fr, 1)
            # webcam: the SAME clock calibration used, so the tracker's timestamps never jump backwards
            t_s = (time.monotonic() - CLOCK0) if is_cam else frame_i / fps_src
            t_first = t_s if t_first is None else t_first
            f = tracker.process(fr, t_s * 1000)
            out = clf.update(f, t_s)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            if writer:
                raw = [f"{v:.3f}" for v in (f.ex, f.ey, f.hy, f.hp)] if f else ["", "", "", ""]
                writer.writerow([f"{t_s - t_first:.3f}", int(out["face"]), int(out["contact"]), out["direction"],
                                 f"{out['nx']:.3f}", f"{out['ny']:.3f}", f"{out['dist']:.3f}", *raw])
            if not a.no_window:
                cv2.imshow(win, draw_overlay(fr, out, fps, cal))
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("c") and is_cam:
                    new = calibrate()
                    if new is not None:
                        cal = new
                        clf = ContactClassifier(cal, a.tolerance, a.hold_ms)
            frame_i += 1
    finally:
        cap.release()
        tracker.close()
        if log_f:
            log_f.close()
        cv2.destroyAllWindows()
    print(f"{frame_i} frames, {frame_i / max(time.monotonic() - t0, 1e-6):.1f} fps average")


if __name__ == "__main__":
    main()
