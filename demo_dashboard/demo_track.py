"""
demo_track.py - demo.py's iris pipeline as a reusable piece: one frame in, one CSV row (plus the drawing data) out.

Shared by demo_extract.py (video files) and demo_live.py (the live webcam window). MUST be run with demo.py's own
.venv Python (it needs the legacy MediaPipe API). Imports demo.py from its folder; never modifies it.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

EC_COLUMNS = ["ec_face", "ec_contact", "ec_dir", "ec_nx", "ec_ny", "ec_hy", "ec_hp", "ec_ear", "ec_tol"]
COLUMNS = ["frame", "timestamp_s", "face", "frame_w", "frame_h",
           "lx", "ly", "rx", "ry", "lr", "rr", "x", "y",
           "ratio_x", "ratio_y", "angle_x", "angle_y"] + EC_COLUMNS


class EyeContactTracker:
    """eye_contact.py's calibrated eye-contact detector, run on the same frames. eye_contact.py itself is NOT modified.

    One addition over the standalone tool: an 'eyes down / closed' limit. A blink only counts for 0.4 s; longer than
    that means the lids are lowered or closed (e.g. looking down at a keyboard), which is not eye contact. The
    standalone tool froze on its previous answer there, so looking down was never detected."""

    def __init__(self, ec_dir: Path, tolerance: float = 0.5):
        ec_dir = Path(ec_dir)
        sys.path.insert(0, str(ec_dir))
        import eye_contact as ec
        self.ec, self.tolerance = ec, tolerance
        cal = json.loads((ec_dir / "calibration.json").read_text(encoding="utf-8"))
        self.tracker = ec.GazeTracker(ec_dir / "face_landmarker.task")

        class BlinkLimited(ec.ContactClassifier):
            def __init__(s, *a, max_blink_s=0.4, **k):
                super().__init__(*a, **k)
                s.max_blink_s, s._blink_t0 = max_blink_s, None

            def update(s, f, t):
                if f is not None and f.ear < s.blink_thr:
                    if s._blink_t0 is None:
                        s._blink_t0 = t
                    if t - s._blink_t0 > s.max_blink_s:          # too long to be a blink: eyes down or closed
                        s._misses = 0
                        s._push(False, t)
                        s.last = dict(s.last, direction="EYES DOWN/CLOSED")
                        return dict(face=True, contact=s.state, blink=False, **s.last)
                else:
                    s._blink_t0 = None
                return super().update(f, t)

        self.clf = BlinkLimited(cal, tolerance)

    def process(self, img, t):
        """-> (row of ec_* CSV columns, classifier output dict for drawing). t = seconds since the recording started."""
        f = self.tracker.process(img, (time.monotonic() - self.ec.CLOCK0) * 1000)
        out = self.clf.update(f, t)
        row = dict(ec_face=int(out["face"]), ec_contact=int(out["contact"]), ec_dir=out["direction"] if not out["contact"] else "",
                   ec_nx=round(out["nx"], 3), ec_ny=round(out["ny"], 3), ec_tol=self.tolerance)
        if f:
            row.update(ec_hy=round(f.hy, 2), ec_hp=round(f.hp, 2), ec_ear=round(f.ear, 3))
        return row, out

    def close(self):
        self.tracker.close()


class IrisTracker:
    def __init__(self, demo_dir: Path, calibration_path=None):
        os.chdir(demo_dir)                                   # demo.py's modules load their models relative to its folder
        sys.path.insert(0, str(demo_dir))
        spec = importlib.util.spec_from_file_location("demo", demo_dir / "demo.py")
        self.demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.demo)
        from face_mesh.face_mesh import FaceMesh
        from iris_landmark.iris_landmark import IrisLandmark
        self.face_mesh, self.iris = FaceMesh(1, 0.7, 0.7), IrisLandmark()
        self.cal = self.demo.load_calibration(str(calibration_path)) if calibration_path else None

    def process(self, img):
        """-> (row, draw). row has face=0/1 plus coordinates when a face was found; draw is the tuple that
        demo.draw_debug_image() expects after the image, or None."""
        row, draw = dict(face=0), None
        try:
            results = self.face_mesh(img)
            if results:
                d = self.demo
                le, re_ = self.face_mesh.calc_around_eye_bbox(results[0])
                li, ri = d.detect_iris(img, self.iris, le, re_)
                lc, lr = d.calc_min_enc_losingCircle(li)
                rc, rr = d.calc_min_enc_losingCircle(ri)
                g = d.calculate_gaze_info(le, re_, lc, rc, 10.0, self.cal, 10.0)
                row = dict(face=1, lx=lc[0], ly=lc[1], rx=rc[0], ry=rc[1], lr=lr, rr=rr,
                           x=round((lc[0] + rc[0]) / 2, 1), y=round((lc[1] + rc[1]) / 2, 1),
                           ratio_x=round(g["raw_ratio_x"], 4), ratio_y=round(g["raw_ratio_y"], 4))
                if self.cal:
                    row.update(angle_x=round(g["angle_x"], 2), angle_y=round(g["angle_y"], 2))
                draw = (li, ri, lc, lr, rc, rr)
        except Exception:
            pass                                             # eye crop fell off the frame etc. -> counts as "no face"
        return row, draw
