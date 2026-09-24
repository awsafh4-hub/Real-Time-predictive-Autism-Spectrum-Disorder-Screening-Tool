"""Helpers for the demo.py dashboard: find the demo + its Python, run the extractor, load a session, compute metrics.
No Streamlit in here, so it can be tested on its own."""
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SESSIONS = HERE / "sessions"
DEFAULT_DEMO_DIR = Path(os.environ.get("DEMO_DIR", HERE.parent / "MediaPipe-Iris"))
DEFAULT_EC_DIR = Path(os.environ.get("EC_DIR", HERE.parent / "eye_contact_portable"))
REQUIRED = ["frame", "timestamp_s", "face", "x", "y", "ratio_x", "ratio_y"]


def ec_status(ec_dir: Path) -> dict:
    return dict(found=(ec_dir / "eye_contact.py").exists(), calibrated=(ec_dir / "calibration.json").exists())


def run_ec_calibration(ec_dir: Path, timeout=600):
    """Runs eye_contact.py's ~45 s calibration (fullscreen, on the user's screen) in the normal Python, then exits.
    Returns True if a new calibration.json was written."""
    cal = ec_dir / "calibration.json"
    before = cal.stat().st_mtime if cal.exists() else 0
    subprocess.run([sys.executable, str(ec_dir / "eye_contact.py"), "--recalibrate", "--max-seconds", "2"],
                   cwd=str(ec_dir), timeout=timeout, capture_output=True)
    return cal.exists() and cal.stat().st_mtime > before


def has_eye_contact(df: pd.DataFrame) -> bool:
    return "ec_contact" in df.columns and df["ec_contact"].notna().any()


def detect_bouts(df: pd.DataFrame, gap=3):
    """Eye-contact bouts, same rule as the FYP dashboard: a bout ends after more than `gap` consecutive non-contact frames."""
    contacts = df["ec_contact"].fillna(0).values
    times = df["timestamp_s"].values
    bouts, in_bout, gap_count, start_s = [], False, 0, 0.0
    for i, (c, t) in enumerate(zip(contacts, times)):
        if c == 1:
            if not in_bout:
                start_s, in_bout = t, True
            gap_count = 0
        elif in_bout:
            gap_count += 1
            if gap_count > gap:
                bouts.append({"start_s": start_s, "end_s": times[i - gap_count], "duration_s": times[i - gap_count] - start_s})
                in_bout, gap_count = False, 0
    if in_bout:
        bouts.append({"start_s": start_s, "end_s": times[-1], "duration_s": times[-1] - start_s})
    return bouts


def eye_contact_metrics(df: pd.DataFrame):
    """Same six measures as the FYP dashboard. Returns (display dict, bouts, duration_s)."""
    dur = float(df["timestamp_s"].max())
    bouts = detect_bouts(df)
    ec_s = sum(b["duration_s"] for b in bouts)
    m = {
        "EC Frequency": f"{len(bouts) / (dur / 60):.1f} /min" if dur else "-",
        "EC Duration Rate": f"{ec_s / dur * 100:.0f}%" if dur else "-",
        "Mean Bout Length": f"{ec_s / len(bouts):.1f} s" if bouts else "0 s",
        "Total EC Time": f"{ec_s:.1f} s",
        "N Bouts": str(len(bouts)),
        "Session Duration": f"{dur:.1f} s",
    }
    return m, bouts, dur


def away_breakdown(df: pd.DataFrame) -> pd.Series:
    """Share of the NOT-eye-contact time spent looking each way (frames with a face only)."""
    away = df[(df["ec_face"] == 1) & (df["ec_contact"] == 0)]
    if len(away) == 0:
        return pd.Series(dtype=float)
    return (away["ec_dir"].fillna("other").replace("", "other").value_counts(normalize=True) * 100).round(0)


def venv_python(demo_dir: Path):
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        p = demo_dir / rel
        if p.exists():
            return p
    return None


def environment_status(demo_dir: Path) -> dict:
    return dict(
        demo=(demo_dir / "demo.py").exists(),
        python=venv_python(demo_dir) is not None,
        calibration=(demo_dir / "gaze_calibration.json").exists(),
    )


def _run(script: str, args: list, out_csv: Path, demo_dir: Path, on_line=None):
    """Runs one of our helper scripts inside the demo's own venv, streaming its stdout to on_line(line).
    Returns (ok, last_log_lines)."""
    py = venv_python(demo_dir)
    if py is None:
        return False, f"No .venv found in {demo_dir}. Set it up as described in the MediaPipe-Iris README."
    cmd = [str(py), str(HERE / script), "--out", str(out_csv), "--demo-dir", str(demo_dir), *args]
    tail = deque(maxlen=25)
    proc = subprocess.Popen(cmd, cwd=str(demo_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        handled = on_line(line.rstrip()) if on_line else False
        if not handled:
            tail.append(line.rstrip())
    rc = proc.wait()
    return (rc == 0 and out_csv.exists()), "\n".join(tail)


def run_extractor(source, out_csv: Path, demo_dir: Path, seconds=0.0, mirror=False, calibration=None, on_progress=None):
    """Video file (or timed webcam clip) -> CSV, no window."""
    args = ["--source", str(source)]
    if seconds:
        args += ["--seconds", str(seconds)]
    if mirror:
        args.append("--mirror")
    if calibration:
        args += ["--calibration", str(calibration)]

    def on_line(line):
        if line.startswith("PROGRESS "):
            try:
                on_progress and on_progress(float(line.split()[1]))
            except ValueError:
                pass
            return True
        return False
    return _run("demo_extract.py", args, out_csv, demo_dir, on_line)


def run_live(out_csv: Path, demo_dir: Path, seconds=0.0, mirror=True, calibration=None, on_status=None,
             ec_dir=None, tolerance=0.5):
    """Opens the live demo window (webcam + demo.py's iris tracking, plus eye-contact detection when ec_dir is given)
    and records to out_csv until the user presses Esc (or `seconds` elapse).
    on_status(elapsed_s, frames) is called about twice a second."""
    args = []
    if seconds:
        args += ["--seconds", str(seconds)]
    if not mirror:
        args.append("--no-mirror")
    if calibration:
        args += ["--calibration", str(calibration)]
    if ec_dir:
        args += ["--ec-dir", str(ec_dir), "--tolerance", str(tolerance)]

    def on_line(line):
        if line.startswith("REC "):
            try:
                _, el, n = line.split()
                on_status and on_status(float(el), int(n))
            except ValueError:
                pass
            return True
        return False
    return _run("demo_live.py", args, out_csv, demo_dir, on_line)


def load_session(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"This CSV is not a demo.py session (missing columns: {', '.join(missing)}).")
    return df


def smoothed(df: pd.DataFrame, cols, window: int) -> pd.DataFrame:
    """Display-only rolling mean. Metrics always use the raw numbers."""
    out = df.copy()
    if window > 1:
        for c in cols:
            if c in out:
                out[c] = out[c].rolling(window, min_periods=1, center=True).mean()
    return out


def session_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    dur = float(df["timestamp_s"].max()) if n else 0.0
    f = df[df["face"] == 1]
    m = {"Duration": f"{dur:.1f} s", "Frames": f"{n}", "Face found": f"{100 * len(f) / n:.0f}%" if n else "-"}
    if len(f) == 0:
        m.update({"Iris position (px)": "-", "Jitter (px/frame)": "-", "Offset spread (x / y)": "-"})
        return m
    m["Iris position (px)"] = f"{f['x'].mean():.0f}, {f['y'].mean():.0f}"
    consecutive = f["frame"].diff() == 1                       # only compare neighbouring frames
    step = np.hypot(f["x"].diff()[consecutive], f["y"].diff()[consecutive])
    m["Jitter (px/frame)"] = f"{step.median():.2f}" if len(step) else "-"
    m["Offset spread (x / y)"] = f"{f['ratio_x'].std():.2f} / {f['ratio_y'].std():.2f}"
    return m
