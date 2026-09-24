"""
Real-Time ASD Screening Tool  (layout copied from the FYP dashboard).

    python -m streamlit run demo_dashboard.py

Live: opens the webcam window with demo.py's iris tracking + eye-contact detection (eye_contact.py's calibrated
detector), records everything to a CSV, and builds the eye-contact metrics and iris x,y graphs from that CSV.

"""
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))      # so `import demo_dash_lib` works however this is launched
import demo_dash_lib as lib


def line(d, cols, ylabel):
    """Line chart whose y axis fits the data (Streamlit's own line_chart always includes 0, which flattens pixel positions)."""
    long = d.melt(id_vars="timestamp_s", value_vars=cols, var_name="series", value_name="value").dropna()
    st.altair_chart(
        alt.Chart(long).mark_line().encode(
            x=alt.X("timestamp_s:Q", title="Time (s)"),
            y=alt.Y("value:Q", title=ylabel, scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(range=["#38bdf8", "#f59e0b"])),      # first series (x) blue, second (y) amber
        ), width="stretch")

def ec_timeline(df, bouts, duration):
    """Green = eye contact, red = not (looking away / no face)."""
    # A step-area of 1 (inside a bout) / 0 on a red background. Built from the same bouts as the metrics above.
    t = df["timestamp_s"].to_numpy()
    inside = pd.Series(0, index=range(len(t)))
    for b in bouts:
        inside[(t >= b["start_s"]) & (t <= b["end_s"])] = 1
    dense = pd.DataFrame({"time": t, "contact": inside.to_numpy()})
    chart = (alt.Chart(dense).mark_area(interpolate="step-after", color="#22c55e", opacity=1).encode(
                x=alt.X("time:Q", title="Time (s)", scale=alt.Scale(domain=[0, duration], nice=False)),
                y=alt.Y("contact:Q", scale=alt.Scale(domain=[0, 1]), axis=None))
             .properties(height=130).configure_view(fill="#b91c1c", strokeWidth=0))   # Streamlit fits axes INSIDE this height
    st.altair_chart(chart, width="stretch")


def ec_score_chart(df):
    d = df[df["ec_face"] == 1].assign(score=lambda x: (x["ec_nx"] ** 2 + x["ec_ny"] ** 2) ** 0.5)
    tol = float(df["ec_tol"].dropna().iloc[0]) if df["ec_tol"].notna().any() else 0.5
    line_ = alt.Chart(d[["timestamp_s", "score"]]).mark_line(color="#38bdf8").encode(
        x=alt.X("timestamp_s:Q", title="Time (s)"), y=alt.Y("score:Q", title="Distance from eye contact"))
    rule = alt.Chart(pd.DataFrame({"y": [tol]})).mark_rule(color="#ef4444", strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart(line_ + rule, width="stretch")


st.set_page_config(page_title="Real-Time ASD Screening Tool", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .metric-box { background: #1e293b; border-radius: 12px; padding: 16px 20px; text-align: center; border: 1px solid #334155; }
  .metric-val  { font-size: 1.6rem; font-weight: 700; color: #38bdf8; white-space: nowrap; }
  .metric-label{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
  .stProgress > div > div { background-color: #22c55e; }
</style>
""", unsafe_allow_html=True)

st.title("Real-Time ASD Screening Tool")
st.caption("Live eye-contact measurement with iris tracking (demo.py).")
st.info("**Research prototype.** It measures eye-contact behaviour only. It does not diagnose autism and has not been "
        "clinically validated - do not use it to make any medical decision.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    demo_dir = lib.DEFAULT_DEMO_DIR
    env = lib.environment_status(demo_dir)
    if not (env["demo"] and env["python"]):
        st.error("Tracker not available on this machine.")
    smooth = st.slider("Display smoothing (frames)", 1, 15, 1, help="Only smooths the charts. The numbers and CSV stay raw.")
    use_cal = st.checkbox("Also compute calibrated angles", value=env["calibration"], disabled=not env["calibration"],
                          help="Uses demo.py's gaze_calibration.json to add an estimated angle in degrees.")
    st.divider()
    st.subheader("Eye contact")
    ec_dir = lib.DEFAULT_EC_DIR
    ec = lib.ec_status(ec_dir)
    if ec["found"] and not ec["calibrated"]:
        st.warning("Not calibrated yet.")
    tolerance = st.slider("Eye-contact tolerance", 0.30, 0.80, 0.50, 0.05,
                          help="How far from your calibrated 'looking at the centre' still counts as eye contact. "
                               "Lower = stricter. Applied while recording.")
    if st.button("Calibrate eye contact (about 45 s)", disabled=not ec["found"],
                 help="Do this once per person and camera. A fullscreen window opens: follow the dots."):
        with st.spinner("Calibration window is open on your screen - follow the dots ..."):
            done = lib.run_ec_calibration(ec_dir)
        st.success("Calibration saved.") if done else st.warning("No new calibration was saved (cancelled?).")
        st.rerun()
    st.divider()
    st.subheader("Input")
    LIVE = "Live demo (webcam window)"
    mode = st.radio("Source", [LIVE, "Upload video file", "Load existing CSV"])
    mirror = st.checkbox("Mirror the picture (like the demo window)", value=(mode == LIVE), key=f"mirror_{mode}",
                         help="Live webcam: on. A normal video file: off, so left stays left.")

cal_path = (demo_dir / "gaze_calibration.json") if (use_cal and env["calibration"]) else None


def analyse(source, label, seconds=0.0):
    lib.SESSIONS.mkdir(exist_ok=True)
    out = lib.SESSIONS / f"{Path(label).stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    prog, stat = st.progress(0.0), st.empty()
    stat.text("Starting the tracker ...")
    with st.spinner("Running demo.py's iris tracker ..."):
        ok, log = lib.run_extractor(source, out, demo_dir, seconds=seconds, mirror=mirror, calibration=cal_path,
                                    on_progress=lambda f: (prog.progress(min(f, 1.0)), stat.text(f"{f * 100:.0f}%")))
    if not ok:
        st.error("The tracker failed.")
        st.code(log or "(no output)")
        return
    prog.progress(1.0)
    stat.text("Done!")
    st.session_state.update(df=lib.load_session(out), name=out.name, path=str(out))


def live(limit):
    """Step 1: open the real demo window. Step 2: record it to a CSV. Step 3 (below): build metrics + graphs from the CSV."""
    lib.SESSIONS.mkdir(exist_ok=True)
    out = lib.SESSIONS / f"live_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    box = st.status("Opening the demo window ... (look for it on your screen, it can take a few seconds)", expanded=True)
    line = box.empty()
    ok, log = lib.run_live(
        out, demo_dir, seconds=limit, mirror=mirror, calibration=cal_path,
        ec_dir=ec_dir if (ec["found"] and ec["calibrated"]) else None, tolerance=tolerance,
        on_status=lambda el, n: line.markdown(f"**Recording** - {el:.0f} s, {n} frames.  Press **Esc** in the demo window to stop and save."))
    if not ok:
        box.update(label="The live demo failed", state="error")
        box.code(log or "(no output)")
        return
    box.update(label=f"Saved {out.name} - the metrics and graphs below are built from that file", state="complete", expanded=False)
    st.session_state.update(df=lib.load_session(out), name=out.name, path=str(out))


# ── Main area ─────────────────────────────────────────────────────────────────
if mode == LIVE:
    st.markdown("**How this works**  \n"
                "1. Press **Start live demo**: a window opens with your webcam, demo.py's iris tracking and a live "
                "**EYE CONTACT / LOOKING AWAY** bar.  \n"
                "2. Look at the camera and move your eyes. Press **Esc** in that window to stop.  \n"
                "3. Every frame is saved to a CSV file, and the eye-contact metrics and graphs below are built from that file.")
    if not (ec["found"] and ec["calibrated"]):
        st.warning("Eye-contact calibration is missing, so the live demo will show iris x,y only. "
                   "Use **Calibrate eye contact** in the sidebar first (once per person and camera).")
    limit = st.slider("Stop automatically after (seconds, 0 = only when I press Esc)", 0, 300, 0)
    if st.button("Start live demo", type="primary"):
        live(limit)

elif mode == "Upload video file":
    up = st.file_uploader("Upload a video file (MP4, AVI, MOV ...)", type=["mp4", "avi", "mov", "mkv"])
    if up:
        st.info(f"Uploaded: **{up.name}**")
        if st.button("Run iris x,y analysis", type="primary"):
            # Written only when the button is pressed (a click reruns the script), and via getvalue(), which does not
            # consume the upload the way read() does.
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
                tmp.write(up.getvalue())
            try:
                analyse(tmp.name, up.name)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

else:
    files = sorted(lib.SESSIONS.glob("*.csv"), reverse=True) if lib.SESSIONS.exists() else []
    if files:
        chosen = st.selectbox("Saved session", files, format_func=lambda p: p.name)
        if st.button("Load", type="primary"):
            try:
                st.session_state.update(df=lib.load_session(chosen), name=chosen.name, path=str(chosen))
            except ValueError as e:
                st.error(str(e))
    else:
        st.warning("No saved sessions yet. Start the live demo or upload a video first.")
    csv_up = st.file_uploader("...or upload a session CSV", type=["csv"])
    if csv_up:
        try:
            st.session_state.update(df=lib.load_session(csv_up), name=csv_up.name, path=None)
        except ValueError as e:
            st.error(str(e))

# ── Results ───────────────────────────────────────────────────────────────────
if "df" in st.session_state:
    df, name = st.session_state["df"], st.session_state.get("name", "session")
    st.divider()
    st.subheader(f"Results - {name}")
    if st.session_state.get("path"):
        st.caption(f"Built from the CSV file: {st.session_state['path']}")

    def boxes(metrics):
        for col, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{value}</div><div class="metric-label">{label}</div></div>',
                            unsafe_allow_html=True)
        st.markdown("")

    if lib.has_eye_contact(df):
        st.subheader("Eye contact")
        em, bouts, dur = lib.eye_contact_metrics(df)
        boxes(em)
        st.caption("Timeline: green = eye contact, red = looking away or no face.")
        ec_timeline(df, bouts, dur)
        st.subheader("Distance from eye contact over time")
        st.caption("0 = looking exactly at your calibrated centre; 1 = as far off as your calibrated left/right/up/down dot. "
                   "Below the red line counts as eye contact.")
        ec_score_chart(df)
        breakdown = lib.away_breakdown(df)
        if len(breakdown):
            st.subheader("Where the gaze went when it was not eye contact")
            st.caption("Share of the looking-away time, by direction (%).")
            st.bar_chart(breakdown, horizontal=True)
        st.divider()
    else:
        st.info("This session has no eye-contact data. Eye contact is measured in **Live demo** sessions, because it needs "
                "the person's own calibration.")

    st.subheader("Iris tracking (demo.py)")
    boxes(lib.session_metrics(df))

    if (df["face"] == 1).sum() == 0:
        st.warning("No face was found in this session, so there is nothing to plot. Check the light and that the face is in view.")
    else:
        d = lib.smoothed(df, ["x", "y", "ratio_x", "ratio_y", "angle_x", "angle_y"], smooth)

        st.subheader("Iris position over time (pixels)")
        st.caption("Centre of the two irises in the frame. x grows to the right, y grows downward (image coordinates).")
        line(d, ["x", "y"], "Pixels")

        st.subheader("Gaze offset over time")
        st.caption("Iris offset from the middle of the eye, in half-eye-widths (demo.py's own ratio). "
                   "x: negative = toward the frame's left. y: negative = up.")
        line(d, ["ratio_x", "ratio_y"], "Offset ratio")

        if "angle_x" in d and d["angle_x"].notna().any():
            st.subheader("Calibrated angle estimate (degrees)")
            st.caption("From gaze_calibration.json: 15 deg = the calibrated LEFT/RIGHT/UP/DOWN dot. "
                       "An estimate, not a measured camera angle.")
            line(d, ["angle_x", "angle_y"], "Degrees")

        st.subheader("Gaze path")
        st.caption("Every frame as a point, darker = earlier. Right = looking toward the frame's right, up = looking up.")
        path = pd.DataFrame({"horizontal (right +)": d["ratio_x"], "vertical (up +)": -d["ratio_y"], "time (s)": d["timestamp_s"]}).dropna()
        st.scatter_chart(path, x="horizontal (right +)", y="vertical (up +)", color="time (s)")

    with st.expander("Raw frame data"):
        st.dataframe(df.tail(200), width="stretch")
    st.download_button("Download session CSV", data=df.to_csv(index=False).encode(), file_name=name, mime="text/csv")
    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **Eye contact** - decided by the calibrated detector (eye_contact.py): your iris position and head angle, "
            "compared with the dots you looked at during calibration. Lowered or closed eyes for over 0.4 s count as "
            "*not* eye contact. **EC Frequency** = bouts per minute, **EC Duration Rate** = share of the session in eye "
            "contact, **Mean Bout Length** = average length of one bout.\n"
            "- **Iris position (px)** - where demo.py finds the iris centre in the picture. It works in whole pixels, "
            "so about 1 px of jitter is normal.\n"
            "- **Jitter** - the typical frame-to-frame move of that point (neighbouring frames only).\n"
            "- **Gaze offset** - iris position relative to the eye box, scaled by eye width. It is not an angle.\n"
            "- **Calibrated angle** - only if you ran `demo.py --calibrate`; it is relative to the dots you looked at.\n"
            "- Vertical readings are less reliable than horizontal ones (the eyelid hides the iris when you look down)."
        )
else:
    st.info("Upload a video, record from the webcam, or load a saved session from the sidebar to begin.")
