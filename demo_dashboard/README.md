# Real-Time ASD Screening Tool (dashboard for demo.py + eye_contact.py)

Live eye-contact measurement with iris tracking. **Research prototype: it measures eye-contact behaviour only. It does
not diagnose autism and has not been clinically validated.**

```
cd demo_dashboard
python -m streamlit run demo_dashboard.py
```

## The demo flow (default source: "Live demo")
1. (Once per person and camera) sidebar -> **Calibrate eye contact** (~45 s, follow the dots).
2. Press **Start live demo**: a window opens with your webcam, demo.py's iris tracking (circles, x,y labels, FPS) and a big
   **EYE CONTACT / LOOKING AWAY** bar.
3. Look at the camera / look around. Press **Esc** in that window to stop (or set an automatic stop time).
4. Every frame is saved to a CSV in `sessions\`, and the dashboard builds everything **from that file** (its path is shown):
   - **Eye contact**: EC frequency (bouts/min), EC duration rate, mean bout length, total EC time, number of bouts,
     a green/red timeline, distance-from-eye-contact over time, and where the gaze went when it was not eye contact.
   - **Iris tracking (demo.py)**: iris x,y position, gaze offset, calibrated angles, gaze path.
   The CSV can be downloaded.

Other sources: upload a video (iris x,y only - eye contact needs the person's own calibration) or load a saved session.

## Settings worth knowing
- **Eye-contact tolerance** (default 0.5): how far from your calibrated centre still counts as eye contact. Lower = stricter.
  0.5 was chosen on one recording, so treat it as a starting point.
- Lowered or closed eyes for more than 0.4 s count as *not* eye contact (looking down at a keyboard, for example).
  The standalone `eye_contact.py` does not have this rule; it lives in `demo_track.py`.

## Files
| file | runs in | what it does |
|---|---|---|
| `demo_dashboard.py`, `demo_dash_lib.py` | your normal Python | the dashboard and its helpers |
| `demo_live.py` | demo.py's `.venv` | the live webcam window + CSV recording |
| `demo_extract.py` | demo.py's `.venv` | video file -> CSV, no window |
| `demo_track.py` | demo.py's `.venv` | demo.py's pipeline + the eye-contact detector, one frame -> one CSV row |

demo.py, eye_contact.py and the FYP folder are never modified; nothing is installed into any environment (the tracker
runs inside demo.py's own `.venv` through a subprocess). Other locations: set `DEMO_DIR` / `EC_DIR` or edit the sidebar boxes.
