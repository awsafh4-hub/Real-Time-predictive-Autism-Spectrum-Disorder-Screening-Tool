# Real-Time Predictive Autism Spectrum Disorder Screening Tool

A Streamlit dashboard that measures **eye-contact behaviour** from a webcam or a video file using
MediaPipe iris tracking, and reports frequency, duration rate, bout length and a green/red timeline.

> **Research prototype.** It measures eye-contact behaviour only. It does not diagnose autism and has
> not been clinically validated - do not use it to make any medical decision.

It runs **locally only**: it needs a webcam and it opens native OpenCV windows, so it cannot be hosted
on Streamlit Cloud or any other server.

## Layout

```
<this repo>/
  demo_dashboard/         the dashboard (Streamlit) + the tracking/extraction scripts
  eye_contact_portable/   standalone eye-contact detector + its calibration routine
  patches/                changes + requirements for the third-party iris demo (see below)
  MediaPipe-Iris/         NOT in this repo - clone it here yourself (step 1)
```

## Setup

1. **Iris tracker** (third-party, not redistributed here):

   ```
   git clone https://github.com/noorkhokhar99/MediaPipe-Iris.git
   cd MediaPipe-Iris
   git apply ../patches/demo.py.patch          # per-eye tracking + gaze calibration used by the dashboard
   python -m venv .venv && .venv\Scripts\pip install -r ../patches/mediapipe-iris-requirements.txt
   cd ..
   ```

   The dashboard runs the tracking scripts with **this** `.venv` (it needs the legacy MediaPipe API).
   Set `DEMO_DIR` / `EC_DIR` if you put the folders somewhere else.

2. **Face landmarker model** - nothing to do: `eye_contact.py` downloads
   `face_landmarker.task` (3.6 MB, Google) into `eye_contact_portable/` on first run.

3. **Dashboard**:

   ```
   pip install -r demo_dashboard/requirements.txt
   cd demo_dashboard
   python -m streamlit run demo_dashboard.py
   ```

Calibrate once per person and camera from the sidebar, then press **Start live demo**.
See [demo_dashboard/README.md](demo_dashboard/README.md) for what each chart means.

## Privacy

Recorded sessions (`demo_dashboard/sessions/*.csv`) and calibration files hold per-person eye data and
are git-ignored on purpose. Keep them out of the repo.
