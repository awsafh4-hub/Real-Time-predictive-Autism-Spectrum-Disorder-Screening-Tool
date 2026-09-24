# eye_contact.py - portable eye-contact detector

One file. Windows, macOS, Linux. Python 3.9+.

```
pip install -r requirements.txt
python eye_contact.py
```

First run calibrates (about 45 seconds, fullscreen): look at five dots (centre, left, right, top,
bottom) moving only your eyes, then keep your eyes on the centre dot while slowly moving your head.
The result is saved to `calibration.json` next to the script. Calibrate again (`--recalibrate`, or
press `c`) when the camera, seat or lighting changes.

| option | what it does |
|---|---|
| `--recalibrate` | redo calibration |
| `--tolerance 0.6` | how far toward your calibrated target counts as still "contact". Lower = stricter |
| `--source clip.mp4 --log out.csv --no-window` | run on a video with the saved calibration and write a CSV |
| `--no-head-comp` | skip the head-movement step |
| `--width/--height` | camera resolution (default 640x480 - bigger is not more accurate, only slower) |

Why it is built this way
- Every direction has its own scale, so a weak side no longer stays inside the threshold.
- Head movement while looking at the camera is measured and subtracted, per person.
- Nothing is hard-coded: thresholds come from your own calibration on your own camera.
- Blinks are ignored and a new answer must hold for about 0.06 s (`--hold-ms`, at least 2 frames) before it is shown, so it does not flicker.
- Low delay: on Windows the faster MSMF camera backend is tried first (30 fps where DSHOW gave 10-15 fps on
  the same webcam), and the tracker uses one continuous clock so MediaPipe's temporal smoothing stays correct.
  If it still lags, the usual cause is a dim room: webcams cut their frame rate in low light.

The face model (`face_landmarker.task`) is downloaded automatically if it is missing.
