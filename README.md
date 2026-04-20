# Traffic Flow Analyzer

Uses Lucas-Kanade optical flow to track motion in traffic footage and classify road conditions as clear, moderate, or dense in real time.

## How it works

Each frame, Shi-Tomasi corner detection finds trackable feature points — edges, corners, anything with texture. Lucas-Kanade then tracks where those points moved in the next frame. Points that barely moved get filtered out (stationary background, parked cars, camera shake). Whatever's left is counted as active motion, and that count drives the traffic label.

One thing worth knowing: LK only tracks corners and edges, so smooth-sided vehicles without much surface texture can slip through undetected. The thresholds were calibrated for the test footage — different cameras and scenes will need retuning.

**Pipeline:**
1. Shi-Tomasi corner detection on each frame
2. Lucas-Kanade optical flow tracks points to the next frame
3. Motion magnitude filter removes near-stationary points
4. Moving point count classifies traffic as CLEAR / MODERATE / DENSE
5. Trails and live metrics render on screen

## Stack

- Python
- OpenCV — video capture, feature detection, optical flow
- NumPy — magnitude calculations

## Setup

```bash
git clone https://github.com/Srujankasturi/Traffic_Flow_Analyzer.git
cd Traffic_Flow_Analyzer
pip install opencv-python numpy
```

Drop your video in the folder, rename it `traffic.mp4`, then:

```bash
python optical_flow.py
```

Press `q` to quit.

## What's shown on screen

| Metric | What it means |
|--------|---------------|
| Moving points | How many tracked points are actively moving |
| Avg motion | Mean pixel displacement across tracked points |
| Traffic status | CLEAR / MODERATE / DENSE |

## Known limitations

LK optical flow misses vehicles with smooth, textureless surfaces — not enough corners to latch onto. The fix would be combining it with a detection model (YOLOv8) to locate vehicles first, then track them. The density thresholds also need manual calibration per camera — what counts as "dense" on a narrow street is different from a highway.

## Author

Srujan Kasturi — [GitHub](https://github.com/Srujankasturi)  
B.Tech CSE, SRM University AP