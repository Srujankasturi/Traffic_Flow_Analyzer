# Real-Time Traffic Flow Analyzer

A computer vision system that analyzes traffic density from video footage using Lucas-Kanade optical flow. Tracks motion across frames and classifies traffic conditions as Clear, Moderate, or Dense.

## How It Works

The system uses Lucas-Kanade sparse optical flow to track feature-rich points (corners, edges) across consecutive frames. Moving points are filtered from stationary background using a motion magnitude threshold, and the count of actively moving points is used to classify traffic density.

**Pipeline:**
1. Extract feature points using Shi-Tomasi corner detection
2. Track points frame-to-frame using Lucas-Kanade optical flow
3. Filter out stationary points using motion magnitude threshold
4. Classify traffic density based on moving point count
5. Display live metrics and colored motion trails on screen

## Features

- Real-time optical flow tracking with colored motion trails
- Motion magnitude filtering to remove camera shake and stationary background
- Live traffic density classification — Clear / Moderate / Dense
- Average motion magnitude display for flow speed estimation
- Automatic feature re-detection when tracked points are lost

## Tech Stack

- Python
- OpenCV — video capture, optical flow, feature detection
- NumPy — motion magnitude calculations

## Installation

```bash
git clone https://github.com/Srujankasturi/Traffic_Flow_Analyzer.git
cd Traffic_Flow_Analyzer
pip install opencv-python numpy
```

## Usage

```bash
python optical_flow.py
```

Place your video file in the project folder and rename it `traffic.mp4`. Press `q` to quit.

## Output Metrics

| Metric | Description |
|--------|-------------|
| Moving points | Count of actively tracked moving feature points |
| Avg motion | Mean displacement magnitude across all tracked points |
| Traffic status | CLEAR / MODERATE / DENSE classification |

## Limitations & Known Issues

- Lucas-Kanade tracks corners and edges — smooth-surfaced vehicles with low texture may not be detected
- Thresholds are calibrated per camera deployment and should be tuned for different scenes
- Minor camera shake can introduce noise — mitigated via motion magnitude filtering

## Use Case

Applicable to smart city infrastructure, CCTV-based traffic monitoring, and AI-powered operations insight platforms that require real-time situational awareness from video feeds.

## Author

Srujan Kasturi — [GitHub](https://github.com/Srujankasturi)  
B.Tech CSE, SRM University AP
