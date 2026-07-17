# Lane Detection — Classic Computer Vision Pipeline

Detects road lane lines in dashcam images using the classic CV pipeline
(no deep learning): grayscale → Gaussian blur → Canny edges → ROI mask →
Hough transform (in progress).

Built as a learning project: each stage is implemented, tuned and
documented by hand before moving to CNN-based approaches.

## Pipeline stages

| Stage | What it does | Why |
|-------|--------------|-----|
| Grayscale | Collapses 3 color channels to 1 intensity channel | Lanes are a brightness feature, not a color feature |
| Gaussian blur | Convolution with a 5x5 bell-curve kernel | Sensor noise and asphalt texture look like edges to a gradient detector |
| Canny | Gradient + non-max suppression + hysteresis thresholds | Finds pixels where brightness changes sharply; hysteresis rescues weak edges connected to strong ones |
| ROI mask | Zeroes everything outside a trapezoid via bitwise AND | The camera is fixed, so lanes always live in a known region of the frame |
| Hough transform | (next) Turns edge pixels into line equations by voting | Disconnected pixels become actual lanes |

## Usage

    python 01_explore.py                    # defaults: canny 50/150
    python 01_explore.py --low 10 --high 50 # experiment with thresholds

Outputs each intermediate stage to output/ for inspection.

## Status

- [x] Edge detection + ROI masking
- [x] Hough transform line detection
- [ ] Lane averaging and overlay
- [ ] Video processing

