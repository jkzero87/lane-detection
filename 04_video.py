import os
import importlib
import argparse

import cv2

# Reuse the full pipeline from 03_average by importing it, not copying it.
# The module name starts with a digit, so a plain `import 03_average` is a
# syntax error -> use importlib. (02_hough is reached through 03's attribute.)
avg = importlib.import_module("03_average")
hough = avg.hough


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="images/solidWhiteRight.mp4")
    parser.add_argument("--out", type=str, default="output/04_lanes.mp4")
    parser.add_argument("--min-slope", type=float, default=0.5)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frames = 0
    left_fallbacks = 0
    right_fallbacks = 0
    last_left = None
    last_right = None
    y_top = int(height * 0.60)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1

            gray, blur, edges = hough.detect_edges(frame)
            masked_edges = hough.region_of_interest(edges)
            lines = hough.detect_lines(masked_edges, threshold=15, min_len=40, max_gap=20)
            left_segs, right_segs = avg.separate_lines(lines, min_slope=args.min_slope)

            # If a side has no fit this frame, reuse the last valid line for
            # that side so the overlay does not flicker on empty frames.
            left_line = avg.fit_lane_line(left_segs, height, y_top)
            if left_line is None:
                left_line = last_left
                if left_line is not None:
                    left_fallbacks += 1
            else:
                last_left = left_line

            right_line = avg.fit_lane_line(right_segs, height, y_top)
            if right_line is None:
                right_line = last_right
                if right_line is not None:
                    right_fallbacks += 1
            else:
                last_right = right_line

            result = avg.overlay(frame, avg.draw_lanes(frame, left_line, right_line))
            writer.write(result)
    finally:
        capture.release()
        writer.release()

    print(f"frames processed: {frames}")
    print(f"left fallbacks (reused previous line): {left_fallbacks}")
    print(f"right fallbacks (reused previous line): {right_fallbacks}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
