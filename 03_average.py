import os
import importlib
import argparse

import cv2
import numpy as np

# Reuse the pipeline from 02_hough by importing it, not copying it.
# The module name starts with a digit, so a plain `import 02_hough` is a
# syntax error -> use importlib.
hough = importlib.import_module("02_hough")


# ---------- pure functions ----------
def separate_lines(lines, min_slope=0.5):
    """Split HoughLinesP output into (left, right) lane segments.

    Image coordinates have y growing downward, so the left lane marking
    has NEGATIVE slope and the right one POSITIVE. Segments with
    abs(slope) < min_slope (horizon / horizontal noise) are discarded.
    """
    left, right = [], []
    if lines is None:
        return left, right
    for line in lines:
        x1, y1, x2, y2 = np.ravel(line)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            slope = np.inf if dy >= 0 else -np.inf
        else:
            slope = dy / dx
        if abs(slope) < min_slope:
            continue
        segment = (x1, y1, x2, y2)
        if slope < 0:
            left.append(segment)
        else:
            right.append(segment)
    return left, right


def fit_lane_line(segments, y_bottom, y_top):
    """Fit one lane line through all segment endpoints (x as a function of y).

    Returns (x1, y1, x2, y2) at the given y values, or None if no segments.
    """
    if not segments:
        return None
    xs = []
    ys = []
    for x1, y1, x2, y2 in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    slope, intercept = np.polyfit(ys, xs, 1)
    x_bottom = int(round(slope * y_bottom + intercept))
    x_top = int(round(slope * y_top + intercept))
    return (x_bottom, y_bottom, x_top, y_top)


def draw_lanes(image, left_line, right_line, color=(0, 0, 255), thickness=8):
    """Draw the fitted lane lines on a copy of image (input not mutated)."""
    canvas = image.copy()
    for lane_line in (left_line, right_line):
        if lane_line is None:
            continue
        x1, y1, x2, y2 = lane_line
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


def overlay(base, lines_img, alpha=0.8, beta=1.0):
    """Blend the lane-line image over base with cv2.addWeighted."""
    return cv2.addWeighted(base, alpha, lines_img, beta, 0)


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="images/solidWhiteRight.jpg")
    parser.add_argument("--min-slope", type=float, default=0.5)
    args = parser.parse_args()

    # ---------- execution ----------
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")

    gray, blur, edges = hough.detect_edges(img)
    masked_edges = hough.region_of_interest(edges)
    lines = hough.detect_lines(masked_edges, threshold=15, min_len=40, max_gap=20)

    total = 0 if lines is None else len(lines)
    left, right = separate_lines(lines, min_slope=args.min_slope)
    discarded = total - len(left) - len(right)

    height = img.shape[0]
    left_line = fit_lane_line(left, height, int(height * 0.60))
    right_line = fit_lane_line(right, height, int(height * 0.60))

    # ---------- diagnostics ----------
    print(f"total Hough segments: {total}")
    print(f"  left  (negative slope): {len(left)}")
    print(f"  right (positive slope): {len(right)}")
    print(f"  discarded (|slope| < {args.min_slope}): {discarded}")
    for name, line in (("left", left_line), ("right", right_line)):
        if line is None:
            print(f"{name} fitted slope: None (no segments to fit)")
        else:
            x1, y1, x2, y2 = line
            if x2 == x1:
                print(f"{name} fitted slope: inf (vertical)")
            else:
                print(f"{name} fitted slope: {(y2 - y1) / (x2 - x1):.4f}")

    # ---------- outputs ----------
    lanes_img = draw_lanes(img, left_line, right_line)
    result = overlay(img, lanes_img)
    os.makedirs("output", exist_ok=True)
    out_path = "output/03_average.jpg"
    cv2.imwrite(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
