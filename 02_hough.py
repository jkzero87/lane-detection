import cv2
import numpy as np
import argparse
import os

def detect_edges(img, kernel_size=5, low=50, high=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur, low, high)
    return gray, blur, edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)          # all-black image, same shape

    polygon = np.array([[
        (int(width * 0.10), height),          # bottom-left
        (int(width * 0.45), int(height * 0.60)),  # top-left
        (int(width * 0.55), int(height * 0.60)),  # top-right
        (int(width * 0.95), height),          # bottom-right
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)     # paint the trapezoid white
    return cv2.bitwise_and(edges, mask)  # keep pixels only where mask is white

def detect_lines(masked_edges, threshold, min_len, max_gap):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    return lines

def draw_lines(shape, lines):
    canvas = np.zeros(shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.ravel(line)
            cv2.line(canvas, (x1, y1), (x2, y2), 255, 5)
    return canvas

def main():
    # ---------- inputs ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-gap", type=int, default=20)
    parser.add_argument("--image", type=str, default="images/solidWhiteRight.jpg")
    args = parser.parse_args()

    # ---------- execution ----------
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    gray, blur, edges = detect_edges(img)
    masked_edges = region_of_interest(edges)
    lines = detect_lines(masked_edges, args.threshold, args.min_len, args.max_gap)
    line_img = draw_lines(masked_edges.shape, lines)

    # ---------- outputs ----------
    count = 0 if lines is None else len(lines)
    print(f"segments detected: {count}")

    os.makedirs("output", exist_ok=True)
    filename = f"output/hough_t{args.threshold}_ml{args.min_len}_mg{args.max_gap}.jpg"
    cv2.imwrite(filename, line_img)
    print(f"saved: {filename}")


if __name__ == "__main__":
    main()
