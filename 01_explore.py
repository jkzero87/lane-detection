# ---------- 1. IMPORTS ----------
import cv2
import argparse
import numpy as np


# ---------- 2. DEFINITIONS (recipes, nothing runs yet) ----------
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

# ---------- 3. INPUTS ----------
parser = argparse.ArgumentParser()
parser.add_argument('--low',  type=int, default=50)
parser.add_argument('--high', type=int, default=150)
args = parser.parse_args()                    # <- args is born HERE

img = cv2.imread('images/solidWhiteRight.jpg')


# ---------- 4. EXECUTION ----------
gray, blur, edges = detect_edges(img, low=args.low, high=args.high)
masked = region_of_interest(edges)

# ---------- 5. OUTPUTS ----------
cv2.imwrite('output/1_gray.jpg', gray)
cv2.imwrite('output/2_blur.jpg', blur)
out_path = f'output/3_edges_{args.low}_{args.high}.jpg'
cv2.imwrite(out_path, edges)
cv2.imwrite('output/4_masked.jpg', masked)
print(f"Saved stages, edges at {out_path}")