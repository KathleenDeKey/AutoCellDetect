import cv2
import numpy as np

# Draw the detected lines on the original image
def draw_Lines(lines, image):
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('detected lines', image)

def detect_parallel_distance(lines):
    parallel_line_pairs = []
    for i in range(len(lines)):
        rho_i, theta_i = lines[i][0]
        for j in range(i + 1, len(lines)):
            rho_j, theta_j = lines[j][0]
            # Define a tolerance for angle similarity
            angle_tolerance = 10
            if abs(theta_i - theta_j) < np.deg2rad(angle_tolerance):
                parallel_line_pairs.append((rho_i, theta_i, rho_j, theta_j))
    total_distance = 0;
    for pair in parallel_line_pairs:
        rho1 = pair[0]
        rho2 = pair[2]
        distance = abs(rho1 - rho2)
        if distance < 100:
            total_distance += distance
    return total_distance / 2

image = cv2.imread('dayZerocell03m0001.tif')
cv2.imshow('original', image)

# Preprocessing (e.g., convert to grayscale and apply Canny edge detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
cv2.imshow("after threshold processing", binary)
edges = cv2.Canny(binary, 100, 150)
cv2.imshow("edge", edges)

# Use the Hough Line Transform to detect the reference
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=170)
draw_Lines(lines, image)
# Detect parallel lines that show the reference and find the average distance between them
reference_pixel = detect_parallel_distance(lines)
print("reference pixel: " + str(reference_pixel))
# print(reference_pixel)
actual_distance = 20 # um
um_per_pixel = actual_distance / reference_pixel
print(um_per_pixel)

cv2.waitKey(0)
cv2.destroyAllWindows()
