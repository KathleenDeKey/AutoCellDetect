import cv2
import numpy as np

# conversion factor
um_per_pixel = 0.3333333333333333

# create slider window
cv2.namedWindow("Enter Values", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Enter Values', 500, 200)
# cv2.moveWindow("Enter Values", 100, 100)

def on_trackbar(value):
    # print("value: " + str(value))
    return

def process_image(minDist, minRad, maxRad):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=0.011, minDist=minDist, param1=0.75, param2=15, minRadius=minRad, maxRadius=maxRad)
    return circles

def display_image(detected, img):
    if detected is not None:
        circles = np.uint16(detected)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            # print(center)
            radius = circle[2]
            # print(radius)
            area = np.pi * (radius ** 2)
            perimeter = 2 * np.pi * radius
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularity_threshold = 0.5
            if circularity > circularity_threshold:
                cv2.circle(img, center, radius, (0, 0, 255), 2)
                text_position = (int(center[0] + radius + 10), int(center[1]))
                formatted_radius = "{:.1f}".format(float(radius * um_per_pixel))
                cv2.putText(img, str(formatted_radius) + "um", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Detected Cells', img)
        cv2.waitKey(0)
        cv2.destroyWindow('Detected Cells')

# Create trackbars
cv2.createTrackbar("Max Radius", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Min Radius", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Min Distance", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Contrast", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Brightness", "Enter Values", 0, 127, on_trackbar)
cv2.setTrackbarMin("Brightness", "Enter Values", -127)

while True:
    # open file
    fileName = 'dayZerocell05m0001.tif'
    img = cv2.imread(fileName)
    cv2.imshow('image', img)

    # sets values
    minRad = cv2.getTrackbarPos("Min Radius", "Enter Values")
    maxRad = cv2.getTrackbarPos("Max Radius", "Enter Values")
    minDist = cv2.getTrackbarPos("Min Distance", "Enter Values")
    alpha = cv2.getTrackbarPos("Contrast", "Enter Values") / 10.0  # Contrast control (1.0 means no change)
    beta = cv2.getTrackbarPos("Brightness", "Enter Values")  # Brightness control (0 means no change)

    # Convert image to grayscale and apply the contrast and brightness adjustment
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    cv2.imshow("adjusted image", gray)

    # Detect cells and display image after pressing 'a'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        cv2.destroyWindow("adjusted image")
        detected = process_image(minDist, minRad, maxRad)
        display_image(detected, img)
    # press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # press 'w' to write processed file
    if key == ord('w'):
        cv2.imwrite('processed' + fileName, img)

cv2.destroyAllWindows()
