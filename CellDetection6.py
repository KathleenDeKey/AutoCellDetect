import cv2
import numpy as np

# conversion factor
um_per_pixel = 0.3333333333333333

# create slider window
cv2.namedWindow("Enter Values", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Enter Values', 500, 300)
# cv2.moveWindow("Enter Values", 100, 100)

def on_trackbar(value):
    # print("value: " + str(value))
    return

def process_image(dp, edge_threshold, accumulator_threshold, minDist, minRad, maxRad):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=edge_threshold, param2=accumulator_threshold, minRadius=minRad, maxRadius=maxRad)
    return circles

def display_image(detected, img):
    if detected is not None:
        circles = np.uint16(detected)
        i = 0;
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
                # formatted_diameter = "{:.1f}".format(float(radius * um_per_pixel) * 2)
                cv2.putText(img, str(i), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                i += 1;
        cv2.imshow('Detected Cells', img)
        cv2.waitKey(0)
        cv2.destroyWindow('Detected Cells')

def store_values(detected, trimage):
    index = 0
    cells = {}
    if detected is not None:
        circles = np.uint16(detected)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            # print(center)
            radius = circle[2]
            diameter = radius * 2
            # print(radius)
            area = np.pi * (radius ** 2)
            perimeter = 2 * np.pi * radius
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularity_threshold = 0.5
            if circularity > circularity_threshold:
                average_intensity = find_intensity(detected, trimage)
                specificCell = {'center': center, 'radius': radius, 'diameter': diameter, 'area': area, 'Intensity': []}
                cells[index] = specificCell
                index += 1
    return cells

def find_intensity(detected, trimage):
    # Calculate the average pixel intensity
    gray = cv2.cvtColor(trimage, cv2.COLOR_BGR2GRAY)
    global_average_intensity = np.mean(gray)
    print(f'Average pixel intensity of the entire image: {global_average_intensity}')

    # traverse through each circle to process intensity
    if detected is not None:
        circles = np.uint16(detected)
        for circle in circles[0, :]:
            x, y = (circle[0], circle[1])
            radius = circle[2]
            # Create an empty mask image with the same size as the input image
            mask = np.zeros(trimage.shape[:2], dtype=np.uint8)
            # Traverse through each pixel and check if it's within the circle
            for i in range(trimage.shape[0]):
                for j in range(trimage.shape[1]):
                    # Calculate the distance from the center (x, y)
                    distance = np.sqrt((j - x) ** 2 + (i - y) ** 2)
                    # If the distance is less than or equal to the radius, the pixel is inside the circle
                    if distance <= radius:
                        # Mark this pixel as part of the circle in the mask
                        mask[i, j] = 255
            # Use the mask to extract the pixels within the circle from the original image
            circle_pixels = cv2.bitwise_and(trimage, trimage, mask=mask)
            # Calculate the average intensity of the pixels within the circle
            average_intensity = np.mean(circle_pixels)
            # corrected total cell fluorescence (CTCF) = Integrated Density – (Area of Selected Cell x Mean Fluorescence of Background readings)
            # CTCF =
    return average_intensity

# Create trackbars
cv2.createTrackbar("Max Radius", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Min Radius", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Min Distance", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Accumulator Resolution", "Enter Values", 0, 10, on_trackbar)
cv2.createTrackbar("Edge Threshold", "Enter Values", 0, 10, on_trackbar)
cv2.createTrackbar("Accumulator Threshold", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Contrast", "Enter Values", 0, 20, on_trackbar)
cv2.createTrackbar("Brightness", "Enter Values", 0, 127, on_trackbar)
cv2.setTrackbarMin("Brightness", "Enter Values", -127)

while True:
    # load images
    bfimg = cv2.imread('bf 6.tif', cv2.IMREAD_GRAYSCALE)
    # bfimgGray = cv2.equalizeHist(bfimg)
    trimg = cv2.imread('tr 6.tif')

    # sets values
    minRad = cv2.getTrackbarPos("Min Radius", "Enter Values")
    maxRad = cv2.getTrackbarPos("Max Radius", "Enter Values")
    minDist = cv2.getTrackbarPos("Min Distance", "Enter Values")
    dp = cv2.getTrackbarPos("Accumulator Resolution", "Enter Values") / 10.0
    edge_threshold = cv2.getTrackbarPos("Edge Threshold", "Enter Values")
    accumulator_threshold = cv2.getTrackbarPos("Accumulator Threshold", "Enter Values") /10.0
    alpha = cv2.getTrackbarPos("Contrast", "Enter Values") / 10.0  # Contrast control (1.0 means no change)
    beta = cv2.getTrackbarPos("Brightness", "Enter Values")  # Brightness control (0 means no change)

    # Convert image to grayscale and apply the contrast and brightness adjustment
    gray = cv2.GaussianBlur(bfimg, (7, 7), 1.5)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    cv2.imshow("adjusted image", gray)

    # Detect cells and display image after pressing 'a'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        cv2.destroyWindow("adjusted image")
        detected_circles = process_image(dp, edge_threshold, accumulator_threshold, minDist, minRad, maxRad)
        display_image(detected_circles, bfimg)

    # press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # press 'w' to write processed file
    if key == ord('w'):
        cells = store_values(detected_circles, trimg)
        for key, value in cells.values():
            print(key + " : " + cells[key]['Intensity']);


cv2.destroyAllWindows()
