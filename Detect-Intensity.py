import cv2
import numpy as np

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

def find_intensity(center, radius, trimage):
    (x, y) = center
    # Create an empty mask image with the same size as the input image
    # mask = np.zeros(trimage.shape[:2], dtype=np.uint8)
    trimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_intensity = 0;
    numpixel = 0
    # Traverse through each pixel and check if it's within the circle
    for i in range(trimage.shape[0]):
        for j in range(trimage.shape[1]):
            # Calculate the distance from the center (x, y)
            distance = np.sqrt((j - x) ** 2 + (i - y) ** 2)
            # If the distance is less than or equal to the radius, the pixel is inside the circle
            if distance <= radius:
                # Mark this pixel as part of the circle in the mask
                intensity = trimage[y, x]
                total_intensity += intensity
                numpixel += 1;
    # Use the mask to extract the pixels within the circle from the original image
    # cv2.imshow('mask', mask)
    # circle_pixels = cv2.bitwise_and(trimage, trimage, mask=mask)
    # cv2.imshow('circle_pixels',circle_pixels)
    # cv2.waitKey(0)
    # Calculate the average intensity of the pixels within the circle
    average_intensity = total_intensity // numpixel
    print(average_intensity)
    # corrected total cell fluorescence (CTCF) = Integrated Density – (Area of Selected Cell x Mean Fluorescence of Background readings)
    # CTCF = average_intensity - (np.pi * (radius ** 2) * global_average_intensity)
    # print(CTCF)

fileName = 'intensityTest.tif'
img = cv2.imread(fileName)
# cv2.imshow('image1', img)
# cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
# cv2.imshow('Gaussian Blur', gray)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=0.011, minDist=10, param1=0.75, param2=15, minRadius=4, maxRadius=20)
# Calculate the average pixel intensity
global_average_intensity = np.mean(gray)
print(f'Average pixel intensity of the entire image: {global_average_intensity}')
circles = np.uint16(circles)
for circle in circles[0, :]:
    x, y = (circle[0], circle[1])
    radius = circle[2]
    find_intensity((x,y), radius, img)
display_image(circles, img)


cv2.waitKey(0)
cv2.destroyAllWindows();

