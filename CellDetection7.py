import cv2
import numpy as np
import csv

# conversion factor
um_per_pixel = 0.3333333333333333

# load images
bfFile = 'bf 3.tif'
bfimg = cv2.imread(bfFile)
trFile = 'tr 3.tif'
trimg = cv2.imread(trFile)
trgray = cv2.cvtColor(trimg, cv2.COLOR_BGR2GRAY)
global_average_intensity = np.mean(trgray)
# cv2.imshow('image1', img)
# cv2.waitKey(0)
gray = cv2.cvtColor(bfimg, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)
# cv2.imshow('equalize', gray)
gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

# change constrast and brightness
alpha = 1.5  # Contrast control (1.0 means no change)
beta = 5   # Brightness control (0 means no change)

# Apply the contrast and brightness adjustment
gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
edges = cv2.Canny(gray, threshold1=10, threshold2=60)
cv2.imshow("canny edges", edges)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=5, param2=15, minRadius=4, maxRadius=15)
if circles is not None:
    circles = np.uint16(circles)
    num = 1;
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
            cv2.circle(bfimg, center, radius, (0, 0, 255), 2)
            cv2.circle(trimg, center, radius, (0, 0, 255), 2)
            text_position = (int(center[0] + radius + 10), int(center[1]))
            cv2.putText(bfimg, str(num), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(trimg, str(num), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            num += 1
cv2.imshow('Bright Field', bfimg)
cv2.imshow('Fluorescence', trimg)

def store_values(detected, trimage):
    index = 0
    cells = {}
    if detected is not None:
        circles = np.uint16(detected)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            # print(center)
            radius = circle[2]
            diameter = radius * 2 * um_per_pixel
            # print(radius)
            area = np.pi * ((diameter/2) ** 2)
            perimeter = 2 * np.pi * (diameter/2)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circularity_threshold = 0.5
            if circularity > circularity_threshold:
                specificCell = {'center': center, 'radius in pixels': radius, 'diameter': diameter, 'area': area, 'Mean Gray Value': 0, 'CTCF': 0}
                cells[index] = specificCell
                index += 1
    return cells

def find_intensity(cells, trimage):
    trimage = cv2.cvtColor(trimage, cv2.COLOR_BGR2GRAY)
    for index in range(len(cells)):
        (x, y) = cells[index]['center']
        radius = cells[index]['radius in pixels']
        total_intensity = 0;
        numpixel = 0
        # Traverse through each pixel and check if it's within the circle
        for i in range(trimage.shape[0]):
            for j in range(trimage.shape[1]):
                # Calculate the distance from the center (x, y)
                distance = np.sqrt((j - x) ** 2 + (i - y) ** 2)
                # If the distance is less than or equal to the radius, the pixel is inside the circle
                if distance <= radius:
                    intensity = trimage[y, x]
                    total_intensity += intensity
                    numpixel += 1;
        # Calculate the Mean Gray Value of the pixels within the circle
        mean_gray_value = total_intensity // numpixel
        # Calculate integrated density
        area = cells[index]['area']
        integrated_density = mean_gray_value * area
        # corrected total cell fluorescence (CTCF) = Integrated Density – (Area of Selected Cell x Mean Fluorescence of Background readings)
        CTCF = integrated_density - (area * global_average_intensity)
        cells[index]['Mean Gray Value'] = mean_gray_value
        cells[index]['CTCF'] = CTCF
        print(mean_gray_value)
        print(CTCF)
    return cells

# cv2.imwrite('processed' + fileName, img)
cells = store_values(circles, trimg)
cells = find_intensity(cells, trimg)

# Specify the CSV file path
csv_file_path = 'output.csv'

# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csv_file:
    # Specify the field names (keys of the inner dictionaries)
    fieldnames = cells[1].keys()

    # Create a DictWriter object with the CSV file and field names
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header to the CSV file
    csv_writer.writeheader()

    # Write the data (values of the inner dictionaries) to the CSV file
    for row in cells.values():
        csv_writer.writerow(row)

cv2.waitKey(0)
cv2.destroyAllWindows()
