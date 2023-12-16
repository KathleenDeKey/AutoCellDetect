import cv2
import numpy as np
import csv

# ** constant values **
# conversion factor
um_per_pixel = 0.3333333333333333
# circularity threshold
circularity_threshold = 0.5
# change contrast and brightness
alpha = 1.5  # Contrast control (1.0 means no change)
beta = 5  # Brightness control (0 means no change)


# ** Functions **

# Store Values to Dictionary
def store_values(detected_cells):
    index = 0
    cells = {}
    if detected_cells is not None:
        circles = np.uint16(detected_cells)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            diameter = radius * 2 * um_per_pixel
            area = np.pi * ((diameter / 2) ** 2)
            perimeter = 2 * np.pi * (diameter / 2)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            false_positive = contains_red(center, radius)
            if circularity > circularity_threshold and (not false_positive):
                cell_intensity = find_cell_intensity(center, radius)
                corrected_cell_intensity = find_corrected_cell_intensity(center, radius, cell_intensity, area)
                specific_cell = {'center': center, 'radius in pixels': radius, 'diameter': diameter, 'area': area,
                                 'Cell Intensity': cell_intensity, 'Corrected Cell Intensity': corrected_cell_intensity}
                cells[index] = specific_cell
                index += 1
    return cells


# determine if there are red pixels in the area
def contains_red(center, radius):
    red = (36, 28, 237)
    mask = np.zeros_like(bfimg)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
    cell = cv2.bitwise_and(bfimg, mask)
    matching_pixels = np.where(np.all(cell == red, axis=-1))
    if len(matching_pixels[0]) == 0:
        return False
    else:
        return True


# Draw Cells on Images
def draw_cells(cells: dict):
    for (index, cell) in cells.items():
        center = cell['center']
        radius = cell['radius in pixels']
        cv2.circle(bfimg, center, radius, (0, 255, 0), 2)
        cv2.circle(trimg, center, radius, (0, 255, 0), 2)
        text_position = (int(center[0] + radius + 10), int(center[1]))
        cv2.putText(bfimg, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(trimg, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Find total intensity of the cell
def find_cell_intensity(center, radius):
    cell_mask = np.zeros_like(trimg)
    cell_mask = cv2.circle(cell_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
    cell_area = cv2.bitwise_and(trimg, cell_mask)
    cell_intensity = np.mean(cell_area[cell_area != 0])
    return cell_intensity


# Fine the Corrected Intensity of the cell
def find_corrected_cell_intensity(center, radius, cell_intensity, area):
    cell_mask = np.zeros_like(trimg)
    cell_mask = cv2.circle(cell_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
    dilated_radius = round(radius * 1.5)
    dilated_area_mask = np.zeros_like(trimg)
    dilated_area_mask = cv2.circle(dilated_area_mask, center, dilated_radius, (255, 255, 255), thickness=cv2.FILLED)
    neighbouring_area = cv2.subtract(dilated_area_mask, cell_mask)
    neighbouring_area = cv2.bitwise_and(trimg, neighbouring_area)
    neighbouring_area_intensity = np.mean(neighbouring_area[neighbouring_area != 0])
    integrated_density = cell_intensity * area
    corrected_cell_intensity = integrated_density - (area * neighbouring_area_intensity)
    return corrected_cell_intensity


# Write Results to a csv file
def write_file(cells: dict):
    csv_file_path = 'output.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ['Index'] + list(cells[1].keys())
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for index, row in cells.items():
            row['Index'] = index
            csv_writer.writerow(row)


# ** load images **
bfFile = 'bf 7.tif'
bfimg = cv2.imread(bfFile)
trFile = 'tr 7.tif'
trimg = cv2.imread(trFile)


# ** preprocessing **
# convert to grayscale
trGray = cv2.cvtColor(trimg, cv2.COLOR_BGR2GRAY)
bfGray = cv2.cvtColor(bfimg, cv2.COLOR_BGR2GRAY)
# change contrast and brightness
bfGray = cv2.convertScaleAbs(bfGray, alpha=alpha, beta=beta)
# apply gaussian blur
bfGray = cv2.GaussianBlur(bfGray, (7, 7), 1.5)
# apply canny edge detection
edges = cv2.Canny(bfGray, threshold1=10, threshold2=60)
cv2.imshow("Preprocessing Done", edges)

# ** Apply Hough Circle Transform **
detected_cells = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=5, param2=15, minRadius=4,
                                  maxRadius=15)

# ** store, save and show result **
cells = store_values(detected_cells)
draw_cells(cells)
write_file(cells)
cv2.imshow('Bright Field', bfimg)
cv2.imshow('Fluorescence', trimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
