import os
import pandas as pd
import cv2
import numpy as np
import math

# ** Choose File **
# path to the folder containing images
input_folder_path = 'delivered'
# number label for the set of images
file_number = 6
# the type of image you want to analyze - True: only bf images are analyzed; False: bf and tr images are analyzed
bf_only = False
# name of output Excel file
output_file_name = 'output_example2.xlsx'
# relative path to the folder to save your results; will create a new folder if it does not exist
output_folder_path = 'delivered-results-ver2'

# ** constant values **

# conversion factor
um_per_pixel = 0.3333333333333333
# circularity threshold
circularity_threshold = 0.5
# red label value
red = (36, 28, 237)


# ** Functions **

# Store Values to Dictionary
def store_values(detected_cells, bf_only, bfimg, processed_img, trimg=None):
    index = 0
    cells = {}
    if detected_cells is not None:
        circles = np.uint16(detected_cells)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius_in_pixel = circle[2]
            diameter = radius_in_pixel * 2 * um_per_pixel
            area = np.pi * ((diameter / 2) ** 2)
            false_positive = contains_red(center, radius_in_pixel, bfimg)
            is_labeled = near_red_label(center, radius_in_pixel, bfimg)
            if (not false_positive) and is_labeled:
                contours = find_contour(center, radius_in_pixel, processed_img)
                ellipse = find_ellipse(contours, radius_in_pixel)
                if ellipse == 'ellipse not found':
                    aspect_ratio = 'NA'
                    taylor = 'NA'
                    eccentricity = 'NA'
                    perimeter = 'NA'
                    circularity = 'NA'
                else:
                    ((centx, centy), (minor_axis_b, major_axis_a), angle) = ellipse
                    perimeter = np.pi * np.sqrt(2 * (major_axis_a ** 2 + minor_axis_b ** 2) -
                                                (major_axis_a - minor_axis_b) ** 2)
                    aspect_ratio = major_axis_a / minor_axis_b
                    taylor = (major_axis_a - minor_axis_b) / (major_axis_a + minor_axis_b)
                    eccentricity = math.sqrt(1 - (minor_axis_b / major_axis_a))
                    circularity = (4 * math.pi * area) / (perimeter ** 2)
                if bf_only:
                    specific_cell = {'center': center, 'radius in pixels': radius_in_pixel, 'diameter': diameter,
                                     'perimeter': perimeter, 'area': area, 'aspect ratio': aspect_ratio,
                                     'taylor': taylor, 'eccentricity': eccentricity, 'circularity': circularity,
                                     'ellipse properties': ellipse}
                    cells[index] = specific_cell
                    index += 1
                else:
                    cell_intensity = find_cell_intensity(center, radius_in_pixel, trimg)
                    corrected_cell_intensity = find_corrected_cell_intensity(center, radius_in_pixel, cell_intensity,
                                                                             area, trimg)
                    specific_cell = {'center': center, 'radius in pixels': radius_in_pixel, 'diameter': diameter,
                                     'perimeter': perimeter, 'area': area, 'aspect ratio': aspect_ratio,
                                     'taylor': taylor, 'eccentricity': eccentricity, 'circularity': circularity,
                                     'ellipse properties': ellipse, 'Cell Intensity': cell_intensity,
                                     'Corrected Cell Intensity': corrected_cell_intensity}
                    cells[index] = specific_cell
                    index += 1
    return cells


# determine if there are red pixels in the area
def contains_red(center, radius, bfimg):
    mask = np.zeros_like(bfimg)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
    cell = cv2.bitwise_and(bfimg, mask)
    matching_pixels = np.where(np.all(cell == red, axis=-1))
    if len(matching_pixels[0]) == 0:
        return False
    else:
        return True


# determine if the detected cell is near a red pixel
def near_red_label(center, radius, bfimg):
    radius *= 10
    return contains_red(center, radius, bfimg)


# Draw Cells on Images
def draw_cells(cells: dict, bf_only, bfimg, trimg=None):
    for (index, cell) in cells.items():
        center = cell['center']
        radius = cell['radius in pixels']
        cv2.circle(bfimg, center, radius, (0, 255, 0), 2)
        text_position = (int(center[0] - radius - 20), int(center[1]))
        cv2.putText(bfimg, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        ellipse = cell['ellipse properties']
        if ellipse != "ellipse not found":
            cv2.ellipse(bfimg_copy, ellipse, (0, 255, 0), 2)
            cv2.putText(bfimg_copy, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if not bf_only:
            cv2.circle(trimg, center, radius, (0, 255, 0), 2)
            cv2.putText(trimg, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Find total intensity of the cell
def find_cell_intensity(center, radius, trimg):
    cell_mask = np.zeros_like(trimg)
    cell_mask = cv2.circle(cell_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
    cell_area = cv2.bitwise_and(trimg, cell_mask)
    cell_intensity = np.mean(cell_area[cell_area != 0])
    return cell_intensity


# Find the Corrected Intensity of the cell
def find_corrected_cell_intensity(center, radius, cell_intensity, area, trimg):
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


# Find the contour of the cell
def find_contour(center, radius, processed_img):
    cell_mask = np.zeros_like(processed_img)
    cell_mask = cv2.circle(cell_mask, center, round(radius * 1.1), (255, 255, 255), thickness=cv2.FILLED)
    cell_area = cv2.bitwise_and(processed_img, cell_mask)
    contours, _ = cv2.findContours(cell_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Find the ellipse of the cell using contours
def find_ellipse(contours, radius):
    for contour in contours:
        if len(contour) >= 5:
            cell_ellipse = cv2.fitEllipse(contour)
            (axes1, axes2) = cell_ellipse[1]
            # the length of axes should be similar to the radius with at most +/- 3 pixels
            similar_radius = (radius + 5 > axes1 > radius - 5) and (radius + 5 > axes1 > radius - 5)
            # the axes cannot be too small
            too_small = axes1 < 5 and axes2 < 7
            if similar_radius and (not too_small):
                return cell_ellipse
    return 'ellipse not found'


# Write Results to an Excel file
def write_excel_file(cells: dict, dataset_number, bf_only):
    if len(cells) == 0:
        print("No cells were detected in image(s) #", file_number)
        return
    else:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        output_file_path = os.path.join(output_folder_path, output_file_name)
        data_frame = pd.DataFrame(cells).T
        data_frame['center'] = data_frame['center'].astype(str)
        if bf_only:
            column_order = ['center', 'radius in pixels', 'diameter', 'area', 'aspect ratio', 'taylor', 'eccentricity',
                            'circularity', 'ellipse properties']
            dataset_name = 'bf ' + str(dataset_number)
        else:
            column_order = ['center', 'radius in pixels', 'diameter', 'area', 'aspect ratio', 'taylor', 'eccentricity',
                            'circularity', 'ellipse properties', 'Cell Intensity', 'Corrected Cell Intensity']
            dataset_name = 'bf-tr ' + str(dataset_number)
        data_frame = data_frame[column_order]
        if os.path.exists(output_file_path):
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                data_frame.to_excel(writer, sheet_name=dataset_name, index_label='Index')
        else:
            with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
                data_frame.to_excel(writer, sheet_name=dataset_name, index_label='Index')


# Save Processed Images
def save_images(bfimg, bfimg_copy, trimg=None):
    bf_name = 'bf hough circle ' + str(file_number) + '.jpg'
    cv2.imwrite(os.path.join(output_folder_path, bf_name), bfimg)
    bf_copy_name = 'bf ellipse ' + str(file_number) + '.jpg'
    cv2.imwrite(os.path.join(output_folder_path, bf_copy_name), bfimg_copy)
    if trimg is not None:
        tr_name = 'Processed tr ' + str(file_number) + '.jpg'
        cv2.imwrite(os.path.join(output_folder_path, tr_name), trimg)


# ** Create slider window and trackbars **
cv2.namedWindow("Adjust Image", cv2.WINDOW_NORMAL)


def on_trackbar(value):
    # print("value: " + str(value))
    return

# Gaussian Blur Values
cv2.createTrackbar("ksize", "Adjust Image", 1, 4, on_trackbar)
cv2.createTrackbar("sigmax (divide by 10)", "Adjust Image", 0, 30, on_trackbar)
# Brightness and Contrast Values
cv2.createTrackbar("alpha - contrast (divide by 10)", "Adjust Image", 0, 100, on_trackbar)
cv2.createTrackbar("beta - brightness", "Adjust Image", -127, 127, on_trackbar)
# Canny Edge Detection Values
cv2.createTrackbar("min threshold", "Adjust Image", 0, 100, on_trackbar)
cv2.createTrackbar("max threshold", "Adjust Image", 0, 100, on_trackbar)

# Set default values for each trackbar
cv2.setTrackbarPos("ksize", "Adjust Image", 3)
cv2.setTrackbarPos("sigmax (divide by 10)", "Adjust Image", 15)
cv2.setTrackbarPos("alpha - contrast (divide by 10)", "Adjust Image", 15)
cv2.setTrackbarPos("beta - brightness", "Adjust Image", 5)
cv2.setTrackbarPos("min threshold", "Adjust Image", 10)
cv2.setTrackbarPos("max threshold", "Adjust Image", 60)


while True:
    # ** load images **
    bfFile = 'bf ' + str(file_number) + '.tif'
    bfimg = cv2.imread(os.path.join(input_folder_path, bfFile))
    bfimg_copy = cv2.imread(os.path.join(input_folder_path, bfFile))
    trimg = None
    if not bf_only:
        trFile = 'tr ' + str(file_number) + '.tif'
        trimg = cv2.imread(os.path.join(input_folder_path, trFile))

    # sets values
    ksize = cv2.getTrackbarPos("ksize", "Adjust Image")
    sigmax = cv2.getTrackbarPos("sigmax (divide by 10)", "Adjust Image") / 10.0
    alpha_contast = cv2.getTrackbarPos("alpha - contrast (divide by 10)", "Adjust Image") / 10.0
    beta_brightness = cv2.getTrackbarPos("beta - brightness", "Adjust Image")
    min_threshold = cv2.getTrackbarPos("min threshold", "Adjust Image")
    max_threshold = cv2.getTrackbarPos("max threshold", "Adjust Image")

    # ** preprocessing **

    # convert to grayscale
    bfGray = cv2.cvtColor(bfimg, cv2.COLOR_BGR2GRAY)
    # change contrast and brightness
    bfGray = cv2.convertScaleAbs(bfGray, alpha=alpha_contast, beta=beta_brightness)
    # apply gaussian blur
    ksize = 2 * ksize + 1
    bfGray = cv2.GaussianBlur(bfGray, (ksize, ksize), sigmax)
    # apply canny edge detection
    edges = cv2.Canny(bfGray, threshold1=min_threshold, threshold2=max_threshold)
    combined_preprocessing_image = cv2.hconcat([bfGray, edges])

    # ** Apply Hough Circle Transform **
    detected_cells = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=5, param2=15, minRadius=4,
                                              maxRadius=15)
    # store values and draw on result image
    cells = store_values(detected_cells, True, bfimg, processed_img=edges)
    draw_cells(cells, True, bfimg)

    # Combine and show panels
    combined_cells_image = cv2.hconcat([bfimg, bfimg_copy])
    combined_preprocessing_image = cv2.cvtColor(combined_preprocessing_image, cv2.COLOR_GRAY2BGR)
    total_combined_image = np.vstack([combined_preprocessing_image, combined_cells_image])
    total_combined_image = cv2.resize(total_combined_image, (0,0), fx=0.7, fy=0.7)
    cv2.imshow("Adjust Image", total_combined_image)

    key = cv2.waitKey(10)  # Wait for 1 millisecond

    # Apply the parameters in processing when 'a' is pressed
    if key == 97:
        cv2.destroyAllWindows()  # clean windows

        # ** store, save and show result **
        if bf_only:
            cells = store_values(detected_cells, bf_only, bfimg, processed_img=edges)
            draw_cells(cells, bf_only, bfimg)
            combined_result_image = cv2.hconcat([bfimg, bfimg_copy, trimg])
            save_images(bfimg, bfimg_copy)

        else:
            cells = store_values(detected_cells, bf_only, bfimg, processed_img=edges, trimg=trimg)
            draw_cells(cells, bf_only, bfimg, trimg)
            combined_result_image = cv2.hconcat([bfimg, bfimg_copy, trimg])
            save_images(bfimg, bfimg_copy, trimg)

        write_excel_file(cells, file_number, bf_only)
        combined_result_image = cv2.resize(combined_result_image, (0,0), fx=0.7, fy=0.7)
        cv2.imshow("Result", combined_result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break

    # Reset to default value if 'r' key is pressed
    if key == 114:
        cv2.setTrackbarPos("ksize", "Adjust Image", 3)
        cv2.setTrackbarPos("sigmax (divide by 10)", "Adjust Image", 15)
        cv2.setTrackbarPos("alpha - contrast (divide by 10)", "Adjust Image", 15)
        cv2.setTrackbarPos("beta - brightness", "Adjust Image", 5)
        cv2.setTrackbarPos("min threshold", "Adjust Image", 10)
        cv2.setTrackbarPos("max threshold", "Adjust Image", 60)

    # Break if key 'q' is pressed
    if key == 27:
        cv2.destroyAllWindows()
        break

print('All Done')

