import os
import pandas as pd
import cv2
import numpy as np
import math

# Analyze and store cell properties to Dictionary after image is processed by hough circles
# Returns a dictionary with the cell properties labeled to corresponding cell index
def store_values(detected_cells, bf_only, bfimg, processed_img, um_per_pixel, red, trimg=None) -> dict:
    index = 0
    cells = {}
    if detected_cells is not None:
        circles = np.uint16(detected_cells)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius_in_pixel = circle[2]
            # If the cell is labeled with red pixels nearby, it is a target cell
            false_positive = contains_red(red, center, radius_in_pixel, bfimg)
            is_labeled = near_red_label(red, center, radius_in_pixel, bfimg)
            if (not false_positive) and is_labeled:
                contours = find_contour(center, radius_in_pixel, processed_img)
                ellipse = find_ellipse(contours, radius_in_pixel)
                if ellipse == 'ellipse not found':
                    aspect_ratio = 'NA'
                    taylor = 'NA'
                    eccentricity = 'NA'
                    perimeter = 'NA'
                    circularity = 'NA'
                    averaged_diameter_in_pixels = 'NA'
                    averaged_diameter = 'NA'
                    area = 'NA'
                else:
                    ((centx, centy), (minor_axis_b, major_axis_a), angle) = ellipse
                    center = (round(centx), round(centy))
                    semi_minor_axis_b = minor_axis_b / 2
                    semi_major_axis_a = major_axis_a / 2
                    averaged_diameter_in_pixels = (minor_axis_b + major_axis_a) / 2 
                    averaged_diameter = averaged_diameter_in_pixels * um_per_pixel
                    perimeter = np.pi * (3 * (semi_major_axis_a + semi_minor_axis_b) - 
                                         np.sqrt((3 * semi_major_axis_a + semi_minor_axis_b) * (semi_major_axis_a + 3 * semi_minor_axis_b)))
                    area = np.pi * (semi_major_axis_a) * (semi_minor_axis_b)
                    aspect_ratio = major_axis_a / minor_axis_b
                    taylor = (major_axis_a - minor_axis_b) / (major_axis_a + minor_axis_b)
                    eccentricity = math.sqrt(1 - (minor_axis_b / major_axis_a))
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                # Store cell properties into a dictionary
                if bf_only:
                    specific_cell = {'center': center, 'averaged diameter in pixels': averaged_diameter_in_pixels, 'averaged diameter': averaged_diameter,
                                     'perimeter': perimeter, 'area': area, 'aspect ratio': aspect_ratio,
                                     'taylor': taylor, 'eccentricity': eccentricity, 'circularity': circularity,
                                     'ellipse properties': ellipse}
                    cells[index] = specific_cell
                    index += 1
                else:
                    cell_intensity = find_cell_intensity(ellipse, trimg)
                    corrected_cell_intensity = find_corrected_cell_intensity(ellipse, cell_intensity,
                                                                             area, trimg)
                    specific_cell = {'center': center, 'averaged diameter in pixels': averaged_diameter_in_pixels, 'averaged diameter': averaged_diameter,
                                     'perimeter': perimeter, 'area': area, 'aspect ratio': aspect_ratio,
                                     'taylor': taylor, 'eccentricity': eccentricity, 'circularity': circularity,
                                     'ellipse properties': ellipse, 'Cell Intensity': cell_intensity,
                                     'Corrected Cell Intensity': corrected_cell_intensity}
                    cells[index] = specific_cell
                    index += 1
    return cells


# return bool of whether there are red pixels in selected area
def contains_red(red, center, radius, bfimg) -> bool:
    mask = np.zeros_like(bfimg)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
    cell = cv2.bitwise_and(bfimg, mask)
    matching_pixels = np.where(np.all(cell == red, axis=-1))
    if len(matching_pixels[0]) == 0:
        return False
    else:
        return True


# determine if the detected cell is near a red pixel
def near_red_label(red, center, radius, bfimg):
    radius *= 5
    return contains_red(red, center, radius, bfimg)


# Draw Cells on Images with ellipses
def draw_cells(cells: dict, bf_only, bfimg, bfimg_copy, trimg=None):
    for (index, cell) in cells.items():
        center = cell['center']
        diameter = cell['averaged diameter in pixels']
        if diameter == 'NA':
            break
        radius = round(diameter / 2)
        cv2.circle(bfimg, center, radius, (0, 255, 0), 1)
        text_position = (int(center[0] - radius - 20), int(center[1]))
        cv2.putText(bfimg, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        ellipse = cell['ellipse properties']
        if ellipse != "ellipse not found":
            cv2.ellipse(bfimg_copy, ellipse, (0, 255, 0), 1)
            cv2.putText(bfimg_copy, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if not bf_only:
            cv2.ellipse(trimg, ellipse, (0, 255, 0), 1)
            cv2.putText(trimg, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Find total intensity of the cell; mean pixel value of selected area
def find_cell_intensity(ellipse, trimg) -> float:
    cell_mask = np.zeros_like(trimg)
    cell_mask = cv2.ellipse(cell_mask, ellipse, (255, 255, 255), thickness=cv2.FILLED)
    cell_area = cv2.bitwise_and(trimg, cell_mask)
    cell_intensity = np.mean(cell_area[cell_area != 0])
    return cell_intensity


# Find the Corrected Intensity of the cell; integrated density - area * neighbouring area intensity
def find_corrected_cell_intensity(ellipse, cell_intensity, area, trimg) -> float:
    ((centx, centy), (minor_axis_b, major_axis_a), angle) = ellipse
    cell_mask = np.zeros_like(trimg)
    cell_mask = cv2.ellipse(cell_mask, ellipse, (255, 255, 255), thickness=cv2.FILLED)
    dilated_minor_axis_b = 2 * minor_axis_b
    dilated_major_axis_a = 2 * major_axis_a
    dilated_ellipse = ((centx, centy), (dilated_minor_axis_b, dilated_major_axis_a), angle)
    dilated_area_mask = np.zeros_like(trimg)
    dilated_area_mask = cv2.ellipse(dilated_area_mask, dilated_ellipse, (255, 255, 255), thickness=cv2.FILLED)
    neighbouring_area = cv2.subtract(dilated_area_mask, cell_mask)
    neighbouring_area = cv2.bitwise_and(trimg, neighbouring_area)
    neighbouring_area_intensity = np.mean(neighbouring_area[neighbouring_area != 0])
    # print("neighbouring_area:", neighbouring_area_intensity)
    # print("cell_intensity:", cell_intensity)
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
            # similar_radius = (radius + 5 > axes1 > radius - 5) and (radius + 5 > axes1 > radius - 5)
            # the axes cannot be too small
            too_small = axes1 < 7 and axes2 < 9
            if not too_small:
                return cell_ellipse
    return 'ellipse not found'


# Write Results to an Excel file
def write_excel_file(cells: dict, dataset_number, file_number, output_folder_path, output_file_name, bf_only):
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
            column_order = ['center', 'averaged diameter in pixels', 'averaged diameter', 'perimeter', 'area', 'aspect ratio', 'taylor', 'eccentricity',
                            'circularity', 'ellipse properties']
            dataset_name = 'bf ' + str(dataset_number)
        else:
            column_order = ['center', 'averaged diameter in pixels', 'averaged diameter', 'perimeter', 'area', 'aspect ratio', 'taylor', 'eccentricity',
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
def save_images(bfimg, bfimg_copy, file_number, output_folder_path, trimg=None):
    bf_name = 'bf hough circle ' + str(file_number) + '.jpg'
    cv2.imwrite(os.path.join(output_folder_path, bf_name), bfimg)
    bf_copy_name = 'bf ellipse ' + str(file_number) + '.jpg'
    cv2.imwrite(os.path.join(output_folder_path, bf_copy_name), bfimg_copy)
    if trimg is not None:
        tr_name = 'Processed tr ' + str(file_number) + '.jpg'
        cv2.imwrite(os.path.join(output_folder_path, tr_name), trimg)
