import os 
import cv2
import numpy as np
import SelectRegion as sr
import CellDetection as cd


# ** Choose File **
# path to the folder containing images
input_folder_path = 'DeviceN'
# number label for the set of images
file_number = 1
# the type of image you want to analyze - True: only bf images are analyzed; False: bf and tr images are analyzed
bf_only = False

# build image path
bfimg = 'bf ' + str(file_number) + '.tif'
bfimg_path = os.path.join(input_folder_path, bfimg)

trimg = 'tr ' + str(file_number) + '.tif'
trimg_path = os.path.join(input_folder_path, trimg)

# Select Region
region = sr.get_region(bfimg_path)
bf_region = sr.load_region(bfimg_path, region)
bf_region_copy = sr.load_region(bfimg_path, region)
tr_region = sr.load_region(trimg_path, region)

# name of output Excel file
output_file_name = 'DeviceN_output.xlsx'
# relative path to the folder to save your results; will create a new folder if it does not exist
output_folder_path = 'DeviceN_results_ver2'

# ** constant values **

# conversion factor
um_per_pixel = 0.3333333333333333
# circularity threshold
circularity_threshold = 0.5
# red label value
red = (36, 28, 237)

# ** Create slider window and trackbars **
cv2.namedWindow("Adjust Image", cv2.WINDOW_FREERATIO)

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
    # sets values
    ksize = cv2.getTrackbarPos("ksize", "Adjust Image")
    sigmax = cv2.getTrackbarPos("sigmax (divide by 10)", "Adjust Image") / 10.0
    alpha_contast = cv2.getTrackbarPos("alpha - contrast (divide by 10)", "Adjust Image") / 10.0
    beta_brightness = cv2.getTrackbarPos("beta - brightness", "Adjust Image")
    min_threshold = cv2.getTrackbarPos("min threshold", "Adjust Image")
    max_threshold = cv2.getTrackbarPos("max threshold", "Adjust Image")

    # ** preprocessing **

    # change contrast and brightness
    bfimg = cv2.convertScaleAbs(bf_region, alpha=alpha_contast, beta=beta_brightness)
    bfimg = cv2.cvtColor(bfimg, cv2.COLOR_RGBA2BGR)
    bfimg_copy = cv2.cvtColor(bf_region_copy, cv2.COLOR_RGBA2BGR)
    # convert to grayscale
    bfGray = cv2.cvtColor(bfimg, cv2.COLOR_BGR2GRAY)
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
    cells = cd.store_values(detected_cells, True, bfimg, edges, um_per_pixel, red)
    cd.draw_cells(cells, True, bfimg, bfimg_copy)

    # Combine and show panels
    combined_cells_image = cv2.hconcat([bfimg, bfimg_copy])
    combined_preprocessing_image = cv2.cvtColor(combined_preprocessing_image, cv2.COLOR_GRAY2BGR)
    total_combined_image = np.vstack([combined_preprocessing_image, combined_cells_image])
    # total_combined_image = cv2.resize(total_combined_image, (0,0), fx=0.7, fy=0.7)
    cv2.imshow("Region", total_combined_image)

    key = cv2.waitKey(10)  # Wait for 1 millisecond

    # Apply the parameters in processing when 'a' is pressed
    if key == 97:
        cv2.destroyAllWindows()  # clean windows

        # ** store, save and show result **
        if bf_only:
            cells = cd.store_values(detected_cells, bf_only, bfimg, processed_img=edges)
            cd.draw_cells(cells, bf_only, bfimg)
            combined_result_image = cv2.hconcat([bfimg, bf_region_copy, trimg])
            cd.save_images(bfimg, bf_region_copy)

        else:
            cells = cd.store_values(detected_cells, bf_only, bfimg, processed_img=edges, trimg=trimg)
            cd.draw_cells(cells, bf_only, bfimg, trimg)
            combined_result_image = cv2.hconcat([bfimg, bf_region_copy, trimg])
            cd.save_images(bfimg, bf_region_copy, trimg)

        cd.write_excel_file(cells, file_number, bf_only)
        cd.save_images(bfimg, bf_region_copy, trimg)
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