import cv2
import numpy as np
import os
import tifffile

# ** Choose File **
# path to the folder containing images
input_folder_path = 'DeviceN'
# number label for the set of images
file_number = 1

# Load the image
bfimg = 'bf ' + str(file_number) + '.tif'
bfimg_path = os.path.join(input_folder_path, bfimg)
rgba_bfimg = tifffile.imread(bfimg_path)
bfimg = cv2.cvtColor(rgba_bfimg, cv2.COLOR_RGBA2BGR)

trimg = 'tr ' + str(file_number) + '.tif'
trimg_path = os.path.join(input_folder_path, trimg)
rgba_trimg = tifffile.imread(trimg_path)
trimg = cv2.cvtColor(rgba_trimg, cv2.COLOR_RGBA2BGR)

# Get image dimensions
height, width, _ = bfimg.shape

# Split the image vertically
bf_left_half = bfimg[:, :width//2]
bf_right_half = bfimg[:, width//2:]

tr_left_half = trimg[:, :width//2]
tr_right_half = trimg[:, width//2:]

# save_image
bf_name_left = 'bf ' + str(file_number) + 'l' + '.tif'
bf_name_right = 'bf ' + str(file_number) + 'r' + '.tif'
cv2.imwrite(os.path.join(input_folder_path, bf_name_left), bf_left_half)
cv2.imwrite(os.path.join(input_folder_path, bf_name_right), bf_right_half)

tr_name_left = 'tr ' + str(file_number) + 'l' + '.tif'
tr_name_right = 'tr ' + str(file_number) + 'r' + '.tif'
cv2.imwrite(os.path.join(input_folder_path, tr_name_left), tr_left_half)
cv2.imwrite(os.path.join(input_folder_path, tr_name_right), tr_right_half)

# Display the split images
cv2.imshow('bf Left Half', bf_left_half)
cv2.imshow('bf Right Half', bf_right_half)
cv2.imshow('tr Left Half', tr_left_half)
cv2.imshow('tr Right Half', tr_right_half)

cv2.waitKey(0)
cv2.destroyAllWindows()