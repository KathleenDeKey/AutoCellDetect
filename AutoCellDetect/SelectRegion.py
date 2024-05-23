import cv2
import numpy as np
import os
import tifffile

# will split the target image vertically or horizontally, will save the image in the same folder
# splitType: defines whether image is split vertically or horizontally;
# valid values for splitType = "vertical", "horizontal"
def splitImage(input_folder_path, file_number, splitType, bfOnly = False):
    # Load the image
    bfimg = 'bf ' + str(file_number) + '.tif'
    bfimg_path = os.path.join(input_folder_path, bfimg)
    rgba_bfimg = tifffile.imread(bfimg_path)
    bfimg = cv2.cvtColor(rgba_bfimg, cv2.COLOR_RGBA2BGR)
    if (not bfOnly):
        trimg = 'tr ' + str(file_number) + '.tif'
        trimg_path = os.path.join(input_folder_path, trimg)
        rgba_trimg = tifffile.imread(trimg_path)
        trimg = cv2.cvtColor(rgba_trimg, cv2.COLOR_RGBA2BGR)

    # Get image dimensions
    height, width, _ = bfimg.shape

    # Check specification is valid
    valid_split_type = {"vertical","horizontal"}
    if splitType not in valid_split_type:
        raise ValueError("please define a valid splitType: horizontal or vertical")
    
    # Split the image vertically
    if (splitType == "vertical"):
        bf_left_half = bfimg[:, :width//2]
        bf_right_half = bfimg[:, width//2:]
        bf_name_left = 'bf ' + str(file_number) + 'l' + '.tif'
        bf_name_right = 'bf ' + str(file_number) + 'r' + '.tif'
        saveAndName(bf_name_left, input_folder_path, bf_left_half)
        saveAndName(bf_name_right, input_folder_path, bf_right_half)

        if (not bfOnly):
            tr_left_half = trimg[:, :width//2]
            tr_right_half = trimg[:, width//2:]
            tr_name_left = 'tr ' + str(file_number) + 'l' + '.tif'
            tr_name_right = 'tr ' + str(file_number) + 'r' + '.tif'
            saveAndName(tr_name_left, input_folder_path, tr_left_half)
            saveAndName(tr_name_right, input_folder_path, tr_right_half)
            
    # Split the image horizontally
    elif (splitType == "horizontal"):
        bf_top_half = bfimg[:height//2,:]
        bf_bottom_half = bfimg[height//2:, :]
        bf_name_top = 'bf ' + str(file_number) + 't' + '.tif'
        bf_name_bottom = 'bf ' + str(file_number) + 'b' + '.tif'
        saveAndName(bf_name_top, input_folder_path, bf_top_half)
        saveAndName(bf_name_bottom, input_folder_path, bf_bottom_half)

        if (not bfOnly):
            tr_top_half = trimg[:height//2,:]
            tr_bottom_half = trimg[height//2:, :]
            tr_name_top = 'tr ' + str(file_number) + 't' + '.tif'
            tr_name_bottom = 'tr ' + str(file_number) + 'b' + '.tif'
            saveAndName(tr_name_top, input_folder_path, tr_top_half)
            saveAndName(tr_name_bottom, input_folder_path, tr_bottom_half)

# save the image in the correct path and give the appropriate name
def saveAndName(file_name, input_folder_path, image):
    path = os.path.join(input_folder_path, file_name)
    cv2.imwrite(path, image)

