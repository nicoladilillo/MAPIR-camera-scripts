# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

from genericpath import samefile
from scipy import stats
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import sys
import glob
import os
import Geometry
from PIL import Image
from ExifUtils import *
from PIL.TiffTags import TAGS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Measurements (inches) for point to point real world distances between MAPIR Calibration Target V2 QR symbol (aruco) and reflectance target panel centers
QR_CORNER_TO_CORNER = 5.5
QR_CORNER_TO_TARG_1_3 = 9.0
QR_CORNER_TO_TARG_2_4 = 10.5

#Computed reflectance (percent) for Survey3 camera (filter) models
refvalues = {
    "550/660/850": [[0.8689592421, 0.2656248359, 0.1961875592, 0.0195576511], [0.8775934407, 0.2661207692, 0.1987265874, 0.0192249327],
                    [0.8653063177, 0.2798126291, 0.2337498097, 0.0193295348]],
    "490/615/808": [[0.8414604806, 0.2594283565, 0.1897271608, 0.0197180224],
                    [0.8751529643, 0.2673261446, 0.2007025375, 0.0192817427],
                    [0.868782908, 0.27845399, 0.2298671821, 0.0211305297]],
    "475/550/850": [[0.8348841674, 0.2580074987, 0.1890252099, 0.01975703], [0.8689592421, 0.2656248359, 0.1961875592, 0.0195576511],
                    [0.8653063177, 0.2798126291, 0.2337498097, 0.0193295348]],
    "725" : [[0.8688518306024209, 0.26302553751154756, 0.2127410973890211, 0.019551020566927594],[0, 0, 0], [0, 0, 0]],
    "850": [[0.8649280907, 0.2800907016, 0.2340131491, 0.0195446727], [0, 0, 0], [0, 0, 0]]
}

calibration_coefficients = {
    "red":   {"slope": 0.00, "intercept": 0.00},
    "green": {"slope": 0.00, "intercept": 0.00},
    "blue":  {"slope": 0.00, "intercept": 0.00},
    "mono":  {"slope": 0.00, "intercept": 0.00}
}

#Read file applying vignette correction
def ApplyVig(target_image_path, FileType_calib, vigImg, vig=True):   
    
    img = cv2.imread(target_image_path)
    
    if not vig:
        return img
    
    if FileType_calib == 'TIFF':
        dc = [120, 119, 119]
    else:
        # return img  
        dc = [0, 0, 0]
        
    #Create the dark current image (subMatrix) to be subtracted from img
    
    subMatrixB = np.full(shape = (3000, 4000),fill_value = dc[0],dtype=int)
    subMatrixG = np.full(shape = (3000, 4000),fill_value = dc[1],dtype=int)
    subMatrixR = np.full(shape = (3000, 4000),fill_value = dc[2],dtype=int)

    if FileType_calib == 'TIFF':
        subMatrixB = subMatrixB.astype("uint32").astype("uint16")
        subMatrixG = subMatrixG.astype("uint32").astype("uint16") 
        subMatrixR = subMatrixR.astype("uint32").astype("uint16") 
        dc = [120, 119, 119]
    else:
        subMatrixB = subMatrixB.astype("uint8")
        subMatrixG = subMatrixG.astype("uint8") 
        subMatrixR = subMatrixR.astype("uint8")
        dc = [0, 0, 0]

    #Split img into 3 channels
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0] 

    #Subtract dark current image from each channel of img
    b -= subMatrixB
    g -= subMatrixG
    r -= subMatrixR
    
    #Split vigImg into 3 channels
    
    vigB = cv2.imread(vigImg[0],-1)
    vigG = cv2.imread(vigImg[1],-1)
    vigR = cv2.imread(vigImg[2],-1)
    
    #Apply flat field (vignette) correction by dividing vigImg and img per channel

    b = np.divide(b,vigB)
    g = np.divide(g,vigG)
    r = np.divide(r,vigR)
    
    #Clip off any values outside the bitdepth range (keep exposure lower to reduce clipping)
    if FileType_calib == 'TIFF':
        b[b > 65535.0] = 65535.0
        b[b < 0.0] = 0.0
        
        g[g > 65535.0] = 65535.0
        g[g < 0.0] = 0.0
        
        r[r > 65535.0] = 65535.0
        r[r< 0.0] = 0.0
    
        color = cv2.merge((b,g,r))
        color = color.astype("uint32").astype("uint16")
    else:
        b[b > 255.0] = 255.0
        b[b < 0.0] = 0.0
        
        g[g > 255.0] = 255.0
        g[g < 0.0] = 0.0
        
        r[r > 255.0] = 255.0
        r[r< 0.0] = 0.0
    
        color = cv2.merge((b,g,r))                
        color = color.astype("uint8") 

    #Return the corrected img
    return color

#Outputs modified calibration photo image showing pixels used in each reflectance panel for calibration values
def print_center_targs(image, target1, target2, target3, target4, sample_diameter):

    image_line = image.split(".")[0] + "_circles." + image.lower().split(".")[1]
    line_image = cv2.imread(image, -1)

    if image.lower().endswith(('jpg','jpeg')):
        cv2.circle(line_image,target1, sample_diameter, (0,0,255), -1) #Red
        cv2.circle(line_image,target2, sample_diameter, (255,0,0), -1) #Blue
        cv2.circle(line_image,target3, sample_diameter, (0,255,255), -1) #Yellow
        cv2.circle(line_image,target4, sample_diameter, (255,0,255), -1) #Pink
    elif image.lower().endswith(('tif','tiff')):
        cv2.circle(line_image,target1, sample_diameter, (0,0,65535), -1) #Red
        cv2.circle(line_image,target2, sample_diameter, (65535,0,0), -1) #Blue
        cv2.circle(line_image,target3, sample_diameter, (0,65535,65535), -1) #Yellow
        cv2.circle(line_image,target4, sample_diameter, (65535,0,65535), -1) #Pink

    cv2.imwrite(image_line, line_image)

#Checks whether the image is a 3 channel RGB
def is_color_image(img):
    return len(img.shape) > 2

#Converts calib photo to grayscale for easier QR detection
def prep_target_image_for_detection(target_img):
    img = cv2.imread(target_img, 0)

    if is_color_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#Stretches the calib photo's contrast to improve detection
def contrast_stretch(img, threshold):
    bit_depth = int(img.dtype.name[4:])
    pixel_range_max = 2**bit_depth-1
    for pixel_row in img:
        pixel_row[pixel_row <= threshold] = 0 # Black
        pixel_row[pixel_row > threshold] = pixel_range_max # White

#Stretches the calib photo's contrast to improve detection, using image's pixel range midpoint
def midpoint_threshold_contrast_stretch(img):
    stretch_img = img.copy()
    threshold = (stretch_img.max() + 1 - stretch_img.min()) / 2 - 1
    contrast_stretch(stretch_img, threshold)
    return stretch_img

#Stretches the calib photo's contrast to improve detection, using image's pixel range mode
def mode_threshold_contrast_stretch(img):
    stretch_img = img.copy()
    threshold = stats.mode(stretch_img.flatten())[0][0]
    contrast_stretch(stretch_img, threshold)
    return stretch_img

def filter_detected_targets_by_id(corners, ids, target_id):
    return [i for i, j in zip(corners, ids) if j == target_id]

#Gets the pixel locations of the QR pattern's corners in the calibration photo
def get_image_corners(target_img_path):
    img = prep_target_image_for_detection(target_img_path)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    corners, ids, _= aruco.detectMarkers(img, aruco_dict)

    #If QR is not detected then contrast stetch around pixel range mode to improve detection
    if corners == []:
        mode_stretched = mode_threshold_contrast_stretch(img)
        corners, ids, _ = aruco.detectMarkers(mode_stretched, aruco_dict)

    #If QR is not detected then contrast stetch around pixel range midpoint to improve detection
    if corners == []:
        mid_stretched = midpoint_threshold_contrast_stretch(img)
        corners, ids, _ = aruco.detectMarkers(mid_stretched, aruco_dict)

    if corners == []:
        raise Exception('Could not find MAPIR Calibration Target V2. Please try a different calibration photo.')

    QR_matches = filter_detected_targets_by_id(corners, ids, 13)[0]
    single_QR = QR_matches[0]

    QR_corner_ints = [[int(corner[0]), int(corner[1])] for corner in single_QR]

    QR_corner_ints_reordered = [
        QR_corner_ints[0],
        QR_corner_ints[1],
        QR_corner_ints[3],
        QR_corner_ints[2]
    ]

    return QR_corner_ints_reordered

def get_version_2_QR_corners(target_image_path):
    return get_image_corners(target_image_path)

#Checks whether the 4 reflectance panels increase/decrease in reflectance as expected
def check_exposure_quality(x, y):
    if (x[0] == 1 and x[-1] == 0):
        x = x[1:]
        y = y[1:]

    elif (x[0] == 1):
        x = x[1:]
        y = y[1:]

    elif (x[-1] == 0):
        x = x[:-1]
        y = y[:-1]

    return x, y

#If reflectnace is not sorted correctly then calibration photo is "bad", we suggest using a different one
def bad_target_photo(channels):
    for channel in channels:
        if channel != sorted(channel, reverse=True):
            return True

        for targ in channel:
            if math.isnan(targ):
                return True

    return False

#Calculates the linear regression of the channel model
def get_channel_model(xr, xg, xb, y):
    try:
        X = np.array([xr, xg, xb]).T
        X = np.append(X, [[0., 0.0, 0.0]], axis=0)
        y = np.append(y, [0], axis=0)
        print(y.shape)
        print(y)
        print(X.shape)
        print(X)

        return LinearRegression().fit(X, y)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error: ", e)
        print("Line: " + str(exc_tb.tb_lineno))

#Calculates the linear regression between the points produced from known reflectance of panels versus calibration photo pixel values
def get_line_of_best_fit(x, y):
    try:
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numer = sum((x - mean_x) * (y - mean_y))
        denom = sum(np.power(x - mean_x, 2))

        slope = numer / denom
        intercept = mean_y - (slope * mean_x)

        return slope, intercept
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error: ", e)
        print("Line: " + str(exc_tb.tb_lineno))

#Calculates the pixel locations for the target panel centers, taking QR rotation into account
def get_target_center_from_QR_corner(qr_corner_to_target_in_pixels, dx, dy, QR_corner, angle):

    y_shift = int(qr_corner_to_target_in_pixels * math.sin(angle))
    x_shift = int(qr_corner_to_target_in_pixels * math.cos(angle))
        
    #angle regions below in clockwise rotation direction

    if dx >= 0: #angle = 0-89d & 270-359d
        target_x = QR_corner[0] - x_shift
        if dy >= 0: #angle = 0-89d            
            target_y = QR_corner[1] - y_shift
        else: #angle = 270-359d
            target_y = QR_corner[1] - y_shift
    else: #angle = 90-269d
        target_x = QR_corner[0] + x_shift
        if dy >= 0: #angle = 90-179d            
            target_y = QR_corner[1] + y_shift
        else: #angle = 180-269d
            target_y = QR_corner[1] + y_shift
    
    return (int(target_x), int(target_y))    

#Averages the pixel values for each panel's sampling area
def get_reflectance_target_sample_pixels(image, target_center, target_sample_area_width_in_pixels):
    half_sample_width = target_sample_area_width_in_pixels / 2

    return image[
        int(target_center[1] - half_sample_width):int(target_center[1] + half_sample_width),
        int(target_center[0] - half_sample_width):int(target_center[0] + half_sample_width)
    ]

#Checks whether input contains RAW, TIFF or JPG image(s)
def check_input_folder_structure(in_folder):
    infiles = []    
    infiles.extend(glob.glob(in_folder + os.sep + "*.[rR][aA][wW]"))
    numFiles = len(infiles)
    if numFiles > 0:
        sys.exit("RAW images not supported. Please convert to TIFF.")

    infiles.extend(glob.glob(in_folder + os.sep + "*.[Tt][Ii][Ff]"))
    infiles.extend(glob.glob(in_folder + os.sep + "*.[Tt][Ii][Ff][Ff]"))
    numFiles = len(infiles)

    if numFiles > 0:
        FileType = "TIFF"
    else:
        infiles.extend(glob.glob(in_folder + os.sep + "*.[jJ][pP][gG]"))
        infiles.extend(glob.glob(in_folder + os.sep + "*.[jJ][pP][eE][gG]"))
        numFiles = len(infiles)
        if numFiles > 0:
            FileType = "JPG"
        else:
            sys.exit("No images to process in " + in_folder)    

    return infiles[0], FileType

#Reads image metadata (exif) to determine camera's model and filter used
def check_images_params(image_path, FileType):
    camera_model, camera_filter = 'CAMERA_MODEL', 'CAMERA_FILTER' 
    
    if FileType == "JPG":
        image = Image.open(image_path)

        exifdata = image.getexif()
        camera_model = exifdata.get(272,272)
        camera_filter = camera_model[-3:]

    else: #TIFF
        image = Image.open(image_path)

        camera_model = image.tag_v2[272]
        camera_filter = camera_model[-3:]

    return camera_model, camera_filter 

#Main function to produce calibration values from calibration photo
def get_calibration_coefficients_from_target_image(target_image_path, in_folder, vigImg):
    
    FileType_calib = FileType_function(target_image_path)
 
    camera_model_calib, camera_filter_calib = check_images_params(target_image_path, FileType_calib)

    img_folder, FileType_img  = check_input_folder_structure(in_folder)
    camera_model_img, camera_filter_img = check_images_params(img_folder, FileType_img)

    # print(f"\nCamera model {camera_model_calib} and camera filter {camera_filter_img}")

    # if camera_model_calib != camera_model_img or camera_filter_calib != camera_filter_img  or FileType_calib !=FileType_img :
    #     sys.exit("Calibration photo does not match input image (EXIF).")

    try:
        calibration_coefficients = {
            "red":   {"slope": 0.00, "intercept": 0.00},
            "green": {"slope": 0.00, "intercept": 0.00},
            "blue":  {"slope": 0.00, "intercept": 0.00},
            "mono":  {"slope": 0.00, "intercept": 0.00}
        }

        QR_corners = get_version_2_QR_corners(target_image_path)

        top_left = QR_corners[0]
        top_right = QR_corners[1]
        bottom_left = QR_corners[2]

        slope = Geometry.slope(top_right, bottom_left)
        dist = top_left[1] - (slope * top_left[0]) + ((slope * bottom_left[0]) - bottom_left[1])
        dist /= np.sqrt(np.power(slope, 2) + 1)

        slope_top_right_to_top_left = Geometry.slope(top_right, top_left)
        angle = math.atan(slope_top_right_to_top_left)
        
        if len(QR_corners) > 0:
            qr_height_in_pixels = Geometry.distance(top_left, bottom_left)
            pixels_per_inch = qr_height_in_pixels / QR_CORNER_TO_CORNER

            corner_to_target_1_3_in_pixels = (pixels_per_inch * QR_CORNER_TO_TARG_1_3)
            corner_to_target_2_4_in_pixels = (pixels_per_inch * QR_CORNER_TO_TARG_2_4)
            
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]

            target1 = get_target_center_from_QR_corner(corner_to_target_1_3_in_pixels, dx, dy, QR_corners[0], angle)
            target2 = get_target_center_from_QR_corner(corner_to_target_2_4_in_pixels, dx, dy, QR_corners[1], angle)
            target3 = get_target_center_from_QR_corner(corner_to_target_1_3_in_pixels, dx, dy, QR_corners[2], angle)
            target4 = get_target_center_from_QR_corner(corner_to_target_2_4_in_pixels, dx, dy, QR_corners[3], angle)

        im2 = ApplyVig(target_image_path, FileType_calib, vigImg)
        # im2 = cv2.imread(target_image_path, -1)
        print(im2.shape)
        target_sample_area_width_in_pixels = int(pixels_per_inch * 0.75)

        try:

            targ1values = get_reflectance_target_sample_pixels(im2, target1, target_sample_area_width_in_pixels)
            targ2values = get_reflectance_target_sample_pixels(im2, target2, target_sample_area_width_in_pixels)
            targ3values = get_reflectance_target_sample_pixels(im2, target3, target_sample_area_width_in_pixels)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(e)
            print("Line: " + str(exc_tb.tb_lineno))


        t1redmean = np.mean(targ1values[:, :, 2])
        t1greenmean = np.mean(targ1values[:, :, 1])
        t1bluemean = np.mean(targ1values[:, :, 0])

        t2redmean = np.mean(targ2values[:, :, 2])
        t2greenmean = np.mean(targ2values[:, :, 1])
        t2bluemean = np.mean(targ2values[:, :, 0])

        t3redmean = np.mean(targ3values[:, :, 2])
        t3greenmean = np.mean(targ3values[:, :, 1])
        t3bluemean = np.mean(targ3values[:, :, 0])

        yred = []
        yblue = []
        ygreen = []

        if len(QR_corners) > 0:
            half_target_sample_area_width_in_pixels = int(pixels_per_inch * 0.75 / 2)
            targ4values = im2[(target4[1] - half_target_sample_area_width_in_pixels):(target4[1] + half_target_sample_area_width_in_pixels),
                            (target4[0] - half_target_sample_area_width_in_pixels):(target4[0] + half_target_sample_area_width_in_pixels)]
            t4redmean = np.mean(targ4values[:, :, 2])
            t4greenmean = np.mean(targ4values[:, :, 1])
            t4bluemean = np.mean(targ4values[:, :, 0])
            yred = [0.87, 0.51, 0.23, 0.0]
            yblue = [0.87, 0.51, 0.23, 0.0]
            ygreen = [0.87, 0.51, 0.23, 0.0]

            xred = [t1redmean, t2redmean, t3redmean, t4redmean]
            xgreen = [t1greenmean, t2greenmean, t3greenmean, t4greenmean]
            xblue = [t1bluemean, t2bluemean, t3bluemean, t4bluemean]

        print_center_targs(target_image_path, target1, target2, target3, target4, target_sample_area_width_in_pixels)

        if "Survey3" in camera_model_calib:
            if camera_filter_calib == "OCN":
                yred = refvalues["490/615/808"][0]
                ygreen = refvalues["490/615/808"][1]
                yblue = refvalues["490/615/808"][2]
            elif camera_filter_calib == "RGN":
                yred = refvalues["550/660/850"][0]
                ygreen = refvalues["550/660/850"][1]
                yblue = refvalues["550/660/850"][2]
            elif camera_filter_calib == "NGB":
                yred = refvalues["475/550/850"][0]
                ygreen = refvalues["475/550/850"][1]
                yblue = refvalues["475/550/850"][2]
            elif camera_filter_calib == "RE":
                yred = refvalues["725"][0]
                ygreen = refvalues["725"][1]
                yblue = refvalues["725"][2]
            elif camera_filter_calib == "NIR":
                yred = refvalues["850"][0]
                ygreen = refvalues["850"][1]
                yblue = refvalues["850"][2]

            if FileType_calib == "JPG":
                xred = [x / 255 for x in xred]
                xgreen = [x / 255 for x in xgreen]
                xblue = [x / 255 for x in xblue]
            elif FileType_calib == "TIFF":
                xred = [x / 65535 for x in xred]
                xgreen = [x / 65535 for x in xgreen]
                xblue = [x / 65535 for x in xblue]

            xred, yred = check_exposure_quality(xred, yred)
            xgreen, ygreen = check_exposure_quality(xgreen, ygreen)
            xblue, yblue = check_exposure_quality(xblue, yblue)

            x_channels = [xred, xgreen, xblue]
            
        # print("\nRed values: " + str(xred))
        # print("Green values: " + str(xgreen))
        # print("Blue values: " + str(xblue))
        
        red_model   = get_channel_model(xred, xgreen, xblue, yred)
        green_model = get_channel_model(xred, xgreen, xblue, ygreen)
        blue_model  = get_channel_model(xred, xgreen, xblue, yblue)
        
        print("\nRed model: " + str(red_model.coef_)   + " * x + " + str(red_model.intercept_))
        print("Green model: " + str(green_model.coef_) + " * x + " + str(green_model.intercept_))
        print("Blue model : " + str(blue_model.coef_)   + " * x + " + str(blue_model.intercept_))

        red_slope, red_intercept = get_line_of_best_fit(xred, yred)
        green_slope, green_intercept = get_line_of_best_fit(xgreen, ygreen)
        blue_slope, blue_intercept = get_line_of_best_fit(xblue, yblue)

        calibration_coefficients["red"]["slope"] = red_slope
        calibration_coefficients["red"]["intercept"] = red_intercept

        calibration_coefficients["green"]["slope"] = green_slope
        calibration_coefficients["green"]["intercept"] = green_intercept

        calibration_coefficients["blue"]["slope"] = blue_slope
        calibration_coefficients["blue"]["intercept"] = blue_intercept

        if len(QR_corners) > 0:
            print("\nFound MAPIR Calibration Target V2, proceeding with calibration.")
        else:
            sys.exit("\nCould not find MAPIR Calibration Target V2. Please try a different calibration photo.")

        if bad_target_photo(x_channels):
            print("\nWARNING: BAD CALIBRATION PHOTO")

        return red_model, green_model, blue_model, calibration_coefficients, FileType_calib

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(str(e) + ' Line: ' + str(exc_tb.tb_lineno))

def FileType_function(target_image_path):
    if target_image_path.lower().endswith(('jpg','jpeg')):
        FileType_calib = 'JPG'
    elif target_image_path.lower().endswith(('tif','tiff')):
        FileType_calib = 'TIFF'
    else:
        sys.exit("Unknown calibration image format. Requires JPG or TIFF.")
    return FileType_calib
