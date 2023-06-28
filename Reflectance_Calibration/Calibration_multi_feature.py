# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

import os, sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
from sklearn import preprocessing
from Calibration_Utils import get_calibration_coefficients_from_target_image, ApplyVig, FileType_function
from ExifUtils import *
from sklearn.preprocessing import PolynomialFeatures, normalize
  
# Calculate NDVI from input image
def create_ndvi_image(ndvi, name_file):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(ndvi.reshape(3000, 4000), cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title('NDVI Image')
    fig.colorbar(im)
    plt.savefig(name_file)

#Apply calibration formula to pixel values
def calibrate_channel(mult_values, value):
    slope = mult_values["slope"]
    intercept = mult_values["intercept"]
    return (slope * value) + intercept

#Stretch image contrast to global calib max/min pixel values
def contrast_stretch_channel(global_cal_max, global_cal_min, rangeMax, value):
    value = rangeMax * ((value - global_cal_min) / (global_cal_max - global_cal_min))    
    return value

#Calibrate the dataset global min/max pixel values
def calibrate_extrema(img):
    max_mins = {"redmaxs": [], "redmins": [], "greenmaxs": [], "greenmins": [], "bluemaxs": [], "bluemins": []}

    max_mins["redmaxs"].append(img[:, :, 2].max())
    max_mins["redmins"].append(img[:, :, 2].min())

    max_mins["greenmaxs"].append(img[:, :, 1].max())
    max_mins["greenmins"].append(img[:, :, 1].min())

    max_mins["bluemaxs"].append(img[:, :, 0].max())
    max_mins["bluemins"].append(img[:, :, 0].min())


    global_maxes = {"red": max(max_mins["redmaxs"]),
                    "green": max(max_mins["greenmaxs"]),
                    "blue": max(max_mins["bluemaxs"])}

    global_mins = {"red": min(max_mins["redmins"]),
                    "green": min(max_mins["greenmins"]),
                    "blue": min(max_mins["bluemins"])}

    extrema = {"calib": {"max": global_maxes, "min": global_mins}}
    return extrema 

#Get pixel min/max for each image channel
def get_channel_extrema_for_image(image):
    channels = [
        image[:, :, 0],
        image[:, :, 1],
        image[:, :, 2]
    ]

    maxes = list(map(np.max, channels))
    mins = list(map(np.min, channels))
    return maxes, mins

#Get pixel min/max for entire input folder of images
def get_channel_extrema_for_project(inFolder):
    max_int = sys.maxsize
    min_int = -max_int - 1

    maxes = [min_int, min_int, min_int]
    mins = [max_int, max_int, max_int]

    for path, subdirs, files, in os.walk(inFolder):
        if files:
            for file_path in files:
                image = cv2.imread(os.path.join(path, file_path))
                tile_maxes, tile_mins = get_channel_extrema_for_image(image)

                for i in range(3):
                    if tile_maxes[i] > maxes[i]:
                        maxes[i] = int(tile_maxes[i])
                    if tile_mins[i] < mins[i]:
                        mins[i] = int(tile_mins[i])

    return maxes, mins

#Apply calibration formulas to input folder's global min/max
def get_global_calib_extrema(calibration_values, global_max, global_min):
    global_cal_maxes = []
    global_cal_mins = []

    global_cal_maxes.append( int(calibrate_channel(calibration_values["red"], global_max[0])) )
    global_cal_mins.append( int(calibrate_channel(calibration_values["red"], global_min[0])) )

    global_cal_maxes.append( int(calibrate_channel(calibration_values["green"], global_max[1])) )
    global_cal_mins.append( int(calibrate_channel(calibration_values["green"], global_min[1])) )
    
    global_cal_maxes.append( int(calibrate_channel(calibration_values["blue"], global_max[2])) )
    global_cal_mins.append( int(calibrate_channel(calibration_values["blue"], global_min[2])) )
    
    global_cal_max = max(global_cal_maxes)
    global_cal_min = min(global_cal_mins)

    return global_cal_max, global_cal_min

#returns all tif files in the current directory
def get_tif_files_in_dir(dir_name):
    file_paths = []
    file_paths.extend(glob.glob(dir_name + os.sep + "*.[tT][iI][fF]"))
    return file_paths


def main():
    
    #Read arguments from bat file
    calib_photo = "calib/calib_3.JPG"
    calib_photo = "calib/calib.tif"
    calib_photo = "calib/calib_2.jpg"
    corrFolder  = "flatFields"
    inFolder    = "inFolder_copy"
    inFolder    = "inFolder"
    outFolder   = "outFolder_multi"
    ndviFolder  = "ndviFolder_multi"
    degree      = 1
    
    #Read the per-band flat field images into a single 3-band VigImg
    vigImg = []
    vigImg.extend(get_tif_files_in_dir(corrFolder))
    vigImg.sort()  # [B,G,R] = [0,1,2]
    
    print('\n(1/3) Computing Calibration Values') #Analyze photo of MAPIR Calibration Target V2
    red_model, green_model, blue_model, calibration_values, FileType_calib = get_calibration_coefficients_from_target_image(calib_photo, inFolder, vigImg, degree)

    print('\n(2/3) Analyzing Input Images') #Analyze input image folder
    # maxes, mins  = get_channel_extrema_for_project(inFolder)
    # global_cal_max, global_cal_min = get_global_calib_extrema(calibration_values, maxes, mins)

    print('\n(3/3) Calibrating Images\n') #Apply calibration formula to input images
    
    for path, subdirs, files, in os.walk(inFolder):
        if files:
            for file_name in files:
                FileType_calib = FileType_function(file_name)
                # print("FileType_calib: ", FileType_calib)
                # img = cv2.imread(os.path.join(path, file_name))
                # img = ApplyVig(os.path.join(path, file_name), FileType_calib, vigImg)
                
                # print("Begining Calibration of...")
                # print("red: ", img[:,:,2].max(), img[:,:,2].min())
                # print("green: ", img[:,:,1].max(), img[:,:,1].min())
                # print("blue: ", img[:,:,0].max(), img[:,:,0].min())
                
                img = ApplyVig(os.path.join(path, file_name), FileType_calib, vigImg).reshape(3000 * 4000, 3)   
                # print(img.max(), img.min())

                full_path_out = os.path.join(outFolder, file_name)
                full_path_ndvi = os.path.join(ndviFolder, file_name)
                      
                if FileType_calib == "JPG":
                    img   = img/255.0
                elif FileType_calib == "TIFF":
                    img   = img/65535.0
                    
                # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                # img = min_max_scaler.fit_transform(img)
                
                # red   = img[:,2].astype("float").reshape(-1,1)
                # green = img[:,1].astype("float").reshape(-1,1)
                # blue  = img[:,0].astype("float").reshape(-1,1)
                # # print(red.max(), red.min())
                # # print(green.max(), green.min())
                # # print(blue.max(), blue.min())
                    
                # poly = PolynomialFeatures(degree=degree, include_bias=False)
                # red   = red_model.predict(poly.fit_transform(red))
                # green = green_model.predict(poly.fit_transform(green))
                # blue  = blue_model.predict(poly.fit_transform(blue))
                
                red   = red_model.predict(img).reshape(-1,1)
                green = green_model.predict(img).reshape(-1,1)
                blue  = blue_model.predict(img).reshape(-1,1)
                # red[red > 1] = 1
                # red[red < 0] = 0
                # green[green > 1] = 1    
                # green[green < 0] = 0    
                # blue[blue > 1] = 1
                # blue[blue < 0] = 0 
                # print(red.max(), red.min())
                # print(green.max(), green.min())
                # print(blue.max(), blue.min())
                # print(red.shape)    
                
                
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                red = min_max_scaler.fit_transform(red)
                print(red.max(), red.min())
                print(red)
                print(blue.max(), blue.min())
                print(blue)
                green = min_max_scaler.fit_transform(green)
                blue = min_max_scaler.fit_transform(blue)
                # print(red.max(), red.min())
                # print(green.max(), green.min())
                # print(blue.max(), blue.min())
                
                img_merge = cv2.merge((blue.reshape(3000,4000), green.reshape(3000,4000), red.reshape(3000,4000)))
                if FileType_calib == "JPG":
                    img_merge = img_merge.astype("uint8")
                    cv2.imencode(".jpg", img_merge)
                elif FileType_calib == "TIFF":
                    img_merge = img_merge.astype("uint32")
                    img_merge = img_merge.astype("uint16")
                    cv2.imencode(".tif", img_merge)
                cv2.imwrite(full_path_out, img_merge) 
                print(full_path_out)
                
                # print("... after Calibration")
                # print("red: ", red.max(), red.min())
                # print("green: ", green.max(), green.min())
                # print("blue: ", blue.max(), blue.min())

                    
                # ndvi = (blue.astype("float") - red.astype("float") + 0.00000001) / (blue.astype("float") + red.astype("float") + 0.00000001)
                ndvi = (blue.astype("float") - red.astype("float")) / (blue.astype("float") + red.astype("float") + 0.00000001)
                # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                # ndvi = min_max_scaler.fit_transform(ndvi)
                # print(ndvi.max(), ndvi.min())
                
                create_ndvi_image(ndvi.reshape(3000, 4000), full_path_ndvi)
                

    print('\nFinished Processing\n') 
    
if __name__ == '__main__':
    main()