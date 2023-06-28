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

# function to perform locally weighted linear regression
def local_weighted_regression(x0, X, Y, tau):
    # print("x0: ", x0)
    # print("X: ", X)
    # print("Y: ", Y)
    
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
     
    # fit model: normal equations with kernel
    xw = X.T * weights_calculate(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    # "@" is used to
    # predict value
    return x0 @ theta
 
# function to perform weight calculation
def weights_calculate(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau **2) ))
  
# Calculate NDVI from input image
def create_ndvi_image(ndvi, name_file):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
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
    ndviFolder  = "ndviFolder_w"
    ndviFolder_s  = "ndviFolder_w_s"
    degree      = 1
    tau         = 0.1
    
    #Read the per-band flat field images into a single 3-band VigImg
    vigImg = []
    vigImg.extend(get_tif_files_in_dir(corrFolder))
    vigImg.sort()  # [B,G,R] = [0,1,2]
    
    print('\n(1/2) Computing Calibration Values') #Analyze photo of MAPIR Calibration Target V2
    xred, xgreen, xblue, yred, ygreen, yblue = get_calibration_coefficients_from_target_image(calib_photo, inFolder, vigImg, degree, "weighted")

    x_n = np.arange(0, 1.0001, 0.0001).round(4)
    red_plot = [local_weighted_regression(x, xred, yred, tau) for x in x_n]
    green_plot = [local_weighted_regression(x, xgreen, ygreen, tau) for x in x_n]
    blue_plot = [local_weighted_regression(x, xblue, yblue, tau) for x in x_n]
    
    plt.plot(x_n, red_plot)
    plt.scatter(xred, yred)
    plt.savefig("red_plot.png")
    plt.close()
    
    plt.plot(x_n, blue_plot)
    plt.scatter(xblue, yblue)
    plt.savefig("blue_plot.png")
    plt.close()
    
    red_predict  = dict(zip(x_n, red_plot))
    blue_predict = dict(zip(x_n, blue_plot))   
    
    print('\n(2/2) Calibrating Images\n') #Apply calibration formula to input images
    
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
                
                img = ApplyVig(os.path.join(path, file_name), FileType_calib, vigImg)
                # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
                shape = img.shape
                s1 = shape[0]
                s2 = shape[1]

                # full_path_out = os.path.join(outFolder, file_name)
                full_path_ndvi = os.path.join(ndviFolder, file_name)
                full_path_ndvi_s = os.path.join(ndviFolder_s, file_name)
                      
                if FileType_calib == "JPG":
                    img   = img/255.0
                elif FileType_calib == "TIFF":
                    img   = img/65535.0
                    
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                red = min_max_scaler.fit_transform(img[:,:,2].astype("float").reshape(-1,1)).round(4)
                blue = min_max_scaler.fit_transform(img[:,:,0].astype("float").reshape(-1,1)).round(4) 
                # print("red: ", red)
                # print("blue: ", blue)
                
                print("Begining Calibration NIR")
                # blue_c   = np.array([local_weighted_regression(x, xblue, yblue, tau) for x in blue]).reshape(-1,1)   
                blue_c = np.array([blue_predict[x] for x in blue.reshape(-1).tolist()]).reshape(-1,1)
                print("Begining Calibration RED")
                # red_c   = np.array([local_weighted_regression(x, xred, yred, tau) for x in red]).reshape(-1,1)
                red_c = np.array([red_predict[x] for x in red.reshape(-1).tolist()]).reshape(-1,1)
                
                red_c = min_max_scaler.fit_transform(red_c)
                blue_c = min_max_scaler.fit_transform(blue_c)
                    
                # ndvi = (blue.astype("float") - red.astype("float") + 0.00000001) / (blue.astype("float") + red.astype("float") + 0.00000001)
                ndvi = (blue_c.astype("float") - red_c.astype("float")) / (blue_c.astype("float") + red_c.astype("float") + 0.00000001)
                print(ndvi.max(), ndvi.min())
                create_ndvi_image(ndvi.reshape((s1, s2)), full_path_ndvi)
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                ndvi = min_max_scaler.fit_transform(ndvi)
                create_ndvi_image(ndvi.reshape((s1, s2)), full_path_ndvi_s)
                
                

    print('\nFinished Processing\n') 
    
if __name__ == '__main__':
    main()