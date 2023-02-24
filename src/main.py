import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import math
import sys
import netCDF4 as nc
import datetime

data_dir = "../data"
out_dir = "../out"
target_cities = ["ba", "va", "pm"]
batch_size = 17
SHOW = False
if int(sys.argv[1]) == 1:
    SHOW = True

val_map_path = "../data/gt/val.png"
bar_map_path = "../data/gt/bar.png"
pal_map_path = "../data/gt/pal.png"

def get_clean_map(city):
    """
    returns the clean map corresponding to the city.
    """
    if city == "va":
        return cv2.cvtColor(cv2.imread(val_map_path), cv2.COLOR_BGR2RGB)
    elif city == "ba":
        return cv2.cvtColor(cv2.imread(bar_map_path), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(cv2.imread(pal_map_path), cv2.COLOR_BGR2RGB)

def groundtruth_map(image_sequences):
    """
    This function samples frames from the sequence of images and uses
    temporal median filter to create the background model. 
    Parameters:
        source (str): the path to the video file
        sampling_prob (float): probaility for each frame to be sampled
        save (bool): determines whether to save the baackground model as image
    Returns:
        np.ndarray: the background mask
    """
    # Background Model using median filter
    is_np = np.array(image_sequences)
    background_model = np.median(is_np, axis=0, keepdims=True).astype('uint8')
    return background_model[0] #cv2.cvtColor(, cv2.COLOR_BGR2RGB)

def get_foreground_mask(gt_mask, img, threshold = 0):
    foreground = cv2.cvtColor(cv2.absdiff(img, gt_mask), cv2.COLOR_RGB2GRAY)
    foreground[foreground > threshold] = 255
    # foreground[foreground<=threshold] = 0
    return foreground

def dataset_creator(gt, bg, img):
    """reconstruct image after removing interference"""
    res = gt.copy()
    for x in range(bg.shape[0]):
        for y in range(bg.shape[1]):
            if bg[x, y] != 0:
                res[x,y,:] = img[x,y,:]
    return res

def process_contours(out_cnt, contours):
    """detect if a contour is an interference based on aspect ratio and angle"""
    for cnt in contours:
        # crit 1:
        rows,cols = out_cnt.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        ori = math.atan2((righty - lefty),(cols-1)) * 180 / np.pi
        c1_score = abs(abs(ori) - 45)
        
        # crit 2:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        (x, y), (width, height), angle = rect
        try:
            aspect_ratio = min(width, height) / max(width, height)
        except:
            continue
        if aspect_ratio < 0.45 or (aspect_ratio < 0.4 and c1_score < 5):
            cv2.drawContours(out_cnt,[box],-1,0,-1)
    return out_cnt

def export_to_netCDF(image, timestamp, outname):
    ncfile = nc.Dataset(outname + ".nc", 'w', format='NETCDF4')

    # Create dimensions for the data
    ncfile.createDimension('time', 1)
    ncfile.createDimension('x', image.shape[0])
    ncfile.createDimension('y', image.shape[1])
    ncfile.createDimension('z', image.shape[2])
    # Create a variable for the image data
    data = ncfile.createVariable('data', np.float32, ('time', 'x', 'y', 'z'))

    # Set the variable attributes
    data.units = 'dBm'
    data.long_name = 'Radar Reading'
    data.coordinates = 'time x y z'

    # Add the data to the netCDF file
    data[0, :, :, :] = image

    # Create a variable for the timestamp
    time = ncfile.createVariable('time', np.str, ('time',))

    # Add the timestamp to the netCDF file
    time[0] = timestamp

    # Close the netCDF file
    ncfile.close()

if __name__=="__main__":
    # main loop
    for dir in os.listdir(data_dir):
        if dir in target_cities:
            full_dir = os.path.join(data_dir, dir)
            clean_city = get_clean_map(dir)
            for date in os.listdir(full_dir):
                full_subdir = os.path.join(full_dir, date)
                try:
                    os.mkdir(os.path.join(out_dir, dir, date))
                except:
                    pass
                img_seq = []
                for i in range(len(os.listdir(full_subdir)) - batch_size - 1):
                    input_batch = [os.path.join(full_subdir, x) for x in os.listdir(full_subdir)[i:i+batch_size]]
                    img_seq = [np.array(Image.open(img).convert("RGB")) for img in input_batch]

                    img_ave = groundtruth_map(img_seq)
                    img_ave_mask = get_foreground_mask(clean_city, img_ave, threshold = 40)

                    input_img = os.path.join(full_subdir, os.listdir(full_subdir)[i+batch_size+1])
                    date_time = input_img.split("_")[-1].split(".")[0]

                    # convert date_time to timestamp
                    timestamp = str(datetime.datetime.strptime(date_time, "%Y%m%d%H%M"))

                    last_img = np.array(Image.open(input_img).convert("RGB"))
                    last_img_mask = get_foreground_mask(clean_city, last_img, threshold = 40)

                    out_cnt = np.zeros_like(last_img_mask)
                    mask_bg = cv2.bitwise_and(last_img_mask, cv2.bitwise_not(img_ave_mask))#last_img_mask.copy()
                    mask_bg[mask_bg < 255] = 0
                    mask_bg = cv2.dilate(mask_bg, np.ones((3,3)))
                    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
                    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15)))
                    contours,hierarchy = cv2.findContours(mask_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    out_cnt[last_img_mask == 255] = 255
                    out_cnt = process_contours(out_cnt, contours)
                    out = dataset_creator(clean_city, out_cnt, last_img)
                    # cv2.imwrite(os.path.join(out_dir, dir, date, str(os.listdir(full_subdir)[i+batch_size+1])+".png"), cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    export_to_netCDF(out, timestamp, os.path.join(out_dir, dir, date, str(os.listdir(full_subdir)[i+batch_size+1])))
                    if SHOW:
                        cv2.imshow("Ouptut", cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                        cv2.imshow("Input", cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB))
                        cv2.waitKey(0)
    cv2.destroyAllWindows()
