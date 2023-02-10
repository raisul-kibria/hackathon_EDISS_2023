#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import math
import sys


# In[11]:


def get_sorted_images(path):
    """Get sorted images by timestamp
    Args:
        path (str): path to images directory (image name example: aemet_ba_202201010000.gif)
    Returns:
        list: list of (timestamp, image_name) sorted by timestamp
    """

    images = os.listdir(path)
    timestamps = []
    for image in images:
        # get timestamp after last underscore
        date_time = image.split("_")[-1].split(".")[0]

        # convert date_time to timestamp
        timestamp = int(datetime.datetime.strptime(
            date_time, "%Y%m%d%H%M").timestamp())

        timestamps.append(timestamp)

    images = [(timestamp, name)
              for timestamp, name in sorted(zip(timestamps, images))]

    return images


# In[12]:


val = cv2.cvtColor(cv2.imread("gt/val.png"), cv2.COLOR_BGR2RGB)
palma = cv2.cvtColor(cv2.imread("gt/palma.png"), cv2.COLOR_BGR2RGB)
barca = cv2.cvtColor(cv2.imread("gt/barca1.png"), cv2.COLOR_BGR2RGB)

# In[13]:


def groundtruth_map(image_sequences, name = "", save = False):
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
    if save:
        if name:
            cv2.imwrite(f"data/{name}_background_model.png", background_model[0])
        else:
            cv2.imwrite(f"data/background_model.png", background_model[0])
    return background_model[0] #cv2.cvtColor(, cv2.COLOR_BGR2RGB)


# In[136]:


def get_foreground_mask(gt_mask, img, name, threshold = 0):
    foreground = cv2.cvtColor(cv2.absdiff(img, gt_mask), cv2.COLOR_RGB2GRAY)
    foreground[foreground > threshold] = 255
#     foreground[foreground<=threshold] = 0
    
#     cv2.imwrite(name + ".png", foreground)
    plt.imshow(foreground, cmap = "gray")
    return foreground


# In[137]:


batch = 17
img_seq = []
cities = os.listdir("./aemet/10min/")

def slide_window(seq, window_size = 18):
    windows = []
    for i in range(len(seq) - window_size + 1):
        windows.append(seq[i: i + window_size])
    return windows
        

def avg_img(path, windows, city):
    print("windows::::::::::::::::::::::::::")
    print(windows)
    print("windows::::::::::::::::::::::::::")
    for window in windows:
        for window_img in window:
            temp = Image.open(path +"/"+ window_img).convert("RGB")
            # temp = np.array()
            #     blur = cv2.bilateralFilter(temp,18,75,75)
            #cv2.erode(blur, np.ones((3,3)))
            img_seq.append(temp)
            img_ave = groundtruth_map(img_seq)
            img_ave_mask = get_foreground_mask(city, img_ave, "with_int" , threshold = 60)
        plt.savefig(img_ave)
            # img_ave_mask = cv2.dilate(last_img_mask, np.ones((3,3)))
    return img_ave, img_ave_mask

for i in range(0,3):
    print(cities[i])
    daily_dirs = os.listdir("./aemet/10min/" + cities[i] + "/")
    print(daily_dirs)
    for daily_dir in daily_dirs:
        # print(daily_dir)
        local_path = "./aemet/10min/" + cities[i] + "/" + daily_dir
        daily_images = os.listdir(local_path)
        print(daily_images[0:3])
        windows = slide_window(daily_images)
        for window in windows:
            img_ave, img_ave_mask = avg_img(local_path, windows, cities[i])
            plt.savefig(img_ave)
        sys.exit()


# In[147]:


plt.imshow(img_ave)


# In[223]:


last_img = np.array(Image.open(os.path.join('./test', os.listdir("./test")[-1])).convert("RGB"))
last_img_mask = get_foreground_mask(barca, last_img, "with_int" , threshold = 60)


# In[239]:


plt.imshow(cv2.bitwise_and(last_img_mask, cv2.bitwise_not(img_ave_mask)), 'gray')


# In[225]:


def dataset_creator(gt, bg, img):
    res = gt.copy()
    for x in range(bg.shape[0]):
        for y in range(bg.shape[1]):
            if bg[x, y] != 0:
                res[x,y,:] = img[x,y,:]
        
#     plt.imshow(res)
#     out = cv2.bitwise_and(gt,gt,mask = res)
    return res


# In[284]:


out_cnt = np.zeros_like(last_img_mask)
mask_bg = cv2.bitwise_and(last_img_mask, cv2.bitwise_not(img_ave_mask))#last_img_mask.copy()
mask_bg[mask_bg < 255] = 0
mask_bg = cv2.dilate(mask_bg, np.ones((3,3)))
mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15)))
contours,hierarchy = cv2.findContours(mask_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
out_cnt[last_img_mask == 255] = 255
# out_cnt = cv2.drawContours(out_cnt, contours, -1, (0,255,0), 3)
plt.imshow(out_cnt)


# In[285]:


for cnt in contours:
    score  = 0
    
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
    if aspect_ratio < 0.3 or (aspect_ratio < 0.8 and c1_score < 10):
        cv2.drawContours(out_cnt,[box],-1,0,-1)


# In[286]:


plt.imshow(out_cnt)


# In[278]:


plt.imshow(last_img_mask, 'gray')


# In[257]:


mask_bg_o = cv2.bitwise_and(last_img_mask, cv2.bitwise_not(img_ave_mask))
# mask_bg = cv2.dilate(mask_bg, np.ones((3,3)))#cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
mask_bg = mask_bg_o.copy()
mask_bg[mask_bg < 255] = 0
mask_bg = cv2.dilate(mask_bg_o, np.ones((3,3)))
mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15)))
contours,hierarchy = cv2.findContours(mask_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    score  = 0
    
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
    if aspect_ratio < 0.3 or (aspect_ratio < 0.8 and c1_score < 10):
        cv2.drawContours(mask_bg_o,[box],-1,(255, 0, 0),5)


# In[258]:


plt.imshow(mask_bg_o,'gray')


# In[287]:


out = dataset_creator(barca, out_cnt, last_img)


# In[288]:


plt.imshow(out)


# In[289]:


plt.imshow(last_img)


# In[ ]:





# In[ ]:




