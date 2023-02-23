import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

val_map_path = "../data/gt/val.png"
bar_map_path = "../data/gt/bar.png"
pal_map_path = "../data/gt/pal.png"

cloud_path   = "../data/training_images/cloud"
cloud_out_path = "../data/training_labels/cloud"
cloud_out2_path = "../data/training_labels/cloud_tm"

interf_path = "../data/training_images/interf"
interf_out_path = "../data/training_labels/interf"

mix_path = "../data/training_images/mix"
mix_out_path = "../data/training_labels/mix"

THRESH_BG = 30
CUTOFF_HEIGHT = 480

def get_clean_map(city):
    """
    returns the clean map corresponding to the city.
    """
    if city == "va":
        return cv2.cvtColor(cv2.imread(val_map_path), cv2.COLOR_BGR2RGB)
    elif city == "ba":
        return cv2.cvtColor(cv2.imread(bar_map_path), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(cv2.imread(pal_map_path), cv2.COLOR_BGR2RGB)

def get_foreground_mask(clean_map, img, name = None, threshold = THRESH_BG):
    """returns the foreground objects after background subtraction"""
    foreground = cv2.cvtColor(cv2.absdiff(img, clean_map), cv2.COLOR_RGB2GRAY)
    foreground[foreground > threshold] = 255
    foreground[foreground <=threshold] = 0
    if name:
        cv2.imwrite(name + ".png", foreground)
    plt.imshow(clean_map, cmap = "gray")
    plt.show()
    plt.imshow(img, cmap = "gray")
    plt.show()
    plt.imshow(foreground, cmap = "gray")
    plt.show()
    return foreground

def create_single_object_data(img_path, out_path):
    """creates training data/labeled with only clouds"""
    for img_name in os.listdir(img_path):
        print(img_name)
        clean_map = get_clean_map(img_name[6:8])
        full_path = os.path.join(img_path, img_name)
        img = np.array(Image.open(full_path).convert("RGB"))
        fg = get_foreground_mask(clean_map, img)
        full_out_path = os.path.join(out_path, img_name)
        print(full_path[:-4] + ".png")
        cv2.imwrite(full_path[:-4] + ".png", cv2.cvtColor(img[:CUTOFF_HEIGHT], cv2.COLOR_BGR2RGB))
        cv2.imwrite(full_out_path[:-4] + ".png", np.zeros(img[:CUTOFF_HEIGHT].shape[:-1]))#fg[:CUTOFF_HEIGHT])


def create_mix_data(cloud_path, cloud_mask_path, interf_path, interf_mask_path, out_path, out_mask_path, num_repeat = 3, show = False):
    """repeats the data and augments training images with known labels and multiple objects"""
    for k in range(num_repeat):
        for cloud_file in os.listdir(cloud_path):
            out_img = cv2.cvtColor(cv2.imread(os.path.join(cloud_path, cloud_file)), cv2.COLOR_BGR2RGB)
            interf_file = np.random.choice(os.listdir(interf_path))
            interf_img = cv2.cvtColor(cv2.imread(os.path.join(interf_path, interf_file)), cv2.COLOR_BGR2RGB)

            interf_mask = cv2.imread(os.path.join(interf_mask_path, interf_file), 0)
            p = np.random.uniform()
            if p > 0.95:
                interf_img = cv2.rotate(interf_img, cv2.ROTATE_90_CLOCKWISE)
                interf_mask = cv2.rotate(interf_mask, cv2.ROTATE_90_CLOCKWISE)
            elif p > 0.9:
                interf_img = cv2.rotate(interf_img, cv2.ROTATE_180)
                interf_mask = cv2.rotate(interf_mask, cv2.ROTATE_180)
            elif p > 0.85:
                interf_img = cv2.rotate(interf_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                interf_mask = cv2.rotate(interf_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

            for i in range(interf_mask.shape[0]):
                for j in range(interf_mask.shape[1]):
                    if interf_mask[i,j] != 0:
                        out_img[i, j, :] = interf_img[i, j, :]

            outname = f'{k}_{cloud_file}'
            cv2.imwrite(os.path.join(out_path, outname), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_mask_path, outname), interf_mask)
            cv2.imwrite(os.path.join(cloud_mask_path, outname), cv2.imread(os.path.join(cloud_mask_path, cloud_file), 0))
            if show:
                plt.imshow(out_img)
                plt.title("Output Image")
                plt.show()

                plt.imshow(interf_mask)
                plt.title("interference lable")
                plt.show()

                plt.imshow(cv2.imread(os.path.join(cloud_mask_path, cloud_file), 0))
                plt.title("cloud label")
                plt.show()                

if __name__ == "__main__":
    # Only cloud
    create_single_object_data(cloud_path, cloud_out_path)

    # Only interference
    create_single_object_data(interf_path, interf_out_path)

    create_mix_data(cloud_path, cloud_out2_path, interf_path, interf_out_path, mix_path, mix_out_path, num_repeat = 5)