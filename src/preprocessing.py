import cv2
import numpy as np
import os
from PIL import Image
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
            cv2.imwrite(f"../data/{name}_background_model.png", background_model[0])
        else:
            cv2.imwrite(f"../data/background_model.png", background_model[0])
    return cv2.cvtColor(background_model[0], cv2.COLOR_BGR2RGB)

def joiner(imageA, posA):
    fig = plt.figure(figsize=(8, 8))
    lons, lats = posA
    m = Basemap(projection='cyl',
            # llcrnrlon=min(lons) - 2, llcrnrlat=min(lats) - 2,
            # urcrnrlon=max(lons) + 2, urcrnrlat=max(lats) + 2,
            resolution='c')
    # m.etopo(scale=0.5, alpha=0.5)

    # Map (long, lat) to (x, y) for plotting
    x, y = m(lons, lats)

    # Add the plane marker at the last point.
    plane = np.array(imageA)
    im = OffsetImage(plane, zoom=1)
    ab = AnnotationBbox(im, (x,y), xycoords='data', frameon=False)

    # Get the axes object from the basemap and add the AnnotationBbox artist
    m._check_ax().add_artist(ab)

    plt.show()
    # plt.plot(x, y, 'ok', markersize=5)
    # plt.text(x, y, 'BA', fontsize=12)
    # plt.show()


if __name__=="__main__":
    # dir = "../data/va/20220102"
    image_sequences = []
    # for x in os.listdir(dir):
    #     # cap = cv2.VideoCapture(cv2.imread(os.path.join(dir, x)))
    #     # ret, image = cap.read()
    #     # cap.release()
    #     print(x)
    #     image = np.array(Image.open(os.path.join(dir, x)).convert("RGB"))
    #     image_sequences.append(image)
    dir = "../data/ba/"
    for z in os.listdir(dir):
        fol_dir = os.path.join(dir, z)
        for x in os.listdir(fol_dir):
            # cap = cv2.VideoCapture(cv2.imread(os.path.join(dir, x)))
            # ret, image = cap.read()
            # cap.release()
            print(x)
            p = np.random.uniform()
            if p > 0.95:
                try:
                    image = np.array(Image.open(os.path.join(fol_dir, x)).convert("RGB"))
                    image_sequences.append(image)
                except:
                    pass
    gt = groundtruth_map(image_sequences)
    cv2.imshow("out", gt)
    cv2.imwrite("barca1.png", gt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im_dir = np.array(Image.open("../data/pm/20220102/aemet_pm_202201021100.gif").convert("RGB"))
    pos = [2.785060, 39.379770]
    joiner(im_dir, pos)


