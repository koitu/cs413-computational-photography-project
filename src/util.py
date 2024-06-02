import cv2
import numpy as np

from skimage.transform import resize
from skimage.segmentation import slic


def load_image(path):
    """
    Load the image from path and normalize its values to [0, 1]
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def downscale(img, n, m):
    """
    Downscale a given image into an (n, m) array
    """
    # return cv2.resize(img, (n, m), interpolation=cv2.INTER_AREA)
    return resize(img, (n, m), anti_aliasing=True)


def get_superpixels(img, n_segments=10000, start_label=1):
    """
    Get the superpixel labels for a given image
    """
    # n, m = np.shape(img)
    #
    # slic_res = slic(img, n_segments=n_segments, start_label=start_label, slic_zero=True)
    # segment_ids = np.unique(slic_res)
    #
    # masks = np.array([(slic_res == i) for i in segment_ids])
    # d_avgs = np.zeros((n, m), dtype='float')
    # for m in masks:
    #     d_avgs[m] = np.mean(slic_res[m])
    #
    # return d_avgs
    return slic(img, n_segments=n_segments, start_label=start_label, slic_zero=True)
