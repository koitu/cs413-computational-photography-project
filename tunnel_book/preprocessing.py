import os
import sys
sys.path.append(os.path.join(sys.path[0], "tunnel_book/DepthAnything"))

import cv2
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from torchvision.transforms import Compose

from sklearn.cluster import KMeans

from skimage.segmentation import slic, mark_boundaries
from skimage.transform import resize

from tunnel_book.DepthAnything.depth_anything.dpt import DepthAnything
from tunnel_book.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# start init for depth anything
encoder = 'vits'  # can also be 'vitb' or 'vitl'

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
# end init for depth anything




# def get_image_from_url(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content)).convert("RGB")
#     return np.array(img) / 255.0


def normalize_image(img):
    n, m = np.shape(img)
    return img / np.max(np.reshape(img, n * m))


def load_image(path):
    """
    load the image from path and normalize its values to [0, 1]
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def downscale(img, n, m):
    # return cv2.resize(img, (n, m), interpolation=cv2.INTER_AREA)
    return resize(img, (n, m), anti_aliasing=True)


def depth_map(img):
    img = transform({'image': img})['image']
    img = torch.from_numpy(img).unsqueeze(0)

    depth = depth_anything(img)  # depth shape: 1xHxW
    # depth = depth.detach().squeeze()
    depth = depth.detach().squeeze().numpy()
    return depth


def kmeans(d_img, n_clusters=4):
    """
    takes an image and returns a kmeans clustering mask
    - img[(mask == 0)] will extract the back-most layer
    - img[(mask == (n-1))] will extract the front-most layer
    """
    # compute the KMeans segmentation layers
    n, m = np.shape(d_img)

    res = np.reshape(d_img, (n * m, 1))

    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(res)
    res = k_means.predict(res)

    res = np.reshape(res, (n, m))

    # order the segmentation layers such that the back-most has id 0
    layer_ids = np.unique(res)
    img_masks = [(res == i) for i in layer_ids]
    img_masks = [(np.mean(d_img[m]), m) for m in img_masks]
    sorted(img_masks, key=lambda x: x[0])

    for i, m in img_masks:
        res[m] = i
    return res


def slic_segmentation(d_img, n_segments, compactness, sigma, start_label=1):
    # for normalized depth map I recommend
    # d_slic = slic(n_d, n_segments=500, compactness=0.03, sigma=1, start_label=1, channel_axis=None)

    n, m = np.shape(d_img)

    d_slic = slic(d_img,
                  n_segments=n_segments,
                  compactness=compactness,
                  sigma=sigma,
                  start_label=start_label,
                  channel_axis=None)
    segment_ids = np.unique(d_slic)

    masks = np.array([(d_slic == i) for i in segment_ids])
    d_avgs = np.zeros((n, m), dtype='float')
    for m in masks:
        d_avgs[m] = np.mean(d_img[m])

    return d_avgs




# import maxflow
# from scipy.spatial import Delaunay
#
# # %%
# # rgb_slic = slic(rescaled_img, n_segments=250, compactness=10, sigma=1, start_label=1)
# rgb_slic = slic(rescaled_img, n_segments=250, compactness=18.5, sigma=1, start_label=1)
# segments_ids = np.unique(rgb_slic)
#
# # centers
# centers = np.array([np.mean(np.nonzero(rgb_slic == i), axis=1) for i in segments_ids])
# print(np.shape(centers))
#
# # neighbors via Delaunay tesselation
# tri = Delaunay(centers)
#
# indptr, indices = tri.vertex_neighbor_vertices
#
# plt.imshow(rgb_slic)
# plt.plot(centers[:, 1], centers[:, 0], '.')
#
# i = 0
# for k in range(len(indptr) - 1):
#     neigh = indices[indptr[k]:indptr[k + 1]]
#     y1, x1 = centers[k]
#
#     for n in neigh:
#         y2, x2 = centers[n]
#         plt.plot((x1, x2), (y1, y2))
