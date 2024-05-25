import os
import sys

# need to manually add the path because base of DepthAnything not have a __init__.py
sys.path.append(os.path.join(sys.path[0], "src/DepthAnything"))

import cv2
import torch
import numpy as np

from torchvision.transforms import Compose

from src.DepthAnything.depth_anything.dpt import DepthAnything
from src.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class DepthModel:
    def __init__(self):
        encoder = 'vits'  # can also be 'vitb' or 'vitl'

        self.transform = Compose([
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

        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()


InitDepthModel = None


def estimate_depth(img):
    """
    Estimate the depth of the given image using Depth-Anything
    """

    # Initialize depth anything on the first call to this function
    global InitDepthModel
    if InitDepthModel is None:
        InitDepthModel = DepthModel()

    img = InitDepthModel.transform({'image': img})['image']
    img = torch.from_numpy(img).unsqueeze(0)

    depth = InitDepthModel.depth_anything(img)  # depth shape: 1xHxW
    depth = depth.detach().squeeze().numpy()

    return depth


def normalize_depth(depth):
    """
    Normalize the depth map to values between [0, 1]
    """
    n, m = np.shape(depth)
    return depth / np.max(np.reshape(depth, n * m))
