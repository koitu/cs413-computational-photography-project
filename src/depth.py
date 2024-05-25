import os
import sys
sys.path.append(os.path.join(sys.path[0], "src/DepthAnything"))

import cv2
import torch
from src.DepthAnything.depth_anything.dpt import DepthAnything
from src.Depth_Antyhing import Resize, NormalizeImage, PrepareForNet


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


def depth_map(img):
    img = transform({'image': img})['image']
    img = torch.from_numpy(img).unsqueeze(0)

    depth = depth_anything(img)  # depth shape: 1xHxW
    # depth = depth.detach().squeeze()
    depth = depth.detach().squeeze().numpy()
    re