import torch
import numpy as np
from PIL import Image, ImageOps

import sys
# import os

inpaint_anything_path = './Inpaint-Anything'
if inpaint_anything_path not in sys.path:
    sys.path.append(inpaint_anything_path)
# sys.path.append('../src')
# sys.path.append('../src/Inpaint-Anything')

from src.InpaintAnything.sam_segment import predict_masks_with_sam
from src.InpaintAnything.lama_inpaint import inpaint_img_with_lama
from src.InpaintAnything.utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

from scipy.ndimage import gaussian_filter

from skimage.segmentation import slic
from skimage.measure import regionprops



class InpaintModel:
    def __init__(self,
                 input_img,
                 # input_img = './data/init_image/1.jpg',
                 resizeshape = None,
                 point_labels=[1],
                 dilate_kernel_size=15,
                 output_dir='../results',
                 sam_model_type='vit_t',
                 sam_ckpt='./InpaintAnything/weights/mobile_sam.pt',
                 lama_config='./InpaintAnything/lama/configs/prediction/default.yaml',
                 lama_ckpt='./InpaintAnything/pretrained_models/big-lama',
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        # self.input_img = input_img
        # img = load_img_to_array(input_img)
        # img_pil = Image.open(input_img)
        # img_pil = ImageOps.exif_transpose(img_pil)
        # if resizeshape != None:
        #     img = img_pil.resize((resizeshape[1], resizeshape[0]))
        #     img_a = img.convert('RGBA')
        # img = input_img
        # img_a = np.array(img_a)

        img = input_img
        n, m, d = img.shape
        img_a = np.ones((n, m, 4))
        img_a[:, :, 0:3] = img

        self.input_img = img
        self.input_img_a = img_a

        self.mask_numbers = 0

        self.masked = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

        self.point_labels = point_labels
        self.dilate_kernel_size = dilate_kernel_size
        self.output_dir = output_dir
        self.sam_model_type = sam_model_type
        self.sam_ckpt = sam_ckpt
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt
        self.device = device

    def load_masks(self, masks_path):
        n = len(masks_path)
        self.mask_numbers = n
        # layers = []
        # layers_a = []
        # mask_layers = []
        # for i in range(n):
        #     masks = np.load(masks_path[i])
        #     # mask = np.zeros_like(masks[0, :, :])
        #     # for j in range(masks.shape[0]):
        #     #     mask |= masks[j, :, :]
        #     # mask_layers.append(mask)
        #     mask_layers.append(masks)
        #     layers.append(self.input_img.copy())
        #     layers_a.append(self.input_img_a.copy())
        layers = []
        layers_a = []
        mask_layers = []
        for i in range(n):
            mask_layers.append(masks_path[i])
            layers.append(self.input_img.copy())
            layers_a.append(self.input_img_a.copy())
        layers.append(self.input_img.copy())
        layers_a.append(self.input_img_a.copy())

        self.mask_layers_origin = mask_layers
        self.mask_layers = mask_layers
        self.layers = layers
        self.layers_a = layers_a


    def mask_filter_process(self, n = 1, sigma=10, threshold = 0.2,filter = 'gaussian', modify = True):
        if filter == 'gaussian':
            smoothed_mask = gaussian_filter(self.mask_layers_origin[n-1].astype(float), sigma=sigma)
            smoothed_mask = (smoothed_mask > threshold)
        else :
            raise ValueError('Filter not supported')
        if modify:
            self.mask_layers[n-1] = smoothed_mask
        return smoothed_mask
    
    
    def create_layer(self, n = 1):
        self.layers[n-1][~self.mask_layers[n-1]] = [0,0,0]
        self.layers_a[n-1][~self.mask_layers[n-1]] = [0,0,0,0]
        return self.layers_a[n-1]
    
    def inpaint_layer(self, mask_idx = 1):
        image = self.layers[mask_idx].copy()
        mask = self.mask_layers[mask_idx-1]
        self.masked |= mask
        image_inpaint = inpaint_img_with_lama(
                        image, 
                        mask, 
                        self.lama_config, 
                        self.lama_ckpt, 
                        device=self.device)
        for i in range(mask_idx, len(self.layers)):
            self.layers[i] = image_inpaint.copy()
            self.layers_a[i] = np.array(Image.fromarray(self.layers[mask_idx]).convert('RGBA'))
        return image_inpaint
    
    def mask_re_segmentation(self, n, sample_method = 'grid', grid_size = 80, slic_segments = 100, slic_compactness = 10, modify = True):
        mask = self.mask_layers[n-1].copy()
        sampled_coords = []
        if sample_method == 'grid':
            for i in range(0, mask.shape[0], grid_size):
                for j in range(0, mask.shape[1], grid_size):
                    if mask[i, j]:
                        sampled_coords.append((j, i))
        elif sample_method == 'superpixel':
            image = self.layers[n-1].copy()
            segments = slic(image, n_segments=slic_segments, compactness=slic_compactness, start_label=1)
            regions = regionprops(segments)
            centers = [region.centroid for region in regions]
            sampled_coords = [center for center in centers if mask[int(center[0]), int(center[1])]]
        else:
            raise ValueError('Sample method not supported')
        
        label = np.ones(len(sampled_coords))
        image = self.layers[n-1].copy()
        masks, _, _ = predict_masks_with_sam(
            image, sampled_coords, label,
            self.sam_model_type, self.sam_ckpt, device=self.device
        )
        score = []
        original_mask = self.mask_layers[n-1].copy()
        for i in range(len(masks)):
            mask = masks[i]
            score.append(np.sum(np.sum(mask & original_mask)))
        
        mask_idx = np.argmin(score)
        samples = np.zeros_like(mask)
        
        if sample_method == 'grid':
            for coords in sampled_coords:
                samples[max(coords[1]-2,0):min(coords[1]+2, samples.shape[0]-1), max(coords[0]-2,0):min(coords[0]+2, samples.shape[1]-1)] = 1
        elif sample_method == 'superpixel':
            for coords in sampled_coords:
                samples[int(coords[0])-2 :int(coords[0])+2, int(coords[1])-2:int(coords[1])+2] = 1
        else:
            raise ValueError('Sample method not supported')      
        mask = masks[mask_idx] & self.masked
        mask = mask | self.mask_layers[n-1]

        if modify:
            self.mask_layers[n-1] = mask
        
        return samples, mask


    def layer_link_ground(self, n):
        mask = self.mask_layers[n-1].copy()
        for y in range(mask.shape[1]):
            for x in range(mask.shape[0]-1,0,-1):
                if mask[x, y]:
                    min_row_index = x
                    mask[min_row_index:, y] = True
                    break
        self.mask_layers[n-1] = mask
        return mask
    
    def auto_generate_layers(self, filter_segma=10, filter_threshold=0.2, link_to_ground = False, sample_method = 'superpixel'):
        for i in range(self.mask_numbers):
            number = i + 1
            if number > 1:
                self.mask_re_segmentation(number)
            self.mask_filter_process(number, sigma=filter_segma, threshold=filter_threshold)
            if link_to_ground:
                self.layer_link_ground(number)
            self.create_layer(number)
            self.inpaint_layer(number)
    
    def save_outputs(self, filepath):
        for i in range(len(self.layers)):
            save_array_to_img(self.layers_a[i], filepath + f'_{i}.png')
