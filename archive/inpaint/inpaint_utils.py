import torch
import numpy as np
from PIL import Image, ImageOps

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

from scipy.ndimage import gaussian_filter

class InpaintModel:
    def __init__(self, input_img = './data/init_image/1.jpg',
                 resizeshape = None,
                point_labels = [1],
                dilate_kernel_size =15,
                output_dir = './results',
                sam_model_type = 'vit_t',
                sam_ckpt = './weights/mobile_sam.pt',
                lama_config = './lama/configs/prediction/default.yaml',
                lama_ckpt = './pretrained_models/big-lama',
                device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.input_img = input_img
        img = load_img_to_array(input_img)
        img_pil = Image.open(input_img)
        img_pil = ImageOps.exif_transpose(img_pil)
        if resizeshape != None:
            img = img_pil.resize((resizeshape[1], resizeshape[0]))
        img = np.array(img)
        self.input_img = img

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
        layers = []
        mask_layers = []
        for i in range(n):
            masks = np.load(masks_path[i])
            # mask = np.zeros_like(masks[0, :, :])
            # for j in range(masks.shape[0]):
            #     mask |= masks[j, :, :]
            # mask_layers.append(mask)
            mask_layers.append(masks)
            layers.append(self.input_img.copy())
        layers.append(self.input_img.copy())

        self.mask_layers_origin = mask_layers
        self.mask_layers = mask_layers
        self.layers = layers


    def mask_filter_process(self, n = 1, sigma=8, threshold = 0.2,filter = 'gaussian', modify = True):
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
        return self.layers[n-1]
    
    def inpaint_layer(self, mask_idx = 1):
        image = self.input_img.copy()
        mask = self.mask_layers[mask_idx-1]
        self.masked |= mask
        image_inpaint = inpaint_img_with_lama(
                        image, 
                        mask, 
                        self.lama_config, 
                        self.lama_ckpt, 
                        device=self.device)
        self.layers[mask_idx] = image_inpaint
        return image_inpaint
    
    def mask_re_segmentation(self, n, grid_size = 80):
        mask = self.mask_layers[n-1].copy()
        sampled_coords = []
        # grid_size = int(min(self.input_img.shape[0], self.input_img.shape[1]) / grid_size)
        for i in range(0, mask.shape[0], grid_size):
            for j in range(0, mask.shape[1], grid_size):
                if mask[i, j]:
                    sampled_coords.append((j, i))
        label = np.ones(len(sampled_coords))
        masks, _, _ = predict_masks_with_sam(
            self.layers[n-1], sampled_coords, label,
            self.sam_model_type, self.sam_ckpt, device=self.device
        )
        score = []
        for i in range(len(masks)):
            mask = masks[i]
            original_mask = self.mask_layers[n-1].copy()
            score.append(np.sum(np.sum(mask & original_mask)))
        
        mask_idx = np.argmin(score)
        samples = np.zeros_like(mask)
        for coords in sampled_coords:
            samples[coords[1], coords[0]] = 1
        mask = masks[mask_idx] & self.masked
        self.mask_layers[n-1] = mask | self.mask_layers[n-1]
        
        return samples, self.mask_layers[n-1]


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
    
    def auto_generate_layers(self, segma=8, threshold=0.3):
        for i in range(self.mask_numbers):
            number = i + 1
            if number > 1:
                self.mask_re_segmentation(number)
            self.mask_filter_process(number, sigma=segma, threshold=threshold)
            self.layer_link_ground(number)
            self.mask_filter_process(number, sigma=segma, threshold=threshold)
            self.create_layer(number)
            self.inpaint_layer(number)
    
    def save_outputs(self, filepath):
        for i in range(len(self.layers)):
            save_array_to_img(self.layers[i], filepath + f'_{i}.png')
