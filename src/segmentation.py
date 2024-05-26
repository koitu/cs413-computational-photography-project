import os

import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator #, SamPredictor
from skimage.segmentation import slic


class SegmentModel:
    def __init__(self, points_per_side=32):
        model_path = "./models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        if not os.path.isfile(model_path):
            urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", model_path)
            # https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)

        self.segment_anything = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)


InitSegmentModel = None


# def get_msk_gen():
#     return segment_anything


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_all_segmts_ind(masks, img):
    for i, m in enumerate(masks):
        plt.figure(figsize=(15,7))
        mask = m['segmentation']
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        res = np.ones_like(img)
        res[mask] = img[mask]
        plt.imshow(res)
        plt.show()


# def downsample_image_opencv(img, output_size):
#     height, width = img.shape[:2]
#     new_height, new_width = output_size
#
#     resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
#
#     return resized_img


def remove_small_masks(masks, threshold=500):
    large_masks =[]
    for m in masks:
        mask = m['segmentation']
        if np.sum(mask) > threshold:
            large_masks.append(m)

    return large_masks


def remove_overlapping(masks, overlap_ratio_threshold=0.05):
    embedded_objects = []

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap_area = np.logical_and(masks[i]['segmentation'], masks[j]['segmentation'])

            overlap_size = np.sum(overlap_area)

            mask_i_size = np.sum(masks[i]['segmentation'])
            mask_j_size = np.sum(masks[j]['segmentation'])

            # Calculate the percentage of overlap, relative to smaller masks
            if overlap_size > 0:  # Ensure overlap
                overlap_ratio = round(overlap_size / min(mask_i_size, mask_j_size),4)
                if overlap_ratio > overlap_ratio_threshold:
                    if mask_i_size > mask_j_size:
                        embedded_objects.append(j)
                    else:
                        embedded_objects.append(i)
                # else:
                #     print(f"There is no overlap between Mask {i} and Mask {j}.")
    non_overlap_object_masks = np.delete(masks, embedded_objects).tolist()

    return embedded_objects, non_overlap_object_masks


def obtain_rest_of_img(object_masks, original_img):
    combined_mask = np.sum([m['segmentation'] for m in object_masks], axis=0)
    unsegmented_part = np.where(combined_mask > 0, 0, 1)
    res_img = np.ones_like(original_img)
    for i in range(3):
        res_img[:, :, i][unsegmented_part == 1] = original_img[:, :, i][unsegmented_part == 1]

    return res_img


def img_white_p_ratio(img):
    return np.sum(img == 1.0)/(img.shape[0]*img.shape[1]*img.shape[2])


def rest_image_pixel_ratio(object_masks):
    num_seg_pixel = 0
    for mask in object_masks:
        m = mask['segmentation']
        num_seg_pixel += np.sum(m)
    return num_seg_pixel/(m.shape[0]*m.shape[1])


def remove_white_canva(res_img, res_masks):
    white_canva_masks = []
    for i, m in enumerate(res_masks):
        mask = m["segmentation"]
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        masked_img = res_img * mask

        total_pixels = np.sum(mask)
        if total_pixels > 0:
            ones_ratio = img_white_p_ratio(masked_img)
            # print(f"Proportion of pixels in Mask {i} with value equal to 1: {ones_ratio:.4f}")
            if ones_ratio > 0.5:
                white_canva_masks.append(i)
        # else:
        # print(f"Mask {i} does not cover any area")

    res_masks_no_white_bg = np.delete(res_masks, white_canva_masks).tolist()

    return res_masks_no_white_bg


def obtain_all_objects(mask_generator, img_to_procd, img_r_thrd=0.90, n_thrd=5, ovlp_r_thrd=0.1, small_thrd=500):
    n = 0
    diff = 1
    object_masks = []
    rows = img_to_procd.shape[0]
    cols = img_to_procd.shape[1]
    img_r = 0

    orignial_img = img_to_procd

    while img_r < img_r_thrd and n <= n_thrd:

        masks = mask_generator.generate(img_to_procd)
        if n > 0:
            masks = remove_white_canva(img_to_procd, masks)

        large_masks = remove_small_masks(masks, small_thrd)
        object_masks.extend(large_masks)
        embedded_objects, object_masks = remove_overlapping(object_masks, ovlp_r_thrd)

        # check_overlapping(object_masks)

        res_img = obtain_rest_of_img(object_masks, orignial_img)

        img_r = round(rest_image_pixel_ratio(object_masks), 2)
        # diff = np.sum(np.abs(res_img - img_to_procd))/(rows*cols)
        n += 1
        print(f"Iteration n={n}: white pixel raito after segmentation = {img_r}")

        img_to_procd = res_img

    return object_masks


def check_overlapping(masks):
    n = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap_area = np.logical_and(masks[i]['segmentation'], masks[j]['segmentation'])

            overlap_size = np.sum(overlap_area)

            mask_i_size = np.sum(masks[i]['segmentation'])
            mask_j_size = np.sum(masks[j]['segmentation'])

            # Calculate the percentage of overlap, relative to smaller masks
            if overlap_size > 0:  # Ensure overlap
                overlap_ratio = round(overlap_size / min(mask_i_size, mask_j_size),4)
                if overlap_ratio > 0.05:
                    n += 1
                    print(f"The overlap ratio between Mask {i} and Mask {j} is. {overlap_ratio}")
        # else:
        #     print(f"There is no overlap between Mask {i} and Mask {j}.")

    if n == 0:
        print("There is no overlap")


def show_layers(img, object_masks, groups):
    plt.figure(figsize=(10, 8))

    for i, group in enumerate(groups):
        if len(group) == 0:
            continue

        ax = plt.subplot(2, 2, i + 1)

        res_img = np.zeros_like(img)
        for idx in group:
            mask = object_masks[idx]['segmentation']
            m = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            res_img += img*m
        ax.imshow(res_img)

        ax.axis('off')
        ax.set_title(f'Group {i+1}')

    plt.tight_layout()
    plt.show()


# def assign2layers_avg_obj(object_masks, depth, n_layers=4):
#     # Step 1: Calculate the average depth of each mask
#     average_depths = []
#     for m in object_masks:
#         mask = m["segmentation"]
#         # Calculate the average depth of the depth map region corresponding to mask
#         average_depth = np.mean(depth[mask])
#         average_depths.append(average_depth)
#
#     # Step 2: Sort objects by average depth
#     sorted_indices = np.argsort(average_depths)
#
#     # Step 3: Assign objects to layers
#     # Calculate the minimum number of objects per layer
#     objects_per_layer = len(object_masks) // n_layers
#     # Calculate the number of objects to be distributed in the first few layers
#     extra_objects = len(object_masks) % n_layers
#
#     layers = []
#     layers_inx = []
#     start_idx = 0
#     for i in range(n_layers):
#         if i < extra_objects:
#             # Assign one extra object to this layer
#             end_idx = start_idx + objects_per_layer + 1
#         else:
#             end_idx = start_idx + objects_per_layer
#
#         # Selected sorted objects for each layer
#         layer_indices = sorted_indices[start_idx:end_idx]
#         layer = [object_masks[index] for index in layer_indices]
#         layers.append(layer)
#         layers_inx.append(layer_indices)
#         start_idx = end_idx
#
#     return layers_inx, layers

def fill_with_superpixels(img_lr, object_masks):
    rgb_slic = slic(img_lr, n_segments=1000, start_label=1, slic_zero=True)

    unique_superpixels, superpixel_counts = np.unique(rgb_slic, return_counts=True)
    superpixel_areas = dict(zip(unique_superpixels, superpixel_counts))
    new_masks = []

    for seg in object_masks:
        mask = seg['segmentation']
        masked_superpixel_areas = {label: np.sum(mask[rgb_slic == label]) for label in unique_superpixels}
        overlapping_threshold = 0.1
        overlapping_superpixels = [label for label, masked_area in masked_superpixel_areas.items()
                                   if masked_area / superpixel_areas[label] > overlapping_threshold]
        new_mask = np.isin(rgb_slic, overlapping_superpixels)
        new_masks.append(new_mask)

    for i, seg in enumerate(object_masks):
        seg['segmentation'] = new_masks[i]

    return object_masks


def segment_anything_tunnel_book_oneshot(img, depth, n_layers=4):
    from src.kmeans import assign2layers_kmeans

    # Dowsample the original image
    # img_lr = downsample_image_opencv(raw_image, depth.shape)

    # object_masks = obtain_all_objects(img_lr)
    object_masks = obtain_all_objects(img)

    check_overlapping(object_masks)

    print(f"There is {len(object_masks)} object masks")

    show_all_segmts_ind(object_masks, img)

    # Assign objects to layers based on depth
    # layers_idx, layers = assign2layers_avg_obj(object_masks, depth.numpy(), 3)

    # show_layers(img_lr, object_masks, layers_idx)

    # layers_idx, layers = assign2layers_kmeans(object_masks, depth.numpy(), 4)

    # show_layers(img_lr, object_masks, layers_idx)

    layers_idx, layers = assign2layers_kmeans(object_masks, depth.numpy(), n_layers=n_layers)
    return layers_idx, layers
