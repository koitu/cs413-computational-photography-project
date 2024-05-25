import cv2
import numpy as np
from skimage.segmentation import slic


def downsample_image_opencv(img, output_size):
    height, width = img.shape[:2]
    new_height, new_width = output_size

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img

def downsample_image_opencv2(img, output_size=(518, 518)):
    # the output size is same as the depth mapping

    height, width = img.shape[:2]

    ratio = min(output_size[0] / width, output_size[1] / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    for n in range(resized_img.shape[0]):
      for m in range(resized_img.shape[1]):
        for k in range(resized_img.shape[2]):
          if resized_img[n,m,k] > 1:
            resized_img[n,m,k] = 1.0

    return resized_img

def remove_small_masks(masks, thrds = 500):
  large_masks =[]
  for i, m in enumerate(masks):
    mask = m['segmentation']
    if np.sum(mask) > thrds:
      large_masks.append(m)

  return large_masks


def remove_overlapping(masks, ovlp_r_thrd=0.05):
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
        if overlap_ratio > ovlp_r_thrd:
          if mask_i_size > mask_j_size:
            embedded_objects.append(i)
          else:
            embedded_objects.append(j)

  non_ovlp_object_masks = np.delete(masks, embedded_objects).tolist()

  return embedded_objects, non_ovlp_object_masks

def obtain_rest_of_img(object_masks, orignial_img):
  combined_mask = np.sum([m['segmentation'] for m in object_masks], axis=0)
  unsegmented_part = np.where(combined_mask > 0, 0, 1)
  res_img = np.ones_like(orignial_img)
  for i in range(3):
    res_img[:, :, i][unsegmented_part == 1] = orignial_img[:, :, i][unsegmented_part == 1]

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
    embedded_objects, object_masks = remove_overlapping(object_masks,ovlp_r_thrd)

    # check_overlapping(object_masks)

    res_img = obtain_rest_of_img(object_masks, orignial_img)

    img_r = round(rest_image_pixel_ratio(object_masks),2)
    # diff = np.sum(np.abs(res_img - img_to_procd))/(rows*cols)
    n += 1
    print(f"Iteration n={n}: white pixel raito after segmentation = {img_r}")

    img_to_procd = res_img

  # plt.imshow(res_img)

  return object_masks

  def normalize_image(img):
    n, m = np.shape(img)
    return img / np.max(np.reshape(img, n * m))

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