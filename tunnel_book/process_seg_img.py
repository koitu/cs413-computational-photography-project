import cv2
import numpy as np

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

def remove_small_masks(masks, thrd = 500):
  large_masks =[]
  for i, m in enumerate(masks):
    mask = m['segmentation']
    if np.sum(mask) > thrd:
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
            embedded_objects.append(j)
          else:
            embedded_objects.append(i)
        # else:
        #     print(f"There is no overlap between Mask {i} and Mask {j}.")
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


def obtain_all_objects(mask_generator, img_to_procd, img_w_r_thrd=0.90, diff_thrd=0.01, n_thrd=3):

  n = 0
  diff = 1
  object_masks = []
  rows = img_to_procd.shape[0]
  cols = img_to_procd.shape[1]
  img_w_r = img_white_p_ratio(img_to_procd)

  orignial_img = img_to_procd

  while img_w_r < img_w_r_thrd and diff > diff_thrd and n <= n_thrd:

    masks = mask_generator.generate(img_to_procd)
    if n > 0:
      masks = remove_white_canva(img_to_procd, masks)

    large_masks = remove_small_masks(masks)
    object_masks.extend(large_masks)
    embedded_objects, object_masks = remove_overlapping(object_masks)

    # check_overlapping(object_masks)

    res_img = obtain_rest_of_img(object_masks, orignial_img)

    img_w_r = img_white_p_ratio(res_img)
    diff = np.sum(np.abs(res_img - img_to_procd))/(rows*cols)
    n += 1
    print(f"Iteration n={n}: white pixel raito after segmentation = {img_w_r}, difference ={diff}")

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