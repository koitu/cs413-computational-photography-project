import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

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


def show_all_segmts_ind(masks, img_lr):
  for i, mask in enumerate(masks):
      plt.figure(figsize=(15,7))
      mask = m['segmentation']
      mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
      res = np.ones_like(img_lr)
      res[mask] = img_lr[mask]
      plt.imshow(res)
      plt.show()


def show_layers(img, object_masks, groups):
    fig = plt.figure(figsize=(10, 8))

    for i, group in enumerate(groups):
        if len(group) == 0:
            continue

        ax = fig.add_subplot(2, 3, i + 1)

        res_img = np.ones((img.shape[0], img.shape[1], 4), dtype=np.float32)

        group_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        for idx in group:
            mask = object_masks[idx]['segmentation']
            group_mask = np.logical_or(group_mask, mask)

        for c in range(3):  # 遍历RGB通道
              res_img[:, :, c] = np.where(group_mask, img[:, :, c], 0)
        res_img[:, :, 3] = np.where(group_mask, 1, 0)

        ax.imshow(res_img)
        ax.axis('off')
        ax.set_title(f'Layer {i+1}')

    plt.tight_layout()

    plt.show(fig)