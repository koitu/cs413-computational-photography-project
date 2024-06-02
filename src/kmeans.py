import torch

import numpy as np

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbow, KElbowVisualizer


def get_optimal_k(object_masks, depth, visualize=False):
    """
    Use the Elbow method to find a reasonable number of layers to segment the image into

    Parameters:
        - object_masks: the result of segment-anything
        - depth: the result of depth-anything
        - visualize: to visualize a graph or not

    Returns:
        - k: the k value
    """
    depth = torch.from_numpy(depth)
    
    depth_features = []
    for m in object_masks:
        mask = m['segmentation']
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        mask_depth = depth[mask_tensor]
        if mask_depth.numel() > 0:
            average_depth = torch.mean(mask_depth.float())
            depth_features.append([average_depth.item()])

    depth_features = np.array(depth_features)
    model = KMeans()
    
    if visualize:
        visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=True)
        visualizer.fit(depth_features)
        optimal_k = visualizer.elbow_value_
    else:
        k_elbow = KElbow(model, k=(2, 10), metric='distortion', timings=True)
        k_elbow.fit(depth_features)
        optimal_k = k_elbow.elbow_value_

    return optimal_k


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


def assign2layers_kmeans(object_masks, depth, n_layers=4):
    """
    Assigns objects to layers using information from depth to perform Kmeans.
    """

    # Step 1: Calculate the average depth of each mask
    average_depths = []
    for m in object_masks:
        mask = m["segmentation"]
        average_depth = np.sum(depth[mask])/np.sum(mask)
        average_depths.append(average_depth)

    # Convert average_depths to shapes suitable for KMeans inputs
    average_depths_np = np.array(average_depths).reshape(-1, 1)

    # Step 2: Use K-means to cluster objects
    kmeans = KMeans(n_clusters=n_layers, random_state=0).fit(average_depths_np)
    labels = kmeans.labels_

    # Get the centers of clustering and sort these centers by depth
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers_indices = np.argsort(centers)

    # Reassign labels based on sorted centers
    sorted_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(sorted_centers_indices):
        sorted_labels[labels == old_label] = new_label

    # Step 3: Assign objects to layers based on K-means clusters
    layers = [[] for _ in range(n_layers)]
    layers_inx = [[] for _ in range(n_layers)]
    for i, label in enumerate(sorted_labels):
        layers[label].append(object_masks[i])
        layers_inx[label].append(i)

    return layers_inx[::-1], layers[::-1], sorted(centers)
