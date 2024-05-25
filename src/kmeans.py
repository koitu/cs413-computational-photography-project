import numpy as np
from sklearn.cluster import KMeans


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
