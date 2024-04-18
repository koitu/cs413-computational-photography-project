import numpy as np
from sklearn.cluster import KMeans

def assign2layers_avg_obj(object_masks, depth, n_layers=4):
    # Step 1: Calculate the average depth of each mask
    average_depths = []
    for m in object_masks:
        mask = m["segmentation"]
        # Calculate the average depth of the depth map region corresponding to mask
        average_depth = np.mean(depth[mask])
        average_depths.append(average_depth)

    # Step 2: Sort objects by average depth
    sorted_indices = np.argsort(average_depths)

    # Step 3: Assign objects to layers
    # Calculate the minimum number of objects per layer
    objects_per_layer = len(object_masks) // n_layers
    # Calculate the number of objects to be distributed in the first few layers
    extra_objects = len(object_masks) % n_layers

    layers = []
    layers_inx = []
    start_idx = 0
    for i in range(n_layers):
        if i < extra_objects:
            # Assign one extra object to this layer
            end_idx = start_idx + objects_per_layer + 1
        else:
            end_idx = start_idx + objects_per_layer

        # Selected sorted objects for each layer
        layer_indices = sorted_indices[start_idx:end_idx]
        layer = [object_masks[index] for index in layer_indices]
        layers.append(layer)
        layers_inx.append(layer_indices)
        start_idx = end_idx

    return layers_inx, layers

def assign2layers_kmeans(object_masks, depth, n_layers=4):
  # Step 1: Calculate the average depth of each mask
  average_depths = []
  for m in object_masks:
    mask = m["segmentation"]
    average_depth = np.mean(depth[mask])
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
  for old_label, new_label in enumerate(sorted_centers_indices):
    sorted_labels[labels == old_label] = new_label

  # Step 3: Assign objects to layers based on K-means clusters
  layers = [[] for _ in range(n_layers)]
  layers_inx = [[] for _ in range(n_layers)]
  for i, label in enumerate(sorted_labels):
    layers[label].append(object_masks[i])
    layers_inx[label].append(i)

  return layers_inx[::-1], layers[::-1]