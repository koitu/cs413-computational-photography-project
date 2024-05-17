import numpy as np
from skimage import data, segmentation, graph, color
import networkx as nx


class RAG:
    def __init__(self, img, depth, labels):
        """
        - img: input image
        - depth: depth of the image
        - labels: the result of slic (should start at 1)
        """
        n, m = np.shape(depth)
        cie_img = color.rgb2lab(img)

        boundary_img = np.zeros((n + 10, m + 10, 3))
        boundary_img[5:-5, 5:-5, :] = img

        boundary_cie_img = np.zeros((n + 10, m + 10, 3))
        boundary_cie_img[5:-5, 5:-5, :] = cie_img

        boundary_depth = np.zeros((n + 10, m + 10))
        boundary_depth[5:-5, 5:-5] = depth

        # expand the superpixels to include a border of 0s
        boundary_labels = np.zeros((n + 10, m + 10), dtype='int64')
        boundary_labels[5:-5, 5:-5] = labels

        g = graph.RAG(boundary_labels, connectivity=2)

        for n in g:
            g.nodes[n].update(
                {
                    'labels': [n],
                    'mask': None,
                    'pixel count': 0,
                    'total color': [],
                    'total depth': [],
                }
            )

        for index in np.ndindex(boundary_labels.shape):
            current = boundary_labels[index]
            g.nodes[current]['pixel count'] += 1
            g.nodes[current]['total color'].append(boundary_cie_img[index])
            g.nodes[current]['total depth'].append(boundary_depth[index])

        for n in g:
            g.nodes[n]['mask'] = labels == n
            g.nodes[n]['total color'] = np.array(g.nodes[n]['total color'], dtype='float64')
            g.nodes[n]['total depth'] = np.array(g.nodes[n]['total depth'], dtype='float64')
            g.nodes[n]['mean color'] = np.sum(g.nodes[n]['total color'] / g.nodes[n]['pixel count'], axis=0)
            g.nodes[n]['mean depth'] = np.sum(g.nodes[n]['total depth'] / g.nodes[n]['pixel count'], axis=0)

        for x, y, edge in g.edges(data=True):
            # some kind of loss function with the distance, color difference, and segment anything result
            edge['weight'] = (
                    color.deltaE_cie76(g.nodes[x]['mean color'], g.nodes[y]['mean color']) ** 2
            )

        self.boundary_graph = g.copy()
        self.boundary_img = boundary_img
        self.boundary_cie_img = boundary_cie_img
        self.boundary_depth = boundary_depth
        self.boundary_labels = boundary_labels

        regions = []
        for node in nx.all_neighbors(g, 0):
            regions.append(node)

        for n in regions:
            g.remove_edge(n, 0)

        g.remove_node(0)

        self.edge_nodes = regions
        self.graph = g
        self.img = img
        self.cie_img = cie_img
        self.depth = depth
        self.labels = labels

def merge_nodes(g, n1, n2):
    pass

# extra stuff
# from tunnel_book.preprocessing import normalize_image, slic_segmentation, kmeans
#
# from skimage.segmentation import slic
# from scipy.spatial import Delaunay
# import maxflow
#
# # %%
# rgb_slic = slic(img, n_segments=250, compactness=18.5, sigma=1, start_label=1)
# segments_ids = np.unique(rgb_slic)
#
# # centers
# centers = np.array([np.mean(np.nonzero(rgb_slic == i), axis=1) for i in segments_ids])
# print(np.shape(centers))
#
# # neighbors via Delaunay tesselation
# tri = Delaunay(centers)
#
# indptr, indices = tri.vertex_neighbor_vertices
#
# plt.imshow(rgb_slic)
# plt.plot(centers[:, 1], centers[:, 0], '.')
#
# i = 0
# for k in range(len(indptr) - 1):
#     neigh = indices[indptr[k]:indptr[k + 1]]
#     y1, x1 = centers[k]
#
#     for n in neigh:
#         y2, x2 = centers[n]
#         plt.plot((x1, x2), (y1, y2))
# # %%
# # basic: consider the nodes on the same layer as the free and use dikstra to get the shortest path to the edge
# # generate a distorition matrix for maxflow
# # - at each layer you put the cost of putting the pixel at such a layer
# # - add another cost to switching layers
# # - somehow force at least some path to edge?
# #   - maybe perform multiple cuts with only two layers (from back to front) until there is little left
# #   - then we can force the cut to somehow include the edges
#
# # binary cut but start from most salient object at the depth and grow until we get to the edge
# # - kmeans segmentation to estimate the layer depths
# # - choose the most salient objects from each layer
# # - each layer grows their object until a certan point (make sure they hit the edge)
# # %%
# depth = normalize_image(depth) * 100
# depth_avgs = slic_segmentation(depth, 500, 0.03, 1, 1)
# depth_kmeans = kmeans(depth_avgs, 4)
#
# plt.imshow(depth_kmeans)
# # %%
# means = np.unique(depth_kmeans)
# print(means)  # why are these integers?
#
# n, m = depth.shape
# loss = np.zeros((n, m, 4))
# for i in range(4):
#     loss[:, :, i] = (depth - means[i]) ** 2  # L2 loss
#
# loss = loss[:, :, ::-1]
#
# plt.subplot(221)
# plt.imshow(loss[:, :, 0])
#
# plt.subplot(222)
# plt.imshow(loss[:, :, 1])
#
# plt.subplot(223)
# plt.imshow(loss[:, :, 2])
#
# plt.subplot(224)
# plt.imshow(loss[:, :, 3])
#
# # %%
# # up t-links should be set to np.inf
# up = np.array(
#     [[[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [1, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]]])
#
# # down t-links should hold photo consistency loss term
# down = np.array(
#     [[[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [0, 0, 1],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]]])
#
# # heigh-wise spatial consistency term
# height_wise = np.array(
#     [[[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [0, 1, 0],
#       [0, 0, 0]]])
#
# # length-wise spatial consistency term
# length_wise = np.array(
#     [[[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [0, 0, 0],
#       [0, 1, 0]],
#      [[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]]])
# # %%
# # create an intensity shift to make it easier to change layers on contrast edges
# sigma = 3
#
# # img = np.array(img, dtype=float)
# h = img - np.roll(img, -1, axis=0)
# l = img - np.roll(img, -1, axis=1)
# # h = depth - np.roll(depth, -1, axis=0)
# # l = depth - np.roll(depth, -1, axis=1)
#
# trans = lambda x: np.exp(-np.sum(x ** 2, axis=2) / (2 * (sigma ** 2)))
# # trans = lambda x : np.exp(-x ** 2 / (2*(sigma**2)))
#
# h_m = trans(h)
# l_m = trans(l)
# plt.imshow(h_m)
# plt.colorbar()
# # %%
# g = maxflow.Graph[float]()
# # nodeids = g.add_grid_nodes((n, m, 2))
# nodeids = g.add_grid_nodes((n, m, 3))
#
# # up t-links cost infinity
# g.add_grid_edges(nodeids, weights=np.inf, structure=up, symmetric=False)
#
# # down t-links start from d_max then go to d_min
# # g.add_grid_edges(nodeids, weights=loss[:,:,1:-1], structure=down, symmetric=False)
# g.add_grid_edges(nodeids, weights=loss[:, :, 1:], structure=down, symmetric=False)
# g.add_grid_tedges(nodeids[:, :, 0], loss[:, :, 0], 0)  # start is d_max
# g.add_grid_tedges(nodeids[:, :, -1], 0, loss[:, :, -1])  # end is d_min
#
# weight = 200
#
# g.add_grid_edges(nodeids, weights=weight * h_m[:, :, None], structure=height_wise, symmetric=True)
# g.add_grid_edges(nodeids, weights=weight * l_m[:, :, None], structure=length_wise, symmetric=True)
#
# # perform the graph cut
# g.maxflow()
# mask = g.get_grid_segments(nodeids)
#
# # sum the amount of nodes below the cut
# mask = np.count_nonzero(mask, axis=2)
# # %%
# plt.imshow(mask)
