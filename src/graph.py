import numpy as np
import networkx as nx

from skimage import graph, color
from scipy.cluster.hierarchy import DisjointSet
from scipy.ndimage import binary_fill_holes


def merge_nodes(g, n1, n2):
    """
    Merges the node n2 into the n1 while combining their mask, pixel count, color, and depth into a larger node
    """
    g.remove_edge(n1, n2)

    g.nodes[n1]['mask'] = np.logical_or(g.nodes[n1]['mask'], g.nodes[n2]['mask'])
    g.nodes[n1]['pixel count'] += g.nodes[n2]['pixel count']
    g.nodes[n1]['total color'] = np.append(g.nodes[n1]['total color'], g.nodes[n2]['total color'], axis=0)
    g.nodes[n1]['total depth'] = np.append(g.nodes[n1]['total depth'], g.nodes[n2]['total depth'], axis=0)
    g.nodes[n1]['mean color'] = np.sum(g.nodes[n1]['total color'] / g.nodes[n1]['pixel count'], axis=0)
    g.nodes[n1]['mean depth'] = np.sum(g.nodes[n1]['total depth'] / g.nodes[n1]['pixel count'], axis=0)

    n1_con = []
    for c in nx.all_neighbors(g, n1):
        n1_con.append(c)
    for c in n1_con:
        g.remove_edge(n1, c)

    n2_con = []
    for c in nx.all_neighbors(g, n2):
        n2_con.append(c)
    for c in n2_con:
        g.remove_edge(n2, c)

    for n in np.unique(n1_con + n2_con):
        g.add_edge(n1, n)
        g[n1][n]['weight'] = (
            color.deltaE_cie76(g.nodes[n1]['mean color'], g.nodes[n]['mean color'])
            # abs(g.nodes[x]['mean depth'] - g.nodes[y]['mean depth'])
        )

    g.remove_node(n2)


def merge_regions_until_done(g, start_regions, regions_left=4):
    """
    Merge regions until there are only "regions_left" regions left

    Parameters
        - g: the graph
        - start_regions: the start regions (usually the edge nodes)
        - regions_left: the number of regions when merging stops

    Returns
        - g: the resulting graph of regions after the merging is done
        - regions: the nodes of the remaining regions
    """
    g = g.copy()
    regions = start_regions.copy()

    while g.number_of_nodes() > regions_left:
        cheapest_edge_val = np.inf
        cheapest_edge = [0, 0]

        # Search for the cheapest adjacent node to a region
        for n1 in regions:
            for n2 in nx.all_neighbors(g, n1):
                dst = 15 * (g.nodes[n1]['mean depth'] - g.nodes[n2]['mean depth']) ** 2 + (
                        g.nodes[n1]['pixel count'] + g.nodes[n2]['pixel count']) + g[n1][n2]['weight']
                if dst < cheapest_edge_val:
                    cheapest_edge_val = dst
                    cheapest_edge = [n1, n2]

        # Merge that cheapest adjacent node into the region
        n1 = cheapest_edge[0]
        n2 = cheapest_edge[1]
        merge_nodes(g, n1, n2)
        if n2 in regions:
            regions.remove(n2)

    return g, regions


def merge_sets_until_done(g, start_regions, regions_left=4):
    """
    Creates disjoint sets for each node and starting from the "start_regions" merge the disjoint sets until there
        are only "regions_left" regions left

    Parameters:
        - g: the graph
        - start_regions: the start regions (usually the edge nodes)
        - regions_left: the number of regions when merging stops

    Returns
        - g: the resulting graph of regions after the merging is done
        - ds: the resulting disjoint sets of nodes
    """
    g = g.copy()
    ds = DisjointSet(g.nodes())
    active = start_regions.copy()  # the active nodes

    while len(ds.subsets()) > regions_left:
        cheapest_edge_val = np.inf
        cheapest_edge = [0, 0]

        # active nodes will not have edges to nodes within their disjoint set
        # for all active nodes determines the the cheapest adjacent node
        for n1 in active:
            for n2 in nx.all_neighbors(g, n1):
                dst = g[n1][n2]['weight'] + (g.nodes[n1]['spixel count'] + g.nodes[n2]['spixel count']) * 40
                if dst < cheapest_edge_val:
                    cheapest_edge_val = dst
                    cheapest_edge = [n1, n2]

        n1 = cheapest_edge[0]
        n2 = cheapest_edge[1]

        # merge the node into the disjoint set
        ds.merge(n1, n2)
        new_spixel_count = g.nodes[n1]['spixel count'] + g.nodes[n2]['spixel count']

        # remove edges from the newly added node to other nodes in its disjoint set
        # (could also instead just count the number of nodes in the disjoint set since each is always one superpixel)
        ds_sub = ds.subset(n1)
        for n in ds_sub:
            # all nodes within a disjoint set hav ethe same super pixel count
            g.nodes[n]['spixel count'] = new_spixel_count
            if n not in active:
                active.append(n)

            n_internal_con = []
            for t in nx.all_neighbors(g, n):
                if t in ds_sub:
                    n_internal_con.append(t)

            for t in n_internal_con:
                g.remove_edge(n, t)

    return g, ds


class RAG:
    def __init__(self, img, depth, labels, objects=None):
        """
        Creates a regional adjacency graph and sets weights based on rgb, depth, and segmentation results
        - img: input image
        - depth: depth of the image
        - labels: the result of slic (should start at 1)
        - objects: the result of segment anything
        """
        n, m = np.shape(depth)
        cie_img = color.rgb2lab(img)

        # create a boundary so that we can find the nodes that are at the border of the image later
        boundary_img = np.zeros((n + 10, m + 10, 3))
        boundary_img[5:-5, 5:-5, :] = img

        boundary_cie_img = np.zeros((n + 10, m + 10, 3))
        boundary_cie_img[5:-5, 5:-5, :] = cie_img

        boundary_depth = np.zeros((n + 10, m + 10))
        boundary_depth[5:-5, 5:-5] = depth

        # expand the superpixels to include a border of 0s
        boundary_labels = np.zeros((n + 10, m + 10), dtype='int64')
        boundary_labels[5:-5, 5:-5] = labels

        # initialize a node for each superpixel
        g = graph.RAG(boundary_labels, connectivity=2)
        for n in g:
            g.nodes[n].update(
                {
                    'labels': [n],
                    'mask': None,
                    'pixel count': 0,
                    'spixel count': 0,
                    'total color': [],
                    'total depth': [],
                }
            )

        # collect information for every pixel in each superpixel
        for index in np.ndindex(boundary_labels.shape):
            current = boundary_labels[index]
            g.nodes[current]['pixel count'] += 1
            g.nodes[current]['total color'].append(boundary_cie_img[index])
            g.nodes[current]['total depth'].append(boundary_depth[index])

        # set up the nodes
        for n in g:
            g.nodes[n]['mask'] = labels == n
            g.nodes[n]['spixel count'] = 1
            g.nodes[n]['total color'] = np.array(g.nodes[n]['total color'], dtype='float64')
            g.nodes[n]['total depth'] = np.array(g.nodes[n]['total depth'], dtype='float64')
            g.nodes[n]['mean color'] = np.sum(g.nodes[n]['total color'] / g.nodes[n]['pixel count'], axis=0)
            g.nodes[n]['mean depth'] = np.sum(g.nodes[n]['total depth'] / g.nodes[n]['pixel count'], axis=0)

        # set the edge weights based on difference in color and mean depth
        for x, y, edge in g.edges(data=True):
            edge['weight'] = (
                    color.deltaE_cie76(g.nodes[x]['mean color'], g.nodes[y]['mean color']) ** 2
                    + (g.nodes[x]['mean depth'] - g.nodes[y]['mean depth']) ** 2
            )

        # save the boundary information for later
        self.boundary_graph = g.copy()
        self.boundary_img = boundary_img
        self.boundary_cie_img = boundary_cie_img
        self.boundary_depth = boundary_depth
        self.boundary_labels = boundary_labels

        # remove the boundary and save the edge nodes
        regions = []
        for node in nx.all_neighbors(g, 0):
            regions.append(node)

        for n in regions:
            g.remove_edge(n, 0)

        g.remove_node(0)

        # multiply the edge weight by 0.2 for edges between nodes that are both have 80% overlap with the same object
        if objects is not None:
            for obj in objects:
                mask = obj['segmentation']

                for n in g:

                    n_mask_overlap = np.count_nonzero(g.nodes[n]['mask'] & mask) / g.nodes[n]['pixel count']
                    if n_mask_overlap > 0.8:
                        for t in nx.all_neighbors(g, n):
                            if n < t:
                                # apply to each edge only once
                                continue

                            t_mask_overlap = np.count_nonzero(g.nodes[t]['mask'] & mask) / g.nodes[t]['pixel count']
                            if t_mask_overlap > 0.8:
                                g[n][t]['weight'] /= 5

        # save the results
        self.edge_nodes = regions
        self.graph = g
        self.img = img
        self.cie_img = cie_img
        self.depth = depth
        self.labels = labels

    def get_masks(self, num_masks=4):
        """
        Get the masks for the image by optimizing over the adjacency graph
        """
        g, s = merge_sets_until_done(self.graph, self.edge_nodes, num_masks)

        masks = []
        for i, reg in enumerate(s.subsets()):
            mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)
            for n in reg:
                mask[g.nodes[n]['mask']] = True
            mask = binary_fill_holes(mask)
            masks.append((np.average(self.depth[mask]), mask))
        
        masks.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in masks]
