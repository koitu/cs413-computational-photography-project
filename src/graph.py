import numpy as np
import networkx as nx

from skimage import graph, color
from scipy.cluster.hierarchy import DisjointSet


def merge_nodes(g, n1, n2):
    """
    merges the node n2 into the n1 while combining their pixel count, color, and depth into a single larger node
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
    - g: the graph
    - start_regions: the start regions (usually the edge nodes)
    - regions_left: the number of regions when merging stops
    """
    g = g.copy()
    regions = start_regions.copy()

    while g.number_of_nodes() > regions_left:
        cheapest_edge_val = np.inf
        cheapest_edge = [0, 0]

        for n1 in regions:
            for n2 in nx.all_neighbors(g, n1):
                dst = 15 * (g.nodes[n1]['mean depth'] - g.nodes[n2]['mean depth']) ** 2 + (
                        g.nodes[n1]['pixel count'] + g.nodes[n2]['pixel count']) + g[n1][n2]['weight']
                if dst < cheapest_edge_val:
                    cheapest_edge_val = dst
                    cheapest_edge = [n1, n2]

        n1 = cheapest_edge[0]
        n2 = cheapest_edge[1]
        merge_nodes(g, n1, n2)
        if n2 in regions:
            regions.remove(n2)

    return g, regions


def merge_sets_until_done(g, start_regions, regions_left=4):
    g = g.copy()
    ds = DisjointSet(g.nodes())
    active = start_regions.copy()

    while len(ds.subsets()) > regions_left:
        cheapest_edge_val = np.inf
        cheapest_edge = [0, 0]

        # active nodes don't have attachments to interior nodes
        for n1 in active:
            for n2 in nx.all_neighbors(g, n1):
                dst = g[n1][n2]['weight'] + (g.nodes[n1]['spixel count'] + g.nodes[n2]['spixel count']) * 40
                if dst < cheapest_edge_val:
                    cheapest_edge_val = dst
                    cheapest_edge = [n1, n2]

        n1 = cheapest_edge[0]
        n2 = cheapest_edge[1]

        ds.merge(n1, n2)
        new_spixel_count = g.nodes[n1]['spixel count'] + g.nodes[n2]['spixel count']

        ds_sub = ds.subset(n1)
        for n in ds_sub:
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
                    'spixel count': 0,
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
            g.nodes[n]['spixel count'] = 1
            g.nodes[n]['total color'] = np.array(g.nodes[n]['total color'], dtype='float64')
            g.nodes[n]['total depth'] = np.array(g.nodes[n]['total depth'], dtype='float64')
            g.nodes[n]['mean color'] = np.sum(g.nodes[n]['total color'] / g.nodes[n]['pixel count'], axis=0)
            g.nodes[n]['mean depth'] = np.sum(g.nodes[n]['total depth'] / g.nodes[n]['pixel count'], axis=0)

        for x, y, edge in g.edges(data=True):
            # some kind of loss function with the distance, color difference, and segment anything result
            # edge['weight'] = (
            #         color.deltaE_cie76(g.nodes[x]['mean color'], g.nodes[y]['mean color']) ** 2
            # )
            edge['weight'] = (
                    color.deltaE_cie76(g.nodes[x]['mean color'], g.nodes[y]['mean color']) ** 2
                    + (g.nodes[x]['mean depth'] - g.nodes[y]['mean depth']) ** 2
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

        self.edge_nodes = regions
        self.graph = g
        self.img = img
        self.cie_img = cie_img
        self.depth = depth
        self.labels = labels

    
    def get_masks(num_masks=4):
        g, s = merge_sets_until_done(self.graph, self.edge_nodes, num_masks)
        
        masks = []
        for i, reg in enumerate(s.subsets()):
            mask = np.zeros((img_n, img_m), dtype=bool)
            for n in reg:
                mask[g.nodes[n]['mask']] = True
            mask = binary_fill_holes(mask)
            masks.append((np.average(self.depth[mask]), mask))
        
        masks.sort(key=lambda x: x[0])
        return [x[1] for x in masks]
