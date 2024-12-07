import logging

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def skeleton_to_graph(skeleton):
    """Convert a 3D skeleton into a graph where each skeleton point is a node.

    Parameters:
        skeleton (numpy.ndarray): 3D binary array of the skeleton.

    Returns:
        G (networkx.Graph): Graph representation of the skeleton.
    """
    # Define a 3x3x3 neighborhood kernel to find neighbors
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0  # Exclude the center point

    # Identify all skeleton points
    skeleton_coords = np.argwhere(skeleton > 0)

    # Initialize the graph
    G = nx.Graph()

    # Add all skeleton points as nodes
    for idx, coord in enumerate(skeleton_coords):
        G.add_node(idx, coordinate=tuple(coord))

    # Create a map of coordinates to node indices for fast lookup
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(skeleton_coords)}

    # Find neighbors for each skeleton point and add edges
    for coord in skeleton_coords:
        # Get neighbors within the kernel
        neighbors = np.argwhere(kernel) - 1  # Offsets for 3x3x3 neighborhood
        for offset in neighbors:
            neighbor_coord = tuple(coord + offset)
            if neighbor_coord in coord_to_index:  # Check if neighbor exists in skeleton
                G.add_edge(coord_to_index[tuple(coord)], coord_to_index[neighbor_coord])

    return G


def reduce_graph(graph):
    """Reduce the graph by keeping only nodes that are endpoints or connecting nodes.

    - Endpoints (nodes with exactly one neighbor).
    - Connecting nodes (nodes with more than two neighbors).

    Parameters:
        graph (networkx.Graph): Input graph.

    Returns:
        reduced_graph (networkx.Graph): Reduced graph.
    """
    # Initialize the reduced graph
    reduced_graph = nx.Graph()

    # Iterate over the nodes in the original graph
    for node in graph.nodes():
        degree = graph.degree[node]  # Number of neighbors

        # Retain the node if it is an endpoint or a connecting node
        if degree == 1 or degree > 2:
            # Add the node to the reduced graph with its attributes
            reduced_graph.add_node(node, **graph.nodes[node])  # Copy node attributes

    for primary_node in reduced_graph.nodes():
        for primary_neighbor in graph.neighbors(primary_node):
            visited = set()
            current_node = primary_neighbor

            irregular_node = False
            while current_node not in reduced_graph.nodes():
                visited.add(current_node)
                current_node_neighbors = set(graph.neighbors(current_node))
                # Develop a more robust strategy for handling irregular nodes
                # assert len(current_node_neighbors) == 2, f"{len(current_node_neighbors)=}"
                diff = current_node_neighbors - visited - {primary_node}
                # assert len(diff) == 1, f"{len(diff)=}"
                if len(diff) == 1:
                    current_node = diff.pop()
                else:
                    irregular_node = True
                    msg = f"{current_node=} is irregular | {len(diff)=}"
                    logger.info(msg)
                    break
            if not irregular_node:
                reduced_graph.add_edge(primary_node, current_node)

    return reduced_graph


def extract_graph(binary_volume: np.ndarray):
    skeleton = skeletonize(binary_volume)
    graph = skeleton_to_graph(skeleton)
    reduced_graph = reduce_graph(graph)
    return reduced_graph, skeleton
