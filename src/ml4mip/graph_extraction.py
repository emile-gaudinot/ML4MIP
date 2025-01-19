import json
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def compute_minimum_distances(component_points, valid_labels):
    num_labels = len(valid_labels)
    distances = np.inf * np.ones(
        (num_labels, num_labels)
    )  # Initialize distance matrix with infinity

    for i in range(num_labels):
        points_a = component_points[valid_labels[i]]
        tree_a = cKDTree(points_a)
        for j in range(i + 1, num_labels):
            points_b = component_points[valid_labels[j]]
            distances[i, j] = distances[j, i] = tree_a.query(points_b, k=1)[0].min()

    return distances


def connected_component_distance_filter(pred, min_size=300, max_dist=50):
    structure = np.ones((3, 3, 3), dtype=np.int64)  # Define connectivity (6, 18, or 26)
    labeled_array, _ = label(pred, structure=structure)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Remove background size
    valid_labels = np.where(component_sizes >= min_size)[0]
    component_points = {label: np.argwhere(labeled_array == label) for label in valid_labels}
    distances = compute_minimum_distances(component_points, valid_labels)
    min_distances = distances.min(axis=1) <= max_dist

    to_keep = valid_labels[min_distances]
    # Create a mask with the filtered components
    filtered_mask = np.isin(labeled_array, list(to_keep)).astype(np.uint8)

    # Count the final number of components
    final_num_components = len(to_keep)

    return filtered_mask, final_num_components


def skeleton_to_graph(skeleton) -> nx.Graph:
    """Convert a 3D skeleton into a graph where each skeleton point is a node.

    Parameters:
        skeleton: 3D binary array of the skeleton.

    Returns:
        Graph representation of the skeleton.
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


def reduce_graph(graph) -> nx.Graph:
    """Reduce the graph by keeping only nodes that are endpoints or connecting nodes.

    - Endpoints (nodes with exactly one neighbor).
    - Connecting nodes (nodes with more than two neighbors).

    Parameters:
        graph (networkx.Graph): Input graph.

    Returns:
        Reduced graph.
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
                # TODO: Develop a more robust strategy for handling irregular nodes
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


def reduce_graph_2(graph) -> nx.Graph:
    reduced_graph = nx.Graph()

    # Step 1: Add branch and endpoint nodes to the reduced graph
    for node in graph.nodes():
        degree = graph.degree[node]
        if degree == 1 or degree > 2:
            reduced_graph.add_node(node, **graph.nodes[node])

    # Step 2: Add edges between retained nodes
    for primary_node in reduced_graph.nodes():
        for primary_neighbor in graph.neighbors(primary_node):
            visited = set()
            current_node = primary_neighbor

            while current_node not in reduced_graph.nodes():
                visited.add(current_node)
                neighbors = set(graph.neighbors(current_node))
                diff = neighbors - visited - {primary_node}

                if len(diff) == 1:
                    current_node = diff.pop()
                else:
                    break

            if current_node in reduced_graph.nodes():
                reduced_graph.add_edge(primary_node, current_node)

    return reduced_graph


def merge_nodes(reduced_graph, distance_threshold) -> nx.Graph:
    # Step 3: Merge nodes that are too close
    merged_graph = nx.Graph()
    node_map = {}  # Maps original nodes to merged nodes
    retained_nodes = list(reduced_graph.nodes(data=True))

    for i, (node_a, attr_a) in enumerate(retained_nodes):
        coord_a = np.array(attr_a["coordinate"])
        merged = False

        for j, (node_b, attr_b) in enumerate(merged_graph.nodes(data=True)):
            coord_b = np.array(attr_b["coordinate"])
            if np.linalg.norm(coord_a - coord_b) < distance_threshold:
                # Merge node_a into node_b
                node_map[node_a] = node_b
                merged_graph.nodes[node_b]["coordinate"] = (
                    merged_graph.nodes[node_b]["coordinate"] + coord_a
                ) / 2  # Update merged position
                merged = True
                break

        if not merged:
            # Add node_a as a new node in the merged graph
            merged_graph.add_node(node_a, **attr_a)
            node_map[node_a] = node_a

    # Step 4: Add edges to the merged graph
    for u, v in reduced_graph.edges():
        if u in node_map and v in node_map:
            new_u = node_map[u]
            new_v = node_map[v]
            if new_u != new_v:  # Avoid self-loops
                merged_graph.add_edge(new_u, new_v)

    return merged_graph


def merge_to_two_components(graph):
    """Merge disjoint connected components in a graph until only two large components remain.

    Parameters:
    graph (networkx.Graph): The input graph with multiple disjoint connected components.

    Returns:
    networkx.Graph: The modified graph with two connected components.
    """
    # Create a copy of the graph to avoid modifying the original
    graph = graph.copy()

    while len(list(nx.connected_components(graph))) > 2:
        # Get all connected components sorted by size (smallest to largest)
        components = sorted(nx.connected_components(graph), key=len)

        # Start with the smallest component
        smallest_component = components[0]

        min_distance = float("inf")
        closest_pair = None

        # Iterate over all nodes in the smallest component
        for node in smallest_component:
            # Get the position of the current node
            pos_node = np.array(graph.nodes[node]["coordinate"])

            # Iterate through all other components (excluding the smallest)
            for other_component in components[1:]:
                for other_node in other_component:
                    # Get the position of the other node
                    pos_other = np.array(graph.nodes[other_node]["coordinate"])

                    # Calculate Euclidean distance
                    distance = np.linalg.norm(pos_node - pos_other)

                    # Update the closest pair if the distance is smaller
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (node, other_node)

        # Add multiple nodes along the edge connecting the closest pair
        node_a, node_b = closest_pair
        pos_a = np.array(graph.nodes[node_a]["coordinate"])
        pos_b = np.array(graph.nodes[node_b]["coordinate"])

        # Calculate the number of steps required (assuming step size of 1 pixel)
        num_steps = int(np.linalg.norm(pos_b - pos_a))
        step_vector = (pos_b - pos_a) / num_steps

        previous_node = node_a
        for step in range(1, num_steps):
            intermediate_pos = pos_a + step * step_vector
            intermediate_node = f"intermediate_{node_a}_{node_b}_{step}"
            graph.add_node(intermediate_node, coordinate=tuple(intermediate_pos))
            graph.add_edge(previous_node, intermediate_node)
            previous_node = intermediate_node

        # Finally connect the last intermediate node to node_b
        graph.add_edge(previous_node, node_b)

    return graph


def extract_evenly_spaced_skeleton_points(reduced_graph, original_graph, spacing=10):
    """Extract evenly spaced skeleton points for each edge in the reduced graph.

    Parameters:
    reduced_graph (networkx.Graph): The reduced graph containing only endnodes and branching nodes.
    original_graph (networkx.Graph): The original graph constructed from the skeleton.
    spacing (int): The step size for selecting skeleton points.

    Returns:
    dict: A dictionary where keys are edges of the reduced graph, and values are lists of evenly spaced skeleton points.
    """
    skeleton_points = {}

    for edge in reduced_graph.edges():
        # Extract the nodes of the edge
        start_node, end_node = edge

        # Find the path in the original graph between the start and end nodes
        try:
            path = nx.shortest_path(original_graph, source=start_node, target=end_node)
        except nx.NetworkXNoPath:
            print(f"No path found between {start_node} and {end_node}.")
            continue

        # Select every `spacing`-th node along the path
        evenly_spaced_points = path[::spacing]

        # Store the result in the dictionary
        skeleton_points[edge] = evenly_spaced_points

    return skeleton_points


def compute_length(edge, graph: nx.Graph, id_, OUTPUT_DIR):
    """Returns the length, in mm, of `edge`"""
    # Get the coordinates of the nodes
    node_a, node_b = edge
    coord_a = np.array(graph.nodes[node_a]["coordinate"])
    coord_b = np.array(graph.nodes[node_b]["coordinate"])

    # Compute the voxel dimensions
    nifti_pred = nib.load(OUTPUT_DIR / f"{id_}.pred.nii.gz")
    voxel_dimensions = np.array(nifti_pred.header["pixdim"][1:4])

    # Scale the coordinates by the voxel dimensions
    scaled_coord_a = coord_a * voxel_dimensions
    scaled_coord_b = coord_b * voxel_dimensions

    # Compute the Euclidean distance in mm
    distance_mm = np.linalg.norm(scaled_coord_a - scaled_coord_b)
    return distance_mm


def nodes_edges2json(d: dict, graph: nx.Graph, id_: str, OUTPUT_DIR):
    node_list = []
    nodes, edges = {}, {}

    for i, edge in enumerate(d.keys()):
        n1, n2 = edge
        node_list += [n1, n2]

        # Transform the skeleton of this edge into the desired `skeletons` dict
        skeletons = []
        for sk_node in d[edge]:
            # Intermediate node
            if type(sk_node) == str:
                sk_node1, sk_node2 = int(sk_node.split("_")[1]), int(sk_node.split("_")[2])
                coos1 = graph.nodes(data=True)[sk_node1]["coordinate"]
                coos2 = graph.nodes(data=True)[sk_node2]["coordinate"]
                if list(coos1) not in skeletons:
                    skeletons += [list(coos1)]
                if list(coos2) not in skeletons:
                    skeletons += [list(coos2)]
            # Not an intermediate node
            else:
                coos = graph.nodes(data=True)[sk_node]["coordinate"]
                if list(coos) not in skeletons:
                    skeletons += [list(coos)]

        # Transform the edge to the desired `edges` dict
        edges[i] = {
            "length": compute_length(
                edge, graph, id_, OUTPUT_DIR
            ),  # LENGTH STILL HAS TO BE COMPUTED
            "skeletons": skeletons,
            "source": n1,
            "target": n2,
        }

    # Add the nodes and their coordinates to the `nodes` dict
    node_list = sorted(list(set(node_list)))
    for i, node in enumerate(node_list):
        coos = graph.nodes(data=True)[node]["coordinate"]
        nodes[i] = {
            "pos": list(coos),
            "is_root": False,  # WE STILL NEED TO DETERMINE THE ROOT
            "id": node,
        }

    return nodes, edges


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export2json(d: dict, graph: nx.Graph, id_, OUTPUT_DIR):
    nodes, edges = nodes_edges2json(d, graph, id_, OUTPUT_DIR)

    # Create the final JSON
    json_dict = {
        "directed": True,
        "multigraph": False,
        "graph": {"coordinateSystem": "RAS"},
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }

    # Export the dictionnary into JSON
    with open("test.graph.json", "w") as json_file:
        json.dump(json_dict, json_file, default=convert_numpy_types, indent=4)


@dataclass
class ExtractionConfig:
    min_size: int = 100
    max_dist: int = 100
    merge_nodes_distance: int = 10
    spacing_skeleton: int = 7


def extract_graph(
    nifti_obj,
    cfg: ExtractionConfig,
    path: str | Path | None,
) -> tuple[nx.Graph, np.ndarray]:
    """Extract a reduced graph from a binary volume."""
    binary_volume = nifti_obj.get_fdata()
    binary_volume = connected_component_distance_filter(
        binary_volume, min_size=cfg.min_size, max_dist=cfg.max_dist
    )
    skeleton = skeletonize(binary_volume)
    graph = skeleton_to_graph(skeleton)
    merged_graph = merge_to_two_components(graph)
    graph = reduce_graph(merged_graph)
    graph = merge_nodes(graph, distance_threshold=cfg.merge_nodes_distance)
    graph = reduce_graph(graph)

    if path is not None:
        evenly_spaced_skeleton_points = extract_evenly_spaced_skeleton_points(
            graph, merged_graph, spacing=cfg.spacing_skeleton
        )
        export2json(evenly_spaced_skeleton_points, graph, path)

    return graph, skeleton
