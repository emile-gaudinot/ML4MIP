import json
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
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


def add_skeleton2edges(reduced_graph, original_graph, spacing=10):
    """Extract evenly spaced skeleton points for each edge in the reduced graph."""
    for edge in reduced_graph.edges():
        start_node, end_node = edge
        try:
            path = nx.shortest_path(original_graph, source=start_node, target=end_node)
        except nx.NetworkXNoPath:
            logger.warning("No path found between %s and %s.", start_node, end_node)
            continue

        nodes = path[::spacing]  # Extract every `spacing`-th node
        reduced_graph.edges[edge]["skeletons"] = [
            {"node": node, "coordinate": original_graph.nodes[node]["coordinate"]} for node in nodes
        ]

    return reduced_graph


def determine_root_nodes(graph: nx.Graph):
    """Determine the root nodes."""
    # Compute the root in each component
    # Find connected components
    components = list(nx.connected_components(graph))
    assert len(components) == 2, "More than 2 components in final_graph"
    # Create subgraphs for each component
    subgraphs = [graph.subgraph(cc).copy() for cc in components]
    root = [-1, -1]
    for i, cc in enumerate(subgraphs):
        z_max = -np.inf
        for node in cc.nodes:
            _, _, z = cc.nodes[node]["coordinate"]
            if z > z_max:
                z_max = z
                root[i] = node

    logger.info("Root nodes: %s", root)
    for node in graph.nodes:
        graph.nodes[node]["root"] = node in root
    
    return graph


def root_based_direct_graph(graph: nx.Graph):
    """Orientate edges according to root nodes."""
    # Find the root nodes
    root_nodes = [node for node in graph.nodes if graph.nodes[node].get("root", False)]
    assert len(root_nodes) == 2, "More than 2 root nodes found."

    visited = set()
    directed_edges = set()

    def traverse_graph(node):
        visited.add(node)
        for neighbor in [n for n in graph.neighbors(node) if n not in visited]:
            directed_edges.add((node, neighbor))
            traverse_graph(neighbor)

    for root_node in root_nodes:
        traverse_graph(root_node)

    # create a new directed graph
    directed_graph = nx.DiGraph()
    for u, v in directed_edges:
        directed_graph.add_edge(u, v)

    # copy the node attributes
    for node in graph.nodes:
        directed_graph.nodes[node].update(graph.nodes[node])

    return directed_graph


def add_edge_length(graph: nx.Graph, pixdim: np.ndarray):
    for edge in graph.edges:
        node_a, node_b = edge
        scaled_coord_a = np.array(graph.nodes[node_a]["coordinate"]) * pixdim
        scaled_coord_b = np.array(graph.nodes[node_b]["coordinate"]) * pixdim
        graph.edges[edge]["length"] = np.linalg.norm(scaled_coord_a - scaled_coord_b)
    return graph


def export2json(graph: nx.Graph):
    nodes, edges = {}, {}

    for i, (n1, n2) in enumerate(graph.edges):
        edge_data = graph[n1][n2]
        edges[i] = {
            "length": edge_data.get("length", None),
            "skeletons": [list(skn["coordinate"]) for skn in edge_data.get("skeletons", [])],
            "source": n1,
            "target": n2,
        }

    for i, node in enumerate(graph.nodes):
        nodes[i] = {
            "pos": list(graph.nodes[node]["coordinate"]),
            "is_root": graph.nodes[node].get("root", False),
            "id": node,
        }

    # Create the final JSON
    return {
        "directed": True,
        "multigraph": False,
        "graph": {"coordinateSystem": "RAS"},
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


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
    binary_volume, _ = connected_component_distance_filter(
        binary_volume, min_size=cfg.min_size, max_dist=cfg.max_dist
    )
    skeleton = skeletonize(binary_volume)
    graph = skeleton_to_graph(skeleton)
    merged_graph = merge_to_two_components(graph)
    graph = reduce_graph(merged_graph)
    graph = merge_nodes(graph, distance_threshold=cfg.merge_nodes_distance)
    graph = reduce_graph(graph)
    graph = determine_root_nodes(graph)
    graph = root_based_direct_graph(graph)
    pixdim = np.array(nifti_obj.header["pixdim"][1:4])
    graph = add_edge_length(graph, pixdim=pixdim)
    graph = add_skeleton2edges(graph, merged_graph, spacing=cfg.spacing_skeleton)

    if path is not None:
        json_dict = export2json(graph)
        with open(path, "w") as json_file:
            json.dump(json_dict, json_file, default=convert_numpy_types, indent=4)
        print("write file done:", path)

    return graph, skeleton
