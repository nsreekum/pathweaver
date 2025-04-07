# src/pathweaver/environment/path_generator.py
"""
Generates agent paths within an environment defined by obstacles.
Includes methods for:
- Generating valid start/end points.
- Ordering transaction items based on proximity.
- Building an 'aisle graph' approximating centerlines using Voronoi diagrams.
- Finding paths along the aisle graph between item locations.
- Generating full, multi-segment shopping paths for multiple users.
"""

import random
import math
import itertools
import time
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Sequence, Union, Any, Set, Iterable

# --- Relative Import for Type Hinting ---
try:
    # Assumes item_placement.py defines ItemPlacement = Dict[str, Any]
    from .item_placement import ItemPlacement
except ImportError:
    print("Warning: Could not import ItemPlacement type alias from .item_placement. Using fallback Dict[str, Any].")
    ItemPlacement = Dict[str, Any]


# --- Type Aliases ---
# Output structure for the main path generation function
PathData = Dict[str, Union[int, Point, List[Point], List[str]]]
# Using coordinate tuples as node IDs in the aisle graph for NetworkX compatibility
NodeID = Tuple[float, float]


# --- Utility Functions ---

def get_map_bounds(obstacles: List[Polygon], padding: float = 10.0) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculates the bounding box encompassing all obstacles, plus padding.
    """
    if not obstacles: return None
    valid_obstacles = [obs for obs in obstacles if isinstance(obs, (Polygon, MultiPolygon)) and obs.is_valid]
    if not valid_obstacles: return None
    try:
        all_obstacles_union = unary_union(valid_obstacles)
        if all_obstacles_union.is_empty: return None
        min_x, min_y, max_x, max_y = all_obstacles_union.bounds
        return (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
    except Exception as e:
        print(f"Error calculating map bounds during unary_union: {e}")
        return None


def generate_valid_point(bounds: Tuple[float, float, float, float],
                         obstacles: List[Polygon],
                         max_attempts: int = 1000) -> Optional[Point]:
    """
    Generates a random Shapely Point within the bounds that is not
    inside any of the provided obstacles.
    """
    min_x, min_y, max_x, max_y = bounds
    # Pre-buffer obstacles slightly for robust intersection check
    buffered_obstacles = [obs.buffer(1e-9) for obs in obstacles if obs.is_valid]

    for _ in range(max_attempts):
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        is_inside_obstacle = False
        for obstacle in buffered_obstacles:
            if obstacle.intersects(point):
                is_inside_obstacle = True
                break
        if not is_inside_obstacle:
            return point # Found a valid point
    # print(f"Warning: Failed to generate a valid point within {max_attempts} attempts.")
    return None


def order_transaction_items(transaction_items: Set[str],
                            item_placements: Dict[str, ItemPlacement],
                            start_item: Optional[str] = None) -> List[str]:
    """
    Orders items in a transaction based on proximity using a greedy
    nearest-neighbor approach. Starts from item closest to origin if not specified.
    """
    if not transaction_items: return []
    relevant_locations = {}
    for item in transaction_items:
        placement = item_placements.get(item)
        if placement and 'location' in placement and isinstance(placement['location'], Point):
            relevant_locations[item] = placement['location']
        # else: print(f"Warning: Item '{item}' skipped in ordering (missing/invalid location).")

    if len(relevant_locations) < 1: return []
    if len(relevant_locations) == 1: return list(relevant_locations.keys())

    items_to_visit = set(relevant_locations.keys())
    ordered_list: List[str] = []

    if start_item and start_item in relevant_locations:
        current_item = start_item
    else:
        origin = Point(0, 0)
        current_item = min(items_to_visit, key=lambda item: relevant_locations[item].distance(origin))

    ordered_list.append(current_item)
    items_to_visit.remove(current_item)

    while items_to_visit:
        last_location = relevant_locations[current_item]
        nearest_item = min(items_to_visit, key=lambda item: relevant_locations[item].distance(last_location))
        ordered_list.append(nearest_item)
        items_to_visit.remove(nearest_item)
        current_item = nearest_item

    return ordered_list


# --- Aisle Graph Construction (Voronoi Based) ---

def build_aisle_graph(obstacles: List[Polygon],
                      map_boundary: Polygon) -> Optional[nx.Graph]:
    """
    Builds a graph representing approximate aisle centerlines using Voronoi
    diagrams of obstacle vertices, filtered to the walkable space.
    Nodes in the graph are coordinate tuples (NodeID).
    """
    print("Building aisle graph from Voronoi diagram...")
    start_time = time.time()
    # 1. Prepare Geometry
    if not map_boundary.is_valid: print("Error: Map boundary invalid."); return None
    valid_obstacles = [obs.buffer(0) for obs in obstacles if isinstance(obs, (Polygon, MultiPolygon)) and obs.is_valid]
    if not valid_obstacles: print("Warning: No valid obstacles for aisle graph."); return None
    try:
        obstacle_union = unary_union(valid_obstacles)
        walkable_space = map_boundary.difference(obstacle_union)
        if walkable_space.is_empty: print("Error: Walkable space is empty."); return None
        walkable_space_buffered = walkable_space.buffer(1e-9) # For robust checks
    except Exception as e: print(f"Error calculating walkable space: {e}"); return None

    # Extract vertices
    obstacle_vertices_coords = []
    # ... (logic to extract coords from Polygons and MultiPolygons as before) ...
    for obs in valid_obstacles: # Simplified extraction loop
        if isinstance(obs, Polygon):
            obstacle_vertices_coords.extend(list(obs.exterior.coords))
            for interior in obs.interiors: obstacle_vertices_coords.extend(list(interior.coords))
        elif isinstance(obs, MultiPolygon):
             for poly in obs.geoms:
                  if isinstance(poly, Polygon):
                       obstacle_vertices_coords.extend(list(poly.exterior.coords))
                       for interior in poly.interiors: obstacle_vertices_coords.extend(list(interior.coords))

    if len(obstacle_vertices_coords) < 3: print("Error: < 3 obstacle vertices."); return None
    unique_obstacle_vertices = np.array(list(set(map(tuple, obstacle_vertices_coords))))
    if len(unique_obstacle_vertices) < 3: print("Error: < 3 unique obstacle vertices."); return None

    # 2. Compute Voronoi
    print(f"  Computing Voronoi for {len(unique_obstacle_vertices)} vertices...")
    try:
        vor = Voronoi(unique_obstacle_vertices)
    except Exception as e: print(f"Error computing Voronoi diagram: {e}"); return None

    # 3. Filter Ridges & Build Graph
    print("  Filtering Voronoi ridges...")
    aisle_graph = nx.Graph()
    added_nodes = set() # Track added NodeIDs (tuples)

    for ridge_points in vor.ridge_vertices:
        if -1 in ridge_points: continue # Skip infinite ridges

        v1_idx, v2_idx = ridge_points
        # Use tuples of coordinates as node IDs
        p1_tuple: NodeID = tuple(vor.vertices[v1_idx])
        p2_tuple: NodeID = tuple(vor.vertices[v2_idx])

        segment = LineString([p1_tuple, p2_tuple])
        if segment.length < 1e-9: continue # Skip zero-length segments

        # Check if segment is within walkable space (using midpoint and contains check)
        mid_point = segment.interpolate(0.5, normalized=True)
        if walkable_space_buffered.contains(mid_point):
        # Optional stricter check: if walkable_space_buffered.contains(segment.buffer(-1e-9)):
            # Add nodes if new
            if p1_tuple not in added_nodes: aisle_graph.add_node(p1_tuple); added_nodes.add(p1_tuple)
            if p2_tuple not in added_nodes: aisle_graph.add_node(p2_tuple); added_nodes.add(p2_tuple)
            # Add edge
            aisle_graph.add_edge(p1_tuple, p2_tuple, weight=segment.length)

    isolated = list(nx.isolates(aisle_graph))
    aisle_graph.remove_nodes_from(isolated)
    if aisle_graph.number_of_nodes() == 0 or aisle_graph.number_of_edges() == 0:
        print("Warning: Aisle graph has no connected segments after filtering.")
        return None # Or maybe return the empty graph?

    end_time = time.time()
    print(f"  Aisle graph built in {end_time - start_time:.2f}s ({aisle_graph.number_of_nodes()} nodes, {aisle_graph.number_of_edges()} edges).")
    return aisle_graph


# --- Pathfinding on Aisle Graph ---

def find_aisle_path(start_point: Point, end_point: Point,
                    aisle_graph: nx.Graph, obstacles: List[Polygon]
                    ) -> Optional[List[Point]]:
    """
    Finds path using the aisle graph, connecting start/end to nearest graph nodes.
    """
    # print(aisle_graph)
    if aisle_graph is None or aisle_graph.number_of_nodes() < 1: return None

    graph_nodes_tuples = list(aisle_graph.nodes())
    graph_node_points = [Point(n) for n in graph_nodes_tuples]

    # 1. Find nearest graph nodes (Points and their corresponding tuple IDs)
    if not graph_node_points: return None # Handle empty graph case
    start_nearest_node_pt = min(graph_node_points, key=lambda p: start_point.distance(p))
    end_nearest_node_pt = min(graph_node_points, key=lambda p: end_point.distance(p))
    start_nearest_node_id = tuple(start_nearest_node_pt.coords[0])
    end_nearest_node_id = tuple(end_nearest_node_pt.coords[0])
    print('s ', start_nearest_node_id)
    print('e ', end_nearest_node_id)

    # 2. Check Line of Sight for connections
    obstacle_interiors = [obs.buffer(0).buffer(-1e-9) for obs in obstacles if obs.is_valid]
    obstacle_interiors = [oi for oi in obstacle_interiors if not oi.is_empty]
    def is_segment_clear(p_a: Point, p_b: Point) -> bool:
        line = LineString([p_a, p_b])
        for obs_int in obstacle_interiors:
            if line.intersects(obs_int): return False
        return True

    start_clear = is_segment_clear(start_point, start_nearest_node_pt)
    end_clear = is_segment_clear(end_point, end_nearest_node_pt)
    # if not start_clear: print(f"  Warning: LOS from start {start_point} to aisle blocked.")
    # if not end_clear: print(f"  Warning: LOS from end {end_point} to aisle blocked.")

    # 3. Find path on aisle graph
    aisle_path_points: List[Point] = []
    try:
        if start_nearest_node_id == end_nearest_node_id:
             aisle_path_points = [start_nearest_node_pt]
        else:
            heuristic = lambda u, v: Point(u).distance(Point(v))
            path_tuples = nx.astar_path(aisle_graph, start_nearest_node_id, end_nearest_node_id, heuristic=heuristic, weight='weight')
            aisle_path_points = [Point(p_tuple) for p_tuple in path_tuples]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print(f"  Warning: No path found on aisle graph nodes {start_nearest_node_id} -> {end_nearest_node_id}.")
        return None

    # 4. Construct Full Path
    full_path: List[Point] = []
    if start_clear and start_point.distance(start_nearest_node_pt) > 1e-6:
        full_path.append(start_point)

    # Add aisle path, handling potential overlap with start/end connection points
    print(aisle_path_points)
    if aisle_path_points:
        if not full_path or full_path[-1].distance(aisle_path_points[0]) > 1e-6:
            full_path.extend(aisle_path_points)
        elif len(aisle_path_points) > 1:
            full_path.extend(aisle_path_points[1:])

    # Add end point
    if end_clear and end_point.distance(end_nearest_node_pt) > 1e-6:
        if not full_path or full_path[-1].distance(end_point) > 1e-6:
            full_path.append(end_point)
    elif not end_clear and full_path and full_path[-1].distance(end_nearest_node_pt) < 1e-6:
        # End connection blocked, path ends at nearest graph node (already there)
        pass
    elif not end_clear and full_path:
         # Path ended somewhere else, connection blocked
         # Add the nearest node if it's not already the last point
         if full_path[-1].distance(end_nearest_node_pt) > 1e-6:
              full_path.append(end_nearest_node_pt)
         print("  Warning: Path ends at nearest aisle node; LOS to endpoint blocked.")
    elif not end_clear and not full_path:
         # Should not happen if aisle_path_points had content
         pass


    # 5. Simplify Path (remove collinear points)
    print(full_path)
    if len(full_path) > 2:
        simplified_path = [full_path[0]]
        # Use numpy for vector math
        v_prev = np.array(full_path[1].coords[0]) - np.array(full_path[0].coords[0])
        norm_prev = np.linalg.norm(v_prev)
        if norm_prev > 1e-9: v_prev /= norm_prev

        for i in range(1, len(full_path) - 1):
            v_curr = np.array(full_path[i+1].coords[0]) - np.array(full_path[i].coords[0])
            norm_curr = np.linalg.norm(v_curr)
            if norm_curr < 1e-9: continue # Skip if segment is zero-length

            v_curr /= norm_curr
            # Check dot product for collinearity
            if norm_prev > 1e-9 and abs(np.dot(v_prev, v_curr)) < 0.9999: # If not collinear
                simplified_path.append(full_path[i])
                v_prev = v_curr # Update previous vector direction
                norm_prev = 1.0 # Already normalized
            elif norm_prev < 1e-9: # Handle case after a zero-length segment
                 v_prev = v_curr
                 norm_prev = 1.0

        simplified_path.append(full_path[-1])
        full_path = simplified_path

    # Ensure path has at least start and end if they are same after simplification
    if not full_path and start_point.distance(end_point) < 1e-6:
         return [start_point]
    elif not full_path:
         return None


    return full_path


# --- Main Generator Function using Aisle Graph ---

def generate_shopping_paths_aisle(
    num_users: int,
    synthetic_transactions: List[Set[str]],
    item_placements: Dict[str, ItemPlacement],
    obstacles: List[Polygon],
    map_boundary: Polygon
    ) -> List[PathData]:
    """
    Generates detailed paths using a pre-built Aisle Graph based on Voronoi.
    Builds the aisle graph once.
    """
    print(f"\nGenerating detailed shopping paths for {num_users} users using Aisle Graph...")
    # --- Input Validation ---
    if not map_boundary: print("Error: Map boundary required."); return []
    if not synthetic_transactions: print("Error: No synthetic transactions."); return []
    if not item_placements: print("Error: No item placements."); return []
    # Note: obstacles can be empty, but graph/pathfinding might be trivial

    # --- 1. Build Aisle Graph (Once) ---
    aisle_graph = build_aisle_graph(obstacles, map_boundary)
    if aisle_graph is None:
        print("Error: Failed to build aisle graph. Cannot generate paths.")
        return []

    # --- Transaction Assignment ---
    num_available = len(synthetic_transactions)
    actual_num_users = min(num_users, num_available)
    if num_users > num_available: print(f"Warning: Using {actual_num_users} transactions.")
    assigned_transactions = random.sample(synthetic_transactions, actual_num_users)

    user_full_paths: List[PathData] = []

    # --- Path Generation Loop ---
    for user_id, transaction_set in enumerate(assigned_transactions):
        print(f"\nProcessing User {user_id + 1}/{actual_num_users}...")
        # --- 2. Order Items ---
        ordered_item_list = order_transaction_items(transaction_set, item_placements)
        # --- 3. Get Location Sequence ---
        location_sequence = []
        valid_ordered_list = []
        for item in ordered_item_list:
             placement = item_placements.get(item)
             if placement and 'location' in placement and isinstance(placement['location'], Point):
                  location_sequence.append(placement['location'])
                  valid_ordered_list.append(item)
        if len(location_sequence) < 2:
            print(f"  Warning: Need >= 2 valid locations. Skipping user."); continue

        # --- 4. Segment Pathfinding ---
        full_path_vertices: List[Point] = []
        path_possible = True
        for i in range(len(location_sequence) - 1):
            loc_a = location_sequence[i]
            loc_b = location_sequence[i+1]
            item_a = valid_ordered_list[i]
            item_b = valid_ordered_list[i+1]
            print(f"  Finding aisle path segment: {item_a} -> {item_b}...")

            print(loc_a, loc_b)
            segment_path = find_aisle_path(loc_a, loc_b, aisle_graph, obstacles)

            if segment_path is None:
                print(f"  ERROR: No aisle path found for segment {item_a} -> {item_b}.")
                path_possible = False; break

            # Append vertices, avoiding duplication
            if not full_path_vertices:
                full_path_vertices = segment_path # Start with the first full segment
            elif len(segment_path) > 1:
                # Check if start of new segment matches end of current path
                if full_path_vertices[-1].distance(segment_path[0]) < 1e-6:
                     full_path_vertices.extend(segment_path[1:])
                else: # Gap detected, just append (indicates issue in connection or pathfinding)
                     print("  Warning: Gap detected between path segments.")
                     full_path_vertices.extend(segment_path)
            # If segment_path has only 1 point, it means start/end mapped to same node
            # and are likely identical to previous end point, so skip adding usually

        # --- 5. Store Result ---
        if path_possible and full_path_vertices:
            user_path_data = {
                'user_id': user_id + 1,
                'transaction_ordered': valid_ordered_list,
                'start_loc': full_path_vertices[0],
                'end_loc': full_path_vertices[-1],
                'full_path': full_path_vertices
            }
            user_full_paths.append(user_path_data)
            print(f"  -> Aisle path generated for User {user_id+1} ({len(full_path_vertices)} vertices).")
        else:
             print(f"  -> Failed to generate complete aisle path for User {user_id+1}.")

    print(f"\nGenerated aisle paths for {len(user_full_paths)} users.")
    return user_full_paths