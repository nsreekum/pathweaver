# src/pathweaver/simulation/recorder.py
"""
Processes generated user paths to simulate timing, record trajectory,
and identify items within Field of View (FoV). Generates output CSV files.
"""

import os
import csv
import numpy as np
from shapely.geometry import Point
from typing import List, Tuple, Dict, Any, Optional

# --- Relative Import for Type Hinting ---
try:
    # Assumes item_placement.py defines ItemPlacement = Dict[str, Any]
    from ..environment.item_placement import ItemPlacement
    # Assumes path_generator.py defines PathData
    from ..environment.path_generator import PathData
except ImportError:
    print("Warning: Could not import type aliases. Using fallback Dict[str, Any].")
    ItemPlacement = Dict[str, Any]
    PathData = Dict[str, Any]


def add_timestamps_to_path(path_vertices: List[Point], speed: float = 1.5) -> List[Tuple[Point, float]]:
    """
    Calculates timestamps for each vertex along a path assuming constant speed.

    Args:
        path_vertices: Ordered list of Shapely Point objects representing the path.
        speed: Assumed constant movement speed (units per second). Default is 1.5.

    Returns:
        A list of tuples, where each tuple is (Point, timestamp_in_seconds).
        Returns empty list if path is empty or speed is non-positive.
    """
    if not path_vertices or speed <= 0:
        return []

    path_with_time: List[Tuple[Point, float]] = []
    current_time = 0.0
    path_with_time.append((path_vertices[0], current_time)) # Start at time 0

    for i in range(1, len(path_vertices)):
        p_prev = path_vertices[i-1]
        p_curr = path_vertices[i]
        distance = p_prev.distance(p_curr)
        time_delta = distance / speed
        current_time += time_delta
        path_with_time.append((p_curr, current_time))

    return path_with_time


def get_items_in_fov(current_point: Point,
                     fov_radius: float,
                     item_placements: Dict[str, ItemPlacement]) -> List[str]:
    """
    Identifies items within a given radius (FoV) of a point.

    Args:
        current_point: The observer's current location (Shapely Point).
        fov_radius: The radius distance for the Field of View.
        item_placements: Dict mapping item_id to placement data (must contain 'location': Point).

    Returns:
        A sorted list of item IDs within the FoV radius.
    """
    items_in_view = []
    if fov_radius <= 0: return []

    for item_id, placement in item_placements.items():
        loc = placement.get('location')
        # Check if location is valid Point
        if isinstance(loc, Point):
            distance = current_point.distance(loc)
            if distance <= fov_radius:
                items_in_view.append(item_id)

    return sorted(items_in_view)


def record_user_simulation_data(
    user_path_data: PathData,
    item_placements: Dict[str, ItemPlacement],
    user_speed: float,
    fov_radius: float,
    output_dir: str,
    coordinate_system: str = 'xy' # 'xz' or 'xy' to label columns correctly
    ):
    """
    Generates trajectory.csv and fov.csv for a single user's path data.

    Args:
        user_path_data: Dict containing user's 'user_id', 'full_path', 'transaction_ordered'.
        item_placements: Dict mapping item_id to placement data.
        user_speed: Assumed constant speed for the user.
        fov_radius: Field of View radius.
        output_dir: Directory to save the CSV files.
        coordinate_system: Specifies if the y-coordinate in points represents 'y' or 'z'.
    """
    user_id = user_path_data.get('user_id', 'unknown_user')
    full_path = user_path_data.get('full_path')
    transaction_ordered = user_path_data.get('transaction_ordered', [])

    if not full_path:
        print(f"Warning: User {user_id} has no path data. Skipping CSV generation.")
        return

    print(f"Recording data for User {user_id}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Calculate Timestamps ---
    path_with_timestamps = add_timestamps_to_path(full_path, user_speed)
    if not path_with_timestamps:
        print(f"Warning: Could not generate timestamps for User {user_id}. Skipping.")
        return

    # --- 2. Prepare and Write Trajectory CSV ---
    traj_filepath = os.path.join(output_dir, f'user_{user_id}_trajectory.csv')
    pos_y_header = 'pos_z' if coordinate_system == 'xz' else 'pos_y'
    traj_headers = ['user_id', 'timestamp', 'pos_x', pos_y_header, 'interacted_item']
    traj_rows = []

    # Create lookup for target item locations
    target_locations = {}
    target_location_sequence = []
    for item_id in transaction_ordered:
        placement = item_placements.get(item_id)
        if placement and 'location' in placement and isinstance(placement['location'], Point):
            loc = placement['location']
            target_locations[loc.wkt] = item_id # Use WKT for robust key matching
            target_location_sequence.append(loc)

    current_target_index = 0
    dist_tolerance = 1e-3 # Tolerance for matching path vertex to target location

    for point, timestamp in path_with_timestamps:
        interacted_item = "" # Default to empty string
        current_target_loc = None
        if current_target_index < len(target_location_sequence):
             current_target_loc = target_location_sequence[current_target_index]

        # Check if current point IS the next target item location
        if current_target_loc and point.distance(current_target_loc) < dist_tolerance:
            interacted_item = transaction_ordered[current_target_index]
            # Move to next target *only* if we matched the current one
            current_target_index += 1

        traj_rows.append({
            'user_id': user_id,
            'timestamp': f"{timestamp:.4f}", # Format timestamp
            'pos_x': f"{point.x:.4f}",
            pos_y_header: f"{point.y:.4f}", # Use correct key based on coordinate system
            'interacted_item': interacted_item
        })

    try:
        with open(traj_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=traj_headers)
            writer.writeheader()
            writer.writerows(traj_rows)
        print(f"  Trajectory data saved to {traj_filepath}")
    except IOError as e:
        print(f"  Error writing trajectory CSV for user {user_id}: {e}")


    # --- 3. Prepare and Write FoV CSV ---
    fov_filepath = os.path.join(output_dir, f'user_{user_id}_fov.csv')
    fov_headers = ['user_id', 'timestamp', 'pos_x', pos_y_header, 'visible_items']
    fov_rows = []

    for point, timestamp in path_with_timestamps:
        items_in_view = get_items_in_fov(point, fov_radius, item_placements)
        # Format list as semi-colon separated string (or choose another format)
        items_list_str = ';'.join(items_in_view)

        fov_rows.append({
            'user_id': user_id,
            'timestamp': f"{timestamp:.4f}",
            'pos_x': f"{point.x:.4f}",
            pos_y_header: f"{point.y:.4f}",
            'visible_items': items_list_str
        })

    try:
        with open(fov_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fov_headers)
            writer.writeheader()
            writer.writerows(fov_rows)
        print(f"  FoV data saved to {fov_filepath}")
    except IOError as e:
        print(f"  Error writing FoV CSV for user {user_id}: {e}")