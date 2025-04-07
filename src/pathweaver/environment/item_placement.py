import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from typing import List, Tuple, Optional, Dict, Any # Import all needed types

# Define type aliases for clarity
SegmentData = Tuple[LineString, np.ndarray] # (Segment, Normal Vector)
ItemPlacement = Dict[str, Any] # {'item_id': str, 'location': Point, 'orientation': np.ndarray, 'group_id': int}

# --- Helper Function for Outward Normal ---
def get_outward_normal(segment: LineString, polygon: Polygon, epsilon: float = 1e-6) -> Optional[np.ndarray]:
    """Calculates the unit normal vector pointing outwards from the polygon for a given segment."""
    if len(segment.coords) != 2:
        return None # Should be a simple segment

    p1 = np.array(segment.coords[0])
    p2 = np.array(segment.coords[1])
    midpoint = (p1 + p2) / 2.0
    segment_vector = p2 - p1

    # Calculate perpendicular vector (candidate normal)
    # Rotate segment vector 90 degrees clockwise or counter-clockwise
    # Normal candidate 1: (y, -x)
    norm1 = np.array([segment_vector[1], -segment_vector[0]])
    # Normal candidate 2: (-y, x)
    # norm2 = np.array([-segment_vector[1], segment_vector[0]])

    # Normalize the candidate normal
    norm1_len = np.linalg.norm(norm1)
    if norm1_len < 1e-9: return None # Zero length segment?
    norm1 /= norm1_len

    # Test point slightly offset along the normal
    test_point = Point(midpoint + norm1 * epsilon)

    # Check if test point is inside the polygon
    # Use buffer(0) for robustness with points exactly on boundary
    if polygon.buffer(0).contains(test_point):
        # If inside, the normal points inwards, so flip it
        outward_normal = -norm1
    else:
        # Otherwise, it was already pointing outwards
        outward_normal = norm1

    return outward_normal


# --- Function to Find Placeable Boundaries ---
def find_placeable_boundaries(obstacles: List[Polygon], check_clearance: bool = True, clearance_dist: float = 0.1) -> List[SegmentData]:
    """
    Identifies boundary segments of obstacles that likely face free space.

    Args:
        obstacles: List of Shapely Polygons representing obstacles.
        check_clearance: If True, checks that the space slightly outward
                         from the segment midpoint is free of *other* obstacles.
        clearance_dist: Distance to check for clearance.

    Returns:
        A list of tuples, where each tuple contains:
          - A LineString representing the placeable boundary segment.
          - A numpy array representing the outward-pointing normal vector.
    """
    placeable_segments: List[SegmentData] = []
    all_obstacles_union = unary_union(obstacles) if check_clearance else None

    print(f"Finding placeable boundaries for {len(obstacles)} obstacles...")
    for i, obs in enumerate(obstacles):
        if not isinstance(obs, Polygon): continue # Skip non-polygons

        boundary = obs.exterior
        # Could also process obs.interiors if items can be placed inside holes

        coords = list(boundary.coords)
        for j in range(len(coords) - 1): # Iterate through segments
            p1_coord = coords[j]
            p2_coord = coords[j+1]
            segment = LineString([p1_coord, p2_coord])

            if segment.length < 1e-6: continue # Skip zero-length segments

            # Get the normal pointing outwards from *this* polygon
            outward_normal = get_outward_normal(segment, obs)
            if outward_normal is None: continue

            is_clear = True
            if check_clearance and all_obstacles_union:
                # Check if a point just outside the segment is clear of ALL obstacles
                midpoint = np.array(segment.interpolate(0.5, normalized=True).coords[0])
                test_point_outside = Point(midpoint + outward_normal * clearance_dist)

                # Check if the test point intersects *any* obstacle
                # Use buffer(0) for robustness
                if all_obstacles_union.buffer(0).intersects(test_point_outside):
                   is_clear = False

                   # Optional: Check specifically against *other* obstacles
                   # is_clear_from_others = True
                   # for k, other_obs in enumerate(obstacles):
                   #     if i == k: continue # Don't check against self
                   #     if other_obs.buffer(0).intersects(test_point_outside):
                   #         is_clear_from_others = False
                   #         break
                   # is_clear = is_clear_from_others

            if is_clear:
                placeable_segments.append((segment, outward_normal))

    print(f"Found {len(placeable_segments)} placeable boundary segments.")
    return placeable_segments


# --- Function to Place Item Groups (Simple Greedy Approach) ---
def place_item_groups_greedy(item_groups: List[List[str]],
                             placeable_boundaries: List[SegmentData],
                             item_width: float = 1.0, # Assumed width each item takes along boundary
                             spacing: float = 0.2 # Spacing between items
                            ) -> List[ItemPlacement]:
    """
    Places items from groups onto available boundary segments using a greedy approach.

    Args:
        item_groups: A list of lists, where each inner list contains item IDs for a group.
        placeable_boundaries: Output from find_placeable_boundaries.
        item_width: The space each item occupies along the boundary segment.
        spacing: Additional spacing between placed items.

    Returns:
        A list of dictionaries, each describing a placed item's location and orientation.
    """
    item_placements: List[ItemPlacement] = []
    boundary_availability = [(seg, norm, seg.length) for seg, norm in placeable_boundaries]
    # Sort boundaries, e.g., by length descending (optional)
    # boundary_availability.sort(key=lambda x: x[2], reverse=True)

    segment_usage = [0.0] * len(boundary_availability) # Track used length on each segment
    total_item_step = item_width + spacing

    print(f"Placing items from {len(item_groups)} groups...")
    group_id_counter = 0
    for group in item_groups:
        group_id_counter += 1
        group_placed = False
        # Try to place the whole group contiguously first
        required_length = len(group) * item_width + max(0, len(group) - 1) * spacing

        for seg_idx, (segment, normal, total_length) in enumerate(boundary_availability):
            available_length = total_length - segment_usage[seg_idx]
            if available_length >= required_length:
                # Place all items of this group on this segment
                start_dist = segment_usage[seg_idx] + item_width / 2.0 # Center of first item
                for item_id in group:
                    location_point = segment.interpolate(start_dist)
                    item_placements.append({
                        'item_id': item_id,
                        'location': location_point,
                        'orientation': normal,
                        'group_id': group_id_counter
                    })
                    start_dist += total_item_step
                segment_usage[seg_idx] += required_length # Mark space as used
                print(f"Placed group {group_id_counter} (size {len(group)}) on segment {seg_idx}")
                group_placed = True
                break # Move to the next group

        if not group_placed:
             # Simple fallback: Place items individually if group didn't fit contiguously
             # More complex logic could split groups across segments if needed
             print(f"Warning: Could not find contiguous space for group {group_id_counter} (size {len(group)}). Placing items individually.")
             items_to_place = list(group) # Copy group list
             while items_to_place:
                 item_placed_this_round = False
                 for seg_idx, (segment, normal, total_length) in enumerate(boundary_availability):
                     if segment_usage[seg_idx] + total_item_step <= total_length:
                         item_id = items_to_place.pop(0) # Take first unplaced item
                         start_dist = segment_usage[seg_idx] + item_width / 2.0
                         location_point = segment.interpolate(start_dist)
                         item_placements.append({
                             'item_id': item_id,
                             'location': location_point,
                             'orientation': normal,
                             'group_id': group_id_counter
                         })
                         segment_usage[seg_idx] += total_item_step
                         print(f"Placed item '{item_id}' (from group {group_id_counter}) on segment {seg_idx}")
                         item_placed_this_round = True
                         if not items_to_place: break # All items from group placed
                 if not item_placed_this_round or not items_to_place:
                     break # Either no more space anywhere, or all items placed
             if items_to_place:
                 print(f"Warning: Ran out of space to place remaining {len(items_to_place)} items from group {group_id_counter}.")


    print(f"Total items placed: {len(item_placements)}")
    return item_placements