import random
from typing import List, Tuple, Optional, Dict
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Define a type alias for clarity
PathData = Dict[str, Point] # e.g., {'start': Point, 'end': Point}

def get_map_bounds(obstacles: List[Polygon], padding: float = 10.0) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculates the bounding box encompassing all obstacles, plus padding.

    Args:
        obstacles: A list of Shapely Polygon objects representing obstacles.
        padding: Optional padding to add around the obstacles' bounds.

    Returns:
        A tuple (min_x, min_y, max_x, max_y) representing the bounds,
        or None if no obstacles are provided.
    """
    if not obstacles:
        return None

    # Combine all obstacles into one geometry to easily get total bounds
    # This handles cases where obstacles might define the only valid space
    all_obstacles_union = unary_union(obstacles)
    min_x, min_y, max_x, max_y = all_obstacles_union.bounds

    # Add padding
    return (min_x - padding, min_y - padding, max_x + padding, max_y + padding)


def generate_valid_point(bounds: Tuple[float, float, float, float],
                         obstacles: List[Polygon],
                         max_attempts: int = 1000) -> Optional[Point]:
    """
    Generates a random Shapely Point within the bounds that is not
    inside any of the provided obstacles.

    Args:
        bounds: Tuple (min_x, min_y, max_x, max_y) defining the sampling area.
        obstacles: List of Shapely Polygons representing forbidden areas.
        max_attempts: Maximum number of tries to find a valid point.

    Returns:
        A valid Shapely Point, or None if max_attempts reached.
    """
    min_x, min_y, max_x, max_y = bounds

    # Pre-calculating the union might be faster for many obstacles, but can be slow itself.
    # For moderate numbers, iterating might be acceptable. Let's iterate for now.
    # combined_obstacles = unary_union(obstacles) # Optimization option

    for _ in range(max_attempts):
        rand_x = random.uniform(min_x, max_x)
        rand_y = random.uniform(min_y, max_y)
        point = Point(rand_x, rand_y)

        is_inside_obstacle = False
        # Check against individual obstacles
        for obstacle in obstacles:
            # Use intersects instead of within to also catch points on the boundary
            if obstacle.intersects(point):
                is_inside_obstacle = True
                break
        # Alternative check using pre-calculated union:
        # if combined_obstacles.intersects(point):
        #     is_inside_obstacle = True

        if not is_inside_obstacle:
            return point # Found a valid point

    print(f"Warning: Failed to generate a valid point within {max_attempts} attempts.")
    return None


def generate_user_paths(n_users: int,
                        obstacles: List[Polygon],
                        min_distance: float = 200.0,
                        padding: float = 10.0,
                        max_endpoint_attempts: int = 100) -> List[PathData]:
    """
    Generates start and end points for a specified number of users.

    Ensures points are outside obstacles and the distance between
    start and end points meets the minimum requirement.

    Args:
        n_users: The number of user paths to generate.
        obstacles: List of Shapely Polygons representing obstacles.
        min_distance: The minimum required Euclidean distance between start and end points.
        padding: Padding around obstacle bounds to define the sampling area.
        max_endpoint_attempts: Max attempts to find a suitable endpoint for a given start point.


    Returns:
        A list of dictionaries, where each dictionary has 'start' and 'end'
        keys with Shapely Point objects as values.
    """
    if not obstacles:
        print("Warning: No obstacles provided, cannot determine map bounds.")
        return []

    bounds = get_map_bounds(obstacles, padding)
    if bounds is None:
         print("Warning: Could not calculate map bounds.")
         return []

    print(f"Generating paths within bounds: {bounds}")
    paths: List[PathData] = []
    generated_count = 0

    while generated_count < n_users:
        print(f"Generating path for user {generated_count + 1}/{n_users}...")
        start_point = generate_valid_point(bounds, obstacles)
        if start_point is None:
            print(f"Failed to generate a valid start point for user {generated_count + 1}. Skipping.")
            # Consider if we should retry entirely or just report failure for this user
            # For now, let's assume we might not be able to generate all n_users if space is tight.
            # Alternatively, could raise an exception if finding *any* point fails.
            continue # Try generating path for the next user index, effectively reducing total generated

        end_point = None
        for _ in range(max_endpoint_attempts):
            candidate_end = generate_valid_point(bounds, obstacles)
            if candidate_end is None:
                # Failed to even find a valid point, continue trying for an endpoint
                continue

            distance = start_point.distance(candidate_end)
            if distance >= min_distance:
                end_point = candidate_end
                break # Found a suitable endpoint

        if end_point is not None:
            paths.append({'start': start_point, 'end': end_point})
            print(f"-> Path {generated_count + 1} generated: Start({start_point.x:.1f}, {start_point.y:.1f}), End({end_point.x:.1f}, {end_point.y:.1f}), Dist({start_point.distance(end_point):.1f})")
            generated_count += 1
        else:
            print(f"Warning: Failed to find a suitable endpoint (Dist >= {min_distance}) for start point ({start_point.x:.1f}, {start_point.y:.1f}) after {max_endpoint_attempts} attempts. Retrying generation for user {generated_count + 1}.")
            # Loop will continue trying to generate a path for the current user index

    print(f"Successfully generated {len(paths)} paths.")
    return paths