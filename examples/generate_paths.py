import matplotlib.pyplot as plt
import os
import sys

# --- Make the src package importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# -----------------------------------------

try:
    # Decide which map view to use (XY or XZ)
    # from pathweaver.environment.map_loader import load_colliders_as_polygons # XZ View
    from pathweaver.environment.map_loader import load_colliders_as_polygons # XY View
    from pathweaver.environment.path_generator import generate_user_paths, get_map_bounds
except ImportError as e:
    print(f"Error importing pathweaver: {e}")
    print("Ensure you have installed the package, possibly in editable mode (`pip install -e .`),")
    print("or that the script correctly modifies sys.path to find the 'src' directory.")
    sys.exit(1)

# --- Configuration ---
COLLIDER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'colliders.csv')
NUM_USERS = 10 # How many user paths to generate
MIN_PATH_DISTANCE = 200.0 # Minimum distance between start and end
MAP_PADDING = 15.0 # Padding around obstacle bounds for sampling

# --- Main execution ---
if __name__ == "__main__":
    print("Running example: Generate and Plot User Paths")

    if not os.path.exists(COLLIDER_FILE_PATH):
         print(f"Error: Collider data file not found at {COLLIDER_FILE_PATH}")
         sys.exit(1)

    try:
        # 1. Load Obstacles (choose XY or XZ loader)
        print(f"Loading obstacles from {COLLIDER_FILE_PATH}...")
        # obstacles = load_colliders_as_polygons(COLLIDER_FILE_PATH) # XZ View
        obstacles = load_colliders_as_polygons(COLLIDER_FILE_PATH) # XY View

        if not obstacles:
            print("No obstacles loaded. Cannot generate paths.")
            sys.exit(1)

        # 2. Generate Paths
        print(f"\nGenerating {NUM_USERS} user paths (min distance: {MIN_PATH_DISTANCE})...")
        user_paths = generate_user_paths(
            n_users=NUM_USERS,
            obstacles=obstacles,
            min_distance=MIN_PATH_DISTANCE,
            padding=MAP_PADDING
        )

        if not user_paths:
            print("No paths were generated.")
            sys.exit(1)

        # 3. Visualize Results
        print("\nVisualizing map, obstacles, and paths...")
        fig, ax = plt.subplots(figsize=(14, 14))

        # Plot obstacles
        print("Plotting obstacles...")
        for poly in obstacles:
            x_coords, y_coords = poly.exterior.xy
            ax.fill(x_coords, y_coords, alpha=0.5, color='gray', edgecolor='black')

        # Plot paths (start/end points and lines)
        print("Plotting paths...")
        for i, path in enumerate(user_paths):
            start_pt = path['start']
            end_pt = path['end']

            # Plot Start Point (Blue Circle)
            ax.plot(start_pt.x, start_pt.y, 'o', color='blue', markersize=8, label='Start' if i == 0 else "")
            # Plot End Point (Red Square)
            ax.plot(end_pt.x, end_pt.y, 's', color='red', markersize=8, label='End' if i == 0 else "")
            # Plot Line between Start and End
            ax.plot([start_pt.x, end_pt.x], [start_pt.y, end_pt.y], '-', color='lime', linewidth=1.5, alpha=0.7)
            # Optional: Add text label near start point
            ax.text(start_pt.x + 2, start_pt.y + 2, f'{i+1}', color='blue', fontsize=9)


        # --- Customize the Plot ---
        ax.set_aspect('equal', adjustable='box')
        # Update title/labels based on the view used (XY or XZ)
        ax.set_title(f'Generated User Paths ({len(user_paths)}/{NUM_USERS}) - XY View')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate') # Change to Z if using XZ view
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend() # Show legend for Start/End markers
        # ax.autoscale_view() # Autoscale might zoom too much, let bounds calculation handle it mostly
        # Optionally set limits based on calculated bounds if needed
        bounds = get_map_bounds(obstacles, MAP_PADDING)
        if bounds:
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])

        print("Displaying plot...")
        plt.show()
        print("Plot window closed.")

    except Exception as e:
        print(f"An error occurred in the example script: {e}")
        raise # Re-raise during development to see traceback