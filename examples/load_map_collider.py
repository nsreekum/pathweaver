import matplotlib
try:
    matplotlib.use('TkAgg')
    print("Attempting to use TkAgg backend.")
except ImportError:
    print("Failed to set TkAgg backend. Trying default.")
    
import matplotlib.pyplot as plt
import os
import sys

# --- Make the src package importable ---
# Get the absolute path to the project root directory (one level up from examples)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Construct the path to the src directory
src_path = os.path.join(project_root, 'src')
# Add the src directory to the Python path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# -----------------------------------------

# Now we can import from our package
try:
    from pathweaver.environment.map_loader import load_colliders_as_polygons
except ImportError as e:
    print(f"Error importing pathweaver: {e}")
    print("Ensure you have installed the package, possibly in editable mode (`pip install -e .`),")
    print("or that the script correctly modifies sys.path to find the 'src' directory.")
    sys.exit(1)


# --- Configuration ---
COLLIDER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'colliders.csv')
# --- Set the distance threshold for merging ---
MERGE_DISTANCE_THRESHOLD = 5.0 # Merge polygons within 5 units of each other
# --- Set to 0 to see only touching/overlap merge ---
# MERGE_DISTANCE_THRESHOLD = 0.0

# --- Main execution ---
if __name__ == "__main__":
    print("Running example: Load and Plot Collider Map")

    # Ensure the data file exists if using local path
    if not os.path.exists(COLLIDER_FILE_PATH):
         print(f"Error: Collider data file not found at {COLLIDER_FILE_PATH}")
         print("Please download it or check the path.")
         # Instructions to download using curl (requires curl command)
         print(f"\nSuggestion: mkdir -p examples/data && curl -o {COLLIDER_FILE_PATH} https://raw.githubusercontent.com/NS-PhD-Research/aprox/main/dataset/maps/colliders.csv\n")
         sys.exit(1)

    try:
        # Load the collider polygons using the function from our package
        collider_polygons = load_colliders_as_polygons(COLLIDER_FILE_PATH, merge_threshold=MERGE_DISTANCE_THRESHOLD)

        if not collider_polygons:
            print("No colliders were loaded or generated. Exiting.")
            sys.exit(1)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))

        print("Plotting loaded polygons...")
        # Plot each polygon
        for poly in collider_polygons:
            x_coords, z_coords = poly.exterior.xy
            ax.fill(x_coords, z_coords, alpha=0.7, color='gray', edgecolor='black')

        # Customize the plot
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'2D Geographic Map')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.autoscale_view() # Adjust view to encompass all plotted objects

        print("Displaying plot...")
        plt.show()
        print("Plot window closed.")

    except Exception as e:
        print(f"An error occurred in the example script: {e}")