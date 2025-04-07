# examples/place_items_example.py

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os
import sys

# --- Make the src package importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# -----------------------------------------

try:
    from pathweaver.environment.map_loader import load_colliders_as_polygons
    from pathweaver.environment.item_placement import find_placeable_boundaries, place_item_groups_greedy
    from pathweaver.environment.path_generator import get_map_bounds
    # --- Import the new analysis function ---
    from pathweaver.analysis.market_basket import generate_item_categories_from_transactions
except ImportError as e:
    print(f"Error importing pathweaver: {e}")
    # Make sure all dependencies including python-louvain are installed
    sys.exit(1)

# --- Configuration ---
COLLIDER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'colliders.csv')
# --- Path to transaction data ---
TRANSACTION_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')

MAP_PADDING = 5.0
ITEM_WIDTH = 2.0
ITEM_SPACING = 20

# --- Main execution ---
if __name__ == "__main__":
    print("Running example: Generate Categories and Place Items")

    # Check input files
    if not os.path.exists(COLLIDER_FILE_PATH):
         print(f"Error: Collider data file not found at {COLLIDER_FILE_PATH}")
         sys.exit(1)
    if not os.path.exists(TRANSACTION_FILE_PATH):
         print(f"Error: Transaction data file not found at {TRANSACTION_FILE_PATH}")
         # Provide sample data or instructions if needed
         print(f"\nPlease create a sample transaction CSV at {TRANSACTION_FILE_PATH}")
         sys.exit(1)


    try:
        # --- Step A: Generate Item Categories from Transactions ---
        print(f"\nGenerating item categories from: {TRANSACTION_FILE_PATH}")
        item_categories = generate_item_categories_from_transactions(
            TRANSACTION_FILE_PATH,
            weighting='count' # Or 'jaccard'
            # Adjust other parameters like resolution if needed
        )

        if not item_categories:
            print("Failed to generate item categories. Exiting.")
            sys.exit(1)

        print("\nGenerated Item Categories:")
        for i, category in enumerate(item_categories):
            print(f"  Category {i+1}: {category}")


        # --- Step B: Load Obstacles ---
        print(f"\nLoading obstacles from {COLLIDER_FILE_PATH}...")
        MERGE_DISTANCE_THRESHOLD = 5.0 # Merge polygons within 5 units of each other
        obstacles = load_colliders_as_polygons(COLLIDER_FILE_PATH, merge_threshold=MERGE_DISTANCE_THRESHOLD)

        if not obstacles:
            print("No obstacles loaded. Cannot place items.")
            sys.exit(1)

        # --- Step C: Find Placeable Boundaries ---
        print("\nFinding placeable boundaries...")
        placeable_boundaries = find_placeable_boundaries(obstacles, check_clearance=True)

        if not placeable_boundaries:
             print("No placeable boundaries found.")
             sys.exit(1)

        # --- Step D: Place Item Groups (using generated categories) ---
        print("\nPlacing items based on generated categories...")
        item_placements = place_item_groups_greedy(
            item_groups=item_categories, # Use the generated categories
            placeable_boundaries=placeable_boundaries,
            item_width=ITEM_WIDTH,
            spacing=ITEM_SPACING
        )

        # --- Step E: Visualize ---
        # (Visualization code remains the same as in the previous item_placement example)
        print("\nVisualizing map, boundaries, and placed items...")
        fig, ax = plt.subplots(figsize=(14, 14))
        # Plot obstacles
        for poly in obstacles:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3, color='gray', edgecolor='black')
        # Plot items and orientation arrows
        plotted_items = 0
        group_colors = plt.cm.tab10(np.linspace(0, 1, len(item_categories))) # Color by category
        for placement in item_placements:
            loc = placement['location']
            orient = placement['orientation']
            group_id = placement.get('group_id', 0) # Get group ID used for coloring
            color = group_colors[ (group_id - 1) % len(group_colors) ] # Cycle colors

            ax.plot(loc.x, loc.y, 'o', color=color, markersize=5)
            arrow_start = (loc.x, loc.y)
            arrow_end = (loc.x + orient[0] * ITEM_WIDTH, loc.y + orient[1] * ITEM_WIDTH)
            arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', color=color, mutation_scale=10, lw=0.5)
            ax.add_patch(arrow)
            plotted_items += 1
        print(f"Visualizing {plotted_items} placed items.")
        # Customize plot
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Item Placements (Data-Driven Categories) - XY View')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, linestyle='--', alpha=0.5)
        bounds = get_map_bounds(obstacles, MAP_PADDING)
        if bounds: ax.set_xlim(bounds[0]-MAP_PADDING, bounds[2]+MAP_PADDING); ax.set_ylim(bounds[1]-MAP_PADDING, bounds[3]+MAP_PADDING)
        print("Displaying plot...")
        plt.show()
        print("Plot window closed.")

    except Exception as e:
        print(f"An error occurred in the example script: {e}")
        raise