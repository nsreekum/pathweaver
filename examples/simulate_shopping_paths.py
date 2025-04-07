# examples/simulate_shopping_paths.py

import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
import os
import sys
import random # For potentially assigning transactions
from typing import Dict

# --- Make src importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# -----------------------------------------

try:
    # Use XY or XZ view consistently
    from pathweaver.environment.map_loader import load_colliders_as_polygons # Using merged XZ view
    # from pathweaver.environment.map_loader import load_colliders_as_xy_polygons # Or merged XY view

    from pathweaver.environment.item_placement import find_placeable_boundaries, place_item_groups_greedy, ItemPlacement
    from pathweaver.environment.path_generator import generate_shopping_paths, get_map_bounds
    from pathweaver.analysis.market_basket import generate_synthetic_transactions, generate_item_categories_from_transactions

except ImportError as e:
    print(f"Error importing pathweaver: {e}")
    # Ensure all dependencies including networkx, python-louvain, mlxtend are installed
    sys.exit(1)

# --- Configuration ---
COLLIDER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'colliders.csv')
TRANSACTION_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')

NUM_USERS_TO_SIMULATE = 5    # Number of user paths to generate
NUM_SYNTHETIC_TRANSACTIONS = 50 # Number of transactions to generate first
MIN_SUPPORT_SYNTH_GEN = 0.10    # Support for finding itemsets for synth generation
MIN_CATEGORY_COOCCURRENCE = 1  # Min co-occurrence for category graph edges

# Item Placement Config
ITEM_WIDTH = 2.0
ITEM_SPACING = 20
MAP_PADDING = 10.0 # Padding around obstacles for plot limits
MERGE_DISTANCE_THRESHOLD = 5.0 # Merge polygons within 5 units of each other


# --- Helper to load original transactions (same as before) ---
def load_original_transactions(file_path):
    transactions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                items = {item.strip() for item in line.strip().split(',') if item.strip()}
                if items: transactions.append(items)
        return transactions
    except Exception as e: print(f"Error loading transactions: {e}"); return None

# --- Main execution ---
if __name__ == "__main__":
    print("--- PathWeaver Simulation Setup ---")

    # --- 1. Load Obstacles (Merged) ---
    print(f"\n[1/5] Loading and merging obstacles ({COLLIDER_FILE_PATH})...")
    # Use the appropriate merged loader (XZ assumed here)
    obstacles = load_colliders_as_polygons(COLLIDER_FILE_PATH, MERGE_DISTANCE_THRESHOLD)
    if not obstacles: sys.exit("Failed to load obstacles.")

    # --- 2. Generate Item Categories ---
    print(f"\n[2/5] Generating item categories ({TRANSACTION_FILE_PATH})...")
    # First load original transactions just for category generation
    original_transactions_for_cats = load_original_transactions(TRANSACTION_FILE_PATH)
    if not original_transactions_for_cats: sys.exit("Failed to load original transactions.")
    # Generate categories
    item_categories = generate_item_categories_from_transactions(
        original_transactions_for_cats,
        min_cooccurrence=MIN_CATEGORY_COOCCURRENCE
    )
    if not item_categories: sys.exit("Failed to generate item categories.")

    # --- 3. Place Items ---
    print("\n[3/5] Finding placeable boundaries and placing items...")
    placeable_boundaries = find_placeable_boundaries(obstacles, check_clearance=True)
    if not placeable_boundaries: sys.exit("No placeable boundaries found.")
    # Place items based on the generated categories
    item_placements_list = place_item_groups_greedy(
        item_groups=item_categories, # Use generated categories
        placeable_boundaries=placeable_boundaries,
        item_width=ITEM_WIDTH,
        spacing=ITEM_SPACING
    )
    if not item_placements_list: sys.exit("Failed to place items.")
    # Convert list of placements to dict for easy lookup by item_id
    item_placements: Dict[str, ItemPlacement] = {p['item_id']: p for p in item_placements_list}
    print(f"Placed {len(item_placements)} unique items.")

    # --- 4. Generate Synthetic Transactions ---
    # Reuse original transactions for generation basis
    print(f"\n[4/5] Generating {NUM_SYNTHETIC_TRANSACTIONS} synthetic transactions...")
    synthetic_transactions = generate_synthetic_transactions(
        original_transactions=original_transactions_for_cats, # Use original data as basis
        num_new_transactions=NUM_SYNTHETIC_TRANSACTIONS,
        min_support=MIN_SUPPORT_SYNTH_GEN
    )
    if not synthetic_transactions: sys.exit("Failed to generate synthetic transactions.")

    # --- 5. Generate Full Shopping Paths ---
    print(f"\n[5/5] Generating detailed paths for {NUM_USERS_TO_SIMULATE} users...")
    shopping_paths_data = generate_shopping_paths(
        num_users=NUM_USERS_TO_SIMULATE,
        synthetic_transactions=synthetic_transactions,
        item_placements=item_placements,
        obstacles=obstacles
    )
    if not shopping_paths_data: sys.exit("Failed to generate any shopping paths.")

    # --- 6. Visualize ---
    print("\n--- Visualization ---")
    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot obstacles
    for poly in obstacles:
        x, y = poly.exterior.xy # Use y for Z coordinate if using XZ view
        ax.fill(x, y, alpha=0.3, color='gray', edgecolor='black')

    # Plot item locations (optional, can make plot busy)
    # for item_id, placement in item_placements.items():
    #    loc = placement['location']
    #    ax.plot(loc.x, loc.y, '.', color='darkorange', markersize=4)
    #    # ax.text(loc.x, loc.y, item_id, fontsize=6) # Very busy

    # Plot shopping paths
    path_colors = plt.cm.turbo(np.linspace(0, 1, len(shopping_paths_data)))
    for i, path_data in enumerate(shopping_paths_data):
        full_path_vertices = path_data['full_path']
        if not full_path_vertices: continue

        path_line = LineString(full_path_vertices)
        x_coords, y_coords = path_line.xy # Use y for Z coordinate if using XZ view
        ax.plot(x_coords, y_coords, '-', color=path_colors[i], linewidth=2, alpha=0.9, label=f'User {path_data["user_id"]}')

        # Mark start (first item loc) and end (last item loc) of the full path
        ax.plot(path_data['start_loc'].x, path_data['start_loc'].y, 'o', color=path_colors[i], markersize=8, markeredgecolor='black')
        ax.plot(path_data['end_loc'].x, path_data['end_loc'].y, 'X', color=path_colors[i], markersize=10, markeredgecolor='black')

    # Customize plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Simulated Shopping Paths ({len(shopping_paths_data)} Users) - XZ View')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate') # Label matches XZ view
    ax.grid(True, linestyle='--', alpha=0.5)
    if len(shopping_paths_data) < 15: # Avoid overly cluttered legend
        ax.legend(fontsize='medium')
    bounds = get_map_bounds(obstacles, MAP_PADDING)
    if bounds: ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])

    print("Displaying final plot...")
    plt.show()
    print("Done.")