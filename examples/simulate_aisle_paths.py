# examples/simulate_aisle_paths.py

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union
import numpy as np
import os
import sys
import time
from typing import List, Tuple, Optional, Dict, Any # Import all needed types

# --- Make src importable ---
# ... (sys.path modification) ...
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # --- Using XZ view (top-down) ---
    from pathweaver.environment.map_loader import load_colliders_as_polygons
    from pathweaver.environment.item_placement import find_placeable_boundaries, place_item_groups_greedy, ItemPlacement
    # --- Import Aisle Graph functions ---
    from pathweaver.environment.path_generator import generate_shopping_paths_aisle, get_map_bounds, build_aisle_graph # Import build_aisle_graph if visualizing it
    from pathweaver.analysis.market_basket import generate_synthetic_transactions, generate_item_categories_from_transactions
except ImportError as e:
    print(f"Error importing pathweaver: {e}")
    # Ensure scipy is installed
    sys.exit(1)

# --- Configuration ---
COLLIDER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'colliders.csv')
TRANSACTION_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')
MERGE_DISTANCE_THRESHOLD = 5.0

NUM_USERS_TO_SIMULATE = 50
NUM_SYNTHETIC_TRANSACTIONS = 50
# ... other configs ...
MAP_PADDING = 15.0 # Increased padding for boundary definition
ITEM_WIDTH = 2.0
ITEM_SPACING = 20


# --- Helper to load transactions ---
def load_original_transactions(file_path): # ... (implementation from before) ...
    transactions = []
    try: # ... (try/except block) ...
        with open(file_path, 'r', encoding='utf-8') as f: # ... (read logic) ...
            for line in f:
                items = {item.strip() for item in line.strip().split(',') if item.strip()}
                if items: transactions.append(items)
    except: pass # Simplified error handling for brevity
    return transactions


# --- Main execution ---
if __name__ == "__main__":
    print("--- PathWeaver Simulation Setup (Using Aisle Graph) ---")
    # --- Check files ---
    # ... (check COLLIDER_FILE_PATH, TRANSACTION_FILE_PATH) ...

    try:
        # --- 1. Load Obstacles ---
        print(f"\n[1/6] Loading and merging obstacles...")
        obstacles = load_colliders_as_polygons(COLLIDER_FILE_PATH, MERGE_DISTANCE_THRESHOLD)
        if not obstacles: sys.exit("Failed to load obstacles.")

        # --- 2. Define Map Boundary ---
        print("[2/6] Defining map boundary...")
        if not obstacles: sys.exit("Need obstacles to define boundary.")
        all_obs_union = unary_union(obstacles)
        min_x, min_y, max_x, max_y = all_obs_union.bounds
        # Use sufficient padding for boundary
        map_boundary_poly = box(min_x - MAP_PADDING, min_y - MAP_PADDING, max_x + MAP_PADDING, max_y + MAP_PADDING)

        # --- 3. Generate Categories & Place Items ---
        # (Combine steps for brevity - same logic as NavMesh example)
        print("\n[3/6] Generating categories and placing items...")
        original_transactions = load_original_transactions(TRANSACTION_FILE_PATH)
        if not original_transactions: sys.exit("Failed to load original transactions.")
        item_categories = generate_item_categories_from_transactions(original_transactions)
        if not item_categories: sys.exit("Failed to generate categories.")
        placeable_boundaries = find_placeable_boundaries(obstacles, True)
        if not placeable_boundaries: sys.exit("No placeable boundaries.")
        item_placements_list = place_item_groups_greedy(item_categories, placeable_boundaries, ITEM_WIDTH, ITEM_SPACING)
        if not item_placements_list: sys.exit("Failed to place items.")
        item_placements: Dict[str, ItemPlacement] = {p['item_id']: p for p in item_placements_list}
        print(f"  Placed {len(item_placements)} unique items.")

        # --- 4. Generate Synthetic Transactions ---
        print(f"\n[4/6] Generating {NUM_SYNTHETIC_TRANSACTIONS} synthetic transactions...")
        MIN_SUPPORT_THRESHOLD = 0.05 # Adjust based on your data (start higher, then lower if needed)
        synthetic_transactions = generate_synthetic_transactions(original_transactions, NUM_SYNTHETIC_TRANSACTIONS, MIN_SUPPORT_THRESHOLD)
        if not synthetic_transactions: sys.exit("Failed to generate transactions.")

        # --- 5. Generate Full Shopping Paths using AISLE GRAPH ---
        # NOTE: build_aisle_graph is called inside generate_shopping_paths_aisle now
        #       (or could be called here once and passed in if refactored)
        print(f"\n[5/6] Generating detailed paths for {NUM_USERS_TO_SIMULATE} users via Aisle Graph...")
        shopping_paths_data = generate_shopping_paths_aisle( # Call the AISLE version
            num_users=NUM_USERS_TO_SIMULATE,
            synthetic_transactions=synthetic_transactions,
            item_placements=item_placements,
            obstacles=obstacles,
            map_boundary=map_boundary_poly # Pass boundary
        )
        if not shopping_paths_data: print("Warning: No shopping paths generated.")

        # --- 6. Visualize ---
        print("\n[6/6] Visualization...")
        fig, ax = plt.subplots(figsize=(16, 16))

        # Plot obstacles
        for poly in obstacles: # ... (fill obstacles) ...
             x, y = poly.exterior.xy
             ax.fill(x, y, alpha=0.3, color='gray', edgecolor='darkgray')

        # Optional: Visualize the aisle graph itself
        # aisle_graph = build_aisle_graph(obstacles, map_boundary_poly) # Rebuild or get from generator
        # if aisle_graph:
        #     print("Plotting aisle graph...")
        #     pos = {node: node for node in aisle_graph.nodes()} # Use node coords as position
        #     nx.draw_networkx_edges(aisle_graph, pos, ax=ax, edge_color='lightblue', alpha=0.7, width=1.0)
        #     # nx.draw_networkx_nodes(aisle_graph, pos, ax=ax, node_size=5, node_color='blue')

        # Plot shopping paths
        path_colors = plt.cm.cool(np.linspace(0, 1, len(shopping_paths_data)))
        for i, path_data in enumerate(shopping_paths_data): # ... (plot paths, start/end markers as before) ...
            full_path_vertices = path_data['full_path']
            if not full_path_vertices: continue
            print(full_path_vertices, len(full_path_vertices))
            if len(full_path_vertices) <= 1:
                continue
            path_line = LineString(full_path_vertices)
            x, y = path_line.xy
            ax.plot(x, y, '-', color=path_colors[i], linewidth=2.0, alpha=0.9, label=f'User {path_data["user_id"]}')
            ax.plot(path_data['start_loc'].x, path_data['start_loc'].y, 'o', color=path_colors[i], markersize=8, markeredgecolor='black')
            ax.plot(path_data['end_loc'].x, path_data['end_loc'].y, 'X', color=path_colors[i], markersize=10, markeredgecolor='black')


        # Customize plot
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Simulated Aisle Paths ({len(shopping_paths_data)} Users) - XZ View')
        # ... (set labels, grid, limits using map_boundary_poly.bounds) ...
        ax.set_xlabel('X Coordinate'); ax.set_ylabel('Z Coordinate')
        ax.grid(True, linestyle='--', alpha=0.5)
        if len(shopping_paths_data) < 15: ax.legend(fontsize='medium')
        bounds = map_boundary_poly.bounds
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])

        print("Displaying final plot...")
        plt.show()
        print("Done.")

    # ... (Exception handling) ...
    except Exception as e: # ... (print traceback) ...
         import traceback; traceback.print_exc()