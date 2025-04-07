# src/pathweaver/environment/map_loader.py

import pandas as pd
# Make sure unary_union is imported
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, Point, box # Add MultiPolygon
from shapely.affinity import translate
# Make sure necessary typing imports are present
from typing import List, Tuple, Optional, Union


# --- Function to load XZ colliders and return MERGED polygons ---
def load_colliders_as_polygons(file_path_or_url: str, merge_threshold: float = 0.1) -> List[Polygon]:
    """
    Loads axis-aligned collider data (XZ plane), merges polygons that are
    within a specified distance threshold using buffering.

    Expects CSV columns: posX, posZ, halfSizeX, halfSizeZ.

    Args:
        file_path_or_url: Path or URL to the CSV file.
        merge_threshold: Max distance between polygons to be considered
                         for merging. If <= 0, performs standard touching/
                         overlap merge only (no buffering).

    Returns:
        A list of Shapely Polygon objects representing the merged obstacles.
        Note: Buffering may alter shapes slightly.

    Raises:
        FileNotFoundError, Exception: As per underlying operations.
    """
    print(f"Attempting to load collider data from: {file_path_or_url} for distance merge (XZ View)")
    # ... (Initial CSV loading and cleaning code remains the same as in
    #      load_colliders_as_merged_polygons - up to creating individual_polygons) ...
    # --- Assume 'individual_polygons: List[Polygon]' has been created ---
    try:
        # (Paste the CSV loading, cleaning, and individual polygon creation logic here
        #  from the previous 'load_colliders_as_merged_polygons' function)

        # Load CSV
        df_colliders = pd.read_csv(file_path_or_url)
        # Required cols for XZ
        required_cols = ['posX', 'posY', 'halfSizeX', 'halfSizeY']
        # Data cleaning... (ensure this part is copied correctly)
        missing_required = [col for col in required_cols if col not in df_colliders.columns]
        if missing_required: raise ValueError(f"Missing required columns: {missing_required}")
        cols_to_convert = required_cols + ['posZ', 'halfSizeZ'] # Optional cols
        cols_to_convert = [col for col in cols_to_convert if col in df_colliders.columns]
        for col in cols_to_convert: df_colliders[col] = pd.to_numeric(df_colliders[col], errors='coerce')
        df_colliders.dropna(subset=required_cols, inplace=True)

        # Create individual polygons
        individual_polygons: List[Polygon] = []
        for index, row in df_colliders.iterrows():
            center_x, center_z = row['posX'], row['posY']
            half_width, half_height = row['halfSizeX'], row['halfSizeY']
            if half_width <= 0 or half_height <= 0: continue
            poly_at_origin = box(-half_width, -half_height, half_width, half_height)
            final_polygon = translate(poly_at_origin, xoff=center_x, yoff=center_z)
            individual_polygons.append(final_polygon)
        # --- End of copied loading logic ---


        if not individual_polygons:
            print("No valid individual polygons created.")
            return []

        print(f"Generated {len(individual_polygons)} individual polygons.")

        # --- Merging Logic ---
        if merge_threshold <= 0:
            print("Merge threshold <= 0, performing standard union only.")
            geometry_to_process = unary_union(individual_polygons)
        else:
            print(f"Buffering polygons by {merge_threshold / 2.0:.2f} for proximity merge...")
            # Buffer outwards - cap_style=2 (FLAT), join_style=2 (MITRE) are common defaults
            # Mitre joins can cause spikes, Bevel (1) is safer but less sharp.
            try:
                buffered_polygons = [p.buffer(merge_threshold / 2.0, join_style=2) for p in individual_polygons]
            except Exception as e:
                 print(f"Error during outward buffering: {e}. Skipping buffer step.")
                 # Fallback to standard union if buffering fails
                 geometry_to_process = unary_union(individual_polygons)
                 merge_threshold = 0 # Ensure no debuffer happens
            else:
                print("Performing union on buffered polygons...")
                geometry_to_process = unary_union(buffered_polygons)
                print("Union complete.")

        # --- Debuffer and Prepare Output List ---
        merged_polygons: List[Polygon] = []
        if geometry_to_process.is_empty:
            print("Warning: Merged geometry is empty.")
            return []

        geometries_to_debuffer = []
        if isinstance(geometry_to_process, Polygon):
            geometries_to_debuffer.append(geometry_to_process)
        elif isinstance(geometry_to_process, MultiPolygon):
            geometries_to_debuffer.extend(list(geometry_to_process.geoms))
        else:
             print(f"Warning: Unexpected geometry type after initial union: {type(geometry_to_process)}")
             # Attempt to process if it looks like a polygon
             if hasattr(geometry_to_process, 'exterior'):
                 geometries_to_debuffer.append(geometry_to_process)


        if merge_threshold > 0:
            print(f"Debuffering results by -{merge_threshold / 2.0:.2f}...")
            for geom in geometries_to_debuffer:
                try:
                    # Buffer inwards - cap_style=3 (SQUARE) is often used for inward
                    debuffered = geom.buffer(-merge_threshold / 2.0, join_style=2, cap_style=3)
                    # Check result type and validity after debuffering
                    if not debuffered.is_empty and debuffered.is_valid:
                        if isinstance(debuffered, Polygon):
                            merged_polygons.append(debuffered)
                        elif isinstance(debuffered, MultiPolygon):
                            # Add valid polygons from the resulting multipolygon
                            merged_polygons.extend([p for p in debuffered.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty])
                except Exception as e:
                     print(f"Error during inward buffering: {e}. Skipping this geometry component.")
            print("Debuffering complete.")
        else:
            # If no buffering was done, just ensure they are valid Polygons
             merged_polygons.extend([g for g in geometries_to_debuffer if isinstance(g, Polygon) and g.is_valid and not g.is_empty])


        print(f"Returning {len(merged_polygons)} final merged polygon components.")
        return merged_polygons

    except FileNotFoundError:
        print(f"Error: File not found at {file_path_or_url}")
        raise
    except Exception as e:
        print(f"An error occurred during collider loading or merging: {e}")
        raise

# --- Optional: Keep previous functions or rename them ---
# e.g., load_colliders_as_individual_polygons(...)
#       load_colliders_merged_by_touch(...) # Standard unary_union
#       load_colliders_merged_by_distance(...) # This new one