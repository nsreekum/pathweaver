# src/pathweaver/analysis/market_basket.py

import pandas as pd
import networkx as nx
import itertools
import community as community_louvain # Requires: pip install python-louvain
from collections import defaultdict
from typing import List, Set, Any, Optional, Union

def generate_item_categories_from_transactions(
    transactions_source: Union[str, List[List[str]], List[Set[str]]],
    weighting: str = 'count', # Options: 'count', 'jaccard'
    min_cooccurrence: int = 1,
    louvain_resolution: float = 1.0,
    louvain_random_state: Optional[int] = 42
    ) -> List[List[str]]:
    """
    Generates item categories by analyzing co-occurrence patterns in transactions
    using graph-based community detection (Louvain).

    Args:
        transactions_source: Either a file path (CSV, one transaction per line,
                             comma-separated items) or a list of lists/sets,
                             where each inner list/set represents a transaction.
        weighting: Method to weight graph edges ('count' or 'jaccard').
        min_cooccurrence: Minimum number of times items must co-occur to form an edge.
        louvain_resolution: Louvain algorithm resolution parameter. Higher values
                            tend to produce more, smaller communities.
        louvain_random_state: Random seed for Louvain for reproducibility.

    Returns:
        A list of lists, where each inner list represents a detected category
        (group of item IDs). Returns an empty list if processing fails.
    """
    print("Starting item category generation...")

    # --- 1. Load and Prepare Data ---
    transactions: List[Set[str]] = []
    if isinstance(transactions_source, str):
        try:
            print(f"Reading transactions line-by-line from file: {transactions_source}")
            with open(transactions_source, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line: # Skip empty lines
                        # Split by comma, strip whitespace from each item, filter out empty items
                        items = {item.strip() for item in stripped_line.split(',') if item.strip()}
                        if items: # Only add if transaction has at least one item
                            transactions.append(items)
        except FileNotFoundError:
            print(f"Error: Transaction file not found at {transactions_source}")
            return []
        except Exception as e:
            print(f"Error reading transaction file {transactions_source}: {e}")
            return []
    elif isinstance(transactions_source, list):
        transactions = [set(t) for t in transactions_source] # Ensure sets
        print(f"Processing {len(transactions)} provided transactions.")
    else:
        print("Error: Invalid transactions_source type.")
        return []

    # Remove empty transactions
    transactions = [t for t in transactions if t]
    if not transactions:
        print("No valid transactions found.")
        return []

    all_items = set(item for transaction in transactions for item in transaction)
    if not all_items:
        print("No items found in transactions.")
        return []
    print(f"Found {len(all_items)} unique items.")

    # --- 2. Build Item Graph ---
    G = nx.Graph()
    G.add_nodes_from(all_items)

    item_pairs = itertools.combinations(all_items, 2)
    edges_added = 0

    # Pre-calculate item counts if using Jaccard
    item_counts = defaultdict(int)
    if weighting == 'jaccard':
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

    # Calculate weights and add edges
    for item_a, item_b in item_pairs:
        pair = tuple(sorted((item_a, item_b)))
        co_occurrence_count = sum(1 for t in transactions if item_a in t and item_b in t)

        if co_occurrence_count >= min_cooccurrence:
            weight = 0.0
            if weighting == 'count':
                weight = float(co_occurrence_count)
            elif weighting == 'jaccard':
                count_a = item_counts[item_a]
                count_b = item_counts[item_b]
                union_count = count_a + count_b - co_occurrence_count # Formula: |A U B| = |A| + |B| - |A n B|
                if union_count > 0:
                    weight = float(co_occurrence_count) / float(union_count)
                else: # Should not happen if count_a/count_b > 0 and co_occurrence > 0
                    weight = 0.0
            else:
                 # Default to count if weighting arg is invalid
                 weight = float(co_occurrence_count)

            if weight > 1e-6: # Add edge if weight is meaningful
                G.add_edge(item_a, item_b, weight=weight)
                edges_added += 1

    print(f"Built graph with {G.number_of_nodes()} nodes and {edges_added} edges (min co-occurrence: {min_cooccurrence}).")

    if G.number_of_edges() == 0:
        print("Warning: No edges created in the item graph. Cannot detect communities.")
        # Return each item as its own category?
        return [[item] for item in all_items]


    # --- 3. Detect Communities (Categories) ---
    print(f"Detecting communities using Louvain (resolution={louvain_resolution})...")
    try:
        partition = community_louvain.best_partition(
            G,
            weight='weight',
            resolution=louvain_resolution,
            random_state=louvain_random_state
        )
        num_communities = len(set(partition.values()))
        print(f"Found {num_communities} communities.")
        if num_communities == 0 : # Handle edge case if partition is empty
             print("Warning: Louvain returned empty partition.")
             return [[item] for item in all_items] # Fallback
    except Exception as e:
        print(f"Error during community detection: {e}")
        return [[item] for item in all_items] # Fallback: each item is a category

    # --- 4. Extract Categories ---
    categories_dict = defaultdict(list)
    for item, community_id in partition.items():
        categories_dict[community_id].append(item)

    # Convert dictionary values to the final list of lists
    category_list = list(categories_dict.values())

    print(f"Extracted {len(category_list)} item categories.")
    return category_list