# src/pathweaver/analysis/market_basket.py

import sys
import random
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import community as community_louvain # Requires: pip install python-louvain
from collections import defaultdict, Counter
from typing import List, Set, Any, Optional, Union

# --- Add mlxtend imports ---
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

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

def generate_synthetic_transactions(
    original_transactions: List[Set[str]],
    num_new_transactions: int,
    min_support: float = 0.01,
    use_fpgrowth: bool = True,
    max_len: Optional[int] = None,
    max_noise_add_attempts: int = 10 # Attempts to add each noise item
    ) -> List[Set[str]]:
    """
    Generates synthetic transactions based on frequent itemsets, attempting
    to match the size distribution of the original transactions and adding
    noise items based on frequency.

    Args:
        original_transactions: List of sets representing original transactions.
        num_new_transactions: Number of synthetic transactions to generate.
        min_support: Minimum support threshold for frequent itemsets.
        use_fpgrowth: If True, use FP-Growth algorithm; otherwise use Apriori.
        max_len: Maximum length of frequent itemsets to consider.
        max_noise_add_attempts: Max attempts to find a suitable unique noise item.

    Returns:
        List of sets representing generated synthetic transactions.
    """
    print(f"\nStarting enhanced synthetic transaction generation ({num_new_transactions} requested)...")
    if not original_transactions:
        print("Error: No original transactions provided."); return []

    # --- 1. Analyze Original Data ---
    print("Analyzing original transaction statistics...")
    # Calculate item frequencies
    all_items_flat = [item for trans in original_transactions for item in trans]
    if not all_items_flat:
         print("Error: No items found in original transactions."); return []
    item_counts = Counter(all_items_flat)
    total_items_occurrences = sum(item_counts.values())
    item_frequencies = {item: count / total_items_occurrences for item, count in item_counts.items()}
    items_list = list(item_frequencies.keys())
    item_probabilities = list(item_frequencies.values())

    # Calculate transaction size distribution
    transaction_sizes = [len(t) for t in original_transactions if t] # Ignore empty ones
    if not transaction_sizes:
         print("Error: No valid transaction sizes found."); return []
    size_counts = Counter(transaction_sizes)
    total_transactions = len(transaction_sizes)
    possible_sizes = list(size_counts.keys())
    size_probabilities = [count / total_transactions for count in size_counts.values()]
    print(f"  Avg size: {np.mean(transaction_sizes):.2f}, Item Frequencies & Size Dist calculated.")

    # --- 2. Find Frequent Itemsets ---
    print(f"Finding frequent itemsets (min_support={min_support})...")
    try:
        te = TransactionEncoder()
        te_ary = te.fit(original_transactions).transform(original_transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

        if use_fpgrowth:
            frequent_itemsets_df = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=max_len)
        else:
            frequent_itemsets_df = apriori(df_onehot, min_support=min_support, use_colnames=True, max_len=max_len)

        if frequent_itemsets_df.empty:
            print("Warning: No frequent itemsets found. Consider lowering min_support.")
            # Fallback: Generate purely based on size and item frequency?
            # For now, return empty or potentially raise error.
            return []

        frequent_itemsets_df['length'] = frequent_itemsets_df['itemsets'].apply(len)
        frequent_itemsets_df['itemsets'] = frequent_itemsets_df['itemsets'].apply(set) # Ensure sets
        print(f"Found {len(frequent_itemsets_df)} frequent itemsets.")

    except Exception as e:
        print(f"Error during frequent itemset mining: {e}"); return []

    # --- 3. Generate Transactions Loop ---
    print("Generating new transactions...")
    synthetic_transactions: List[Set[str]] = []
    generation_attempts = 0
    max_total_attempts = num_new_transactions * 5 # Allow some retries if sampling fails

    while len(synthetic_transactions) < num_new_transactions and generation_attempts < max_total_attempts:
        generation_attempts += 1
        try:
            # --- a) Sample Target Size ---
            target_size = np.random.choice(possible_sizes, p=size_probabilities)

            # --- b) Sample Suitable Base Itemset ---
            # Filter itemsets smaller or equal to target size
            suitable_itemsets = frequent_itemsets_df[frequent_itemsets_df['length'] <= target_size]

            if suitable_itemsets.empty:
                 # If no suitable frequent itemsets (e.g., target size is too small)
                 # Option 1: Generate purely randomly based on frequency up to target_size
                 # Option 2: Skip this attempt and try again
                 print(f"  Warning: No frequent itemset found <= target size {target_size}. Skipping attempt {generation_attempts}.")
                 continue # Try again with a different target size

            # Sample from suitable itemsets, weighted by support
            weights = suitable_itemsets['support'].values
            if weights.sum() <= 0 : continue # Should not happen if not empty
            probabilities = weights / weights.sum()
            chosen_index = np.random.choice(suitable_itemsets.index, p=probabilities)
            base_itemset = suitable_itemsets.loc[chosen_index, 'itemsets']

            new_transaction = set(base_itemset) # Start with a copy

            # --- c) Add Noise Items if needed ---
            num_items_to_add = target_size - len(new_transaction)
            items_added_count = 0
            noise_attempts = 0

            while items_added_count < num_items_to_add and noise_attempts < max_noise_add_attempts * num_items_to_add :
                noise_attempts += 1
                # Sample noise item based on overall frequency
                noise_item = np.random.choice(items_list, p=item_probabilities)

                if noise_item not in new_transaction:
                    # Optional: Add more sophisticated checks here later if needed
                    # (e.g., check against negative associations)
                    new_transaction.add(noise_item)
                    items_added_count += 1

            if items_added_count < num_items_to_add:
                 print(f"  Warning: Could only add {items_added_count}/{num_items_to_add} noise items for attempt {generation_attempts} (target size {target_size}).")

            synthetic_transactions.append(new_transaction)

            if len(synthetic_transactions) % 100 == 0:
                 print(f"  Generated {len(synthetic_transactions)}/{num_new_transactions} transactions...")

        except Exception as e:
             print(f"Error during generation for attempt {generation_attempts}: {e}")
             # Decide whether to stop or continue

    if len(synthetic_transactions) < num_new_transactions:
         print(f"Warning: Only generated {len(synthetic_transactions)} out of {num_new_transactions} requested transactions after {max_total_attempts} attempts.")

    print(f"Finished generating {len(synthetic_transactions)} synthetic transactions.")
    return synthetic_transactions