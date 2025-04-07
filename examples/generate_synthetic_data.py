# examples/generate_synthetic_data.py

import os
import sys
import pandas as pd # Needed if loading transactions via pandas initially
from collections import Counter

# --- Make the src package importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# -----------------------------------------

try:
    from pathweaver.analysis.market_basket import (
        generate_item_categories_from_transactions, # To analyze original/synthetic
        generate_synthetic_transactions           # The new function
    )
except ImportError as e:
    print(f"Error importing pathweaver: {e}")
    # Make sure all dependencies including mlxtend are installed
    sys.exit(1)

# --- Configuration ---
ORIGINAL_TRANSACTION_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')
NUM_SYNTHETIC_TO_GENERATE = 100 # How many new transactions to create
MIN_SUPPORT_THRESHOLD = 0.05 # Adjust based on your data (start higher, then lower if needed)

# --- Helper to load original transactions ---
def load_original_transactions(file_path):
    transactions = []
    try:
        print(f"Reading original transactions from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    items = {item.strip() for item in stripped_line.split(',') if item.strip()}
                    if items:
                        transactions.append(items)
        print(f"Loaded {len(transactions)} original transactions.")
        return transactions
    except FileNotFoundError:
        print(f"Error: Original transaction file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading original transaction file {file_path}: {e}")
        return None

# --- Main execution ---
if __name__ == "__main__":
    print("Running example: Generate Synthetic Transactions")

    # 1. Load Original Transactions
    original_transactions = load_original_transactions(ORIGINAL_TRANSACTION_FILE_PATH)
    if not original_transactions:
        sys.exit(1)

    # 2. Generate Synthetic Transactions
    synthetic_transactions = generate_synthetic_transactions(
        original_transactions=original_transactions,
        num_new_transactions=NUM_SYNTHETIC_TO_GENERATE,
        min_support=MIN_SUPPORT_THRESHOLD,
        use_fpgrowth=True # Generally faster
    )

    if not synthetic_transactions:
        print("Failed to generate synthetic transactions.")
        sys.exit(1)

    # 3. Basic Analysis/Comparison (Optional)
    print(f"\n--- Analysis ---")
    print(f"Generated {len(synthetic_transactions)} transactions.")

    # Compare average basket size
    avg_orig_size = sum(len(t) for t in original_transactions) / len(original_transactions)
    avg_synth_size = sum(len(t) for t in synthetic_transactions) / len(synthetic_transactions)
    print(f"Avg. Basket Size: Original={avg_orig_size:.2f}, Synthetic={avg_synth_size:.2f}")

    # Compare top N item frequencies
    orig_item_counts = Counter(item for trans in original_transactions for item in trans)
    synth_item_counts = Counter(item for trans in synthetic_transactions for item in trans)
    print("\nTop 5 Item Frequencies (Original):")
    for item, count in orig_item_counts.most_common(5):
        print(f"  {item}: {count}")
    print("\nTop 5 Item Frequencies (Synthetic):")
    for item, count in synth_item_counts.most_common(5):
        print(f"  {item}: {count}")

    # Optional: Generate categories from *both* original and synthetic, compare them
    print("\nGenerating categories from original data:")
    original_categories = generate_item_categories_from_transactions(original_transactions)
    print("\nGenerating categories from synthetic data:")
    synthetic_categories = generate_item_categories_from_transactions(synthetic_transactions)

    # (Further comparison logic could go here)

    # Example: Print first few synthetic transactions
    print("\nFirst 10 Synthetic Transactions:")
    for i, trans in enumerate(synthetic_transactions[:10]):
        print(f"  {i+1}: {sorted(list(trans))}")

    # You could now save synthetic_transactions to a file if needed
    # with open("synthetic_transactions.csv", "w") as f:
    #    for trans in synthetic_transactions:
    #        f.write(",".join(sorted(list(trans))) + "\n")