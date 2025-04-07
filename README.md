# PathWeaver
PathWeaver is a Python simulation framework designed for generating and analyzing user movement paths in 2D environments with obstacles. It incorporates real-world data patterns, such as market basket analysis for item placement and transaction generation, making it suitable for applications like simulating customer behavior in retail spaces or generating realistic workloads for Augmented Reality (AR) prefetching studies.

The framework allows you to define an environment from geometric data, analyze transaction patterns to inform item layout, generate synthetic user goals (shopping lists), and simulate how users might navigate the space to achieve those goals while avoiding obstacles.

## Features

* **Environment Loading:** Load 2D environment maps from collider data (CSV format expected).
* **Obstacle Processing:** Merge adjacent or nearby obstacle polygons into unified shapes using `unary_union` or distance-based buffering.
* **Market Basket Analysis:**
    * Generate item categories/clusters from transaction data using graph-based community detection (Louvain algorithm).
    * Generate synthetic user transactions that mimic statistical patterns (co-occurrence, size) of real data using Frequent Itemset Mining (Apriori/FP-Growth).
* **Item Placement:** Place items onto the boundaries of obstacles based on generated categories using a greedy allocation strategy.
* **Pathfinding:**
    * Generate valid random start/end points within the environment.
    * Implement obstacle-avoiding pathfinding between two points using:
        * **Voronoi-based Aisle Graphs:** Approximates centerlines between obstacles for more "aisle-like" paths.
* **Shopping Simulation:** Generate full, multi-segment shopping paths for users based on (synthetic) transactions, ordered by proximity, using Aisle Graph pathfinding.
* **Visualization:** Example scripts provided to visualize maps, obstacles, item placements, and generated user paths using Matplotlib.
* **Modular Structure:** Organized into sub-packages for environment representation, analysis, and path generation.

## Installation

### Prerequisites

* Python (>= 3.8 recommended)
* Standard C/C++ build tools
    * Ubuntu/Debian: `sudo apt-get install build-essential python3-dev`
    * macOS: Xcode Command Line Tools (`xcode-select --install`)
    * Windows: Microsoft C++ Build Tools (available via Visual Studio Installer)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nsreekum/pathweaver.git
    cd pathweaver
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install the package:**
    * **For development (recommended):** Installs the package in editable mode with all optional dependencies for analysis, visualization, and testing.
        ```bash
        pip install -e '.[dev,visualization]'
        ```
        *(Note: If using Zsh, you might need quotes around `'.[dev,visualization]'`)*
    * **Basic installation (core library):**
        ```bash
        pip install -e .
        ```
        *(You may need to install optional dependencies like `matplotlib`, `scipy`, `mlxtend`, `python-louvain` manually if needed later)*

### Key Dependencies

* `shapely`: Geometric operations
* `pandas`: Data loading (CSV)
* `numpy`: Numerical operations
* `networkx`: Graph creation and analysis (Visibility/Aisle graphs, Community Detection)
* `scipy`: Spatial algorithms (Voronoi diagrams)
* `mlxtend`: Frequent itemset mining (Apriori/FP-Growth)
* `python-louvain`: Louvain community detection
* `matplotlib`: Plotting for examples/visualization

*(See `pyproject.toml` for specific version requirements)*

## Quick Start
* Check out the examples folder