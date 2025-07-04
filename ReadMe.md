
# COMSOL 2D Phononic Crystal Analyzer

This script automates the process of analyzing a 2D phononic crystal using COMSOL Multiphysics. It programmatically builds a model of a unit cell, applies Floquet periodic boundary conditions, and calculates the eigenfrequency band structure by sweeping through various wave vectors (**k-points**) in the first Brillouin zone.

The primary goal is to efficiently calculate the dispersion relation for a given material configuration without requiring manual operation of the COMSOL graphical user interface. All parameters and results are systematically stored in an SQLite database for robust data management and post-processing.

## Features

  - **Automated Model Generation**: Creates a complete 2D COMSOL model from scratch, including geometry, materials, physics, and meshing.
  - **Random Material Distribution**: Generates a 28x28 grid and assigns two different materials (Soil and Concrete) based on a randomly generated but symmetric pattern.
  - **Efficient "Manual Sweep"**: Builds the complex model and mesh only once, then programmatically loops through k-points, updating parameters and re-solving for maximum efficiency.
  - **Data Persistence**: Logs all simulation parameters and the resulting eigenfrequencies for every k-point into a structured SQLite database.
  - **Reproducibility**: Saves the final COMSOL model file (`.mph`) and a database stamped with a unique random seed for each complete run.

-----

## How It Works

The script executes a series of automated steps to perform the analysis.

### 1\. Initialization and Configuration

  - **Constants**: Defines global constants for the simulation, such as the lattice constant (`A`), grid size (`GRID_SIZE`), and the number of k-points to sample (`N_K`).
  - **Database Setup**: Prepares an SQLite database file (`results/simulation_results.db`) with two tables:
      - `simulations`: Stores a record for each complete sweep, identified by a unique `random_seed`.
      - `eigenfrequencies`: Stores the calculated eigenfrequencies for every mode at each specific (kx, ky) point.

### 2\. Model Construction (Single Execution)

The core of the script's efficiency comes from building the model only one time per run.

  - **COMSOL Connection**: It starts a single COMSOL client session using the `mph` library.
  - **Geometry**: A 2D geometry is created, consisting of a 28x28 grid of individual square domains. This grid represents the pixels of the material unit cell.
  - **Materials**: A random, symmetric material map is generated. The script then assigns "Soil" and "Concrete" properties to the corresponding domains (pixels) based on this map.
  - **Physics**: A "Solid Mechanics" physics interface is added. Floquet periodic boundary conditions are applied to the outer edges of the grid using pre-calculated boundary index lists to simulate an infinite periodic lattice.
  - **Meshing**: A physics-controlled, free triangular mesh is generated over the entire geometry.

### 3\. Manual k-Point Sweep (Python Loop)

After the model is built, the script begins the "manual sweep" to calculate the band structure.

  - **Generate k-Grid**: A triangular grid of (kx, ky) points is generated within the first Brillouin zone.
  - **Iterate and Solve**: The script loops through each (kx, ky) pair. In each iteration, it:
    1.  Updates the `kx` and `ky` global parameters within the existing COMSOL model.
    2.  Runs the pre-defined "Eigenfrequency" study to solve for the first 10 modes.
    3.  Evaluates and retrieves the list of resulting eigenfrequencies.
    4.  Logs the frequencies, along with the corresponding `kx` and `ky`, to the `eigenfrequencies` table in the database.

### 4\. Finalization

  - After the loop completes, the final state of the COMSOL model (containing the solution for the last k-point) is saved to a `.mph` file in the `results/` directory.
  - The connection to the COMSOL client is closed.

-----

## Prerequisites

To run this script, you will need:

  - A working installation of **COMSOL Multiphysics®** (with a valid license).
  - **COMSOL LiveLink™ for Python**.
  - **Python 3.x**.
  - The **`mph`** and **`numpy`** Python libraries. You can install them using pip:
    ```bash
    pip install mph numpy
    ```

-----

## Usage

1.  Place the script in your project directory.
2.  Ensure COMSOL is correctly installed and accessible from your system's command line environment.
3.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```

The script will create a `results/` directory, where it will save the database and the final `.mph` model file.

## Customization

You can easily modify the script's behavior by changing the global constants at the top of the file:

  - `GRID_SIZE`: To change the resolution of the unit cell. **Note**: If you change this from `28`, you must regenerate the hardcoded boundary lists in the `_outer_bnd_lists` function, as they are specific to a 28x28 grid.
  - `N_K`: To increase or decrease the density of the k-point sweep.
  - `N_MODES`: To change the number of eigenfrequencies calculated.
  - `SOIL_PROPS`, `CONCRETE_PROPS`: To change the physical properties of the materials.