# 6_result_grid.py
#
# This script extends the functionality of 'study_5.py'.
# It programmatically creates a COMSOL model from scratch, defines its
# geometry, materials, physics, and mesh, and runs an eigenfrequency study.
#
# MODIFIED FEATURE: Geometry is a 28x28 grid of two materials (Soil/Concrete)
# assigned randomly. All input parameters and output results (eigenfrequencies)
# are systematically saved to a structured SQLite3 database for robust
# data management and reproducibility.

import mph
from pathlib import Path
import logging
import sqlite3
import datetime
import numpy as np
import math

# ==============================================================================
# PART 1: SCRIPT CONFIGURATION AND BOILERPLATE
# ==============================================================================

# Configure logging for informative output.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define file paths using pathlib for OS-agnostic handling.
OUTPUT_MPH_FILE = Path("grid_model_28x28.mph")
DATABASE_FILE = Path("simulation_results.db")

# Define material properties from the provided table
SOIL_PROPS = {
    "name": "Soil",
    "youngs_modulus": 20e6,  # 20 MPa
    "poissons_ratio": 0.3,
    "density": 1800.0  # kg/m^3
}
CONCRETE_PROPS = {
    "name": "Concrete",
    "youngs_modulus": 20e9,  # 20 GPa
    "poissons_ratio": 0.2,
    "density": 2400.0  # kg/m^3
}


# ==============================================================================
# PART 2: DATABASE SETUP AND MANAGEMENT
# ==============================================================================

def setup_database(db_file):
    """
    Connects to the SQLite database and creates the necessary tables if they
    do not already exist.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Table to store all input parameters for a specific simulation run.
    # Modified to remove r_circle and add random_seed
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simulations (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp TEXT NOT NULL,
        model_filename TEXT,
        geometry_type TEXT,
        param_a REAL,
        random_seed INTEGER,
        param_kx_str TEXT,
        param_ky_str TEXT,
        mesh_size_setting INTEGER,
        num_eigenvalues INTEGER,
        eigenvalue_shift REAL
    )
    ''')

    # Table to store the output results (eigenfrequencies) for each run.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS eigenfrequencies (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        mode_number INTEGER NOT NULL,
        frequency_hz REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES simulations (run_id)
    )
    ''')

    conn.commit()
    conn.close()
    logging.info(f"Database '{db_file}' is set up and ready.")


def log_simulation_run(db_file, run_data):
    """
    Logs the input parameters of a new simulation run to the database.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO simulations (
        run_timestamp, model_filename, geometry_type, param_a, random_seed,
        param_kx_str, param_ky_str, mesh_size_setting, num_eigenvalues, eigenvalue_shift
    ) VALUES (
        :run_timestamp, :model_filename, :geometry_type, :param_a, :random_seed,
        :param_kx_str, :param_ky_str, :mesh_size_setting, :num_eigenvalues, :eigenvalue_shift
    )
    ''', run_data)
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Logged new simulation with run_id: {run_id}")
    return run_id


def log_eigenfrequencies(db_file, run_id, frequencies):
    """
    Logs the calculated eigenfrequencies for a given simulation run.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    for i, freq_hz in enumerate(frequencies):
        cursor.execute('''
        INSERT INTO eigenfrequencies (run_id, mode_number, frequency_hz)
        VALUES (?, ?, ?)
        ''', (run_id, i + 1, freq_hz))
    conn.commit()
    conn.close()
    logging.info(f"Saved {len(frequencies)} eigenfrequencies for run_id {run_id}.")


# ==============================================================================
# PART 3: COMSOL MODEL CREATION AND SIMULATION
# ==============================================================================

def run_simulation_and_save_results(mph_filename, db_filename, grid_size=28, random_seed=None):
    """
    Creates a COMSOL model with a grid geometry, runs an eigenfrequency study,
    and saves all inputs and results to a SQLite database.
    """
    # Initialize the database.
    setup_database(db_filename)

    # Use provided seed or generate a new one
    if random_seed is None:
        random_seed = np.random.randint(0, 2 ** 31 - 1)
    np.random.seed(random_seed)
    logging.info(f"Using random seed: {random_seed}")

    # --- Gather all input parameters for this run ---
    inputs = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "model_filename": str(mph_filename),
        "geometry_type": f"Grid{grid_size}x{grid_size}_SoilConcrete",
        "param_a": 1.0,
        "random_seed": int(random_seed),
        "param_kx_str": "pi/a",
        "param_ky_str": "0",
        "mesh_size_setting": 5,  # 'normal'
        "num_eigenvalues": 10,
        "eigenvalue_shift": 0.0,
    }

    if mph_filename.exists():
        logging.warning(f"Model file '{mph_filename}' already exists. Deleting it to start fresh.")
        mph_filename.unlink()

    logging.info(f"Starting process to create and solve model: '{mph_filename}'")

    client = None
    try:
        # --- Connect to COMSOL and create model ---
        client = mph.start()
        model_mph = client.create('GridModel')
        logging.info(f"Created new model named: '{model_mph.name()}'")

        # --- Define Global Parameters from inputs dict ---
        logging.info("Defining global parameters...")
        model_mph.parameter('a', f"{inputs['param_a']}[m]")
        model_mph.parameter('kx', inputs['param_kx_str'])
        model_mph.parameter('ky', inputs['param_ky_str'])

        # --- Create Component and Geometry ---
        logging.info(f"Creating a 2D component and defining {grid_size}x{grid_size} grid geometry...")
        comp1_java = model_mph.java.component().create("comp1", True)
        geom1_java = comp1_java.geom().create("geom1", 2)

        # --- PATCHED GEOMETRY LOGIC ---
        # The redundant outer square is removed. We only create the grid.
        # The union of the grid cells will form the final square geometry.
        cell_size = inputs["param_a"] / grid_size
        start_pos = -inputs["param_a"] / 2

        for i in range(grid_size):
            for j in range(grid_size):
                sq_tag = f"sq_{i}_{j}"
                x_pos = start_pos + j * cell_size
                y_pos = start_pos + i * cell_size
                cell = geom1_java.create(sq_tag, "Square")
                cell.set("size", cell_size)
                cell.set("pos", [x_pos, y_pos])

        geom1_java.run()
        logging.info("Geometry created: a 28x28 grid.")
        # --- END PATCH ---

        # --- Create and Configure Materials ---
        logging.info("Creating Soil and Concrete materials...")

        mat_soil = comp1_java.material().create("mat_soil", "Common")
        mat_soil.label(SOIL_PROPS["name"])
        prop_group_soil = mat_soil.propertyGroup("def")
        prop_group_soil.set("youngsmodulus", f"{SOIL_PROPS['youngs_modulus']}[Pa]")
        prop_group_soil.set("poissonsratio", str(SOIL_PROPS['poissons_ratio']))
        prop_group_soil.set("density", f"{SOIL_PROPS['density']}[kg/m^3]")

        mat_concrete = comp1_java.material().create("mat_concrete", "Common")
        mat_concrete.label(CONCRETE_PROPS["name"])
        prop_group_concrete = mat_concrete.propertyGroup("def")
        prop_group_concrete.set("youngsmodulus", f"{CONCRETE_PROPS['youngs_modulus']}[Pa]")
        prop_group_concrete.set("poissonsratio", str(CONCRETE_PROPS['poissons_ratio']))
        prop_group_concrete.set("density", f"{CONCRETE_PROPS['density']}[kg/m^3]")

        # --- Assign Materials Randomly ---
        logging.info("Assigning materials to domains randomly...")

        # --- PATCHED DOMAIN INDEXING ---
        # With the outer square removed, domains are now indexed from 1 to grid_size*grid_size.
        domain_indices = np.arange(1, grid_size * grid_size + 1)
        # --- END PATCH ---

        material_map = np.random.randint(0, 2, size=(grid_size * grid_size))

        soil_domains = tuple(domain_indices[material_map == 0])
        concrete_domains = tuple(domain_indices[material_map == 1])

        if soil_domains:
            mat_soil.selection().set(soil_domains)
        if concrete_domains:
            mat_concrete.selection().set(concrete_domains)

        logging.info(f"Assigned Soil to {len(soil_domains)} domains and Concrete to {len(concrete_domains)} domains.")

        # --- Add Solid Mechanics & Floquet PBCs ---
        logging.info("Adding Solid Mechanics and Floquet boundary conditions...")
        solid = comp1_java.physics().create("solid", "SolidMechanics", "geom1")

        # --- PATCHED BOUNDARY CONDITION LOGIC ---
        # Manually specify source and destination to ensure robust pairing on the
        # complex, segmented boundary of the grid geometry.

        # --- X-Periodicity ---
        pbc_x = solid.create("pbc_x", "PeriodicCondition", 1)
        pbc_x.selection().set(
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53,
             55] + list(range(1597, 1625)))
        pbc_x.set("PeriodicType", "Floquet")
        pbc_x.set("kFloquet", ["kx", "0", "0"])

        # --- Y-Periodicity ---
        pbc_y = solid.create("pbc_y", "PeriodicCondition", 1)
        pbc_y.selection().set(
            [2, 57, 59, 114, 116, 171, 173, 228, 230, 285, 287, 342, 344, 399, 401, 456, 458, 513, 515, 570, 572, 627,
             629,
             684, 686, 741, 743, 798, 800, 855, 857, 912, 914, 969, 971, 1026, 1028, 1083, 1085, 1140, 1142, 1197, 1199,
             1254, 1256, 1311, 1313, 1368, 1370, 1425, 1427, 1482, 1484, 1539, 1541, 1596]
        )
        pbc_y.set("PeriodicType", "Floquet")
        pbc_y.set("kFloquet", ["0", "ky", "0"])
        # --- END PATCH ---

        # --- Add Mesh ---
        logging.info("Creating a free triangular mesh...")
        mesh = comp1_java.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(inputs['mesh_size_setting'])
        ftri = mesh.feature().create("ftri1", "FreeTri")
        ftri.selection().geom("geom1", 2)
        ftri.selection().all()
        mesh.run()
        logging.info("Mesh generated.")

        # --- Create Eigenfrequency Study ---
        logging.info("Creating and configuring eigenfrequency study...")
        study = model_mph.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(inputs['num_eigenvalues']))
        eig.set("shiftactive", "on")
        eig.set("shift", str(inputs['eigenvalue_shift']))
        eig.activate("solid", True)

        # --- Log inputs to database BEFORE solving ---
        current_run_id = log_simulation_run(db_filename, inputs)

        # --- Solve the study ---
        logging.info(f"Solving eigenfrequencies for run_id {current_run_id}...")
        study.run()
        logging.info("Eigenfrequency study finished.")

        # --- Extract results and save to database ---
        eigenvalues_rad_s = model_mph.evaluate("freq")
        frequencies_hz = [np.real(val) for val in eigenvalues_rad_s]

        log_eigenfrequencies(db_filename, current_run_id, frequencies_hz)

        # --- Save the COMSOL model file ---
        logging.info(f"Saving the final model to '{mph_filename}'...")
        model_mph.save(mph_filename)
        logging.info("Model saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during model creation or solving: {e}", exc_info=True)
    finally:
        if client:
            client.clear()
            logging.info("Disconnected from COMSOL client.")


# ==============================================================================
# PART 4: SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    run_simulation_and_save_results(OUTPUT_MPH_FILE, DATABASE_FILE, grid_size=28)
    logging.info("Script finished.")
