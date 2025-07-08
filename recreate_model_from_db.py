import logging
import sqlite3
import datetime
from pathlib import Path
import math
import numpy as np
import mph
from typing import Tuple, Dict, Any, List

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: These constants must match the original grid_model.py script
# as they are not stored in the database.
# ──────────────────────────────────────────────────────────────────────────────
A: float = 1.0
GRID_SIZE: int = 32
N_MODES: int = 10

# Define the location of the database file
RESULTS_DIR: Path = Path("results")
DATABASE_FILE: Path = RESULTS_DIR / "simulation_results.db"

# Material properties must be identical to the original script
SOIL_PROPS = {
    "name": "Soil", "youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0,
}
CONCRETE_PROPS = {
    "name": "Concrete", "youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0,
}

# Standardized logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (Copied from grid_model.py for consistency)
# ──────────────────────────────────────────────────────────────────────────────

def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    """Generates the identical material distribution using the stored random seed."""
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))


def _outer_bnd_lists(size: int) -> Tuple[List[int], List[int]]:
    """Provides the boundary indices for Floquet periodicity. Calibrated for 28x28 grid."""
    if size == 28:
        pbc_x_nodes = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
            *range(1597, 1625),
        ]
        pbc_y_nodes = [
            2, 57, 59, 114, 116, 171, 173, 228, 230, 285, 287, 342, 344, 399, 401, 456, 458, 513, 515, 570, 572, 627,
            629,
            684, 686, 741, 743, 798, 800, 855, 857, 912, 914, 969, 971, 1026, 1028, 1083, 1085, 1140, 1142, 1197, 1199,
            1254, 1256, 1311, 1313, 1368, 1370, 1425, 1427, 1482, 1484, 1539, 1541, 1596,
        ]
        return pbc_x_nodes, pbc_y_nodes
    elif size == 32:
        pbc_x_nodes_32: List[int] = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
            57, 59, 61, 63, *range(2081, 2113)
        ]  # ← left+right edges
        pbc_y_nodes_32: List[int] = [
            2, 65, 67, 130, 132, 195, 197, 260, 262, 325, 327, 390, 392, 455, 457, 520, 522, 585, 587, 650, 652, 715,
            717, 780, 782, 845, 847, 910, 912, 975, 977, 1040, 1042, 1105, 1107, 1170, 1172, 1235, 1237, 1300, 1302,
            1365, 1367, 1430, 1432, 1495, 1497, 1560, 1562, 1625, 1627, 1690, 1692, 1755, 1757, 1820, 1822, 1885, 1887,
            1950, 1952, 2015, 2017, 2080
        ]  # ← bottom+top edges
        return pbc_x_nodes_32, pbc_y_nodes_32
    else:
        raise ValueError("Boundary lists only calibrated for GRID_SIZE = 28 or 32.")


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE INTERACTION
# ──────────────────────────────────────────────────────────────────────────────

def fetch_simulation_data(db_file: Path, result_id: int) -> Dict[str, Any]:
    """
    Fetches the necessary parameters for a single simulation point from the database.
    """
    if not db_file.exists():
        logging.error(f"Database file not found at '{db_file}'. Please run grid_model.py first.")
        raise FileNotFoundError

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
    SELECT
        s.random_seed,
        e.kx,
        e.ky
    FROM eigenfrequencies AS e
    JOIN simulations AS s ON e.run_id = s.run_id
    WHERE e.result_id = ?
    """

    result = cur.execute(query, (result_id,)).fetchone()
    conn.close()

    if result is None:
        logging.error(f"No data found for result_id = {result_id}. Please choose a valid ID.")
        raise ValueError(f"Result ID {result_id} not found in the database.")

    logging.info(
        f"Found data for result_id={result_id}: Seed={result['random_seed']}, kx={result['kx']:.4f}, ky={result['ky']:.4f}")
    return dict(result)


# ──────────────────────────────────────────────────────────────────────────────
# COMSOL MODEL RECREATION
# ──────────────────────────────────────────────────────────────────────────────

def recreate_model(params: Dict[str, Any], result_id: int) -> None:
    """
    Builds a COMSOL model from scratch using parameters read from the database.
    """
    output_mph_file = RESULTS_DIR / f"recreated_model_result_{result_id}.mph"

    client = None
    try:
        # Start client and create a new model
        client = mph.start()
        model = client.create(f"RecreatedModel_Result_{result_id}")
        logging.info(f"Client started. Building model for result_id={result_id}...")

        # === Set Global Parameters using data fetched from the database ===
        model.parameter("a", f"{A}[m]")
        model.parameter("kx", str(params['kx']))
        model.parameter("ky", str(params['ky']))
        logging.info("Global parameters kx and ky set from database values.")

        # === Geometry: Identical build process ===
        comp = model.java.component().create("comp1", True)
        geom = comp.geom().create("geom1", 2)
        cell = A / GRID_SIZE
        start = -A / 2
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                sq = geom.create(f"sq_{i}_{j}", "Square")
                sq.set("size", cell)
                sq.set("pos", [start + j * cell, start + i * cell])
        geom.run()
        logging.info(f"Built {GRID_SIZE}x{GRID_SIZE} grid geometry.")

        # === Materials: Use the random_seed from DB to recreate the exact map ===
        mat_soil = comp.material().create("mat_soil", "Common")
        mat_soil.label(SOIL_PROPS["name"])
        pg_soil = mat_soil.propertyGroup("def")
        pg_soil.set("youngsmodulus", f"{SOIL_PROPS['youngs_modulus']}[Pa]")
        pg_soil.set("poissonsratio", str(SOIL_PROPS["poissons_ratio"]))
        pg_soil.set("density", f"{SOIL_PROPS['density']}[kg/m^3]")

        mat_con = comp.material().create("mat_con", "Common")
        mat_con.label(CONCRETE_PROPS["name"])
        pg_con = mat_con.propertyGroup("def")
        pg_con.set("youngsmodulus", f"{CONCRETE_PROPS['youngs_modulus']}[Pa]")
        pg_con.set("poissonsratio", str(CONCRETE_PROPS["poissons_ratio"]))
        pg_con.set("density", f"{CONCRETE_PROPS['density']}[kg/m^3]")

        material_map = create_symmetric_material_map(GRID_SIZE, params['random_seed']).flatten()
        domain_ids = np.arange(1, GRID_SIZE * GRID_SIZE + 1)

        mat_soil.selection().set(tuple(domain_ids[material_map == 0].tolist()))
        mat_con.selection().set(tuple(domain_ids[material_map == 1].tolist()))
        logging.info(f"Recreated material distribution using seed {params['random_seed']}.")

        # === Physics, Mesh, and Study: Identical build process ===
        solid = comp.physics().create("solid", "SolidMechanics", "geom1")
        pbc_x_nodes, pbc_y_nodes = _outer_bnd_lists(GRID_SIZE)
        pbc_x = solid.create("pbc_x", "PeriodicCondition", 1)
        pbc_x.selection().set(pbc_x_nodes)
        pbc_x.set("PeriodicType", "Floquet")
        pbc_x.set("kFloquet", ["kx", "0", "0"])

        pbc_y = solid.create("pbc_y", "PeriodicCondition", 1)
        pbc_y.selection().set(pbc_y_nodes)
        pbc_y.set("PeriodicType", "Floquet")
        pbc_y.set("kFloquet", ["0", "ky", "0"])
        logging.info("Solid mechanics physics and Floquet conditions added.")

        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(5)
        mesh.feature().create("ftri1", "FreeTri").selection().geom("geom1", 2).all()
        mesh.run()
        logging.info("Mesh generated.")

        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(N_MODES))
        eig.set("shiftactive", "on")
        eig.set("shift", "0.0")
        eig.activate("solid", True)
        logging.info("Eigenfrequency study configured. The model is ready for inspection.")

        # === Save the final model file ===
        model.save(output_mph_file)
        logging.info(f"SUCCESS: Model recreated and saved to '{output_mph_file}'")

    except Exception as exc:
        logging.exception("Failed to recreate COMSOL model: %s", exc)
    finally:
        if client:
            client.clear()
            logging.info("Disconnected from COMSOL client.")


# ──────────────────────────────────────────────────────────────────────────────
# SCRIPT EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main function to drive the script."""
    try:
        result_id_str = input("Enter the 'result_id' from the database to recreate the model: ")
        result_id = int(result_id_str)

        model_params = fetch_simulation_data(DATABASE_FILE, result_id)
        recreate_model(model_params, result_id)

    except (ValueError, TypeError):
        logging.error("Invalid input. Please enter a valid integer for the result_id.")
    except FileNotFoundError:
        logging.error("Operation aborted because the database was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
    logging.info("Script finished.")
