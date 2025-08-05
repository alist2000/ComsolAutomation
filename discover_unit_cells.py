from __future__ import annotations
import logging
import datetime
import math
import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import mph
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Global configuration ────────────────────────────────────────────
A: float = 1.0
GRID_SIZE: int = 32
N_K_PER_SEGMENT: int = 9  # From your base code, for the Γ-X-M-Γ path
K_MAX: float = math.pi / A
N_EIG: int = 10
SEED_TO_RUN: int = 24000  # Running for one specific seed as requested

# --- Output paths for this new workflow ---
OUTPUT_DIR = Path(f"results_discovery_seed_{SEED_TO_RUN}")
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUT_DIR / f"simulation_results_seed_{SEED_TO_RUN}.db"

# --- Material properties (from your base code) ---
MATERIALS = {
    "soil": {"youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0},
    "concrete": {"youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(levelname)s · %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logfile.log"),
        logging.StreamHandler()
    ]
)


# ── DB Helpers (copied from your base code) ─────────────────────────
def _setup_db(db_file: Path) -> None:
    """Initializes the database with the schema."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS simulations (
                     run_id         INTEGER PRIMARY KEY,
                     run_timestamp  TEXT,
                     model_file     TEXT,
                     atlas_type     TEXT,
                     seed           INTEGER,
                     atlas_path     TEXT,
                     materials_json TEXT)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS eigenfrequencies (
                     result_id          INTEGER PRIMARY KEY,
                     run_id             INTEGER,
                     kx                 REAL,
                     ky                 REAL,
                     mode_number        INTEGER,
                     frequency_hz       REAL)""")
    conn.commit()
    conn.close()


def _log_run_and_get_id(db_file: Path, seed: int, mph_name: str, atlas_type: str, atlas_path: str) -> int:
    """Logs the initial run info and returns the new run_id."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    materials_json = json.dumps(MATERIALS)
    cur.execute("""INSERT INTO simulations 
                   (run_timestamp, model_file, atlas_type, seed, atlas_path, materials_json) 
                   VALUES (?,?,?,?,?,?)""",
                (datetime.datetime.now().isoformat(timespec="seconds"),
                 mph_name, atlas_type, seed, atlas_path, materials_json))
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


# ── Util Functions (copied and adapted from your base code) ────────
def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    """Creates the original, center-symmetric material map."""
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))


def generate_k_path(n_k_segment: int, k_max: float) -> List[Tuple[float, float]]:
    """Generates k-points along the high-symmetry path Γ-X-M-Γ."""
    gamma_x = np.linspace(0, k_max, n_k_segment)
    path_gx = list(zip(gamma_x, np.zeros_like(gamma_x)))
    x_m = np.linspace(0, k_max, n_k_segment)[1:]
    path_xm = list(zip(np.full_like(x_m, k_max), x_m))
    m_gamma = np.linspace(k_max, 0, n_k_segment)[1:]
    path_mg = list(zip(m_gamma, m_gamma))
    k_points = path_gx + path_xm + path_mg
    logging.info(f"Generated {len(k_points)} k-points for the Γ-X-M-Γ path.")
    return [(float(kx), float(ky)) for kx, ky in k_points]


def _outer_bnd_lists(size: int) -> Tuple[List[int], List[int]]:
    if size == 32:
        pbc_x_nodes_32 = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
            57, 59, 61, 63, *range(2081, 2113)
        ]
        pbc_y_nodes_32 = [
            2, 65, 67, 130, 132, 195, 197, 260, 262, 325, 327, 390, 392, 455, 457, 520, 522, 585, 587, 650, 652, 715,
            717, 780, 782, 845, 847, 910, 912, 975, 977, 1040, 1042, 1105, 1107, 1170, 1172, 1235, 1237, 1300, 1302,
            1365, 1367, 1430, 1432, 1495, 1497, 1560, 1562, 1625, 1627, 1690, 1692, 1755, 1757, 1820, 1822, 1885, 1887,
            1950, 1952, 2015, 2017, 2080
        ]
        return pbc_x_nodes_32, pbc_y_nodes_32
    raise ValueError(f"Boundary lists not calibrated for GRID_SIZE = {size}.")


# ── New Discovery and Visualization Function ───────────────────────
def generate_and_visualize_atlases(seed: int) -> Dict[str, np.ndarray]:
    """
    Generates, visualizes, and returns different unit cell atlases.

    This function performs the core discovery task:
    1. Creates the base symmetric atlas.
    2. Tiles it to a 2x2 super-cell.
    3. Extracts a new "corner-centered" atlas from the tiled version.
    4. Creates and saves a visualization of this process.
    """
    logging.info(f"Generating and visualizing atlases for seed {seed}")

    # 1. Create the original symmetric atlas
    original_atlas = create_symmetric_material_map(GRID_SIZE, seed)

    # 2. Create a 2x2 tiled version to find other periodic shapes
    tiled_map = np.tile(original_atlas, (2, 2))

    # 3. Extract a new "corner-centered" unit cell
    # This slices the tiled map to create a new 32x32 grid
    # whose center is the corner of four original cells.
    s = GRID_SIZE // 2
    corner_centered_atlas = tiled_map[s: s + GRID_SIZE, s: s + GRID_SIZE]

    # 4. Create and save the visualization (MANDATORY DELIVERABLE)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(tiled_map, cmap='Greys', interpolation='nearest')

    # Highlight the original unit cell (top-left)
    rect_orig = patches.Rectangle((-0.5, -0.5), GRID_SIZE, GRID_SIZE,
                                  linewidth=2, edgecolor='cyan', facecolor='none',
                                  label='Original Unit Cell')
    ax.add_patch(rect_orig)

    # Highlight the new corner-centered unit cell
    rect_corner = patches.Rectangle((s - 0.5, s - 0.5), GRID_SIZE, GRID_SIZE,
                                    linewidth=2, edgecolor='magenta', facecolor='none',
                                    label='Corner-Centered Unit Cell')
    ax.add_patch(rect_corner)

    ax.set_title(f'Unit Cell Discovery for Seed {seed}')
    ax.legend()
    image_path = OUTPUT_DIR / f"unit_cell_discovery_seed_{seed}.png"
    plt.savefig(image_path)
    logging.info(f"Saved unit cell visualization to {image_path}")
    plt.close(fig)

    return {
        "original": original_atlas,
        "corner_centered": corner_centered_atlas
    }


# ── Refactored Worker Function ─────────────────────────────────────
def run_simulation_for_atlas(
        atlas_type: str,
        material_map: np.ndarray,
        seed: int
):
    """
    Builds, solves, and saves a COMSOL model for a given material atlas.
    """
    client = None
    model_name = f"seed_{seed}_type_{atlas_type}"
    mph_filename = f"{model_name}.mph"

    try:
        logging.info(f"[{model_name}] Starting COMSOL client.")
        client = mph.start(cores=1)
        model = client.create(model_name)
        logging.info(f"[{model_name}] Building model.")

        # --- Model setup (from your base code) ---
        model.parameter("a", f"{A}[m]")
        model.parameter("kx", "0")
        model.parameter("ky", "0")
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
        soil, concrete = MATERIALS["soil"], MATERIALS["concrete"]
        mat_soil = comp.material().create("mat_soil", "Common")
        pg = mat_soil.propertyGroup("def")
        pg.set("youngsmodulus", f"{soil['youngs_modulus']}[Pa]")
        pg.set("poissonsratio", str(soil['poissons_ratio']))
        pg.set("density", f"{soil['density']}[kg/m^3]")
        mat_con = comp.material().create("mat_con", "Common")
        pgc = mat_con.propertyGroup("def")
        pgc.set("youngsmodulus", f"{concrete['youngs_modulus']}[Pa]")
        pgc.set("poissonsratio", str(concrete['poissons_ratio']))
        pgc.set("density", f"{concrete['density']}[kg/m^3]")

        mmap_flat = material_map.flatten()
        dom_ids = np.arange(1, GRID_SIZE * GRID_SIZE + 1)
        mat_soil.selection().set(tuple(dom_ids[mmap_flat == 0]))
        mat_con.selection().set(tuple(dom_ids[mmap_flat == 1]))

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
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(5)
        mesh.feature().create("ftri1", "FreeTri").selection().geom("geom1", 2).all()
        mesh.run()
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(N_EIG))
        eig.set("shiftactive", "on")
        eig.set("shift", "1")
        eig.activate("solid", True)

        # --- Database and Atlas Logging ---
        atlas_path_str = f"atlas_{model_name}.npy"
        np.save(OUTPUT_DIR / atlas_path_str, material_map)
        run_id = _log_run_and_get_id(DB_PATH, seed, mph_filename, atlas_type, atlas_path_str)

        # --- Solving and Batch DB Insert ---
        results_for_db_batch = []
        k_points = generate_k_path(N_K_PER_SEGMENT, K_MAX)

        for kx, ky in k_points:
            logging.info(f"[{model_name}] Solving for k=({kx:.3f}, {ky:.3f})")
            model.parameter("kx", str(kx))
            model.parameter("ky", str(ky))
            study.run()
            freqs = [float(np.real(v)) for v in model.evaluate("freq")]
            for i, f in enumerate(freqs):
                results_for_db_batch.append((run_id, kx, ky, i + 1, f))

        logging.info(f"[{model_name}] Writing all {len(results_for_db_batch)} results to database.")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.executemany("""INSERT INTO eigenfrequencies
                           (run_id, kx, ky, mode_number, frequency_hz)
                           VALUES (?,?,?,?,?)""", results_for_db_batch)
        conn.commit()
        conn.close()

        # --- Save the final model to .mph file (MANDATORY DELIVERABLE) ---
        save_path = OUTPUT_DIR / mph_filename
        logging.info(f"[{model_name}] Saving model to {save_path}")
        model.save(save_path)
        logging.info(f"[{model_name}] Successfully saved model.")

    except Exception as e:
        logging.exception(f"[{model_name}] Worker failed with error: {e}")
    finally:
        if client:
            logging.info(f"[{model_name}] Clearing client.")
            client.clear()


# ── Main Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("--- Starting Unit Cell Discovery Workflow ---")

    # Setup database for this seed's runs
    _setup_db(DB_PATH)

    # 1. Generate the atlases and the visualization image
    atlases = generate_and_visualize_atlases(SEED_TO_RUN)

    # 2. Sequentially run simulation for each discovered atlas type
    for atlas_type, material_map in atlases.items():
        logging.info(f"--- Preparing to run simulation for atlas type: '{atlas_type}' ---")
        run_simulation_for_atlas(
            atlas_type=atlas_type,
            material_map=material_map,
            seed=SEED_TO_RUN
        )
        logging.info(f"--- Finished simulation for atlas type: '{atlas_type}' ---")

    logging.info("--- All Workflows Completed ---")
