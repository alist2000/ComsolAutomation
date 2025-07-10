from __future__ import annotations
import logging
import datetime
import math
import multiprocessing as mp
import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import mph

# ── Global configuration ────────────────────────────────────────────
A: float = 1.0
GRID_SIZE: int = 32
N_K: int = 8
K_MAX: float = math.pi / A
N_EIG: int = 10
N_SEEDS: int = 2

# --- Updated output paths ---
OUTPUT_DIR = Path("results_v5.3")
OUTPUT_DIR.mkdir(exist_ok=True)
# Create a dedicated directory for the exported text files
EXPORT_DATA_DIR = OUTPUT_DIR / "displacement_data"
EXPORT_DATA_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUT_DIR / "simulation_results_v5.3.db"

# --- Material properties ---
MATERIALS = {
    "soil": {"youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0},
    "concrete": {"youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0},
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")


# ── DB helpers (Schema simplified as sampling points are now implicit) ───
def _setup_db(db_file: Path) -> None:
    """Initializes the database with the updated schema."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS simulations (
                     run_id         INTEGER PRIMARY KEY,
                     run_timestamp  TEXT,
                     model_file     TEXT,
                     seed           INTEGER,
                     atlas_path     TEXT,
                     materials_json TEXT)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS eigenfrequencies (
                     result_id          INTEGER PRIMARY KEY,
                     run_id             INTEGER,
                     kx                 REAL,
                     ky                 REAL,
                     mode_number        INTEGER,
                     frequency_hz       REAL,
                     displacement_path  TEXT)""")  # Renamed from modes_path
    conn.commit()
    conn.close()


def _log_run_and_get_id(db_file: Path, seed: int, mph_name: str) -> int:
    """Logs the initial run info and returns the new run_id."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("INSERT INTO simulations (run_timestamp, model_file, seed) VALUES (?,?,?)",
                (datetime.datetime.now().isoformat(timespec="seconds"),
                 mph_name, seed))
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def _update_run_with_paths(db_file: Path, run_id: int, atlas_path: str,
                           materials_json: str):
    """Updates the run record with atlas and material info."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""UPDATE simulations SET atlas_path = ?, materials_json = ?
                   WHERE run_id = ?""", (atlas_path, materials_json, run_id))
    conn.commit()
    conn.close()


def _log_freqs_and_displacements(db_file: Path, run_id: int, kx: float, ky: float,
                                 freqs: List[float], displacement_path: str):
    """Logs frequencies and path to the exported displacement file."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany("""INSERT INTO eigenfrequencies
                       (run_id, kx, ky, mode_number, frequency_hz, displacement_path)
                       VALUES (?,?,?,?,?,?)""",
                    [(run_id, kx, ky, i + 1, f, displacement_path)
                     for i, f in enumerate(freqs)])
    conn.commit()
    conn.close()


# ── Util functions ──────────────────────────────────────────────────
def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))


def generate_kgrid(n_k: int, k_max: float) -> List[Tuple[float, float]]:
    lin = np.linspace(0.0, k_max, n_k)
    return [(float(kx), float(ky)) for i, kx in enumerate(lin) for ky in lin[: i + 1]]


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


# ── Worker function (UPDATED) ─────────────────────────────────────

# ── Worker function (UPDATED) ─────────────────────────────────────

# ── Worker function (UPDATED) ─────────────────────────────────────

def _build_and_solve(seed: int):
    client = None
    try:
        client = mph.start(cores=1)
        model = client.create(f"GridModel_seed{seed}")
        logging.info("Seed %d – building model", seed)

        # --- Model setup remains the same ---
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

        # --- Material, Physics, Mesh setup remains the same ---
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
        mmap = create_symmetric_material_map(GRID_SIZE, seed)
        mmap_flat = mmap.flatten()
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

        # --- Study setup remains the same ---
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(N_EIG))
        eig.set("shiftactive", "on")
        eig.set("shift", "1")
        eig.activate("solid", True)

        # --- Database logging setup remains the same ---
        mph_name = f"grid32_seed_{seed}.mph"
        run_id = _log_run_and_get_id(DB_PATH, seed, mph_name)
        atlas_path = f"atlas_run_{run_id}.npy"
        np.save(OUTPUT_DIR / atlas_path, mmap)
        materials_json = json.dumps(MATERIALS)
        _update_run_with_paths(DB_PATH, run_id, atlas_path, materials_json)

        # Solve for each k-point
        k_points = generate_kgrid(N_K, K_MAX)
        for kx, ky in k_points:
            logging.info(f"Seed {seed}, k=({kx:.2f},{ky:.2f}) - Solving...")
            model.parameter("kx", str(kx))
            model.parameter("ky", str(ky))
            study.run()

            freqs = [float(np.real(v)) for v in model.evaluate("freq")]

            # 🚨 FIX: Create a temporary, fresh dataset AND export node for this specific solution.
            cpt_tag_temp = "temp_cpt_export"
            export_tag_temp = "temp_data_export"

            # Remove the nodes if they exist from a previous iteration to be safe
            try:
                model.java.result().dataset().remove(cpt_tag_temp)
                model.java.result().export().remove(export_tag_temp)
            except Exception:
                pass  # It's okay if they don't exist

            # Create the dataset
            cpt_node = model.java.result().dataset().create(cpt_tag_temp, "CutPoint2D")
            cpt_node.set("method", "grid")
            cpt_node.set("gridx", f"range(-a/2,a/{GRID_SIZE - 1},a/2)")
            cpt_node.set("gridy", f"range(-a/2,a/{GRID_SIZE - 1},a/2)")
            cpt_node.run()

            # Create the export node
            export_node = model.java.result().export().create(export_tag_temp, "Data")
            export_node.set("data", cpt_tag_temp)
            export_node.setIndex("expr", "u", 0)
            export_node.setIndex("expr", "v", 1)

            displacement_filename = f"atlas_{seed}_kx_{kx:.4f}_ky_{ky:.4f}.txt"
            displacement_filepath = EXPORT_DATA_DIR / displacement_filename
            safe_filepath_str = str(displacement_filepath.resolve()).replace('\\', '\\\\')

            export_node.set("filename", safe_filepath_str)
            export_node.run()
            logging.info(
                f"Seed {seed}, k=({kx:.2f},{ky:.2f}) - Successfully exported displacements to {displacement_filename}")

            _log_freqs_and_displacements(DB_PATH, run_id, kx, ky, freqs, str(displacement_filepath))

        logging.info("Seed %d finished", seed)

    except Exception:
        logging.exception("Seed %d failed", seed)
    finally:
        if client:
            client.clear()
# ── Main entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    _setup_db(DB_PATH)

    logging.info("Launching %d parallel workers…", N_SEEDS)
    seeds_to_run = range(N_SEEDS)
    with mp.Pool(processes=min(mp.cpu_count(), N_SEEDS)) as pool:
        pool.map(_build_and_solve, seeds_to_run)
        pool.close()
        pool.join()
    logging.info("All workers finished.")
