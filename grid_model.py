from __future__ import annotations
import logging
import sqlite3
import datetime
import json
from pathlib import Path
import math
from typing import List, Tuple
import numpy as np
import mph

# ──────────────────────────────────────────────────────────────────────────────
# Global constants & configuration (from your v3 script)
# ──────────────────────────────────────────────────────────────────────────────
A: float = 1.0
GRID_SIZE: int = 28
N_K: int = 8
K_MAX: float = math.pi / A
N_MODES: int = 10

OUTPUT_DIR: Path = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
DATABASE_FILE: Path = OUTPUT_DIR / "simulation_results.db"

SOIL_PROPS = {
    "name": "Soil", "youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0,
}
CONCRETE_PROPS = {
    "name": "Concrete", "youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Database functions (adapted for manual sweep)
# ──────────────────────────────────────────────────────────────────────────────
def setup_database(db_file: Path) -> None:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS simulations (
            run_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp      TEXT    NOT NULL,
            model_filename     TEXT,
            random_seed        INTEGER
        );
        """)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eigenfrequencies (
            result_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        INTEGER NOT NULL,
            kx            REAL    NOT NULL,
            ky            REAL    NOT NULL,
            mode_number   INTEGER NOT NULL,
            frequency_hz  REAL    NOT NULL,
            FOREIGN KEY (run_id) REFERENCES simulations (run_id)
        );
        """)
    conn.commit()
    conn.close()


def log_simulation_run(db_file: Path, run_data: dict) -> int:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO simulations (run_timestamp, model_filename, random_seed)
        VALUES (:run_timestamp, :model_filename, :random_seed);
        """, run_data)
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def log_single_frequency_set(db_file: Path, run_id: int, kx: float, ky: float, freqs: List[float]) -> None:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO eigenfrequencies (run_id, kx, ky, mode_number, frequency_hz) VALUES (?, ?, ?, ?, ?);",
        [(run_id, kx, ky, i + 1, f) for i, f in enumerate(freqs)],
    )
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions (from your v3 script)
# ──────────────────────────────────────────────────────────────────────────────
def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))


def _outer_bnd_lists(size: int) -> Tuple[List[int], List[int]]:
    if size != 28:
        raise ValueError("Boundary lists only calibrated for GRID_SIZE = 28.")
    pbc_x_nodes = [
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
        *range(1597, 1625),
    ]
    pbc_y_nodes = [
        2, 57, 59, 114, 116, 171, 173, 228, 230, 285, 287, 342, 344, 399, 401, 456, 458, 513, 515, 570, 572, 627, 629,
        684, 686, 741, 743, 798, 800, 855, 857, 912, 914, 969, 971, 1026, 1028, 1083, 1085, 1140, 1142, 1197, 1199,
        1254, 1256, 1311, 1313, 1368, 1370, 1425, 1427, 1482, 1484, 1539, 1541, 1596,
    ]
    return pbc_x_nodes, pbc_y_nodes


def generate_kgrid(n_k: int, k_max: float) -> List[Tuple[float, float]]:
    lin = np.linspace(0.0, k_max, n_k)
    return [(float(kx), float(ky)) for i, kx in enumerate(lin) for ky in lin[: i + 1]]


# ──────────────────────────────────────────────────────────────────────────────
# RATIONALE FOR CHANGE: This new function builds the model once, then loops
# in Python to update parameters and resolve. This is efficient and avoids
# the buggy Parametric sweep node configuration.
# ──────────────────────────────────────────────────────────────────────────────
def run_manual_sweep(k_points: List[Tuple[float, float]]) -> None:
    random_seed = np.random.randint(0, 2 ** 31 - 1)
    mph_file = OUTPUT_DIR / f"grid_manual_sweep_seed_{random_seed}.mph"
    setup_database(DATABASE_FILE)

    client = None
    try:
        # Start client and build model ONCE
        client = mph.start()
        model = client.create("GridModel")
        logging.info("Client started and model created. Building geometry and mesh...")

        # All model creation steps are identical to your v3 script
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

        mat_soil = comp.material().create("mat_soil", "Common")
        mat_soil.label(SOIL_PROPS["name"])
        pg = mat_soil.propertyGroup("def")
        pg.set("youngsmodulus", f"{SOIL_PROPS['youngs_modulus']}[Pa]")
        pg.set("poissonsratio", str(SOIL_PROPS["poissons_ratio"]))
        pg.set("density", f"{SOIL_PROPS['density']}[kg/m^3]")
        mat_con = comp.material().create("mat_con", "Common")
        mat_con.label(CONCRETE_PROPS["name"])
        pgc = mat_con.propertyGroup("def")
        pgc.set("youngsmodulus", f"{CONCRETE_PROPS['youngs_modulus']}[Pa]")
        pgc.set("poissonsratio", str(CONCRETE_PROPS["poissons_ratio"]))
        pgc.set("density", f"{CONCRETE_PROPS['density']}[kg/m^3]")
        mmap = create_symmetric_material_map(GRID_SIZE, random_seed).flatten()
        dom_ids = np.arange(1, GRID_SIZE * GRID_SIZE + 1)
        mat_soil.selection().set(tuple(dom_ids[mmap == 0].tolist()))
        mat_con.selection().set(tuple(dom_ids[mmap == 1].tolist()))

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

        # Setup study ONCE
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(N_MODES))
        eig.set("shiftactive", "on")
        eig.set("shift", "0.0")
        eig.activate("solid", True)
        logging.info("Model built successfully.")

        # Log the simulation run once
        run_data = {
            "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_filename": str(mph_file.name),
            "random_seed": random_seed,
        }
        run_id = log_simulation_run(DATABASE_FILE, run_data)

        # Loop through k-points, update parameters, and re-solve
        for i, (kx, ky) in enumerate(k_points):
            logging.info(f"Solving point {i + 1}/{len(k_points)}: (kx={kx:.4f}, ky={ky:.4f})")
            model.parameter("kx", str(kx))
            model.parameter("ky", str(ky))
            study.run()
            freqs_hz = [np.real(v) for v in model.evaluate("freq")]
            log_single_frequency_set(DATABASE_FILE, run_id, kx, ky, freqs_hz)

        model.save(mph_file)
        logging.info(f"Manual sweep finished. Model saved to '{mph_file}'")

    except Exception as exc:
        logging.exception("COMSOL manual sweep failed: %s", exc)
    finally:
        if client:
            client.clear()


def main() -> None:
    k_points = generate_kgrid(N_K, K_MAX)
    logging.info("Starting manual sweep for %d k-points...", len(k_points))
    run_manual_sweep(k_points)


if __name__ == "__main__":
    main()
    logging.info("Script completed.")
