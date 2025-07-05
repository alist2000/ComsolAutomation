"""
*********************************************************************
 grid_model_v5.0.py
 --------------------------------------------------------------------
  • Grid size bumped to 32 × 32.
  • Keeps the **identical creation code path** from your v4 script – only
    two things changed:
      1. `_outer_bnd_lists()` now contains a *second* hard‑coded entry for
         `size == 32` (fill the numbers you just measured with the probe
         script).
      2. A small wrapper runs the same build loop in parallel for 100
         seeds (exactly as you requested earlier).

*********************************************************************
"""

from __future__ import annotations
import logging
import datetime
import math
import multiprocessing as mp
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mph

# ── Global configuration ────────────────────────────────────────────
A: float = 1.0  # unit‑cell size [m]
GRID_SIZE: int = 32  # *** new: 32 instead of 28 ***
N_K: int = 8  # k‑grid resolution per axis
K_MAX: float = math.pi / A
N_EIG: int = 10  # modes per solve
N_SEEDS: int = 10  # number of random masks

OUTPUT_DIR = Path("results_v5");
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUT_DIR / "simulation_results_v5.db"

SOIL_PROPS = {
    "name": "Soil",
    "youngs_modulus": 20e6,
    "poissons_ratio": 0.30,
    "density": 1800.0,
}
CONCRETE_PROPS = {
    "name": "Concrete",
    "youngs_modulus": 20e9,
    "poissons_ratio": 0.20,
    "density": 2400.0,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")


# ── DB helpers (unchanged from v4) ──────────────────────────────────

def _setup_db(db_file: Path) -> None:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS simulations (
                     run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     run_timestamp TEXT, model_file TEXT, seed INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS eigenfrequencies (
                     result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     run_id INTEGER, kx REAL, ky REAL,
                     mode INTEGER, freq REAL)""")
    conn.commit();
    conn.close()


def _log_run(db_file: Path, seed: int, mph_name: str) -> int:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("INSERT INTO simulations VALUES (NULL,?,?,?)",
                (datetime.datetime.now().isoformat(timespec="seconds"),
                 mph_name, seed))
    rid = cur.lastrowid;
    conn.commit();
    conn.close();
    return rid


def _log_freqs(db_file: Path, run_id: int, kx: float, ky: float, freqs: List[float]):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany("INSERT INTO eigenfrequencies VALUES (NULL,?,?,?,?,?)",
                    [(run_id, kx, ky, i + 1, f) for i, f in enumerate(freqs)])
    conn.commit();
    conn.close()


# ── Util functions copied verbatim from v4 ─────────────────────────

def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:  # same name as v4
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))


def generate_kgrid(n_k: int, k_max: float) -> List[Tuple[float, float]]:
    lin = np.linspace(0.0, k_max, n_k)
    return [(float(kx), float(ky)) for i, kx in enumerate(lin) for ky in lin[: i + 1]]


# ── **Hard‑coded PBC boundary lists** (keep identical API) ─────────

def _outer_bnd_lists(size: int) -> Tuple[List[int], List[int]]:
    """Return (pbc_x_nodes, pbc_y_nodes) for the requested `size`.  Identical
    signature to v4 so downstream code is untouched.
    """
    if size == 28:
        # original v4 constants (unchanged) ──────────────────────────
        pbc_x_nodes = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
            37, 39, 41, 43, 45, 47, 49, 51, 53, 55, *range(1597, 1625),
        ]
        pbc_y_nodes = [
            2, 57, 59, 114, 116, 171, 173, 228, 230, 285, 287, 342, 344, 399,
            401, 456, 458, 513, 515, 570, 572, 627, 629, 684, 686, 741, 743,
            798, 800, 855, 857, 912, 914, 969, 971, 1026, 1028, 1083, 1085,
            1140, 1142, 1197, 1199, 1254, 1256, 1311, 1313, 1368, 1370, 1425,
            1427, 1482, 1484, 1539, 1541, 1596,
        ]
        return pbc_x_nodes, pbc_y_nodes

    if size == 32:
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
        if not pbc_x_nodes_32 or not pbc_y_nodes_32:
            raise ValueError(
                "Fill `pbc_x_nodes_32` and `pbc_y_nodes_32` with your measured boundary IDs before running.")
        return pbc_x_nodes_32, pbc_y_nodes_32

    raise ValueError(f"Boundary lists not calibrated for GRID_SIZE = {size}.")


# ── Worker: identical creation routine from v4 ─────────────────────

def _build_and_solve(seed: int):
    client = None
    try:
        client = mph.start(cores=1)
        model = client.create(f"GridModel_seed{seed}")
        logging.info("Seed %d – building model", seed)

        # global parameters ---------------------------------------------------
        model.parameter("a", f"{A}[m]")
        model.parameter("kx", "0");
        model.parameter("ky", "0")

        # component & geometry (same loops, just GRID_SIZE=32) ---------------
        comp = model.java.component().create("comp1", True)
        geom = comp.geom().create("geom1", 2)
        cell = A / GRID_SIZE;
        start = -A / 2
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                sq = geom.create(f"sq_{i}_{j}", "Square")
                sq.set("size", cell)
                sq.set("pos", [start + j * cell, start + i * cell])
        geom.run()

        # materials -----------------------------------------------------------
        mat_soil = comp.material().create("mat_soil", "Common")
        pg = mat_soil.propertyGroup("def")
        pg.set("youngsmodulus", f"{SOIL_PROPS['youngs_modulus']}[Pa]")
        pg.set("poissonsratio", str(SOIL_PROPS['poissons_ratio']))
        pg.set("density", f"{SOIL_PROPS['density']}[kg/m^3]")

        mat_con = comp.material().create("mat_con", "Common")
        pgc = mat_con.propertyGroup("def")
        pgc.set("youngsmodulus", f"{CONCRETE_PROPS['youngs_modulus']}[Pa]")
        pgc.set("poissonsratio", str(CONCRETE_PROPS['poissons_ratio']))
        pgc.set("density", f"{CONCRETE_PROPS['density']}[kg/m^3]")

        mmap = create_symmetric_material_map(GRID_SIZE, seed).flatten()
        dom_ids = np.arange(1, GRID_SIZE * GRID_SIZE + 1)
        mat_soil.selection().set(tuple(dom_ids[mmap == 0]))
        mat_con.selection().set(tuple(dom_ids[mmap == 1]))

        # physics (Solid + Floquet PBCs) --------------------------------------
        solid = comp.physics().create("solid", "SolidMechanics", "geom1")
        pbc_x_nodes, pbc_y_nodes = _outer_bnd_lists(GRID_SIZE)

        pbc_x = solid.create("pbc_x", "PeriodicCondition", 1)
        pbc_x.selection().set(pbc_x_nodes)
        pbc_x.set("PeriodicType", "Floquet");
        pbc_x.set("kFloquet", ["kx", "0", "0"])

        pbc_y = solid.create("pbc_y", "PeriodicCondition", 1)
        pbc_y.selection().set(pbc_y_nodes)
        pbc_y.set("PeriodicType", "Floquet");
        pbc_y.set("kFloquet", ["0", "ky", "0"])

        # mesh ---------------------------------------------------------------
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(5)
        mesh.feature().create("ftri1", "FreeTri").selection().geom("geom1", 2).all()
        mesh.run()

        # study --------------------------------------------------------------
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on");
        eig.set("neigs", str(N_EIG))
        eig.set("shiftactive", "on");
        eig.set("shift", "0")
        eig.activate("solid", True)

        # DB bookkeeping -----------------------------------------------------
        mph_name = f"grid32_seed_{seed}.mph";
        run_id = _log_run(DB_PATH, seed, mph_name)

        k_points = generate_kgrid(N_K, K_MAX)
        for kx, ky in k_points:
            model.parameter("kx", str(kx));
            model.parameter("ky", str(ky))
            study.run()
            freqs = [float(np.real(v)) for v in model.evaluate("freq")]
            _log_freqs(DB_PATH, run_id, kx, ky, freqs)

        # model.save(OUTPUT_DIR / mph_name)  # It is unnecessary
        logging.info("Seed %d finished", seed)

    except Exception:
        logging.exception("Seed %d failed", seed)
    finally:
        if client:
            client.clear()


# ── Main entry point: fire up parallel pool ────────────────────────

if __name__ == "__main__":
    _setup_db(DB_PATH)
    logging.info("Launching %d parallel workers…", N_SEEDS)
    with mp.Pool(processes=min(mp.cpu_count(), N_SEEDS)) as pool:
        pool.map(_build_and_solve, range(N_SEEDS))
        pool.close();
        pool.join()
