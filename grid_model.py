# grid_model.py — v3 (parametric k‑sweep + robust Floquet PBCs)
# -----------------------------------------------------------------------------
# This revision extends the previous single‑point script so it can *sweep* across
# many (kx,ky) pairs inside the first Brillouin zone (0≤ky≤kx≤π/a).
#
#  • Set up a triangular grid of k‑points (default: N_K=8samples per axis).
#  • For every pair we spin up a minimal COMSOL session, build/solve the model
#    via the mph‑Python API, store eigen‑results in an SQLite DB, and save the
#    .mph file under results/.
#  • The core "run_simulation_and_save_results()" routine from v1 is re‑used
#    but now takes *numerical* kx, ky arguments.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import sqlite3
import datetime
import json
from pathlib import Path
import math
from typing import List, Tuple

import numpy as np
import mph  # COMSOL LiveLink (mph‑python)

# ──────────────────────────────────────────────────────────────────────────────
# Global constants & configuration
# ──────────────────────────────────────────────────────────────────────────────

A: float = 1.0               # lattice constant (m)
GRID_SIZE: int = 28          # 28 × 28 material pixels (keep in sync with bnd lists)
N_K: int = 8                 # samples along Γ→X (inclusive)
K_MAX: float = math.pi / A   # first‑BZ limit

OUTPUT_DIR: Path = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
DATABASE_FILE: Path = OUTPUT_DIR / "simulation_results.db"

SOIL_PROPS = {  # toy values
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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — DB & material map (unchanged from v2)
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
            geometry_type      TEXT,
            param_a            REAL,
            random_seed        INTEGER,
            param_kx_str       TEXT,
            param_ky_str       TEXT,
            mesh_size_setting  INTEGER,
            num_eigenvalues    INTEGER,
            eigenvalue_shift   REAL,
            pbc_x_selection    TEXT,
            pbc_y_selection    TEXT,
            mesh_num_elements  INTEGER,
            mesh_min_quality   REAL
        );
        """)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eigenfrequencies (
            result_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        INTEGER NOT NULL,
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
        INSERT INTO simulations (
            run_timestamp, model_filename, geometry_type, param_a, random_seed,
            param_kx_str, param_ky_str, mesh_size_setting, num_eigenvalues,
            eigenvalue_shift, pbc_x_selection, pbc_y_selection,
            mesh_num_elements, mesh_min_quality
        ) VALUES (
            :run_timestamp, :model_filename, :geometry_type, :param_a, :random_seed,
            :param_kx_str, :param_ky_str, :mesh_size_setting, :num_eigenvalues,
            :eigenvalue_shift, :pbc_x_selection, :pbc_y_selection,
            :mesh_num_elements, :mesh_min_quality
        );
        """,
        run_data,
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def log_eigenfrequencies(db_file: Path, run_id: int, freqs: List[float]) -> None:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO eigenfrequencies (run_id, mode_number, frequency_hz) VALUES (?, ?, ?);",
        [(run_id, i + 1, f) for i, f in enumerate(freqs)],
    )
    conn.commit()
    conn.close()


def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))

# ──────────────────────────────────────────────────────────────────────────────
# Floquet‑PBC boundary lists (for 28×28 grid)
# ──────────────────────────────────────────────────────────────────────────────

def _outer_bnd_lists(size: int) -> Tuple[List[int], List[int]]:
    """Return (pbc_x_nodes, pbc_y_nodes) for the given *size*.
    For now only size==28 is supported — raise otherwise."""
    if size != 28:
        raise ValueError("Boundary lists only calibrated for GRID_SIZE = 28.")
    pbc_x_nodes = [
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
        *range(1597, 1625),  # right edge counterparts
    ]

    pbc_y_nodes = [
        2, 57, 59, 114, 116, 171, 173, 228, 230, 285, 287, 342, 344,
        399, 401, 456, 458, 513, 515, 570, 572, 627, 629, 684, 686,
        741, 743, 798, 800, 855, 857, 912, 914, 969, 971, 1026, 1028,
        1083, 1085, 1140, 1142, 1197, 1199, 1254, 1256, 1311, 1313,
        1368, 1370, 1425, 1427, 1482, 1484, 1539, 1541, 1596,
    ]
    return pbc_x_nodes, pbc_y_nodes

# ──────────────────────────────────────────────────────────────────────────────
# Core single‑run routine
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation_and_save_results(
    mph_file: Path,
    db_file: Path,
    grid_size: int,
    random_seed: int | None,
    kx_val: float,
    ky_val: float,
) -> None:

    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**31 - 1))

    setup_database(db_file)

    # ---------------- metadata dict ----------------
    inputs = {
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_filename": str(mph_file),
        "geometry_type": f"Grid{grid_size}x{grid_size}_OctantSym",
        "param_a": A,
        "random_seed": random_seed,
        "param_kx_str": f"{kx_val}",
        "param_ky_str": f"{ky_val}",
        "mesh_size_setting": 5,
        "num_eigenvalues": 10,
        "eigenvalue_shift": 0.0,
        "pbc_x_selection": "[]",
        "pbc_y_selection": "[]",
        "mesh_num_elements": -1,
        "mesh_min_quality": -1.0,
    }

    if mph_file.exists():
        mph_file.unlink()

    client = None
    try:
        client = mph.start()
        model = client.create("GridModel")

        # global parameters
        model.parameter("a", f"{A}[m]")
        model.parameter("kx", str(kx_val))
        model.parameter("ky", str(ky_val))

        comp = model.java.component().create("comp1", True)
        geom = comp.geom().create("geom1", 2)

        cell = A / grid_size
        start = -A / 2
        for i in range(grid_size):
            for j in range(grid_size):
                sq = geom.create(f"sq_{i}_{j}", "Square")
                sq.set("size", cell)
                sq.set("pos", [start + j * cell, start + i * cell])
        geom.run()

        # materials
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

        mmap = create_symmetric_material_map(grid_size, random_seed).flatten()
        dom_ids = np.arange(1, grid_size * grid_size + 1)
        mat_soil.selection().set(tuple(dom_ids[mmap == 0]))
        mat_con.selection().set(tuple(dom_ids[mmap == 1]))

        # physics + PBCs
        solid = comp.physics().create("solid", "SolidMechanics", "geom1")
        pbc_x_nodes, pbc_y_nodes = _outer_bnd_lists(grid_size)

        pbc_x = solid.create("pbc_x", "PeriodicCondition", 1)
        pbc_x.selection().set(pbc_x_nodes)
        pbc_x.set("PeriodicType", "Floquet")
        pbc_x.set("kFloquet", ["kx", "0", "0"])
        inputs["pbc_x_selection"] = json.dumps(pbc_x_nodes)

        pbc_y = solid.create("pbc_y", "PeriodicCondition", 1)
        pbc_y.selection().set(pbc_y_nodes)
        pbc_y.set("PeriodicType", "Floquet")
        pbc_y.set("kFloquet", ["0", "ky", "0"])
        inputs["pbc_y_selection"] = json.dumps(pbc_y_nodes)

        # mesh
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(inputs["mesh_size_setting"])
        mesh.feature().create("ftri1", "FreeTri").selection().geom("geom1", 2).all()
        mesh.run()
        stats = comp.mesh("mesh1").stat()
        inputs["mesh_num_elements"] = stats.getNumElem()
        inputs["mesh_min_quality"] = stats.getMinQuality()

        # study
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(inputs["num_eigenvalues"]))
        eig.set("shiftactive", "on")
        eig.set("shift", str(inputs["eigenvalue_shift"]))
        eig.activate("solid", True)

        run_id = log_simulation_run(db_file, inputs)
        logging.info("Solving eigenfrequencies (run_id=%d, kx=%.4f, ky=%.4f)", run_id, kx_val, ky_val)
        study.run()

        freqs_hz = [float(np.real(v)) for v in model.evaluate("freq") if np.isfinite(np.real(v))]
        log_eigenfrequencies(db_file, run_id, freqs_hz)
        model.save(mph_file)

    except Exception as exc:
        logging.exception("COMSOL run failed: %s", exc)
    finally:
        if client:
            client.clear()

# ──────────────────────────────────────────────────────────────────────────────
# k‑grid generator & sweep loop (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────────────

def generate_kgrid(n_k: int, k_max: float) -> List[Tuple[float, float]]:
    lin = np.linspace(0.0, k_max, n_k)
    return [(float(kx), float(ky)) for i, kx in enumerate(lin) for ky in lin[: i + 1]]


def main() -> None:
    k_points = generate_kgrid(N_K, K_MAX)
    logging.info("k‑sweep: %d points (triangular Γ–X first‑BZ)", len(k_points))

    for idx, (kx, ky) in enumerate(k_points, 1):
        logging.info("—— %d / %d : kx=%.4f, ky=%.4f ——", idx, len(k_points), kx, ky)
        mph_file = OUTPUT_DIR / f"grid_kx{kx:.4f}_ky{ky:.4f}.mph"
        run_simulation_and_save_results(
            mph_file,
            DATABASE_FILE,
            grid_size=GRID_SIZE,
            random_seed=None,
            kx_val=kx,
            ky_val=ky,
        )


if __name__ == "__main__":
    main()
    logging.info("Parametric k‑sweep completed.")
