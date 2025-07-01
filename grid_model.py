# 6_result_grid_final_octant_fixed.py
#
# This script is based on 6_result.py but now:
#   • Builds a symmetric (octant‑symmetric) 28×28 material map.
#   • Logs periodic‑boundary selections and detailed mesh stats to SQLite.
#   • Fixes the mesh‑statistics retrieval that previously raised a NoneType error.
#
# ------------------------------------------------------------------------------
# PART 1 · Imports & configuration
# ------------------------------------------------------------------------------

import logging
import sqlite3
import datetime
import json
from pathlib import Path

import math  # kept for future extensions that might need math functions
import numpy as np
import mph  # COMSOL LiveLink for Python

# File paths
OUTPUT_MPH_FILE = Path("grid_model_28x28_octant.mph")
DATABASE_FILE = Path("simulation_results.db")

# Material properties (example values)
SOIL_PROPS = {
    "name": "Soil",
    "youngs_modulus": 20e6,  # Pa
    "poissons_ratio": 0.3,
    "density": 1800.0,       # kg/m³
}
CONCRETE_PROPS = {
    "name": "Concrete",
    "youngs_modulus": 20e9,  # Pa
    "poissons_ratio": 0.2,
    "density": 2400.0,       # kg/m³
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(levelname)s · %(message)s"
)

# ------------------------------------------------------------------------------
# PART 2 · Database helpers
# ------------------------------------------------------------------------------

def setup_database(db_file: Path) -> None:
    """Create the SQLite tables if they don’t exist."""
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
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eigenfrequencies (
            result_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        INTEGER NOT NULL,
            mode_number   INTEGER NOT NULL,
            frequency_hz  REAL    NOT NULL,
            FOREIGN KEY (run_id) REFERENCES simulations (run_id)
        );
        """
    )
    conn.commit()
    conn.close()
    logging.info("Database ready (with mesh‑stats fields).")


def log_simulation_run(db_file: Path, run_data: dict) -> int:
    """Insert the simulation metadata and return the generated run_id."""
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
    logging.info("Logged simulation metadata (run_id=%s).", run_id)
    return run_id


def log_eigenfrequencies(db_file: Path, run_id: int, freqs: list[float]) -> None:
    """Store each eigenfrequency for the given run."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO eigenfrequencies (run_id, mode_number, frequency_hz) VALUES (?, ?, ?);",
        [(run_id, i + 1, f) for i, f in enumerate(freqs)],
    )
    conn.commit()
    conn.close()
    logging.info("Saved %d eigenfrequencies.", len(freqs))

# ------------------------------------------------------------------------------
# PART 3 · Octant‑symmetric material map
# ------------------------------------------------------------------------------

def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    """Return a *size×size* array of 0/1 with full octant symmetry."""
    rng = np.random.default_rng(seed)
    quad = np.zeros((size // 2, size // 2), dtype=int)
    for i in range(quad.shape[0]):
        for j in range(i, quad.shape[1]):  # ensure diagonal symmetry in quadrant
            val = rng.integers(0, 2)
            quad[i, j] = quad[j, i] = val
    top = np.hstack([np.fliplr(quad), quad])
    full = np.vstack([np.flipud(top), top])
    return full

# ------------------------------------------------------------------------------
# PART 4 · COMSOL model build & solve
# ------------------------------------------------------------------------------

def run_simulation_and_save_results(
    mph_file: Path,
    db_file: Path,
    grid_size: int = 28,
    random_seed: int | None = None,
) -> None:
    """Create model, solve eigenfrequencies, and persist everything."""

    setup_database(db_file)

    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**31 - 1))
    logging.info("Random seed: %d", random_seed)

    # Metadata dict that we will log later
    inputs: dict[str, object] = {
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_filename": str(mph_file),
        "geometry_type": f"Grid{grid_size}x{grid_size}_OctantSym",  # 28×28_OctantSym
        "param_a": 1.0,              # unit‑cell size (m)
        "random_seed": random_seed,
        "param_kx_str": "pi/a",     # Bloch wavenumber kx
        "param_ky_str": "0",        # ky
        "mesh_size_setting": 5,      # COMSOL auto size (1 = extremely coarse … 9)
        "num_eigenvalues": 10,
        "eigenvalue_shift": 0.0,
        # placeholders to be filled later
        "pbc_x_selection": "[]",
        "pbc_y_selection": "[]",
        "mesh_num_elements": -1,
        "mesh_min_quality": -1.0,
    }

    # Delete old mph file so we can save afresh
    if mph_file.exists():
        mph_file.unlink()

    client = None
    try:
        # ------------------------------------------------------------------
        # Connect to COMSOL server and create base model
        # ------------------------------------------------------------------
        client = mph.start()
        model = client.create("GridModel")

        # Parameters a, kx, ky
        model.parameter("a", f"{inputs['param_a']}[m]")
        model.parameter("kx", inputs["param_kx_str"])
        model.parameter("ky", inputs["param_ky_str"])

        comp = model.java.component().create("comp1", True)
        geom = comp.geom().create("geom1", 2)

        logging.info("Building %dx%d square grid …", grid_size, grid_size)
        cell = inputs["param_a"] / grid_size
        start = -inputs["param_a"] / 2
        for i in range(grid_size):
            for j in range(grid_size):
                sq = geom.create(f"sq_{i}_{j}", "Square")
                sq.set("size", cell)
                sq.set("pos", [start + j * cell, start + i * cell])
        geom.run()

        # ------------------------------------------------------------------
        # Materials assignment
        # ------------------------------------------------------------------
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

        logging.info("Assigning octant‑symmetric material map …")
        mmap = create_symmetric_material_map(grid_size, random_seed).flatten()
        dom_ids = np.arange(1, grid_size * grid_size + 1)
        soil_ids = tuple(dom_ids[mmap == 0])
        conc_ids = tuple(dom_ids[mmap == 1])
        if soil_ids:
            mat_soil.selection().set(soil_ids)
        if conc_ids:
            mat_con.selection().set(conc_ids)
        logging.info("Soil domains: %d · Concrete domains: %d", len(soil_ids), len(conc_ids))

        # ------------------------------------------------------------------
        # Solid Mechanics + Floquet PBCs
        # ------------------------------------------------------------------
        solid = comp.physics().create("solid", "SolidMechanics", "geom1")

        pbc_x = solid.create("pbc_x", "PeriodicCondition", 1)
        pbc_x_nodes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                       33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55] + list(range(1597, 1625))
        pbc_x.selection().set(pbc_x_nodes)
        pbc_x.set("PeriodicType", "Floquet")
        pbc_x.set("kFloquet", ["kx", "0", "0"])
        inputs["pbc_x_selection"] = json.dumps(pbc_x_nodes)

        pbc_y = solid.create("pbc_y", "PeriodicCondition", 1)
        pbc_y_nodes = [2, 57, 59, 114, 116, 171, 173, 228, 230, 285, 287, 342, 344,
                       399, 401, 456, 458, 513, 515, 570, 572, 627, 629, 684, 686,
                       741, 743, 798, 800, 855, 857, 912, 914, 969, 971, 1026, 1028,
                       1083, 1085, 1140, 1142, 1197, 1199, 1254, 1256, 1311, 1313,
                       1368, 1370, 1425, 1427, 1482, 1484, 1539, 1541, 1596]
        pbc_y.selection().set(pbc_y_nodes)
        pbc_y.set("PeriodicType", "Floquet")
        pbc_y.set("kFloquet", ["0", "ky", "0"])
        inputs["pbc_y_selection"] = json.dumps(pbc_y_nodes)

        # ------------------------------------------------------------------
        # Mesh (free triangular) and statistics
        # ------------------------------------------------------------------
        logging.info("Creating a free triangular mesh...")
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(inputs['mesh_size_setting'])
        ftri = mesh.feature().create("ftri1", "FreeTri")
        ftri.selection().geom("geom1", 2)
        ftri.selection().all()
        mesh.run()
        logging.info("Mesh generated.")

        # -----------------------------------------------
        # Mesh statistics (after mesh_java.run())
        # -----------------------------------------------
        mesh_seq_java = comp.mesh("mesh1")  # Java MeshSequence
        mesh_stats_java = mesh_seq_java.stat()  # MeshStatistics handle

        inputs["mesh_num_elements"] = mesh_stats_java.getNumElem()
        inputs["mesh_min_quality"] = mesh_stats_java.getMinQuality()

        logging.info(
            "Mesh stats: %d elements · min quality %.4f",
            inputs["mesh_num_elements"],
            inputs["mesh_min_quality"],
        )

        # ------------------------------------------------------------------
        # Eigenfrequency study
        # ------------------------------------------------------------------
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(inputs["num_eigenvalues"]))
        eig.set("shiftactive", "on")
        eig.set("shift", str(inputs["eigenvalue_shift"]))
        eig.activate("solid", True)

        # Log inputs (needs mesh stats first!)
        run_id = log_simulation_run(db_file, inputs)

        logging.info("Running eigenfrequency solver (run_id=%d) …", run_id)
        study.run()

        evals = model.evaluate("freq")
        freqs_hz = [float(np.real(v)) for v in evals if np.isfinite(np.real(v))]
        log_eigenfrequencies(db_file, run_id, freqs_hz)

        # Save model file
        model.save(mph_file)
        logging.info("Simulation finished and saved → %s", mph_file)

    except Exception as exc:
        logging.exception("Error during COMSOL workflow: %s", exc)
    finally:
        if client:
            client.clear()
            logging.info("COMSOL client closed.")

# ------------------------------------------------------------------------------
# PART 5 · Script entry
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run_simulation_and_save_results(OUTPUT_MPH_FILE, DATABASE_FILE, grid_size=28)
    logging.info("Script completed.")
