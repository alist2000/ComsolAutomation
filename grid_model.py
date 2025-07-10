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
N_SEEDS: int = 100

# --- Updated output paths for v5.2 ---
OUTPUT_DIR = Path("results_v5.2")
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUT_DIR / "simulation_results_v5.2.db"

# --- Material properties ---
MATERIALS = {
    "soil": {"youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0},
    "concrete": {"youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0},
}

# --- NEW: Sampling configuration for efficient data storage ---
SAMPLING_CONFIG = {
    "method": "boundary_plus_interior",  # Options: "structured_grid", "boundary_plus_interior", "adaptive"
    "grid_resolution": 32,  # For structured grid sampling
    "boundary_density": GRID_SIZE,  # For boundary sampling
    "interior_density": 8,  # For interior sampling
    "save_boundary_nodes": True,  # Always save boundary nodes for PBC analysis
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")


# ── NEW: Sampling strategy functions ────────────────────────────────

def create_structured_sampling_points(resolution: int = 32) -> np.ndarray:
    """
    Create a structured grid of sampling points within the domain [-A/2, A/2]².
    Returns array of shape (resolution², 2) with (x, y) coordinates.
    """
    x = np.linspace(-A / 2, A / 2, resolution)
    y = np.linspace(-A / 2, A / 2, resolution)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def create_boundary_plus_interior_sampling(boundary_density: int = 16,
                                           interior_density: int = 8) -> np.ndarray:
    """
    Create sampling points focused on boundaries (for PBC) plus sparse interior.
    """
    # Boundary points (edges of the domain)
    boundary_points = []

    # Bottom and top edges
    x_boundary = np.linspace(-A / 2, A / 2, boundary_density)
    boundary_points.extend([(x, -A / 2) for x in x_boundary])  # Bottom
    boundary_points.extend([(x, A / 2) for x in x_boundary])  # Top

    # Left and right edges (excluding corners to avoid duplication)
    y_boundary = np.linspace(-A / 2, A / 2, boundary_density)[1:-1]
    boundary_points.extend([(-A / 2, y) for y in y_boundary])  # Left
    boundary_points.extend([(A / 2, y) for y in y_boundary])  # Right

    # Interior points (sparse sampling)
    interior_x = np.linspace(-A / 2 + A / 10, A / 2 - A / 10, interior_density)
    interior_y = np.linspace(-A / 2 + A / 10, A / 2 - A / 10, interior_density)
    for x in interior_x:
        for y in interior_y:
            boundary_points.append((x, y))

    return np.array(boundary_points)


def create_adaptive_sampling_points(material_map: np.ndarray,
                                    base_resolution: int = 16) -> np.ndarray:
    """
    Create adaptive sampling points with higher density at material interfaces.
    """
    # Start with base structured grid
    base_points = create_structured_sampling_points(base_resolution)

    # Find material interfaces and add extra points there
    # This is a simplified version - you could make it more sophisticated
    grad_x = np.gradient(material_map, axis=1)
    grad_y = np.gradient(material_map, axis=0)
    interface_mask = (np.abs(grad_x) + np.abs(grad_y)) > 0.1

    # Add extra points near interfaces
    interface_points = []
    for i in range(material_map.shape[0]):
        for j in range(material_map.shape[1]):
            if interface_mask[i, j]:
                x = -A / 2 + (j + 0.5) * A / material_map.shape[1]
                y = -A / 2 + (i + 0.5) * A / material_map.shape[0]
                # Add 4 points around the interface
                offset = A / (4 * material_map.shape[0])
                interface_points.extend([
                    (x + offset, y), (x - offset, y),
                    (x, y + offset), (x, y - offset)
                ])

    if interface_points:
        all_points = np.vstack([base_points, np.array(interface_points)])
    else:
        all_points = base_points

    return all_points


def get_sampling_points(method: str, material_map: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get sampling points based on the configured method.
    """
    if method == "structured_grid":
        return create_structured_sampling_points(SAMPLING_CONFIG["grid_resolution"])
    elif method == "boundary_plus_interior":
        return create_boundary_plus_interior_sampling(
            SAMPLING_CONFIG["boundary_density"],
            SAMPLING_CONFIG["interior_density"]
        )
    elif method == "adaptive" and material_map is not None:
        return create_adaptive_sampling_points(material_map, 16)
    else:
        # Fallback to structured grid
        return create_structured_sampling_points(32)


# ── NEW: Efficient displacement evaluation ──────────────────────────

def evaluate_displacements_at_points(model, sample_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate u and v displacements at specific sampling points efficiently.
    Args:
        model: COMSOL model object
        sample_points: Array of (x, y) coordinates, shape (n_points, 2)
    Returns:
        Tuple of (u_values, v_values) arrays, each shape (n_points, n_modes)
    """
    from scipy.interpolate import griddata
    full_u = model.evaluate('u')
    full_v = model.evaluate('v')
    mesh_coords = model.evaluate(['x', 'y'])

    # Flatten mesh coordinates
    mesh_x, mesh_y = mesh_coords[0].flatten(), mesh_coords[1].flatten()

    # Interpolate displacement values
    u_interp = griddata(np.column_stack([mesh_x, mesh_y]), full_u.flatten(), sample_points, method='linear',
                        fill_value=0.0)
    v_interp = griddata(np.column_stack([mesh_x, mesh_y]), full_v.flatten(), sample_points, method='linear',
                        fill_value=0.0)

    return u_interp, v_interp
    # try:
    #     # Direct coordinate evaluation
    #     result = model.java.result().create("res1", "Solution")
    #
    #     # Check if direct point evaluation is available in your COMSOL version
    #     u_at_points = result.evaluate("u", points=sample_points.T)
    #     v_at_points = result.evaluate("v", points=sample_points.T)
    #
    #     return np.array(u_at_points), np.array(v_at_points)
    # except Exception as e:
    #     logging.warning(f"Direct coordinate evaluation failed: {e}")
    #     # If direct evaluation fails, fall back to interpolation (using griddata)
    #     try:
    #         from scipy.interpolate import griddata
    #         full_u = model.evaluate('u')
    #         full_v = model.evaluate('v')
    #         mesh_coords = model.evaluate(['x', 'y'])
    #
    #         # Flatten mesh coordinates
    #         mesh_x, mesh_y = mesh_coords[0].flatten(), mesh_coords[1].flatten()
    #
    #         # Interpolate displacement values
    #         u_interp = griddata(np.column_stack([mesh_x, mesh_y]), full_u.flatten(), sample_points, method='linear',
    #                             fill_value=0.0)
    #         v_interp = griddata(np.column_stack([mesh_x, mesh_y]), full_v.flatten(), sample_points, method='linear',
    #                             fill_value=0.0)
    #
    #         return u_interp, v_interp
    #     except Exception as e:
    #         logging.error(f"Interpolation method also failed: {e}")
    #         return np.zeros((len(sample_points), N_EIG)), np.zeros((len(sample_points), N_EIG))


def save_displacements(u_sampled, v_sampled, freqs, kx, ky, run_id, seed):
    """
    Save the displacement data efficiently for the sampled points.
    """
    modes_path = f"modes_run_{run_id}_kx_{kx:.2f}_ky_{ky:.2f}_seed{seed}.npz"
    try:
        # Compress and save the displacement data for efficiency
        np.savez_compressed(
            OUTPUT_DIR / modes_path,
            u=u_sampled,
            v=v_sampled,
            frequencies=freqs,
            kx=kx,
            ky=ky
        )
        logging.info(f"Seed {seed}, k=({kx},{ky}) - Saved {len(u_sampled)} displacement samples to {modes_path}")
    except Exception as e:
        logging.error(f"Failed to save displacement data for seed {seed}, k=({kx},{ky}): {e}")
        # Save empty arrays as fallback
        np.savez_compressed(
            OUTPUT_DIR / modes_path,
            u=np.array([]),
            v=np.array([]),
            frequencies=freqs,
            kx=kx,
            ky=ky
        )


# ── DB helpers (updated schema) ──────────────────────────────────────

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
                     materials_json TEXT,
                     sampling_config TEXT)""")  # NEW: Added sampling config

    cur.execute("""CREATE TABLE IF NOT EXISTS eigenfrequencies (
                     result_id    INTEGER PRIMARY KEY,
                     run_id       INTEGER,
                     kx           REAL,
                     ky           REAL,
                     mode_number  INTEGER,
                     frequency_hz REAL,
                     modes_path   TEXT,
                     sample_points_path TEXT)""")  # NEW: Added sampling points path
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
                           materials_json: str, sampling_config: str):
    """Updates the run record with atlas, material, and sampling info."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""UPDATE simulations SET atlas_path = ?, materials_json = ?, sampling_config = ?
                   WHERE run_id = ?""", (atlas_path, materials_json, sampling_config, run_id))
    conn.commit()
    conn.close()


def _log_freqs_and_modes(db_file: Path, run_id: int, kx: float, ky: float,
                         freqs: List[float], modes_path: str, sample_points_path: str):
    """Logs frequencies and paths to mode shape and sampling point files."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany("""INSERT INTO eigenfrequencies
                       (run_id, kx, ky, mode_number, frequency_hz, modes_path, sample_points_path)
                       VALUES (?,?,?,?,?,?,?)""",
                    [(run_id, kx, ky, i + 1, f, modes_path, sample_points_path)
                     for i, f in enumerate(freqs)])
    conn.commit()
    conn.close()


# ── Util functions (unchanged) ─────────────────────────────────────

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


# ── Worker function (OPTIMIZED) ─────────────────────────────────────

def _build_and_solve(seed: int):
    client = None
    try:
        client = mph.start(cores=1)
        model = client.create(f"GridModel_seed{seed}")
        logging.info("Seed %d – building model", seed)

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

        # Material setup
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

        # Physics and boundary conditions
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

        # Mesh
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(5)
        mesh.feature().create("ftri1", "FreeTri").selection().geom("geom1", 2).all()
        mesh.run()

        # Study
        study = model.java.study().create("std1")
        eig = study.create("eig", "Eigenfrequency")
        eig.set("neigsactive", "on")
        eig.set("neigs", str(N_EIG))
        eig.set("shiftactive", "on")
        eig.set("shift", "0")
        eig.activate("solid", True)

        # Database logging
        mph_name = f"grid32_seed_{seed}.mph"
        run_id = _log_run_and_get_id(DB_PATH, seed, mph_name)

        # Save material atlas
        atlas_path = f"atlas_run_{run_id}.npy"
        np.save(OUTPUT_DIR / atlas_path, mmap)

        # NEW: Generate sampling points based on configuration
        sample_points = get_sampling_points(SAMPLING_CONFIG["method"], mmap)
        sample_points_path = f"sample_points_run_{run_id}.npy"
        np.save(OUTPUT_DIR / sample_points_path, sample_points)

        # Update database with paths and config
        materials_json = json.dumps(MATERIALS)
        sampling_config_json = json.dumps(SAMPLING_CONFIG)
        _update_run_with_paths(DB_PATH, run_id, atlas_path, materials_json, sampling_config_json)

        logging.info("Seed %d - Generated %d sampling points using %s method",
                     seed, len(sample_points), SAMPLING_CONFIG["method"])

        # Solve for each k-point
        k_points = generate_kgrid(N_K, K_MAX)
        for kx, ky in k_points:
            model.parameter("kx", str(kx))
            model.parameter("ky", str(ky))
            study.run()

            freqs = [float(np.real(v)) for v in model.evaluate("freq")]

            modes_path = f"modes_run_{run_id}_kx_{kx:.2f}_ky_{ky:.2f}_seed{seed}.npz"

            sample_points = get_sampling_points(SAMPLING_CONFIG["method"], mmap)

            # Efficient displacement evaluation for sampling points only
            u_sampled, v_sampled = evaluate_displacements_at_points(model, sample_points)

            # Save displacements
            save_displacements(u_sampled, v_sampled, freqs, kx, ky, run_id, seed)

            # Logging frequencies and modes
            _log_freqs_and_modes(DB_PATH, run_id, kx, ky, freqs, modes_path, sample_points_path)
            # NEW: Efficient displacement evaluation at sampling points only

            # try:
            #     u_sampled, v_sampled = evaluate_displacements_at_points(model, sample_points)
            #
            #     # Save only the sampled displacements
            #     np.savez_compressed(
            #         OUTPUT_DIR / modes_path,
            #         u=u_sampled,
            #         v=v_sampled,
            #         frequencies=freqs,
            #         kx=kx,
            #         ky=ky
            #     )
            #
            #     logging.info("Seed %d, k=(%0.2f,%0.2f) - Saved %d displacement samples",
            #                  seed, kx, ky, len(sample_points))
            #
            # except Exception as e:
            #     logging.error("Seed %d, k=(%0.2f,%0.2f) - Failed to sample displacements: %s",
            #                   seed, kx, ky, e)
            #     # Create empty file to maintain consistency
            #     np.savez_compressed(
            #         OUTPUT_DIR / modes_path,
            #         u=np.array([]),
            #         v=np.array([]),
            #         frequencies=freqs,
            #         kx=kx,
            #         ky=ky
            #     )
            #
            # _log_freqs_and_modes(DB_PATH, run_id, kx, ky, freqs, modes_path, sample_points_path)

        logging.info("Seed %d finished", seed)

    except Exception:
        logging.exception("Seed %d failed", seed)
    finally:
        if client:
            client.clear()


# ── Main entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    _setup_db(DB_PATH)

    # Log sampling configuration
    logging.info("Using sampling method: %s", SAMPLING_CONFIG["method"])
    if SAMPLING_CONFIG["method"] == "structured_grid":
        logging.info("Grid resolution: %d x %d = %d points",
                     SAMPLING_CONFIG["grid_resolution"],
                     SAMPLING_CONFIG["grid_resolution"],
                     SAMPLING_CONFIG["grid_resolution"] ** 2)

    logging.info("Launching %d parallel workers…", N_SEEDS)
    seeds_to_run = range(N_SEEDS)
    with mp.Pool(processes=min(mp.cpu_count(), N_SEEDS)) as pool:
        pool.map(_build_and_solve, seeds_to_run)
        pool.close()
        pool.join()
    logging.info("All workers finished.")
