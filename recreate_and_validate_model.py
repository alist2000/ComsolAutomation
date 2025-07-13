#!/usr/bin/env python3
"""
recreate_and_validate_model.py  –  validate ω, u, v for one DB row
Compatible with grid_model.py V5.5.
Usage
-----
python recreate_and_validate_model.py 698
python recreate_and_validate_model.py       # prompts for id
"""

from __future__ import annotations
import argparse, logging, sqlite3, sys, os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import mph
from numpy.linalg import norm

# ── CONFIG ─────────────────────────────────────────────────────────
A, GRID_SIZE, N_MODES = 1.0, 32, 10
TOL_FREQ = 1e-8
TOL_DISP_MESH = 1e-12  # when mesh & phase aligned
TOL_DISP_REBL = 1e-4  # when we had to rebuild

# Point to the correct results directory and database file
RESULTS_DIR = Path("results_v5.5")
DB_FILE = RESULTS_DIR / "simulation_results_v5.5.db"
MPH_PATTERN = "grid32_seed_{seed}.mph"

SOIL = dict(E=20e6, nu=0.30, rho=1800.0)
CONC = dict(E=20e9, nu=0.20, rho=2400.0)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")


# ── helper functions (unchanged from generator) ────────────────────
def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    return np.vstack((np.flipud(np.hstack((np.fliplr(quad), quad))),
                      np.hstack((np.fliplr(quad), quad))))


def _outer_bnd_lists(size: int) -> Tuple[List[int], List[int]]:
    if size != 32: raise ValueError("GRID_SIZE must be 32.")
    pbc_x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
             41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, *range(2081, 2113)]
    pbc_y = [2, 65, 67, 130, 132, 195, 197, 260, 262, 325, 327, 390, 392, 455, 457,
             520, 522, 585, 587, 650, 652, 715, 717, 780, 782, 845, 847, 910, 912,
             975, 977, 1040, 1042, 1105, 1107, 1170, 1172, 1235, 1237, 1300, 1302,
             1365, 1367, 1430, 1432, 1495, 1497, 1560, 1562, 1625, 1627, 1690, 1692,
             1755, 1757, 1820, 1822, 1885, 1887, 1950, 1952, 2015, 2017, 2080]
    return pbc_x, pbc_y


# ── DB access ──────────────────────────────────────────────────────
def fetch_record(rid: int) -> Dict[str, Any]:
    with sqlite3.connect(DB_FILE) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        row = cur.execute("""SELECT e.run_id,e.kx,e.ky,e.displacement_path,
                                  s.seed
                           FROM   eigenfrequencies e
                           JOIN   simulations s USING(run_id)
                           WHERE  e.result_id=?""", (rid,)).fetchone()
        if row is None:
            sys.exit(f"❌ result_id {rid} not found.")
        freqs = [r[0] for r in cur.execute("""SELECT frequency_hz
                                            FROM   eigenfrequencies
                                            WHERE  run_id=? AND ABS(kx-? )<1e-9
                                                  AND ABS(ky-? )<1e-9
                                            ORDER  BY mode_number""",
                                           (row['run_id'], row['kx'], row['ky']))]
    return dict(row) | {"freqs": np.asarray(freqs, float)}


# ── phase alignment helper ─────────────────────────────────────────
def align_phase(u_ref: np.ndarray, u_new: np.ndarray) -> np.ndarray:
    """Rotate u_new by a scalar phase so it best matches u_ref."""
    phi = np.angle(np.vdot(u_ref, u_new))
    return u_new * np.exp(1j * phi)


# ★★★ NEW: Copied from grid_model.py to handle data conversion ★★★
def _convert_txt_to_npz(txt_path: Path):
    if not txt_path.exists():
        logging.error(f"Cannot convert non-existent file: {txt_path}")
        return
    column_names = ['x', 'y']
    for i in range(1, N_MODES + 1):
        column_names.append(f'u_mode_{i}')
        column_names.append(f'v_mode_{i}')
    df = pd.read_csv(txt_path, comment='%', sep=r'\s+', header=None, names=column_names)
    coords = df[['x', 'y']].to_numpy(dtype=np.float32)
    displacements = np.empty((coords.shape[0], N_MODES, 2), dtype=np.complex64)
    for i in range(N_MODES):
        u_complex = df[f'u_mode_{i + 1}'].astype(str).apply(lambda v: complex(v.replace('i', 'j'))).to_numpy()
        v_complex = df[f'v_mode_{i + 1}'].astype(str).apply(lambda v: complex(v.replace('i', 'j'))).to_numpy()
        displacements[:, i, 0] = u_complex
        displacements[:, i, 1] = v_complex
    npz_path = txt_path.with_suffix('.npz')
    np.savez_compressed(npz_path, coordinates=coords, displacements=displacements)
    os.remove(txt_path)
    return npz_path


# ★★★ NEW: Helper function to export model data to NPZ ★★★
def export_to_npz(model: mph.Model, output_dir: Path, base_name: str) -> Path:
    """Exports displacement data from a solved model to an NPZ file."""
    cpt_tag, export_tag = "temp_cpt_export", "temp_data_export"
    try:
        model.java.result().dataset().remove(cpt_tag)
        model.java.result().export().remove(export_tag)
    except Exception:
        pass  # OK if they don't exist

    # Create 32x32 grid dataset
    cpt_node = model.java.result().dataset().create(cpt_tag, "CutPoint2D")
    grid_step = f"a/({GRID_SIZE})"
    cpt_node.set("method", "grid")
    cpt_node.set("gridx", f"range(-a/2,{grid_step},a/2)")
    cpt_node.set("gridy", f"range(-a/2,{grid_step},a/2)")
    cpt_node.run()

    # Create and configure data export node
    export_node = model.java.result().export().create(export_tag, "Data")
    export_node.set("data", cpt_tag)
    export_node.setIndex("expr", "u", 0)
    export_node.setIndex("expr", "v", 1)

    # Export to a temporary .txt file
    txt_filepath = output_dir / f"{base_name}.txt"
    export_node.set("filename", str(txt_filepath.resolve()).replace('\\', '\\\\'))
    export_node.run()
    logging.info(f"Exported temporary displacement data to {txt_filepath.name}")

    # Convert .txt to .npz and clean up
    npz_filepath = _convert_txt_to_npz(txt_filepath)
    return npz_filepath


# ── COMSOL routines (UPDATED) ──────────────────────────────────────
def load_and_solve(mph_path: Path, kx: float, ky: float, result_id: int):
    client = mph.start(cores=1)
    model = client.load(str(mph_path))
    model.parameter("kx", str(kx));
    model.parameter("ky", str(ky))
    tag = model.java.study().tags()[0]
    model.java.study(tag).run()
    freqs = np.asarray(model.evaluate("freq"))
    # ★★★ CHANGE: Export results to NPZ instead of using evaluate() ★★★
    base_name = f"recreated_result_{result_id}_kx_{kx:.4f}_ky_{ky:.4f}"
    new_npz_path = export_to_npz(model, RESULTS_DIR, base_name)
    return freqs, new_npz_path, model


def build_and_solve(seed: int, kx: float, ky: float, result_id: int):
    client = mph.start(cores=1)
    model = client.create(f"build_seed{seed}")
    model.parameter("a", f"{A}[m]");
    model.parameter("kx", str(kx));
    model.parameter("ky", str(ky))
    comp = model.java.component().create("comp1", True)
    geom = comp.geom().create("geom1", 2)
    cell, start = A / GRID_SIZE, -A / 2
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            sq = geom.create(f"s{i}_{j}", "Square")
            sq.set("size", cell);
            sq.set("pos", [start + j * cell, start + i * cell])
    geom.run()
    # materials
    m_s = comp.material().create("soil", "Common");
    pg = m_s.propertyGroup("def")
    pg.set("youngsmodulus", f"{SOIL['E']}[Pa]");
    pg.set("poissonsratio", str(SOIL['nu']));
    pg.set("density", f"{SOIL['rho']}[kg/m^3]")
    m_c = comp.material().create("con", "Common");
    pgc = m_c.propertyGroup("def")
    pgc.set("youngsmodulus", f"{CONC['E']}[Pa]");
    pgc.set("poissonsratio", str(CONC['nu']));
    pgc.set("density", f"{CONC['rho']}[kg/m^3]")
    mmap = create_symmetric_material_map(GRID_SIZE, seed).flatten();
    ids = np.arange(1, GRID_SIZE ** 2 + 1)
    m_s.selection().set(tuple(ids[mmap == 0]));
    m_c.selection().set(tuple(ids[mmap == 1]))
    solid = comp.physics().create("solid", "SolidMechanics", "geom1")
    bx, by = _outer_bnd_lists(GRID_SIZE)
    pbcx = solid.create("pbcx", "PeriodicCondition", 1);
    pbcx.selection().set(bx);
    pbcx.set("PeriodicType", "Floquet");
    pbcx.set("kFloquet", ["kx", "0", "0"])
    pbcy = solid.create("pbcy", "PeriodicCondition", 1);
    pbcy.selection().set(by);
    pbcy.set("PeriodicType", "Floquet");
    pbcy.set("kFloquet", ["0", "ky", "0"])
    mesh = comp.mesh().create("mesh1", "geom1")
    mesh.autoMeshSize(5);
    mesh.feature().create("f1", "FreeTri").selection().geom("geom1", 2).all();
    mesh.run()
    study = model.java.study().create("std1");
    eig = study.create("eig", "Eigenfrequency")
    eig.set("neigsactive", "on");
    eig.set("neigs", str(N_MODES));
    eig.set("shiftactive", "on");

    # Change solver shift from 0 to 1 to match the original study
    eig.set("shift", "1");

    eig.activate("solid", True)
    study.run()
    freqs = np.asarray(model.evaluate("freq"))

    base_name = f"recreated_result_{result_id}_kx_{kx:.4f}_ky_{ky:.4f}"
    new_npz_path = export_to_npz(model, RESULTS_DIR, base_name)
    return freqs, new_npz_path, model


# ── MAIN (UPDATED) ─────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_id", nargs="?", type=int)
    args = ap.parse_args()
    if args.result_id is None:
        try:
            args.result_id = int(input("Enter result_id to validate: "))
        except ValueError:
            sys.exit("❌ integer required.")

    rec = fetch_record(args.result_id)
    logging.info("run_id=%d  seed=%d  kx=%g  ky=%g",
                 rec["run_id"], rec["seed"], rec["kx"], rec["ky"])

    # Load reference data
    ref_data_path = Path(rec["displacement_path"])
    ref_data = np.load(ref_data_path)
    u_ref = ref_data["displacements"][:, :, 0]
    v_ref = ref_data["displacements"][:, :, 1]
    f_ref = rec["freqs"]
    ref_data.close()

    new_npz_path = None
    # try to reuse mesh
    mph_path = RESULTS_DIR / MPH_PATTERN.format(seed=rec["seed"])
    if mph_path.exists():
        logging.info("Using original mesh: %s", mph_path.name)
        f_new, new_npz_path, model = load_and_solve(mph_path, rec["kx"], rec["ky"], args.result_id)

        # Load the newly created data for comparison
        new_data = np.load(new_npz_path)
        u_new = new_data["displacements"][:, :, 0]
        v_new = new_data["displacements"][:, :, 1]
        new_data.close()

        # Phase-align new eigenvector to reference
        u_new = align_phase(u_ref, u_new)
        v_new = align_phase(v_ref, v_new)

        tol_disp = TOL_DISP_MESH
        rel_u = norm(u_new - u_ref) / norm(u_ref)
        rel_v = norm(v_new - v_ref) / norm(v_ref)
    else:
        logging.info("Original .mph not found – rebuilding cell.")
        f_new, new_npz_path, model = build_and_solve(rec["seed"], rec["kx"], rec["ky"], args.result_id)

        # Load the newly created data for comparison
        new_data = np.load(new_npz_path)
        u_new = new_data["displacements"][:, :, 0]
        v_new = new_data["displacements"][:, :, 1]
        new_data.close()

        tol_disp = TOL_DISP_REBL
        rel_u = norm(np.abs(u_new) - np.abs(u_ref)) / norm(np.abs(u_ref))
        rel_v = norm(np.abs(v_new) - np.abs(v_ref)) / norm(np.abs(v_ref))

    # Clean up the newly created NPZ file
    if new_npz_path and os.path.exists(new_npz_path):
        os.remove(new_npz_path)
        logging.info(f"Cleaned up temporary file: {new_npz_path.name}")

    rel_f = norm(f_new - f_ref) / norm(f_ref)
    logging.info("Δfreq = %.3e  Δu = %.3e  Δv = %.3e  (tol u,v = %.1e)",
                 rel_f, rel_u, rel_v, tol_disp)

    out_mph = RESULTS_DIR / f"recreated_result_{args.result_id}_kx_{rec['kx']:.3f}_ky_{rec['ky']:.3f}.mph"
    model.save(str(out_mph));
    logging.info("Saved model to %s", out_mph)

    if rel_f < TOL_FREQ and rel_u < tol_disp and rel_v < tol_disp:
        print("🎉  Validation PASSED")
        sys.exit(0)
    else:
        print("❌  Validation FAILED")
        sys.exit(1)


# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
