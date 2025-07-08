#!/usr/bin/env python3
"""
recreate_and_validate_model.py  –  validate ω, u, v for one DB row

Usage
-----
python recreate_and_validate_model.py 698
python recreate_and_validate_model.py       # prompts for id
"""

from __future__ import annotations
import argparse, logging, sqlite3, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import mph
from numpy.linalg import norm

# ── CONFIG ─────────────────────────────────────────────────────────
A, GRID_SIZE, N_MODES = 1.0, 32, 10
TOL_FREQ      = 1e-8
TOL_DISP_MESH = 1e-12        # when mesh & phase aligned
TOL_DISP_REBL = 1e-4         # when we had to rebuild

RESULTS_DIR = Path("results_v5.1")
DB_FILE     = RESULTS_DIR / "simulation_results_v5.1.db"
MPH_PATTERN = "grid32_seed_{seed}.mph"

SOIL = dict(E=20e6, nu=0.30, rho=1800.0)
CONC = dict(E=20e9, nu=0.20, rho=2400.0)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(levelname)s · %(message)s")

# ── helper functions (unchanged from generator) ────────────────────
def create_symmetric_material_map(size:int, seed:int)->np.ndarray:
    rng  = np.random.default_rng(seed)
    quad = rng.integers(0,2,size=(size//2,size//2))
    quad = np.triu(quad)+np.triu(quad,1).T
    return np.vstack((np.flipud(np.hstack((np.fliplr(quad), quad))),
                      np.hstack((np.fliplr(quad), quad))))

def _outer_bnd_lists(size:int)->Tuple[List[int],List[int]]:
    if size!=32: raise ValueError("GRID_SIZE must be 32.")
    pbc_x=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,
           41,43,45,47,49,51,53,55,57,59,61,63,*range(2081,2113)]
    pbc_y=[2,65,67,130,132,195,197,260,262,325,327,390,392,455,457,
           520,522,585,587,650,652,715,717,780,782,845,847,910,912,
           975,977,1040,1042,1105,1107,1170,1172,1235,1237,1300,1302,
           1365,1367,1430,1432,1495,1497,1560,1562,1625,1627,1690,1692,
           1755,1757,1820,1822,1885,1887,1950,1952,2015,2017,2080]
    return pbc_x,pbc_y

# ── DB access ──────────────────────────────────────────────────────
def fetch_record(rid:int)->Dict[str,Any]:
    with sqlite3.connect(DB_FILE) as con:
        con.row_factory=sqlite3.Row
        cur=con.cursor()
        row=cur.execute("""SELECT e.run_id,e.kx,e.ky,e.modes_path,
                                  s.seed
                           FROM   eigenfrequencies e
                           JOIN   simulations s USING(run_id)
                           WHERE  e.result_id=?""",(rid,)).fetchone()
        if row is None:
            sys.exit(f"❌ result_id {rid} not found.")
        freqs=[r[0] for r in cur.execute("""SELECT frequency_hz
                                            FROM   eigenfrequencies
                                            WHERE  run_id=? AND ABS(kx-? )<1e-9
                                                  AND ABS(ky-? )<1e-9
                                            ORDER  BY mode_number""",
                                         (row['run_id'],row['kx'],row['ky']))]
    return dict(row)|{"freqs":np.asarray(freqs,float)}

# ── phase alignment helper ─────────────────────────────────────────
def align_phase(u_ref:np.ndarray, u_new:np.ndarray)->np.ndarray:
    """Rotate u_new by a scalar phase so it best matches u_ref."""
    # projection gives phase that minimises ‖u_ref - e^{iφ} u_new‖
    phi = np.angle(np.vdot(u_ref, u_new))
    return u_new * np.exp(1j*phi)

# ── COMSOL routines ────────────────────────────────────────────────
def load_and_solve(mph_path:Path,kx:float,ky:float):
    client=mph.start(cores=1)
    model=client.load(str(mph_path))
    model.parameter("kx",str(kx)); model.parameter("ky",str(ky))
    tag=model.java.study().tags()[0]
    model.java.study(tag).run()
    freqs=np.asarray(model.evaluate("freq"))
    u=np.asarray(model.evaluate("u"))
    v=np.asarray(model.evaluate("v"))
    return freqs,u,v,model

def build_and_solve(seed:int,kx:float,ky:float):
    client=mph.start(cores=1)
    model=client.create(f"build_seed{seed}")
    model.parameter("a",f"{A}[m]"); model.parameter("kx",str(kx)); model.parameter("ky",str(ky))
    comp=model.java.component().create("comp1",True)
    geom=comp.geom().create("geom1",2)
    cell,start=A/GRID_SIZE,-A/2
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            sq=geom.create(f"s{i}_{j}","Square")
            sq.set("size",cell); sq.set("pos",[start+j*cell,start+i*cell])
    geom.run()
    # materials
    m_s=comp.material().create("soil","Common"); pg=m_s.propertyGroup("def")
    pg.set("youngsmodulus",f"{SOIL['E']}[Pa]"); pg.set("poissonsratio",str(SOIL['nu'])); pg.set("density",f"{SOIL['rho']}[kg/m^3]")
    m_c=comp.material().create("con","Common"); pgc=m_c.propertyGroup("def")
    pgc.set("youngsmodulus",f"{CONC['E']}[Pa]"); pgc.set("poissonsratio",str(CONC['nu'])); pgc.set("density",f"{CONC['rho']}[kg/m^3]")
    mmap=create_symmetric_material_map(GRID_SIZE,seed).flatten(); ids=np.arange(1,GRID_SIZE**2+1)
    m_s.selection().set(tuple(ids[mmap==0])); m_c.selection().set(tuple(ids[mmap==1]))
    solid=comp.physics().create("solid","SolidMechanics","geom1")
    bx,by=_outer_bnd_lists(GRID_SIZE)
    pbcx=solid.create("pbcx","PeriodicCondition",1); pbcx.selection().set(bx); pbcx.set("PeriodicType","Floquet"); pbcx.set("kFloquet",["kx","0","0"])
    pbcy=solid.create("pbcy","PeriodicCondition",1); pbcy.selection().set(by); pbcy.set("PeriodicType","Floquet"); pbcy.set("kFloquet",["0","ky","0"])
    mesh=comp.mesh().create("mesh1","geom1")
    mesh.autoMeshSize(5); mesh.feature().create("f1","FreeTri").selection().geom("geom1",2).all(); mesh.run()
    study=model.java.study().create("std1"); eig=study.create("eig","Eigenfrequency")
    eig.set("neigsactive","on"); eig.set("neigs",str(N_MODES)); eig.set("shiftactive","on"); eig.set("shift","0"); eig.activate("solid",True)
    study.run()
    freqs=np.asarray(model.evaluate("freq"))
    u=np.asarray(model.evaluate("u"))
    v=np.asarray(model.evaluate("v"))
    return freqs,u,v,model

# ── MAIN ───────────────────────────────────────────────────────────
def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("result_id",nargs="?",type=int)
    args=ap.parse_args()
    if args.result_id is None:
        try: args.result_id=int(input("Enter result_id to validate: "))
        except ValueError: sys.exit("❌ integer required.")

    rec=fetch_record(args.result_id)
    logging.info("run_id=%d  seed=%d  kx=%g  ky=%g",
                 rec["run_id"],rec["seed"],rec["kx"],rec["ky"])

    # reference data
    ref=np.load(RESULTS_DIR/rec["modes_path"])
    u_ref,v_ref,f_ref=ref["u"],ref["v"],rec["freqs"]

    # try to reuse mesh
    mph_path=RESULTS_DIR/MPH_PATTERN.format(seed=rec["seed"])
    if mph_path.exists():
        logging.info("Using original mesh: %s", mph_path.name)
        f_new,u_new,v_new,model = load_and_solve(mph_path, rec["kx"], rec["ky"])

        # phase-align new eigenvector to reference
        u_new = align_phase(u_ref, u_new)
        v_new = align_phase(v_ref, v_new)

        tol_disp = TOL_DISP_MESH
        rel_u = norm(u_new - u_ref) / norm(u_ref)
        rel_v = norm(v_new - v_ref) / norm(v_ref)
    else:
        logging.info("Original .mph not found – rebuilding cell.")
        f_new,u_new,v_new,model = build_and_solve(rec["seed"], rec["kx"], rec["ky"])
        tol_disp = TOL_DISP_REBL
        rel_u = norm(np.abs(u_new) - np.abs(u_ref)) / norm(np.abs(u_ref))
        rel_v = norm(np.abs(v_new) - np.abs(v_ref)) / norm(np.abs(v_ref))

    rel_f = norm(f_new - f_ref) / norm(f_ref)
    logging.info("Δfreq = %.3e  Δu = %.3e  Δv = %.3e  (tol u,v = %.1e)",
                 rel_f, rel_u, rel_v, tol_disp)

    out = RESULTS_DIR / f"recreated_result_{args.result_id}_kx_{rec['kx']:.3f}_ky_{rec['ky']:.3f}.mph"
    model.save(str(out)); logging.info("Saved model to %s", out)

    if rel_f < TOL_FREQ and rel_u < tol_disp and rel_v < tol_disp:
        print("🎉  Validation PASSED")
        sys.exit(0)
    else:
        print("❌  Validation FAILED")
        sys.exit(1)

# ───────────────────────────────────────────────────────────────────
if __name__=="__main__":
    main()
