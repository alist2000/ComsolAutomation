import mph
import numpy as np
import logging
from pathlib import Path

# --- Configuration ---
GRID_SIZE = 32
logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)s · %(message)s")

# ── 1. Match "First Code" logic for material map ──────────────────
def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))

# ── 2. Match "First Code" logic for Boundary IDs ──────────────────
def _get_boundary_ids() -> tuple:
    """
    Returns the lists of Face IDs for boundaries, exactly like 
    _outer_bnd_lists() in your first code.
    
    ACTION REQUIRED: 
    1. Run this script once. It will save the .mph file but fail to solve.
    2. Open the .mph file in COMSOL.
    3. Find the Face IDs for:
       - Periodic Condition (Front/Back faces)
       - Low Reflecting Boundary (Left/Right/Bottom faces)
    4. Update the lists below with those numbers.
    """
    
    # Placeholder IDs - UPDATE THESE AFTER CHECKING GEOMETRY
    pbc_faces = [1,6,7]          # Front and Back Faces
    lrb_faces = [2, 3, 4]    # Left, Right, and Bottom Faces
    
    return pbc_faces, lrb_faces

# ── 3. Main Build Function ────────────────────────────────────────
def build_model_exact_method(
    seed: int,
    a: float,
    h_pile: float,
    h_model: float,
    num_piles: int,
    dist_gap: float,
    len_plane: float,
    total_len_x: float,
    materials: dict,
    freq_range: str = "range(0, 5, 10)"
):
    client = mph.start()
    try:
        model_name = f"Grid3D_Seed{seed}_ManualMethod"
        model = client.create(model_name)
        logging.info(f"Building {model_name}...")

        # Setup Parameters
        model.parameter("ky", "0[1/m]") 

        comp = model.java.component().create("comp1", True)
        geom = comp.geom().create("geom1", 3)
        
        # --- Geometry Generation ---
        h_bottom = h_model - h_pile
        
        # 1. Bottom Soil
        blk_bot = geom.create("blk_bottom", "Block")
        blk_bot.set("size", [total_len_x, a, h_bottom])
        blk_bot.set("pos", [0, 0, 0])
        
        # 2. Grid Generation
        mmap = create_symmetric_material_map(GRID_SIZE, seed)
        cell_dim = a / GRID_SIZE
        len_piles = num_piles * a
        
        # Top Remainder (Soil after piles)
        rem_len = total_len_x - len_piles
        if rem_len > 0:
            blk_rem = geom.create("blk_top_remainder", "Block")
            blk_rem.set("size", [rem_len, a, h_pile])
            blk_rem.set("pos", [len_piles, 0, h_bottom])

        # Track IDs for Materials (standard loop)
        soil_ids = [1]
        current_id = 2
        if rem_len > 0:
            soil_ids.append(2)
            current_id = 3
        conc_ids = []

        for n in range(num_piles):
            x_offset = n * a
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    y_pos = i * cell_dim
                    x_pos = x_offset + (j * cell_dim)
                    
                    blk = geom.create(f"c_{n}_{i}_{j}", "Block")
                    blk.set("size", [cell_dim, cell_dim, h_pile])
                    blk.set("pos", [x_pos, y_pos, h_bottom])
                    
                    if mmap[i, j] == 1:
                        conc_ids.append(current_id)
                    else:
                        soil_ids.append(current_id)
                    current_id += 1

        # 3. Output Plane (Work Plane)
        wp = geom.create("wp1", "WorkPlane")
        wp.set("planetype", "quick")
        wp.set("quickplane", "zx")
        wp.set("quicky", str(a / 2))
        
        rect = wp.geom().create("r1", "Rectangle")
        plane_x_start = len_piles + dist_gap
        rect.set("pos", [h_bottom, plane_x_start]) 
        rect.set("size", [h_pile, len_plane])

        geom.run()
        logging.info("Geometry built.")

        # --- Materials (Standard Selection) ---
        soil_mat = comp.material().create("mat_soil", "Common")
        conc_mat = comp.material().create("mat_concrete", "Common")
        
        for mat_obj, key in [(soil_mat, "soil"), (conc_mat, "concrete")]:
            pg = mat_obj.propertyGroup("def")
            pg.set("youngsmodulus", f"{materials[key]['youngs_modulus']}[Pa]")
            pg.set("poissonsratio", str(materials[key]['poissons_ratio']))
            pg.set("density", f"{materials[key]['density']}[kg/m^3]")

        soil_mat.selection().set(soil_ids)
        conc_mat.selection().set(conc_ids)

        # --- Physics (Using "First Code" Method) ---
        solid = comp.physics().create("solid", "SolidMechanics", "geom1")
        
        # 1. Get IDs from function (User defined)
        pbc_faces, lrb_faces = _get_boundary_ids()
        
        # 2. Set Low Reflecting Boundary
        lrb = solid.create("lrb1", "LowReflectingBoundary", 2)
        # Using exact method from first code: .selection().set(list)
        lrb.selection().set(lrb_faces)
        
        # 3. Set Periodic Condition
        pbc = solid.create("pbc1", "PeriodicCondition", 2)
        # Using exact method from first code: .selection().set(list)
        pbc.selection().set(pbc_faces)
        pbc.set("PeriodicType", "Floquet")
        pbc.set("kFloquet", ["0", "ky", "0"])

        # --- Mesh & Study ---
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(6) 
        mesh.run()

        study = model.java.study().create("std1")
        freq = study.create("freq", "Frequency")
        freq.set("plist", freq_range)
        
        output_file = Path(f"{model_name}.mph").absolute()
        model.save(str(output_file))
        logging.info(f"Model saved to {output_file}")
        
        logging.info("Solving...")
        study.run()
        logging.info("Done.")

    except Exception as e:
        logging.error(f"Execution failed: {e}")
        logging.info("REMINDER: Check your _get_boundary_ids() list matches the geometry IDs.")
    finally:
        client.clear()

if __name__ == "__main__":
    MATS = {
        "soil": {"youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0},
        "concrete": {"youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0},
    }
    
    build_model_exact_method(
        seed=42,
        a=1.0,
        h_pile=5.0,
        h_model=8.5,
        num_piles=10,
        dist_gap=1.5,
        len_plane=1.5,
        total_len_x=13.5,
        materials=MATS
    )