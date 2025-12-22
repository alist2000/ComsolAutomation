import mph
import numpy as np
import logging
from pathlib import Path

# --- Configuration ---
GRID_SIZE = 32
logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)s · %(message)s")

def create_symmetric_material_map(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quad = rng.integers(0, 2, size=(size // 2, size // 2))
    quad = np.triu(quad) + np.triu(quad, 1).T
    top = np.hstack((np.fliplr(quad), quad))
    return np.vstack((np.flipud(top), top))

def build_model_fast(seed, a, h_pile, h_model, num_piles, dist_gap, len_plane, total_len_x, materials):
    client = mph.start()
    try:
        model_name = f"Grid3D_Seed{seed}_Final"
        model = client.create(model_name)
        logging.info(f"Building {model_name}...")

        comp = model.java.component().create("comp1", True)
        geom = comp.geom().create("geom1", 3)
        
        # --- 1. Base Geometry ---
        h_bottom = h_model - h_pile
        cell_dim = a / GRID_SIZE
        
        # Create the main volumes
        geom.create("blk_bottom", "Block").set("size", [total_len_x, a, h_bottom])
        
        pile_zone = geom.create("pile_zone", "Block")
        pile_zone.set("size", [num_piles * a, a, h_pile])
        pile_zone.set("pos", [0, 0, h_bottom])

        rem_len = total_len_x - (num_piles * a)
        if rem_len > 0:
            geom.create("blk_rem", "Block").set("size", [rem_len, a, h_pile]).set("pos", [num_piles * a, 0, h_bottom])

        # --- 2. Slicing with Workplanes ---
        wp_list = []
        # X-Slices
        for i in range(1, num_piles * GRID_SIZE):
            name = f"wp_x_{i}"
            wp = geom.create(name, "WorkPlane")
            wp.set("quickplane", "yz").set("quickx", i * cell_dim)
            wp_list.append(name)
        # Y-Slices
        for j in range(1, GRID_SIZE):
            name = f"wp_y_{j}"
            wp = geom.create(name, "WorkPlane")
            wp.set("quickplane", "zx").set("quicky", j * cell_dim)
            wp_list.append(name)

        # --- 3. Partitioning (Universal Syntax) ---
        # We use the feature tag directly to set the selection
        part = geom.create("part1", "PartitionDomains")
        # Try "input" - if this fails, COMSOL version requires "objs" 
        # but "input" is the standard for PartitionDomains selections
        try:
            part.selection("input").set("pile_zone")
        except:
            part.set("objs", "pile_zone")
            
        part.set("partitionwith", "workplane")
        part.set("workplane", wp_list)

        geom.run()
        logging.info("Partitioning complete.")

        # --- 4. Material Mapping ---
        mmap = create_symmetric_material_map(GRID_SIZE, seed)
        soil_ids, conc_ids = [], []
        
        # Access domains from the built geometry
        all_doms = comp.geom("geom1").getDomainData("dom")
        for d_id in all_doms:
            cx = comp.geom("geom1").getDomainData("dom", d_id, "centerx")
            cy = comp.geom("geom1").getDomainData("dom", d_id, "centery")
            cz = comp.geom("geom1").getDomainData("dom", d_id, "centerz")
            
            # Logic: Bottom layer or beyond the pile length is always soil
            if cz < (h_bottom + 1e-6) or cx > (num_piles * a - 1e-6):
                soil_ids.append(int(d_id))
            else:
                # Calculate grid index based on position
                ix = int((cx % a) // cell_dim)
                iy = int(cy // cell_dim)
                # Clamp indices
                ix = max(0, min(ix, GRID_SIZE - 1))
                iy = max(0, min(iy, GRID_SIZE - 1))
                
                if mmap[iy, ix] == 1:
                    conc_ids.append(int(d_id))
                else:
                    soil_ids.append(int(d_id))

        # Create Materials
        soil_mat = comp.material().create("mat_soil", "Common")
        conc_mat = comp.material().create("mat_concrete", "Common")
        
        for mat_obj, key in [(soil_mat, "soil"), (conc_mat, "concrete")]:
            pg = mat_obj.propertyGroup("def")
            pg.set("youngsmodulus", f"{materials[key]['youngs_modulus']}[Pa]")
            pg.set("poissonsratio", str(materials[key]['poissons_ratio']))
            pg.set("density", f"{materials[key]['density']}[kg/m^3]")

        soil_mat.selection().set([int(x) for x in soil_ids])
        conc_mat.selection().set([int(x) for x in conc_ids])

        # --- 5. Physics & Automatic Selections ---
        solid = comp.physics().create("solid", "SolidMechanics", "geom1")
        
        # Periodic Conditions
        pbc = solid.create("pbc1", "PeriodicCondition", 2)
        pbc.set("PeriodicType", "Floquet")
        
        # Create Coordinate-based selections for Y-boundaries
        sel_y0 = comp.selection().create("sel_y0", "Box")
        sel_y0.set("entitydim", 2)
        sel_y0.set("ymin", -1e-4).set("ymax", 1e-4)
        
        sel_ya = comp.selection().create("sel_ya", "Box")
        sel_ya.set("entitydim", 2)
        sel_ya.set("ymin", a - 1e-4).set("ymax", a + 1e-4)
        
        # Combine the face IDs found by the boxes
        pbc_faces = list(comp.selection("sel_y0").entities()) + list(comp.selection("sel_ya").entities())
        pbc.selection().set([int(f) for f in pbc_faces])

        # --- 6. Finalizing ---
        mesh = comp.mesh().create("mesh1", "geom1")
        mesh.autoMeshSize(8) # Extremely coarse for fast verification
        
        output_path = Path(f"{model_name}.mph").absolute()
        model.save(str(output_path))
        logging.info(f"Model saved: {output_path}")

    except Exception as e:
        logging.error(f"Failed at: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        client.clear()

if __name__ == "__main__":
    MATS = {
        "soil": {"youngs_modulus": 20e6, "poissons_ratio": 0.30, "density": 1800.0},
        "concrete": {"youngs_modulus": 20e9, "poissons_ratio": 0.20, "density": 2400.0},
    }
    build_model_fast(seed=42, a=1.0, h_pile=5.0, h_model=8.5, num_piles=10, 
                     dist_gap=1.5, len_plane=1.5, total_len_x=13.5, materials=MATS)