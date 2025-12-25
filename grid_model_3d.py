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
    pbc_faces = [2, 5, 132, 135, 264, 393, 520, 522, 649, 651, 778, 780, 907, 909, 1036, 1038, 1165, 1167, 1294, 1296, 1423, 1425, 1552, 1554, 1681, 1683, 1810, 1812, 1939, 1941, 2068, 2070, 2197, 2199, 2326, 2328, 2455, 2457, 2584, 2586, 2713, 2715, 2842, 2844, 2971, 2973, 3100, 3102, 3229, 3231, 3358, 3360, 3487, 3489, 3616, 3618, 3745, 3747, 3874, 3876, 4003, 4005, 4132, 4134, 4261, 4263, 4390, 4392, 4519, 4521, 4648, 4650, 4777, 4779, 4906, 4908, 5035, 5037, 5164, 5166, 5293, 5295, 5422, 5424, 5551, 5553, 5680, 5682, 5809, 5811, 5938, 5940, 6067, 6069, 6196, 6198, 6325, 6327, 6454, 6456, 6583, 6585, 6712, 6714, 6841, 6843, 6970, 6972, 7099, 7101, 7228, 7230, 7357, 7359, 7486, 7488, 7615, 7617, 7744, 7746, 7873, 7875, 8002, 8004, 8131, 8133, 8260, 8262, 8389, 8391, 8518, 8520, 8647, 8649, 8776, 8778, 8905, 8907, 9034, 9036, 9163, 9165, 9292, 9294, 9421, 9423, 9550, 9552, 9679, 9681, 9808, 9810, 9937, 9939, 10066, 10068, 10195, 10197, 10324, 10326, 10453, 10455, 10582, 10584, 10711, 10713, 10840, 10842, 10969, 10971, 11098, 11100, 11227, 11229, 11356, 11358, 11485, 11487, 11614, 11616, 11743, 11745, 11872, 11874, 12001, 12003, 12130, 12132, 12259, 12261, 12388, 12390, 12517, 12519, 12646, 12648, 12775, 12777, 12904, 12906, 13033, 13035, 13162, 13164, 13291, 13293, 13420, 13422, 13549, 13551, 13678, 13680, 13807, 13809, 13936, 13938, 14065, 14067, 14194, 14196, 14323, 14325, 14452, 14454, 14581, 14583, 14710, 14712, 14839, 14841, 14968, 14970, 15097, 15099, 15226, 15228, 15355, 15357, 15484, 15486, 15613, 15615, 15742, 15744, 15871, 15873, 16000, 16002, 16129, 16131, 16258, 16260, 16387, 16389, 16516, 16518, 16645, 16647, 16774, 16776, 16903, 16905, 17032, 17034, 17161, 17163, 17290, 17292, 17419, 17421, 17548, 17550, 17677, 17679, 17806, 17808, 17935, 17937, 18064, 18066, 18193, 18195, 18322, 18324, 18451, 18453, 18580, 18582, 18709, 18711, 18838, 18840, 18967, 18969, 19096, 19098, 19225, 19227, 19354, 19356, 19483, 19485, 19612, 19614, 19741, 19743, 19870, 19872, 19999, 20001, 20128, 20130, 20257, 20259, 20386, 20388, 20515, 20517, 20644, 20646, 20773, 20775, 20902, 20904, 21031, 21033, 21160, 21162, 21289, 21291, 21418, 21420, 21547, 21549, 21676, 21678, 21805, 21807, 21934, 21936, 22063, 22065, 22192, 22194, 22321, 22323, 22450, 22452, 22579, 22581, 22708, 22710, 22837, 22839, 22966, 22968, 23095, 23097, 23224, 23226, 23353, 23355, 23482, 23484, 23611, 23613, 23740, 23742, 23869, 23871, 23998, 24000, 24127, 24129, 24256, 24258, 24385, 24387, 24514, 24516, 24643, 24645, 24772, 24774, 24901, 24903, 25030, 25032, 25159, 25161, 25288, 25290, 25417, 25419, 25546, 25548, 25675, 25677, 25804, 25806, 25933, 25935, 26062, 26064, 26191, 26193, 26320, 26322, 26449, 26451, 26578, 26580, 26707, 26709, 26836, 26838, 26965, 26967, 27094, 27096, 27223, 27225, 27352, 27354, 27481, 27483, 27610, 27612, 27739, 27741, 27868, 27870, 27997, 27999, 28126, 28128, 28255, 28257, 28384, 28386, 28513, 28515, 28642, 28644, 28771, 28773, 28900, 28902, 29029, 29031, 29158, 29160, 29287, 29289, 29416, 29418, 29545, 29547, 29674, 29676, 29803, 29805, 29932, 29934, 30061, 30063, 30190, 30192, 30319, 30321, 30448, 30450, 30577, 30579, 30706, 30708, 30835, 30837, 30964, 30966, 31093, 31095, 31222, 31224, 31351, 31353, 31480, 31482, 31609, 31611, 31738, 31740, 31867, 31869, 31996, 31998, 32125, 32127, 32254, 32256, 32383, 32385, 32512, 32514, 32641, 32643, 32770, 32772, 32899, 32901, 33028, 33030, 33157, 33159, 33286, 33288, 33415, 33417, 33544, 33546, 33673, 33675, 33802, 33804, 33931, 33933, 34060, 34062, 34189, 34191, 34318, 34320, 34447, 34449, 34576, 34578, 34705, 34707, 34834, 34836, 34963, 34965, 35092, 35094, 35221, 35223, 35350, 35352, 35479, 35481, 35608, 35610, 35737, 35739, 35866, 35868, 35995, 35997, 36124, 36126, 36253, 36255, 36382, 36384, 36511, 36513, 36640, 36642, 36769, 36771, 36898, 36900, 37027, 37029, 37156, 37158, 37285, 37287, 37414, 37416, 37543, 37545, 37672, 37674, 37801, 37803, 37930, 37932, 38059, 38061, 38188, 38190, 38317, 38319, 38446, 38448, 38575, 38577, 38704, 38706, 38833, 38835, 38962, 38964, 39091, 39093, 39220, 39222, 39349, 39351, 39478, 39480, 39607, 39609, 39736, 39738, 39865, 39867, 39994, 39996, 40123, 40125, 40252, 40254, 40381, 40383, 40510, 40512, 40639, 40641, 40768, 40770, 40897, 40899, 41026, 41028, 41155, 41157, 41284, 41286, 41320]          # Front and Back Faces
    lrb_faces = [1, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 41322, 41323]    # Left, Right, and Bottom Faces
    
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

        layer_thick = a / GRID_SIZE 
        
        # --- 2. Unit Cell with Layers ---
        unit = geom.create("unit_cell", "Block")
        unit.set("pos", [0, 0, h_bottom])
        unit.set("size", [a, a, h_pile])
        
        # Enable layers on two sides
        unit.set("layerleft", True)
        unit.set("layerfront", True)
        unit.set("layerbottom", False)
        
        # Create a 1D array of layer thicknesses
        # Note: We use 31 divisions to create 32 cells
        layer_vals = np.full(GRID_SIZE - 1, float(layer_thick))
        for i in range(GRID_SIZE - 1):
            unit.setIndex("layer", str(layer_vals[i]), i)

        # --- 3. Array the Unit Cell ---
        arr = geom.create("arr1", "Array")
        arr.selection("input").set("unit_cell")
        arr.set("displ", [float(a), 0.0, 0.0])
        arr.set("fullsize", [num_piles, 1, 1])
        
        geom.run()
        logging.info("Geometry construction complete.")

        # --- Assign Materials to Grids ---
        soil_ids = []
        conc_ids = []
        current_id = 1

        # 1. Bottom Soil
        soil_ids.append(current_id)
        current_id += 1


        # 2. Piles (Array of Unit Cells)
        # COMSOL numbers domains Y-fastest (Column-Major). 
        # We must iterate X (j) then Y (i) to match domain IDs 2, 3, 4... to (0,0), (1,0), (2,0)...
        for n in range(num_piles):
            for j in range(GRID_SIZE):          # Col (x) - Outer Loop
                for i in range(GRID_SIZE):      # Row (y) - Inner Loop (Fastest)
                    
                    # Map is accessed as [row, col] -> [i, j]
                    if mmap[i, j] == 1:
                        conc_ids.append(current_id)
                    else:
                        soil_ids.append(current_id)
                    current_id += 1


        # 3. Remainder (if exists)
        if rem_len > 0:
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