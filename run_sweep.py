# run_sweep.py
#
# Automates the COMSOL eigenfrequency analysis for different material layouts (atlases)
# and Floquet-Bloch wave vectors (k) as per the provided "cook-book" job card.
#
# This script uses the 'mph' library (pip install mph).
#
# To Run:
# 1. Start your COMSOL server:
#    On Windows: "C:\Program Files\COMSOL\COMSOL62\Multiphysics\bin\win64\comsolmphserver.exe"
#    On Linux:   /usr/local/comsol62/multiphysics/bin/comsol mphserver
# 2. Run this script from your terminal:
#    python run_sweep.py

import mph
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# 0. CONFIGURATION & SETUP (Matches "Pre-flight parameters")
# ==============================================================================

# --- Simulation Parameters ---
A_LATTICE = 2.0  # m, lattice length
NX, NY = 32, 32  # Atlas resolution

# --- Material Properties ---
# Soil (background material)
SOIL_PROPS = {'E': '20e6[Pa]', 'nu': '0.3', 'rho': '1800[kg/m^3]'}
# Concrete (inclusion material)
CONCRETE_PROPS = {'E': '30e9[Pa]', 'nu': '0.2', 'rho': '2400[kg/m^3]'}

# --- Study Parameters ---
N_EIGENVALUES = 10  # Number of eigenfrequencies to compute

# --- Sweep Configuration ---
# Define the wave vectors (q_x, q_y) in [0, 1] to sweep over.
K_VECTORS = [
    (0.0, 0.0),
    (0.25, 0.0),
    (0.5, 0.0),
    (0.5, 0.25),
    (0.5, 0.5),
    (0.25, 0.25),
]

# --- File & Directory Setup ---
TEMPLATE_FILE = Path("metamaterial_template.mph")
OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_dummy_atlases(num_atlases=2):
    """Generates placeholder atlas data."""
    atlases = {}
    atlas1 = np.zeros((NX, NY), dtype=int)
    atlas1[NX // 2 - 2: NX // 2 + 2, :] = 1
    atlas1[:, NY // 2 - 2: NY // 2 + 2] = 1
    atlases['atlas_0001'] = atlas1

    atlas2 = np.zeros((NX, NY), dtype=int)
    x, y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    atlas2[(x // 4 + y // 4) % 2 == 0] = 1
    atlases['atlas_0002'] = atlas2

    return atlases


# ==============================================================================
# 1. TEMPLATE GENERATION (Steps 1-5 performed once)
# ==============================================================================

def create_comsol_template(filename=TEMPLATE_FILE):
    """
    Programmatically creates the base COMSOL model file if it doesn't exist.
    This version explicitly creates every required model feature.
    """
    if filename.exists():
        print(f"Template file '{filename}' already exists. Skipping creation.")
        return

    print(f"Creating new COMSOL template file: '{filename}'...")
    client = mph.start()
    model = client.create('Model')

    print("Step 1: Defining Global Parameters...")
    model.java.param().create('par1')
    params_java = model.java.param('par1')
    params_java.set('a', f'{A_LATTICE}[m]', 'Lattice length')
    params_java.set('nx', str(NX), 'Atlas resolution (x)')
    params_java.set('ny', str(NY), 'Atlas resolution (y)')
    params_java.set('dx', 'a/nx', 'Pixel width')
    params_java.set('dy', 'a/ny', 'Pixel height')
    params_java.set('qx', '0.0', 'Normalized wave vector (x)')
    params_java.set('qy', '0.0', 'Normalized wave vector (y)')
    params_java.set('kx', 'pi/a * qx', 'Floquet kx')
    params_java.set('ky', 'pi/a * qy', 'Floquet ky')

    print("Step 2: Creating Component, Geometry, and Materials...")
    model.java.component().create("comp1", True)
    model.java.component("comp1").geom().create("geom1", 2)
    geom_java = model.java.component("comp1").geom("geom1")
    geom_java.create("R1", "Rectangle")
    geom_java.feature("R1").set("size", ['a', 'a'])
    geom_java.run()

    materials_group = model.java.component("comp1").material()
    soil = materials_group.create("Soil", "Common")
    soil.propertyGroup("def").set("youngsmodulus", SOIL_PROPS['E'])
    soil.propertyGroup("def").set("poissonsratio", SOIL_PROPS['nu'])
    soil.propertyGroup("def").set("density", SOIL_PROPS['rho'])
    soil.selection().all()

    concrete = materials_group.create("Concrete", "Common")
    concrete.propertyGroup("def").set("youngsmodulus", CONCRETE_PROPS['E'])
    concrete.propertyGroup("def").set("poissonsratio", CONCRETE_PROPS['nu'])
    concrete.propertyGroup("def").set("density", CONCRETE_PROPS['rho'])

    print("Step 3: Setting up Physics and Boundary Conditions...")
    model.java.component("comp1").physics().create("solid", "SolidMechanics", "geom1")

    # --- DEFINITIVE FIX ---
    # Get the physics object and use .setProperty() to set the AnalysisType.
    solid_java = model.java.component("comp1").physics("solid")
    solid_java.setProperty("AnalysisType", "planeStrain")

    pbc = solid_java.create('pbc1', 'PeriodicCondition', 1)
    pbc.selection().allb()
    pbc.set('PeriodicType', 'Floquet')
    pbc.set('k_user', ['kx', 'ky'])

    print("Step 4: Creating Mesh...")
    model.java.component("comp1").mesh().create("mesh1")
    mesh = model.mesh("mesh1")
    mesh.create("map1", "Mapped")
    mesh.feature("map1").property("ndiv", [str(NX), str(NY)])

    print("Step 5: Creating Study...")
    model.java.study().create("eig")
    study_feature = model.java.study("eig").create("eig1", "Eigenfrequency")
    study_feature.activate("solid", True)

    model.java.solver().create("sol1")
    model.java.solver("sol1").study("eig")
    eig_solver = model.java.solver("sol1").create("e1", "Eigen")
    eig_solver.set("neig", str(N_EIGENVALUES))
    eig_solver.set("shift", "0")

    print("Saving template file...")
    model.save(filename)
    client.remove(model)
    print("Template creation complete.")


# ==============================================================================
# 2. AUTOMATED SWEEP (Step 6)
# ==============================================================================

def run_parametric_sweep():
    """Main function to run the high-throughput campaign."""
    atlas_pool = generate_dummy_atlases()
    print(f"Loaded {len(atlas_pool)} atlases for processing.")

    client = mph.start()
    model = client.load(TEMPLATE_FILE)
    print(f"Loaded template '{TEMPLATE_FILE}'. Starting sweep...")

    # Get the Java parameter object once for reuse.
    params_java = model.java.param('par1')

    for atlas_id, atlas_grid in atlas_pool.items():
        print(f"\n--- Processing {atlas_id} ---")
        np.save(OUTPUT_DIR / f"{atlas_id}.npy", atlas_grid)

        # 1. UPDATE GEOMETRY
        geom1_java = model.java.component("comp1").geom("geom1")
        # Clear previous micro-rectangles if any
        for tag in list(geom1_java.feature().tags()):  # Create list to avoid modification-during-iteration issues
            if "pixel_" in tag:
                geom1_java.feature().remove(tag)

        # CORRECTED: Robustly evaluate parameter values using Java API
        dx = params_java.evaluate('dx')
        dy = params_java.evaluate('dy')
        concrete_selection_tags = []

        for i in range(NX):
            for j in range(NY):
                if atlas_grid[i, j] == 1:
                    tag = f"pixel_{i}_{j}"
                    x0, y0 = i * dx, j * dy
                    rect = geom1_java.feature().create(tag, "Rectangle")
                    rect.set("size", [dx, dy])
                    rect.set("pos", [x0, y0])
                    concrete_selection_tags.append(tag)

        geom1_java.run()

        # 2. UPDATE MATERIAL ASSIGNMENT
        if "sel_concrete" in model.selection():
            model.selection("sel_concrete").remove()
        concrete_selection = model.selection().create("sel_concrete", "geom1")
        # Ensure the selection is from the final geometry by getting entities
        finalized_geom = model.java.component("comp1").geom("geom1").resolve()
        domain_indices = finalized_geom.getEntities("domain", concrete_selection_tags)
        concrete_selection.java.set(domain_indices)
        model.material("Concrete").selection().named("geom1_sel_concrete")

        # Inner loop for k-vectors
        for qx, qy in K_VECTORS:
            print(f"  Running k-vector: (qx={qx:.2f}, qy={qy:.2f})...")

            # 3. UPDATE PARAMETERS (kx, ky)
            # CORRECTED: Use the robust Java API object to set parameters
            params_java.set('qx', str(qx))
            params_java.set('qy', str(qy))

            # 4. RUN STUDY
            model.mesh()
            model.solve("eig")

            # 5. EXPORT RESULTS
            frequencies = model.evaluate('solid.freq', study='eig')
            output_df = pd.DataFrame({
                'mode_index': range(1, len(frequencies) + 1),
                'frequency_Hz': frequencies
            })
            csv_filename = OUTPUT_DIR / f"{atlas_id}_k_{qx:.2f}_{qy:.2f}.csv"
            output_df.to_csv(csv_filename, index=False)
            print(f"    -> Saved results to {csv_filename}")

    client.remove(model)
    print("\nHigh-throughput sweep complete.")


# ==============================================================================
# 3. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    create_comsol_template()
    run_parametric_sweep()
