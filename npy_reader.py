import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Assume the results from the previous script are in this directory
output_dir = Path("results_v5.5_12")
# Use an example atlas file, e.g., for run_id=1.
# In a real scenario, you would get this path from your database.
atlas_files = [output_dir / "atlas_run_4.npy",
               output_dir / "atlas_run_12.npy",
               output_dir / "atlas_run_15.npy",
               output_dir / "atlas_run_46.npy",
               output_dir / "atlas_run_156.npy",
               output_dir / "atlas_run_179.npy",
               ]
for atlas_file in atlas_files:
    # Check if the file exists to prevent errors
    if not atlas_file.exists():
        # Create a dummy file for demonstration if it doesn't exist
        print(f"Warning: '{atlas_file}' not found. Creating a dummy atlas for demonstration.")
        dummy_atlas = np.random.randint(0, 2, size=(32, 32))
        np.save(atlas_file, dummy_atlas)

    # Load the atlas grid from the .npy file
    atlas_grid = np.load(atlas_file)

    # --- Visualization ---

    # 1. Define custom colors: 0=Soil (brown), 1=Concrete (gray)
    cmap = ListedColormap(['saddlebrown', 'darkgray'])

    # 2. Create the plot
    plt.imshow(atlas_grid, cmap=cmap, interpolation='nearest')

    # 3. Add a title and remove axes for a cleaner look
    plt.title(f"Visualization of {atlas_file.name}")
    plt.axis('off')

    # 4. Create a custom colorbar to label the materials
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_ticklabels(['Soil (0)', 'Concrete (1)'])

    # 5. Save the figure
    image_filename = f'atlas_visualization_{atlas_file.name}.png'
    plt.savefig(image_filename)
    print(f"Image saved as {image_filename}")
    plt.close()