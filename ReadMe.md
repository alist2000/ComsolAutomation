# COMSOL 2D Phononic CrystalAnalyzer

*A family of Pythonscripts – written with the `mph` LiveLink – for high‑throughput eigen‑analysis of a $28\times28$ pixel 2‑D phononic crystal unit cell.* Each edition adds new automation features; all store inputs+results in SQLite for reproducibility and downstream PINN training.

---

## 1· Editions at a glance

| Ver.   | Python file | Headline capability                                                                                                            | k‑sweepstrategy                                                  | DB schema                                                                                             | COMSOL runs                                       |
| -------- |-------------| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **v1** | `grid_model.py-v1`    | **Baseline**: random 28×28 material map, hard‑coded Floquet PBC node lists, logs every run to SQLite                             | Single point (Γ→X end‑pointkx=π/a,ky=0)                      | `simulations`+ `eigenfrequencies`(tables w/o mesh stats)                                              | Builds & solves **once per run**                    |
| **v2** | `grid_model.py-v2` | **Octant‑symmetry** generator, meshstatistics logging, PBC selections recorded                                                 | Single point (π/a,0)                                              | Adds`mesh_num_elements`+ `mesh_min_quality`, stores PBC node lists as JSON                           | One run per script                                 |
| **v3** | `grid_model.py—v3` | **Parametric k‑sweep** over triangular Γ–X grid, robust DB logging, separate `.mph` per(kx,ky)                                | External Python loop; spins up a new COMSOL session **per k‑point** | `simulations` row per *k*; `eigenfrequencies` unchanged                                                 | *N* sessions (slow but simple)                      |
| **v4** | `grid_model.py`(current) | **Manual in‑memory sweep** –builds the model **once**, loops through k‑grid by only updating parameters; writes a single `.mph` | Internal loop inside one COMSOL session (fast)                      | Normalised schema: one `simulations` row per sweep; `eigenfrequencies` gains explicit `kx`,`ky` columns | **1** COMSOL solve sequence reused for *N* k‑points |

> **Tip:** choose **v4** for production sweeps; keep **v3** around when debugging individual k‑points or COMSOL memory‑leak issues.

---

## 2· Common workflow

1. **Pick a version**

   * `v1`/`v2`for quick single‑point tests or teaching examples.
   * `v3` if you need completelystateless, crash‑resilient sweeps on a cluster.
   * `v4` for the fastest laptop/desktop band‑diagram runs.
2. **Install requirements** `pip install mph numpy`
3. **Launch COMSOLserver** or rely on`mph.start()` to spawn a stand‑alone client.
4. **Run the script**\`\`\`bash
   python grid\_model.py       # or the file of the edition you need


5. **Post‑process**
   * `.mph` file → inspect modes visually in COMSOL Desktop if desired.
   * `results/simulation_results.db` → query with SQLiteor Pandas; each eigen‑row already tagged with its k‑vector.

---

## 3· Feature matrix

| Capability |v1 |v2 |v3 |v4 |
|------------|:--:|:--:|:--:|:--:|
| Random **octant‑symmetric** material map |✖|✔|✔|✔|
| Floquet PBC hard‑coded node lists |✔|✔|✔|✔|
| Mesh quality & element count logged |✖|✔|✔|(irrelevant– single mesh) |
| Parametric k‑sweep |✖|✖|✔|✔|
| Single COMSOL build reused |✖|✖|✖|✔|
| One `.mph` file per sweep |✖|✖|✔|✖|
| SQLite schema normalised (`kx`,`ky` columns) |✖|✖|✖|✔|

---

## 4· Customisation hooks

* **Grid resolution**– change `GRID_SIZE`; regenerate PBC lists in `_outer_bnd_lists()`.
* **k‑grid density**– tweak `N_K` (v3/v4).
* **Material parameters**– edit `SOIL_PROPS` / `CONCRETE_PROPS` dictionaries.
* **Number of modes**– set `N_MODES` (v4) or `num_eigenvalues` in earlier scripts.

---

## 5. Usage

1.  Place the script in your project directory.
2.  Ensure COMSOL is correctly installed and accessible from your system's command line environment.
3.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```

The script will create a `results/` directory, where it will save the database and the final `.mph` model file.

---

## 6· Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---------|--------------|------|
| COMSOL exits after first k‑point (v3) | server started without `-multion` | Restart server: `comsol mphserver -multi on -port2036` |
| `ValueError` about boundary lists after changing `GRID_SIZE` | Hard‑coded lists only valid for 28×28 | Re‑generate lists or derive them programmatically |
| Out‑of‑memory for large sweeps | v3 spawns many COMSOL instances | Use v4 (single process) or batch k‑points |

---

©2025— Phononic Crystalautomation demo.Licensed under MIT.


