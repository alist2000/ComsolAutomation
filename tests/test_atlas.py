import sys
import types
from pathlib import Path
import numpy as np

# Ensure the repository root is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide dummy modules to satisfy imports in run_sweep
for mod_name in ('mph', 'pandas'):
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

from run_sweep import generate_dummy_atlases, NX, NY


def test_generate_dummy_atlases():
    atlases = generate_dummy_atlases()
    assert isinstance(atlases, dict)
    assert len(atlases) == 2
    for atlas in atlases.values():
        assert atlas.shape == (NX, NY)
        assert np.isin(atlas, [0, 1]).all()
