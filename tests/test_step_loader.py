import os
import pytest
from engine_thermal_visualiser.io.step_loader import load_step_file

def test_load_step_file():
    # Path to test STEP file
    test_file = os.path.join("data", "example_model.step")
    assert os.path.exists(test_file), "STEP file not found for testing."

    # Load the STEP file
    solids = load_step_file(test_file)

    # Check returned structure
    assert isinstance(solids, list), "Expected a list of solids"
    assert len(solids) > 0, "No solids found in STEP file"

    for solid in solids:
        assert 'name' in solid, "Missing 'name' key"
        assert 'solid' in solid, "Missing 'solid' key"
        assert 'center' in solid, "Missing 'center' key"

        x, y, z = solid['center']
        assert isinstance(x, (int, float)), "Invalid X coordinate"
        assert isinstance(y, (int, float)), "Invalid Y coordinate"
        assert isinstance(z, (int, float)), "Invalid Z coordinate"
