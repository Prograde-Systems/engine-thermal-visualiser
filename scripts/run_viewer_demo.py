from engine_thermal_visualiser.visualisation.viewer import render_solids_plotly
from engine_thermal_visualiser.io.step_loader import load_step_file

filepath = "data/example_model.step"
print("Loading step file")
solids, tc_points = load_step_file(filepath)

print(type(solids))  # This is what you're currently printing

print(f"Found {len(tc_points)} thermocouples:")

for name, pos in tc_points.items():
    print(f"  {name}: {pos}")


print("rendering solids")
render_solids_plotly(solids)
