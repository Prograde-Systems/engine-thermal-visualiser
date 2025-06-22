from engine_thermal_visualiser.visualisation.viewer import render_solids_plotly
from engine_thermal_visualiser.io.step_loader import load_step_file
from engine_thermal_visualiser.io.config_loader import load_tc_config  # new
import pandas as pd

# --- Filepaths ---
step_path = "data/example_model.step"
tc_config_path = "data/TC_positions.yaml"  # your config
sensor_csv_path = "data/temperature_data.csv"

# --- Load geometry ---
print("Loading STEP file")
solids, _ = load_step_file(step_path)

# --- Load thermocouple locations ---
print("Loading thermocouple positions from config...")
tc_positions = load_tc_config(tc_config_path)
print(f"Loaded {len(tc_positions)} thermocouple positions:")
for name, coords in tc_positions.items():
    print(f"  {name}: {coords}")

# --- Load temperature sensor values ---
df = pd.read_csv(sensor_csv_path)
t0_data = df.iloc[0].to_dict()
t0_data.pop("t", None)  # remove time column if present

print(f"Visualising t = 0.0s with {len(t0_data)} sensor readings...")
render_solids_plotly(solids, tc_positions=tc_positions, tc_values=t0_data)
