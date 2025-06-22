import yaml
import csv
from collections import OrderedDict

from engine_thermal_visualiser.io.step_loader import load_step_file
from engine_thermal_visualiser.visualisation.viewer import render_solids_with_time_slider

# --- Load STEP geometry ---
print("Loading STEP file...")
solids, _ = load_step_file("data/example_model.step")

# --- Step 1: Load thermocouple positions ---
print("Loading TC positions...")
with open("data/TC_positions.yaml", "r") as f:
    tc_positions = yaml.safe_load(f)

# --- Step 2: Load time series temperature data ---
print("Loading temperature data...")
time_series_data = OrderedDict()
with open("data/temperature_data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        t = float(row["t"])
        tc_readings = {k: float(v) for k, v in row.items() if k != "t"}
        time_series_data[t] = tc_readings

# --- Render interactive plot ---
print("Rendering with time slider...")
render_solids_with_time_slider(solids, tc_positions, time_series_data)
