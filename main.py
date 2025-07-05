import os
import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
import pyvista as pv
import subprocess
from multiprocessing import Pool

# ----------------------------
# Configuration
# ----------------------------
FPS = 20
DURATION = 10.0
FRAME_RES = (1024, 768)
CMAP = "plasma"
KERNEL = "linear"

# ----------------------------
# Folder selection
# ----------------------------
base_input_dir = "input"
folders = [f for f in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, f))]

if not folders:
    raise FileNotFoundError("No folders found in 'input/'")

print("Available input folders:")
for i, f in enumerate(folders):
    print(f"[{i}] {f}")

selection = int(input("\nEnter folder number: "))
folder_path = os.path.join(base_input_dir, folders[selection])

# ----------------------------
# Load data
# ----------------------------
thermo_pos = pd.read_csv(os.path.join(folder_path, "thermocouple_positions.csv"))
temp_data = pd.read_csv(os.path.join(folder_path, "temperature_data.csv"))

id_to_pos = {row["id"]: (row["x"], row["y"], row["z"]) for _, row in thermo_pos.iterrows()}

xt_points = []
xt_values = []
for _, row in temp_data.iterrows():
    t = row["t"]
    for col in row.index[1:]:
        if pd.notna(row[col]) and col in id_to_pos:
            x = id_to_pos[col][0]
            xt_points.append([x, t])
            xt_values.append(row[col])

xt_points = np.array(xt_points)
xt_values = np.array(xt_values)

print("Creating 2D RBF interpolator for (x, t)...")
rbf_xt = RBFInterpolator(xt_points, xt_values, kernel=KERNEL)
print("Interpolator ready.")

# ----------------------------
# Load mesh
# ----------------------------
mesh_path = os.path.join(folder_path, "engine_model.obj")
mesh = pv.read(mesh_path)
mesh.scale([10, 10, 10], inplace=True)
print(f"Mesh loaded: {mesh.n_points} vertices")

# ----------------------------
# Prepare for rendering
# ----------------------------
frame_dir = os.path.join(folder_path, "frames_temp")
os.makedirs(frame_dir, exist_ok=True)

global_min = xt_values.min()
global_max = xt_values.max()
time_range = np.linspace(temp_data["t"].min(), temp_data["t"].max(), int(DURATION * FPS))

# ----------------------------
# Frame rendering function
# ----------------------------
def render_single_frame(args):
    i, t = args
    print(f"🎬 [Worker] Rendering frame {i+1}/{len(time_range)} at t = {t:.2f}s")

    mesh_t = mesh.copy()
    x_coords = mesh_t.points[:, 0]
    xt_query = np.column_stack((x_coords, np.full_like(x_coords, t)))
    temps = rbf_xt(xt_query)

    frame_min = np.interp(t, temp_data["t"], temp_data.iloc[:, 1:].min(axis=1))
    frame_max = np.interp(t, temp_data["t"], temp_data.iloc[:, 1:].max(axis=1))
    temps_clipped = np.clip(temps, frame_min, frame_max)
    temps_norm = (temps_clipped - global_min) / (global_max - global_min)
    temps_norm = temps_norm ** 0.6

    mesh_t.point_data["Temperature"] = temps_norm

    plotter = pv.Plotter(off_screen=True, window_size=FRAME_RES)
    plotter.set_background("black")
    plotter.view_isometric()
    plotter.add_mesh(mesh_t, scalars="Temperature", cmap=CMAP, clim=[0, 1], show_edges=False)
    plotter.reset_camera()

    img_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
    plotter.screenshot(img_path)
    plotter.close()

    return f"✅ Frame {i+1} done"

# ----------------------------
# Parallel rendering
# ----------------------------
args_list = list(enumerate(time_range))
print(f"\n🚀 Starting parallel rendering with {os.cpu_count()} workers...\n")

with Pool() as pool:
    for result in pool.imap_unordered(render_single_frame, args_list):
        print(result)

print(f"✅ All frames saved to: {frame_dir}")

# ----------------------------
# Compile video
# ----------------------------
video_path = os.path.join(folder_path, f"thermal_animation_{FPS}fps.mp4")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(frame_dir, "frame_%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
    video_path
]

print(f"\n🎞️ Compiling video at {FPS} FPS...")
try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ Video saved to: {video_path}")
except FileNotFoundError:
    print("❌ ffmpeg not found. Please install it.")
except subprocess.CalledProcessError:
    print("❌ ffmpeg failed.")
