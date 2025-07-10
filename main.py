import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pyvista as pv
import subprocess
from multiprocessing import Pool
import yaml
from tqdm import tqdm
import time

# ----------------------------
# Folder selection
# ----------------------------
base_input_dir = "data"
folders = [f for f in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, f))]

if not folders:
    raise FileNotFoundError("No folders found in 'input/'")

print("Available input folders:")
for i, f in enumerate(folders):
    print(f"[{i}] {f}")

selection = int(input("\nEnter folder number: "))
folder_path = os.path.join(base_input_dir, folders[selection])

# ----------------------------
# Configuration
# ----------------------------
with open(os.path.join(folder_path, "input/config.yaml")) as stream:
    try:
        cfg = yaml.safe_load(stream)
        print(cfg)
    except yaml.YAMLError as exc:
        print(exc)
        cfg = {}

FPS = cfg['FPS']
T_START = cfg["t_start"]
T_END = cfg["t_end"]
PLAYBACK_SPEED = cfg["playback_speed"]
FRAME_RES = cfg["frame_resolution"]
CMAP = cfg["colour_map"]

# ----------------------------
# Load data
# ----------------------------
thermo_pos = pd.read_csv(os.path.join(folder_path, "input/TC_positions.csv"))
temp_data = pd.read_csv(os.path.join(folder_path, "input/TC_temperatures.csv"))

id_to_pos = {row["id"]: (row["x"], row["y"], row["z"]) for _, row in thermo_pos.iterrows()}

# Normalize x for interpolator stability
x_all = thermo_pos["x"].values
x_mean = x_all.mean()
x_std = x_all.std()
NORMALIZATION = {"x_mean": x_mean, "x_std": x_std}

# Create spatial interpolators per time step
print("\nCreating spatial (x-only) interpolators per time step...")
interpolators_by_time = {}

for _, row in temp_data.iterrows():
    t = row["t"]
    valid_ids = [col for col in row.index[1:] if pd.notna(row[col]) and col in id_to_pos]
    x_vals = np.array([id_to_pos[col][0] for col in valid_ids])
    temps = np.array([row[col] for col in valid_ids])

    if len(x_vals) < 2:
        continue  # Skip if not enough points to interpolate

    sort_idx = np.argsort(x_vals)
    x_vals_sorted = x_vals[sort_idx]
    temps_sorted = temps[sort_idx]

    interpolators_by_time[t] = interp1d(
        x_vals_sorted,
        temps_sorted,
        kind=cfg['interpolation_method'],
        bounds_error=False,
        fill_value="extrapolate"
    )

print(f"✅ {len(interpolators_by_time)} interpolators ready.")

# ----------------------------
# Load mesh
# ----------------------------
mesh_path = os.path.join(folder_path, "input/engine_model.obj")
mesh = pv.read(mesh_path)

sampling_factor = cfg.get("mesh_sampling", 1.0)

if sampling_factor > 1.0:
    subdivisions = int(np.floor(np.log2(sampling_factor)))
    mesh = mesh.subdivide(subdivisions, 'loop')
    print(f"Mesh refined with {subdivisions} subdivisions → {mesh.n_points} vertices")
elif sampling_factor < 1.0:
    target_reduction = 1.0 - sampling_factor
    mesh = mesh.decimate(target_reduction=target_reduction)
    print(f"Mesh simplified by {int(target_reduction * 100)}% → {mesh.n_points} vertices")
else:
    print("Mesh sampling factor = 1.0 → using original resolution")

mesh.scale([1, 1, 1], inplace=True)
print(f"Mesh loaded: {mesh.n_points} vertices")

# ----------------------------
# Prepare for rendering
# ----------------------------
frame_dir = os.path.join(folder_path, "output/frames_temp")
os.makedirs(frame_dir, exist_ok=True)

temp_data_filtered = temp_data[(temp_data["t"] >= T_START) & (temp_data["t"] <= T_END)]
global_min = int(temp_data_filtered.iloc[:, 1:].min().min())
global_max = int(temp_data_filtered.iloc[:, 1:].max().max())
time_range = np.linspace(T_START, T_END, int((T_END - T_START) * FPS))

# ----------------------------
# Frame rendering function
# ----------------------------
def render_single_frame(args):
    i, t = args
    start_time = time.perf_counter()
    print(f"🎬 Rendering frame {i+1}/{len(time_range)} at t = {t:.2f}s", flush=True)

    mesh_t = mesh.copy()
    x_coords = mesh_t.points[:, 0]

    # Find nearest timestamp
    nearest_t = min(interpolators_by_time.keys(), key=lambda ti: abs(ti - t))
    interp = interpolators_by_time[nearest_t]
    temps = interp(x_coords)

    frame_min = np.interp(t, temp_data["t"], temp_data.iloc[:, 1:].min(axis=1))
    frame_max = np.interp(t, temp_data["t"], temp_data.iloc[:, 1:].max(axis=1))
    temps_clipped = np.clip(temps, frame_min, frame_max)

    mesh_t.point_data["Temperature"] = temps_clipped

    plotter = pv.Plotter(off_screen=True, window_size=FRAME_RES)
    plotter.set_background("black")
    plotter.view_isometric()

    plotter.camera.position = cfg["camera"]["position"]
    plotter.camera.focal_point = cfg["camera"]["focal_point"]
    plotter.camera.up = cfg["camera"]["up"]

    if cfg["camera"]["enable_parallel_projection"]:
        plotter.enable_parallel_projection()

    plotter.add_mesh(
        mesh_t,
        scalars="Temperature",
        cmap=CMAP,
        clim=[global_min, global_max],
        show_edges=False,
        scalar_bar_args={
            "title": "",
            "color": "white",
            "vertical": False,
            "title_font_size": 1,
            "label_font_size": 25,
            "position_x": 0.3,
            "position_y": 0.05,
            "width": 0.4,
            "height": 0.04
        }
    )

    plotter.add_text("Temperature (°C)", color="white", position="lower_edge", font_size=15)
    plotter.add_text(cfg["title"], position="upper_left", font_size=15, color="white")
    plotter.add_text(f"t = {t:.2f} s", position="lower_right", font_size=15, color="white")

    plotter.reset_camera()
    img_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
    plotter.screenshot(img_path)
    plotter.close()

    elapsed = time.perf_counter() - start_time
    return f"✅ Frame {i+1}/{len(time_range)} (t={t:.2f}s) completed in {elapsed:.2f}s"

# ----------------------------
# Parallel rendering
# ----------------------------
args_list = list(enumerate(time_range))
print(f"\n🚀 Starting parallel rendering with {os.cpu_count()} workers for {len(args_list)} frames...\n")

with Pool() as pool:
    for result in tqdm(pool.imap_unordered(render_single_frame, args_list), total=len(args_list)):
        print(result, flush=True)

print(f"✅ Rendering complete — {len(args_list)} frames saved to: {frame_dir}")

# ----------------------------
# Compile video
# ----------------------------
video_path = os.path.join(folder_path, f"output/thermal_animation_{FPS}fps.mp4")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS * PLAYBACK_SPEED),
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


    # ----------------------------
# Compile GIF
# ----------------------------
gif_path = os.path.join(folder_path, f"output/thermal_animation_{FPS}fps.gif")
ffmpeg_gif_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS * PLAYBACK_SPEED),
    "-i", os.path.join(frame_dir, "frame_%03d.png"),
    "-vf", "scale=iw:-1:flags=lanczos",  # optional: maintain aspect ratio with high-quality scaling
    "-loop", "0",
    gif_path
]

print(f"\n🖼️ Compiling GIF at {FPS} FPS...")
try:
    subprocess.run(ffmpeg_gif_cmd, check=True)
    print(f"✅ GIF saved to: {gif_path}")
except FileNotFoundError:
    print("❌ ffmpeg not found. Please install it.")
except subprocess.CalledProcessError:
    print("❌ ffmpeg failed during GIF creation.")
