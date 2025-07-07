import os
import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
import pyvista as pv
import subprocess
from multiprocessing import Pool
import yaml

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
        print(cfg)  # Optional: just for debugging
    except yaml.YAMLError as exc:
        print(exc)
        cfg = {}  # Optional: fallback if error occurs

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

# Normalize xt_points for RBF stability
x_mean = xt_points[:, 0].mean()
x_std = xt_points[:, 0].std()
t_mean = xt_points[:, 1].mean()
t_std = xt_points[:, 1].std()

# Save normalization constants for query normalization
NORMALIZATION = {
    "x_mean": x_mean,
    "x_std": x_std,
    "t_mean": t_mean,
    "t_std": t_std
}


xt_points[:, 0] = (xt_points[:, 0] - x_mean) / x_std
xt_points[:, 1] = (xt_points[:, 1] - t_mean) / t_std


print("Creating 2D RBF interpolator for (x, t)...")
rbf_xt = RBFInterpolator(xt_points, xt_values, kernel="multiquadric", epsilon=1.0, smoothing=1e-2)
print("Interpolator ready.")

# ----------------------------
# Load mesh
# ----------------------------
mesh_path = os.path.join(folder_path, "input/engine_model.obj")
mesh = pv.read(mesh_path)

sampling_factor = cfg.get("mesh_sampling", 1.0)

if sampling_factor > 1.0:
    # 🔼 Refine: subdivide mesh
    subdivisions = int(np.floor(np.log2(sampling_factor)))
    mesh = mesh.subdivide(subdivisions, 'loop')

    print(f"Mesh refined with {subdivisions} subdivisions → {mesh.n_points} vertices")

elif sampling_factor < 1.0:
    # 🔽 Simplify: reduce number of faces
    target_reduction = 1.0 - sampling_factor  # e.g., 0.5 means 50% fewer faces
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

global_min = int(xt_values.min())
global_max = int(xt_values.max())
time_range = np.linspace(T_START, T_END, int((T_END - T_START) * FPS))

# ----------------------------
# Frame rendering function
# ----------------------------
def render_single_frame(args):
    i, t = args
    print(f"🎬 [Worker] Rendering frame {i+1}/{len(time_range)} at t = {t:.2f}s")

    mesh_t = mesh.copy()
    x_coords = mesh_t.points[:, 0]
    x_norm = (x_coords - NORMALIZATION["x_mean"]) / NORMALIZATION["x_std"]
    t_norm = (t - NORMALIZATION["t_mean"]) / NORMALIZATION["t_std"]
    xt_query = np.column_stack((x_norm, np.full_like(x_norm, t_norm)))

    temps = rbf_xt(xt_query)

    # Optional: per-frame clipping
    frame_min = np.interp(t, temp_data["t"], temp_data.iloc[:, 1:].min(axis=1))
    frame_max = np.interp(t, temp_data["t"], temp_data.iloc[:, 1:].max(axis=1))
    temps_clipped = np.clip(temps, frame_min, frame_max)

    # Assign raw physical temperature data to mesh
    mesh_t.point_data["Temperature"] = temps_clipped

    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=FRAME_RES)
    plotter.set_background("black")
    plotter.view_isometric()

    plotter.camera.position = cfg["camera"]["position"]
    plotter.camera.focal_point = cfg["camera"]["focal_point"]
    plotter.camera.up = cfg["camera"]["up"]
    
    # Add mesh with temperature scalar bar
    plotter.add_mesh(
        mesh_t,
        scalars="Temperature",
        cmap=CMAP,
        clim=[global_min, global_max],  # Global scale for consistency
        show_edges=False,
        scalar_bar_args={
            "title": "",              # No title
            "color": "white",         # White tick labels
            "vertical": False,        # Horizontal bar
            "title_font_size": 1,     # Hide title spacing
            "label_font_size": 25,
            "position_x": 0.3,        # Adjust as needed
            "position_y": 0.05,
            "width": 0.4,
            "height": 0.04
        }
    )

    plotter.add_text(
        text="Temperature (K)",
        color= "white",
        position="lower_edge",
        font_size=15
    )


    plotter.reset_camera()

    # Add main title (top-left or top-center)
    plotter.add_text(
        text=cfg["title"],
        position="upper_left",  # or "upper_center"
        font_size=15,
        color="white"
    )

    # Add time annotation (bottom-right)
    plotter.add_text(
        text=f"t = {t:.2f} s",
        position="lower_right",
        font_size=15,
        color="white"
    )

    # Render and save frame
    img_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
    plotter.screenshot(img_path)
    plotter.close()

    return f"[INFO]Frame {i+1} complete"


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
