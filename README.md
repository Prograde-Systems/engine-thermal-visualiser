# Engine Thermal Visualiser

A Python-based visualisation tool for animating spatially interpolated temperature data on 3D engine geometries. Developed for design evaluation.

---

## Overview

**Engine Thermal Visualiser** maps time-varying thermocouple data onto a 3D mesh of an engine component using spatial interpolation. The tool generates:
- MP4 video animation  
- GIF animations  

It supports configurable resolution, playback speed, color maps, and camera positions via the YAML configuration file.

---

## Features

- Spatial interpolation (1D along engine length)
- High-resolution frame rendering via PyVista
- Customizable color map and camera configuration
- Parallelised frame generation for speed
- Export to MP4 (video) and palette-optimised GIF

---

## Input Folder Structure

Place your input data in a subdirectory of the `data/` folder, like so:

```
data/
└── MyCase/
    └── input/
        ├── config.yaml               # Configuration file
        ├── TC_positions.csv          # Thermocouple positions (x, y, z, id)
        ├── TC_temperatures.csv       # Time-varying temperature data
        └── engine_model.obj          # 3D geometry in OBJ format
```

---

## Configuration

All runtime options are specified in `config.yaml`:

```yaml
FPS: 15
t_start: 0
t_end: 10
frame_resolution: [1200, 900]
playback_speed: 1.0
colour_map: inferno
interpolation_method: linear
mesh_sampling: 1.0
title: "Wall Temperature Distribution"
camera:
  position: [2, 2, 1]
  focal_point: [0, 0, 0]
  up: [0, 0, 1]
  enable_parallel_projection: true
```

---

## Usage

Run the script from the project root:

```bash
python render_thermal.py
```

You will be prompted to select a case folder within `data/`.

Outputs will be saved under:

```
data/YourCase/output/
├── frames_temp/                    # Rendered PNG frames
├── thermal_animation_15fps.mp4    # MP4 video animation
└── thermal_animation_15fps.gif    # Looping GIF (Slides-ready)
```

---

## Installation

### Python Dependencies

Install required Python packages with:

```bash
pip install -r requirements.txt
```

This includes:
- pyvista
- pandas
- numpy
- scipy
- tqdm
- pyyaml

### System Dependencies

Ensure [`ffmpeg`](https://ffmpeg.org/download.html) is installed and accessible via system `PATH`.

---

## Troubleshooting

| Issue                                | Cause / Fix                                              |
|-------------------------------------|-----------------------------------------------------------|
| `ffmpeg not found`                  | Install ffmpeg and add it to your system's PATH          |
| No frames generated                 | Ensure valid data exists between `t_start` and `t_end`   |
| GIF loops incorrectly               | Use the two-pass optimized palette GIF creation          |
| Visual artefacts in GIF             | Check interpolation method or increase resolution        |

---