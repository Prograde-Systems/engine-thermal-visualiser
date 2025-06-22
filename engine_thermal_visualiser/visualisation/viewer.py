import cadquery as cq
import plotly.graph_objects as go
import numpy as np
import os


def compute_vertex_temperatures(vertices, tc_positions, tc_values, max_distance=60.0):
    vertex_temps = []

    for vx, vy, vz in vertices:
        weights, temps = [], []

        for tc_name, (tx, ty, tz) in tc_positions.items():
            d = np.linalg.norm([vx - tx, vy - ty, vz - tz])
            if d < max_distance:
                w = 1 / (d + 1e-6)
                weights.append(w)
                temps.append(tc_values.get(tc_name, 20.0))

        if weights:
            vertex_temps.append(np.average(temps, weights=weights))
        else:
            vertex_temps.append(20.0)  # ambient fallback

    return vertex_temps


def tessellate_solids(solids_info, angular_tolerance=1.0, linear_tolerance=1.0):
    all_coords, all_faces = [], []
    offset = 0

    for solid_info in solids_info:
        solid = solid_info["solid"]
        try:
            tess = solid.tessellate(angular_tolerance, linear_tolerance)
            vertices = [(v.x, v.y, v.z) for v in tess[0]]
            triangles = tess[1]
        except Exception as e:
            print(f"[WARN] Skipping solid due to tessellation error: {e}")
            continue

        if not vertices or not triangles:
            continue

        face_indices = [
            (tri[0] + offset, tri[1] + offset, tri[2] + offset)
            for tri in triangles
        ]

        all_coords.extend(vertices)
        all_faces.extend(face_indices)
        offset += len(vertices)

    return all_coords, all_faces


def make_mesh_frame(vertices, faces, intensities):
    x, y, z = zip(*vertices)
    i, j, k = zip(*faces)

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=intensities,
        colorscale='Hot',
        opacity=1.0,
        flatshading=True,
        showscale=True
    )


def render_solids_with_time_slider(solids_info, tc_positions, time_series_data, output_file="output/thermal_animation.html"):
    print(f"[INFO] Rendering {len(solids_info)} solids with time slider...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_coords, all_faces = tessellate_solids(solids_info)
    if not all_coords or not all_faces:
        raise ValueError("No mesh data available for rendering.")

    times = sorted(time_series_data.keys())
    frames = []
    slider_steps = []

    # Build frames
    for t in times:
        tc_values = time_series_data[t]
        intensities = compute_vertex_temperatures(all_coords, tc_positions, tc_values)
        mesh = make_mesh_frame(all_coords, all_faces, intensities)

        frames.append(go.Frame(data=[mesh], name=f"{t:.2f}s"))
        slider_steps.append({
            "args": [[f"{t:.2f}s"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f"{t:.2f}s",
            "method": "animate"
        })

    # Base (initial) mesh
    base_intensities = compute_vertex_temperatures(all_coords, tc_positions, time_series_data[times[0]])
    base_mesh = make_mesh_frame(all_coords, all_faces, base_intensities)

    # Full interactive figure
    fig = go.Figure(
        data=[base_mesh],
        layout=go.Layout(
            title="Engine Thermal Visualisation (Time Slider)",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
            ),
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": True,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "pad": {"t": 50},
                "currentvalue": {"prefix": "Time: "},
                "steps": slider_steps
            }]
        ),
        frames=frames
    )

    fig.write_html(output_file)
    print(f"[INFO] Animated thermal plot saved to {output_file}")
