import cadquery as cq
import plotly.graph_objects as go
import numpy as np
import os


def compute_vertex_temperatures(vertices, tc_positions, tc_values, max_distance=60.0):
    vertex_temps = []

    for idx, (vx, vy, vz) in enumerate(vertices):
        distances = []
        temps = []
        for tc_name, (tx, ty, tz) in tc_positions.items():
            d = np.linalg.norm([vx - tx, vy - ty, vz - tz])
            if d < max_distance:
                weight = 1 / (d + 1e-6)
                distances.append(weight)
                temps.append(tc_values[tc_name])

        if distances:
            weighted_temp = np.average(temps, weights=distances)
            vertex_temps.append(weighted_temp)
        else:
            vertex_temps.append(20.0)  # or a neutral temperature, like the ambient value


    return vertex_temps




def combine_solids_to_colored_plotly_mesh(solids_info, tc_positions=None, tc_values=None,
                                          angular_tolerance=1.0, linear_tolerance=1.0):
    """
    Combine solids into a single Plotly mesh, interpolating only where TC coverage exists.
    Faces with insufficient data are excluded.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    all_intensity = []
    vertex_map = {}  # map from original index to new index
    vertex_offset = 0

    for idx, solid_info in enumerate(solids_info):
        solid = solid_info["solid"]

        try:
            tess = solid.tessellate(angular_tolerance, linear_tolerance)
            vertices, triangles = tess
        except Exception as e:
            print(f"[WARN] Skipping solid_{idx} due to tessellation error: {e}")
            continue

        if not vertices or not triangles:
            print(f"[WARN] Skipping solid_{idx}: empty tessellation.")
            continue

        coords = [(v.x, v.y, v.z) for v in vertices]

        if tc_positions and tc_values:
            intensities = compute_vertex_temperatures(coords, tc_positions, tc_values)
        else:
            intensities = [z for _, _, z in coords]

        local_index_map = {}
        for local_idx, ((x, y, z), temp) in enumerate(zip(coords, intensities)):
            if temp is not None:
                new_idx = len(all_x)
                local_index_map[local_idx] = new_idx
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)
                all_intensity.append(temp)

        for tri in triangles:
            try:
                new_i = local_index_map[tri[0]]
                new_j = local_index_map[tri[1]]
                new_k = local_index_map[tri[2]]
                all_i.append(new_i)
                all_j.append(new_j)
                all_k.append(new_k)
            except KeyError:
                # At least one vertex missing data — skip this face
                continue

    if not all_x or not all_i:
        raise ValueError("No mesh data with valid TC coverage to render.")

    return go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        intensity=all_intensity,
        colorscale='Hot',
        opacity=1.0,
        name='Thermal Map',
        showscale=True,
        flatshading=True,
    )



def render_solids_plotly(solids_info, tc_positions=None, tc_values=None, output_file="output/plot.html"):
    """
    Render all solids with mapped thermal data using Plotly.
    """
    print(f"[INFO] Rendering {len(solids_info)} solids...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mesh = combine_solids_to_colored_plotly_mesh(solids_info, tc_positions, tc_values)

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Engine Thermal Visualisation',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(output_file)
    print(f"[INFO] Plot saved to {output_file}. Open it in your browser.")
