import cadquery as cq
import plotly.graph_objects as go
import numpy as np
import os


def combine_solids_to_colored_plotly_mesh(solids_info, angular_tolerance=1.0, linear_tolerance=1.0):
    """
    Combines multiple CadQuery solids into a single Plotly Mesh3D with pseudo-thermal coloring.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
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

        x, y, z = zip(*[(v.x, v.y, v.z) for v in vertices])
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)

        for tri in triangles:
            all_i.append(tri[0] + vertex_offset)
            all_j.append(tri[1] + vertex_offset)
            all_k.append(tri[2] + vertex_offset)

        vertex_offset += len(vertices)

    if not all_x:
        raise ValueError("No mesh data to render. Ensure the STEP file has valid solids.")

    # Use Z as a stand-in for temperature (placeholder)
    scalar_field = all_z

    return go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        intensity=scalar_field,
        colorscale='Viridis',
        opacity=0.9,
        name='Thermal Map',
        showscale=True,
        flatshading=True,
    )


def render_solids_plotly(solids_info, output_file="output/plot.html"):
    """
    Render all solids in a merged view using Plotly and save as an interactive HTML file.
    """
    print(f"[INFO] Rendering {len(solids_info)} solids...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mesh = combine_solids_to_colored_plotly_mesh(solids_info)

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Engine Thermal Visualisation (Merged)',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(output_file)
    print(f"[INFO] Plot saved to {output_file}. Open it in your browser.")
