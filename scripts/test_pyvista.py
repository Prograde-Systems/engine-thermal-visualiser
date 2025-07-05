import pyvista as pv
import os

engine_path = "input/example/engine_model.obj"
mesh = pv.read(engine_path)

print("Mesh bounds:", mesh.bounds)
print("Number of points:", mesh.n_points)

# Create dummy scalar field for testing
import numpy as np
scalars = np.linalg.norm(mesh.points, axis=1)
mesh.point_data["DummyScalar"] = scalars

plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
plotter.set_background("black")
plotter.add_mesh(mesh, scalars="DummyScalar", cmap="plasma", show_edges=False)
plotter.view_isometric()
plotter.reset_camera()
plotter.add_scalar_bar(title="Distance from Origin")
plotter.show(screenshot="test_engine.png")

print("✅ Saved test_engine.png")
