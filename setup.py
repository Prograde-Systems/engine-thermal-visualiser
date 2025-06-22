from setuptools import setup, find_packages

setup(
    name="engine_thermal_visualiser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pythonocc-core",  # or the correct OCC library you're using
        # Add other dependencies like numpy, pyvista etc. as needed
    ],
    author="Your Name",
    description="3D visualisation tool for temperature data on rocket engine assemblies.",
    include_package_data=True,
)
