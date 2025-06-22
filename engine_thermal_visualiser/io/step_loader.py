# File: engine_thermal_visualiser/io/step_loader.py

import cadquery as cq
import os
import re


def load_step_file(filepath):
    """
    Load a STEP file and return:
    - a list of solids with their names and center points
    - a dictionary of thermocouple locations (label -> (x, y, z))
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STEP file not found: {filepath}")

    print("[INFO] Importing STEP assembly...")
    assembly = cq.importers.importStep(filepath)

    solids_info = []
    tc_points = []

    for i, shape in enumerate(assembly.solids().vals()):
        print(f"\n--- Shape {i} Attributes ---")
        for attr in dir(shape):
            if not attr.startswith("__"):  # skip internal dunder methods
                try:
                    value = getattr(shape, attr)
                    print(f"{attr}: {value}")
                except Exception as e:
                    print(f"{attr}: <error reading attribute: {e}>")
        center = shape.Center()
        label = getattr(shape, "label", None)
        name = label.strip() if label else f"solid_{i}"


        print(name.upper())

        if name.upper().startswith("TC:"):
            tc_points.append((name, (center.x, center.y, center.z)))

        solids_info.append({
            "name": name,
            "solid": shape,
            "center": (center.x, center.y, center.z)
        })

    print(f"Found {len(tc_points)} thermocouples:")
    for name, pos in tc_points:
        print(f"  {name}: {pos}")

    return solids_info, dict(tc_points)





def extract_tc_locations(solids_info):
    """
    Extracts thermocouple positions labeled as 'TC1', 'TC2', etc.
    Returns a dictionary { 'TC1': (x, y, z), ... }
    """
    tc_dict = {}
    for solid in solids_info:
        name = solid['name']
        if re.fullmatch(r"TC\d+", name):  # match exactly TC followed by digits
            tc_dict[name] = solid['center']
    print(f"[INFO] Found {len(tc_dict)} thermocouples.")
    return tc_dict
