import numpy as np

def map_temps_to_mesh(vertices, tc_positions, tc_temperatures):
    """
    Assign temperatures to mesh vertices using nearest TC (or inverse-distance weighting).
    """
    vertex_temps = []
    for v in vertices:
        distances = []
        temps = []
        for tc_name, tc_pos in tc_positions.items():
            dist = np.linalg.norm(np.array([v.x, v.y, v.z]) - np.array(tc_pos))
            if dist == 0:
                distances = [1.0]
                temps = [tc_temperatures[tc_name]]
                break
            distances.append(1.0 / dist)
            temps.append(tc_temperatures[tc_name])

        # Weighted average
        weights = np.array(distances)
        values = np.array(temps)
        temp = np.sum(weights * values) / np.sum(weights)
        vertex_temps.append(temp)

    return vertex_temps
