import pandas as pd
import yaml

def load_sensor_data(csv_path):
    """
    Load sensor data CSV with format: t, TC1, TC2, ...
    Returns: DataFrame with time as index
    """
    df = pd.read_csv(csv_path)
    df.set_index("t", inplace=True)
    return df

def load_tc_positions(yaml_path):
    """
    Load thermocouple positions from a YAML config file.
    Returns: dict like { 'TC1': (x, y, z), ... }
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return {k: tuple(v) for k, v in data.items()}
