# utils_config.py
import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    """Load YAML config into a dictionary with Path objects."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: Path(v) for k, v in cfg["paths"].items()}
    # Ensure outputs dir exists
    cfg["paths"]["outputs"].mkdir(parents=True, exist_ok=True)
    return cfg
