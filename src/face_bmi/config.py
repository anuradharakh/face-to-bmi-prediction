from pathlib import Path
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)
