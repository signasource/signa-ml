"""
Utilidades para mapear labels <-> índices a partir del config.
"""
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "signs_config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_label_map(config_path: Path = CONFIG_PATH) -> dict:
    """Retorna {id: label}, ej: {0: 'hola', 1: 'gracias'}"""
    config = load_config(config_path)
    return {sign["id"]: sign["label"] for sign in config["signs"]}


def get_signs(config_path: Path = CONFIG_PATH) -> list[dict]:
    """Retorna la lista completa de señas del config."""
    return load_config(config_path)["signs"]


def get_num_classes(config_path: Path = CONFIG_PATH) -> int:
    return len(get_signs(config_path))


if __name__ == "__main__":
    print(get_label_map())