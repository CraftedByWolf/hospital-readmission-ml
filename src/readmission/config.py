from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the project root so scripts work from any cwd.
    for section, key in [("data", "raw_dir"), ("data", "processed_dir"),
                         ("paths", "models_dir"), ("paths", "reports_dir")]:
        cfg[section][key] = ROOT / cfg[section][key]
    return cfg
