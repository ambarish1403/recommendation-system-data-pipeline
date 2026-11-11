from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT / "configs" / "config.json", encoding="utf-8") as f:
        return json.load(f)

def ensure_dirs():
    for name in ["data/raw", "data/processed", "data/features", "data/errors", "models", "reports"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
