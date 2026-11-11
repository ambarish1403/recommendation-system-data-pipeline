import numpy as np
import pandas as pd
from src.utils import ROOT, load_config

def generate_synthetic_data():
    cfg = load_config()
    rng = np.random.default_rng(cfg["random_seed"])
    n_users, n_items = cfg["n_users"], cfg["n_items"]
    users = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(n_users)],
        "age": rng.integers(18, 66, n_users),
        "country": rng.choice(["IN", "US", "GB", "CA", "AU"], n_users)
    })
    categories = ["Action", "Comedy", "Drama", "Sci-Fi", "Documentary", "Romance"]
    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(n_items)],
        "title": [f"Movie {i:03d}" for i in range(n_items)],
        "category": rng.choice(categories, n_items)
    })
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for u in users.user_id:
        preferred = rng.choice(categories)
        chosen = rng.choice(n_items, size=min(cfg["interactions_per_user"], n_items), replace=False)
        for j, item_idx in enumerate(chosen):
            cat = items.iloc[item_idx].category
            score = 3.5 + (1.0 if cat == preferred else 0.0) + rng.normal(0, 1.0)
            rating = int(np.clip(np.rint(score), 1, 5))
            timestamp = start + pd.Timedelta(days=int(j * 4 + rng.integers(0, 3)))
            rows.append((u, f"I{item_idx:04d}", rating, timestamp.isoformat()))
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    # Add a few deliberate quality issues so quarantine behavior is demonstrable.
    bad = pd.DataFrame([
        ("U0000", "I0000", 8, "2024-12-01T00:00:00Z"),
        ("UNKNOWN", "I0001", 4, "2024-12-02T00:00:00Z"),
        ("U0001", "NOT_AN_ITEM", 5, "not-a-date"),
        ("U0002", "I0002", np.nan, "2024-12-03T00:00:00Z")
    ], columns=interactions.columns)
    interactions = pd.concat([interactions, bad], ignore_index=True)
    for name, frame in [("users", users), ("items", items), ("interactions", interactions)]:
        frame.to_csv(ROOT / "data/raw" / f"{name}.csv", index=False)

def ingest():
    cfg = load_config()
    if cfg.get("generate_synthetic", True):
        generate_synthetic_data()
    paths = {n: ROOT / "data/raw" / f"{n}.csv" for n in ["users", "items", "interactions"]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input CSV(s): {missing}")
    return {n: pd.read_csv(p) for n, p in paths.items()}
