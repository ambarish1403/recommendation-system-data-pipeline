import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import ROOT

def train_models(data):
    train, items = data["train"], data["items"]
    # Popularity baseline: global ranking and global mean as rating estimate.
    pop = train.groupby("item_id").agg(popularity=("user_id", "nunique"),
                                      mean_rating=("rating", "mean")).reset_index()
    pop = pop.sort_values(["popularity", "mean_rating", "item_id"], ascending=[False, False, True])
    pop.to_json(ROOT / "models/popularity_model.json", orient="records", indent=2)
    matrix = data["features"]["matrix"]
    # Cosine similarity between items; zero means missing interaction.
    item_user = matrix.T.to_numpy(dtype=float)
    similarity = cosine_similarity(item_user)
    np.savez_compressed(ROOT / "models/item_item_model.npz", similarity=similarity,
                        item_ids=matrix.columns.to_numpy(dtype=str),
                        user_ids=matrix.index.to_numpy(dtype=str))
    meta = {"algorithm": "item-item collaborative filtering", "similarity": "cosine",
            "rating_threshold_for_recommendation": 4.0, "n_items": int(len(matrix.columns))}
    (ROOT / "models/item_item_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"popularity": pop, "similarity": similarity, "item_ids": list(matrix.columns),
            "matrix": matrix}
