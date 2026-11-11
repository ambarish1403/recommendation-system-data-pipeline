import pandas as pd
from src.utils import ROOT

def prepare(data):
    users, items, interactions = data["users"], data["items"], data["interactions"].copy()
    interactions = interactions.drop_duplicates(["user_id", "item_id", "timestamp"])
    interactions = interactions.sort_values(["user_id", "timestamp", "item_id"])
    counts = interactions.groupby("user_id").size()
    eligible = counts[counts >= 2].index
    test_idx = interactions[interactions.user_id.isin(eligible)].groupby("user_id").tail(1).index
    test = interactions.loc[test_idx].copy()
    train = interactions.drop(index=test_idx).copy()
    train.to_csv(ROOT / "data/processed/train.csv", index=False)
    test.to_csv(ROOT / "data/processed/test.csv", index=False)
    activity = train.groupby("user_id").agg(
        user_activity_count=("item_id", "count"), user_average_rating=("rating", "mean"),
        last_interaction=("timestamp", "max")).reset_index()
    popularity = train.groupby("item_id").agg(
        item_popularity=("user_id", "nunique"), item_average_rating=("rating", "mean"),
        item_rating_count=("rating", "count")).reset_index()
    activity.to_csv(ROOT / "data/processed/user_activity.csv", index=False)
    popularity.to_csv(ROOT / "data/processed/item_popularity.csv", index=False)
    return {"users": users, "items": items, "train": train, "test": test,
            "activity": activity, "popularity": popularity}
