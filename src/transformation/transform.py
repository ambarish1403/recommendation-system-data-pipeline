import pandas as pd
from src.utils import ROOT

def transform(data):
    # Standardize text, IDs, timestamps and rating types; derive interaction fields.
    train = data["train"].copy()
    test = data["test"].copy()
    items = data["items"].copy()

    for frame in (train, test):
        frame["user_id"] = frame["user_id"].astype(str).str.strip()
        frame["item_id"] = frame["item_id"].astype(str).str.strip()
        frame["rating"] = pd.to_numeric(frame["rating"], errors="raise").astype(float)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
        frame["interaction_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
        frame["interaction_hour"] = frame["timestamp"].dt.hour.astype(int)
        frame["rating_bucket"] = pd.cut(
            frame["rating"], bins=[0, 2, 3, 5],
            labels=["low", "medium", "high"], include_lowest=True
        ).astype(str)

    items["title"] = items["title"].astype(str).str.strip()
    items["category"] = items["category"].astype(str).str.strip().str.title()
    train = train.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    test = test.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    train.to_csv(ROOT / "data/processed/transformed_train.csv", index=False)
    test.to_csv(ROOT / "data/processed/transformed_test.csv", index=False)
    items.to_csv(ROOT / "data/processed/transformed_items.csv", index=False)

    data["train"], data["test"], data["items"] = train, test, items
    return data
