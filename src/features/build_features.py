import numpy as np
import pandas as pd
from src.utils import ROOT

def build_features(data):
    users, items, train = data["users"], data["items"], data["train"].copy()
    now = train["timestamp"].max()
    uf = users.merge(data["activity"], on="user_id", how="left")
    uf["user_activity_count"] = uf["user_activity_count"].fillna(0).astype(int)
    uf["user_average_rating"] = uf["user_average_rating"].fillna(train.rating.mean())
    uf["days_since_last_interaction"] = (now - pd.to_datetime(uf["last_interaction"], utc=True)).dt.days
    uf["days_since_last_interaction"] = uf["days_since_last_interaction"].fillna(-1)
    itemf = items.merge(data["popularity"], on="item_id", how="left")
    for col in ["item_popularity", "item_average_rating", "item_rating_count"]:
        itemf[col] = itemf[col].fillna(0)
    preferred = train.merge(items[["item_id", "category"]], on="item_id").groupby(
        ["user_id", "category"]).size().reset_index(name="n")
    preferred = preferred.sort_values(["user_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("user_id")
    uf = uf.merge(preferred[["user_id", "category"]].rename(columns={"category": "preferred_category"}),
                  on="user_id", how="left")
    interactionf = train.merge(uf[["user_id", "user_average_rating", "user_activity_count"]],
                              on="user_id", how="left").merge(
        itemf[["item_id", "item_popularity", "item_average_rating"]], on="item_id", how="left")
    interactionf["rating_deviation_from_user_mean"] = interactionf.rating - interactionf.user_average_rating
    matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    uf.to_csv(ROOT / "data/features/user_features.csv", index=False)
    itemf.to_csv(ROOT / "data/features/item_features.csv", index=False)
    interactionf.to_csv(ROOT / "data/features/interaction_features.csv", index=False)
    matrix.to_csv(ROOT / "data/features/user_item_matrix.csv")
    catalog = [
        ("user_activity_count", "User", "Count of training interactions", "Training", "Count"),
        ("user_average_rating", "User", "Mean rating given by user", "Training", "Float"),
        ("preferred_category", "User", "Most-interacted item category", "Training", "Category"),
        ("days_since_last_interaction", "User", "Days from latest training interaction", "Training", "Days"),
        ("item_popularity", "Item", "Distinct users interacting with item", "Training", "Count"),
        ("item_average_rating", "Item", "Mean training rating for item", "Training", "Float"),
        ("item_rating_count", "Item", "Number of training ratings", "Training", "Count"),
        ("rating_deviation_from_user_mean", "Interaction", "Rating minus user's training mean", "Training", "Float")
    ]
    pd.DataFrame(catalog, columns=["feature", "entity", "definition", "availability", "data_type"]).to_csv(
        ROOT / "reports/feature_catalog.csv", index=False)
    return {"user_features": uf, "item_features": itemf, "interaction_features": interactionf, "matrix": matrix}
