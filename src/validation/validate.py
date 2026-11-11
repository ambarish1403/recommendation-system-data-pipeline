import pandas as pd
from src.utils import ROOT, write_json

def validate(data):
    users, items, interactions = data["users"], data["items"], data["interactions"]
    checks = {}
    required = {
        "users": ["user_id", "age", "country"],
        "items": ["item_id", "title", "category"],
        "interactions": ["user_id", "item_id", "rating", "timestamp"]
    }
    for name, cols in required.items():
        missing_cols = sorted(set(cols) - set(data[name].columns))
        if missing_cols:
            raise ValueError(f"{name}.csv missing required columns: {missing_cols}")
        checks[f"{name}_missing_values"] = int(data[name][cols].isna().any(axis=1).sum())
        checks[f"{name}_duplicate_rows"] = int(data[name].duplicated().sum())
    x = interactions.copy()
    x["rating"] = pd.to_numeric(x["rating"], errors="coerce")
    parsed_ts = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
    reasons = pd.Series("", index=x.index, dtype="object")
    def add(mask, reason):
        reasons.loc[mask] = reasons.loc[mask].apply(lambda old: old + (";" if old else "") + reason)
    add(x[["user_id", "item_id", "rating", "timestamp"]].isna().any(axis=1), "missing_required_value")
    add(x["rating"].isna(), "invalid_rating_type")
    add(~x["rating"].between(1, 5), "rating_out_of_range")
    add(parsed_ts.isna(), "invalid_timestamp")
    add(~x["user_id"].isin(users["user_id"]), "unknown_user_id")
    add(~x["item_id"].isin(items["item_id"]), "unknown_item_id")
    duplicate_mask = x.duplicated(subset=["user_id", "item_id", "timestamp"], keep="first")
    add(duplicate_mask, "duplicate_interaction")
    invalid = x.loc[reasons.ne("")].copy()
    invalid["validation_errors"] = reasons.loc[reasons.ne("")]
    valid = x.loc[reasons.eq("")].copy()
    valid["timestamp"] = parsed_ts.loc[reasons.eq("")]
    valid["rating"] = valid["rating"].astype(float)
    report = {
        "row_counts": {"users": len(users), "items": len(items), "interactions_input": len(interactions),
                       "interactions_valid": len(valid), "interactions_invalid": len(invalid)},
        "checks": checks,
        "invalid_reason_counts": {r: int(reasons.str.contains(r, regex=False).sum()) for r in
                                  ["missing_required_value", "invalid_rating_type", "rating_out_of_range",
                                   "invalid_timestamp", "unknown_user_id", "unknown_item_id", "duplicate_interaction"]},
        "status": "passed_with_quarantined_records" if len(invalid) else "passed"
    }
    invalid.to_csv(ROOT / "data/errors/invalid_interactions.csv", index=False)
    write_json(ROOT / "data/errors/validation_report.json", report)
    return {"users": users.drop_duplicates("user_id"), "items": items.drop_duplicates("item_id"),
            "interactions": valid}, report
