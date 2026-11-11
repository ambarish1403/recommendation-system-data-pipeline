import numpy as np
import pandas as pd
from src.utils import ROOT, load_config

def _recommend_popular(model, seen, k):
    return [i for i in model.item_id.tolist() if i not in seen][:k]

def _recommend_cf(user, model, seen, k):
    matrix, sim, item_ids = model["matrix"], model["similarity"], model["item_ids"]
    if user not in matrix.index:
        return _recommend_popular(model["popularity"], seen, k)
    ratings = matrix.loc[user].to_numpy(dtype=float)
    scores = ratings @ sim
    candidates = [(item_ids[j], scores[j]) for j in range(len(item_ids)) if item_ids[j] not in seen]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [x[0] for x in candidates[:k]]

def _metrics(recs_by_user, test, catalog, k, threshold):
    relevant = test[test.rating >= threshold].groupby("user_id").item_id.apply(set).to_dict()
    users = [u for u in recs_by_user if relevant.get(u)]
    if not users:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0,
                "coverage": 0.0, "diversity": 0.0}
    precisions, recalls, ndcgs, diversities = [], [], [], []
    recommended_all = set()
    # Category diversity is the fraction of distinct categories in each list.
    item_categories = catalog.set_index("item_id").category.to_dict()
    for u in users:
        recs = recs_by_user[u][:k]
        hits = [1 if x in relevant[u] else 0 for x in recs]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant[u]))
        dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
        ideal = sum(1 / np.log2(idx + 2) for idx in range(min(len(relevant[u]), k)))
        ndcgs.append(dcg / ideal if ideal else 0)
        recommended_all.update(recs)
        cats = [item_categories.get(x) for x in recs if item_categories.get(x)]
        diversities.append(len(set(cats)) / len(cats) if cats else 0.0)
    return {"precision_at_k": float(np.mean(precisions)), "recall_at_k": float(np.mean(recalls)),
            "ndcg_at_k": float(np.mean(ndcgs)), "coverage": len(recommended_all) / max(len(catalog), 1),
            "diversity": float(np.mean(diversities))}

def evaluate(data, models):
    cfg = load_config()
    k, threshold = cfg["k"], cfg["test_relevance_threshold"]
    train = data["train"]
    seen = train.groupby("user_id").item_id.apply(set).to_dict()
    test_users = data["test"].user_id.unique()
    pop_recs, cf_recs = {}, {}
    for u in test_users:
        pop_recs[u] = _recommend_popular(models["popularity"], seen.get(u, set()), k)
        cf_recs[u] = _recommend_cf(u, models, seen.get(u, set()), k)
    catalog = data["items"]
    rows = []
    for name, recs in [("Popularity baseline", pop_recs), ("Item-item collaborative filtering", cf_recs)]:
        rows.append({"model": name, "k": k, **_metrics(recs, data["test"], catalog, k, threshold)})
    # Rating error for popularity's global-mean estimate (explicit rating task).
    global_mean = float(train.rating.mean())
    actual = data["test"].rating.to_numpy(dtype=float)
    rows[0]["mae"] = float(np.mean(np.abs(actual - global_mean)))
    rows[0]["rmse"] = float(np.sqrt(np.mean((actual - global_mean) ** 2)))
    rows[1]["mae"] = None
    rows[1]["rmse"] = None
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "reports/evaluation_results.csv", index=False)
    return result
