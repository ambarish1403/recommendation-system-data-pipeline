# Final Report – Recommendation Pipeline

## Objective and design
This project implements the required end-to-end data management pipeline for a movie recommendation use case. Synthetic users, movie items and timestamped ratings are generated with a fixed random seed so the run is reproducible. CSV files are used for transparent intermediate storage.

## Data quality and preparation
Validation checks required fields, missing values, duplicates, numeric ratings in the 1–5 range, parseable UTC timestamps, and user/item referential integrity. Invalid interactions are quarantined rather than silently discarded. The last chronological interaction for each user with at least two interactions is held out as test data.

## Features
User features include activity count, mean rating, preferred category and recency. Item features include popularity, mean rating and rating count. Interaction features include rating deviation from the user's mean. The user-item matrix is reused by collaborative filtering.

## Models and evaluation
Two models are compared: a global popularity baseline and item-item collaborative filtering using cosine similarity. Relevance is defined as a held-out rating >= 4.0; ranking metrics use K=5. Diversity is measured as the proportion of distinct item categories in a recommendation list. MAE/RMSE are reported for the popularity model's global-mean rating estimate; the ranking-only collaborative model does not predict explicit ratings.

### Results
| model                             |   k |   precision_at_k |   recall_at_k |   ndcg_at_k |   coverage |   diversity |        mae |       rmse |
|:----------------------------------|----:|-----------------:|--------------:|------------:|-----------:|------------:|-----------:|-----------:|
| Popularity baseline               |   5 |        0.0176471 |     0.0882353 |   0.0461038 |     0.1375 |    0.797059 |   0.738072 |   0.867046 |
| Item-item collaborative filtering |   5 |        0.0117647 |     0.0588235 |   0.0406156 |     0.65   |    0.735294 | nan        | nan        |

## Orchestration and tracking
A dependency-ordered Python workflow executes ingestion, validation, preparation, feature generation, training, evaluation and reporting. Parameters and metrics are stored in `reports/experiment_tracking.json`; MLflow logging is enabled when installed.

## Limitations
The synthetic dataset is small and does not represent real-world behavior. Item-item similarity can be sparse for low-activity users/items. The time split holds out one event per eligible user rather than a global date cutoff. Diversity is category-based and does not capture semantic novelty or serendipity. A production deployment would require monitoring, incremental feature updates, privacy controls and online evaluation.
