# Architecture

```mermaid
flowchart LR
  A[CSV source data] --> B[Ingestion]
  B --> C[Validation]
  C -->|valid interactions| D[Cleaning and time split]
  C -->|invalid rows| E[Error quarantine]
  D --> F[Derived datasets]
  F --> G[Feature management]
  G --> H1[Popularity baseline]
  G --> H2[Item-item collaborative filtering]
  H1 --> I[Evaluation]
  H2 --> I
  I --> J[Metrics and experiment tracking]
  J --> K[Reports and artifacts]
```

The Python orchestrator runs each stage in dependency order. CSV and JSON files keep intermediate outputs inspectable and portable. The feature layer is a set of versionable, reusable tables shared by training and inference.
