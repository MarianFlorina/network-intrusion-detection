# Pipeline diagram (ASCII fallback)

This document is the ASCII/text equivalent of the Mermaid diagram in
`README.md`. Use it anywhere Mermaid doesn't render — GitHub markdown
preview, rubrics, printouts, plain-text diffs.

```
                          Network Logs / Traffic Data
                                      │
                                      ▼
                              ┌───────────────┐
                              │   Airflow     │
                              │ (optional)    │
                              └───────┬───────┘
                                      │
          ┌───────────────────────────┼─────────────────────────────┐
          ▼                           ▼                             ▼
   ┌───────────────┐        ┌──────────────────┐      ┌─────────────────┐
   │  Data         │        │  Preprocessing + │      │  Model          │
   │  Validation   │───────▶│  Feature Eng.    │──────▶│  Training       │
   │               │        │                  │      │  5-model zoo    │
   └───────────────┘        └──────────────────┘      └────────┬────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  MLflow         │
                                                     │  (params,       │
                                                     │   metrics,      │
                                                     │   artifacts)    │
                                                     └────────┬────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  Model Registry │
                                                     │  promotion gate │
                                                     │  → Production   │
                                                     │  (champion      │
                                                     │   alias)        │
                                                     └────────┬────────┘
                                                              │
                                                              ▼
   ┌─────────────────┐         ┌──────────────────────────┐
   │  Anomaly         │◀───────│  Serving + monitoring     │
   │  Detection       │        │  loop                     │
   │  (FastAPI)       │        │                           │
   └────────┬────────┘        │  ┌─────────────────────┐  │
            │                 │  │ Model Monitoring     │  │
            ▼                 │  │ hourly DAG: PSI · KS │  │
   ┌─────────────────┐        │  │ · live F1           │  │
   │  Dashboard       │◀──────│  └──────────┬──────────┘  │
   │  (Streamlit)     │        │             │             │
   └─────────────────┘        │             ▼             │
                              │   ┌──────────────┐       │
                              │   │ Drift        │       │
                              │   │ Detection    │       │
                              │   └──────┬───────┘       │
                              │          │               │
                              │   ┌──────┴───────┐       │
                              │   ▼              ▼       │
                              │ No Drift    Drift        │
                              │              Detected    │
                              │                        │
                              │           ┌───────────▼───┐
                              │           │ Airflow       │
                              │           │ triggers      │
                              │           │ Retraining →  │
                              │           │ v2            │
                              │           └───────┬───────┘
                              │                   │
                              └───────────────────┘
```

Read it top-to-bottom, left-to-right:

1. Traffic/data enters via Airflow (or directly for the no-Docker local path).
2. Daily training DAG: validate → preprocess + feature engineering → 5-model
   training zoo → MLflow → Model Registry with gated promotion (`champion` alias).
3. Production model served by FastAPI, consumed by Streamlit dashboard and batch
   scoring.
4. Hourly monitoring DAG scores live traffic, measures drift (PSI / KS / live F1),
   and on threshold fires retraining back into the registry.

Files:
- Source diagram: `README.md` (Mermaid block).
- This fallback: `docs/pipeline.md`.
- Project layout: `README.md` → "Project layout".
