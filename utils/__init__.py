"""Shared utilities for the teaching Streamlit app.

All submodules are deliberately UI-free (no Streamlit imports outside
``data_generator``), so pages can cache/render them and tests can exercise them.

Submodules:
    data_generator    -- cached random dataset generation
    db_utils          -- PostgreSQL/SQLite connection + CRUD (Database Basics page)
    compute_utils     -- serial/parallel/vectorized benchmarks (Parallelization page)
    io_utils          -- sequential/threaded/async/multiprocess I/O benchmarks
    teaching          -- source_of, script splitting, architecture diagrams
    recommender_utils -- content-based / collaborative / association-rule recommenders
    propensity_utils  -- churn classification, tuning, metrics, risk segments
    survival_utils    -- Kaplan-Meier, Cox/RSF/SVM, censoring-aware evaluation
    segmentation_utils-- RFM, cohort-measured CLTV, clustering (adapts the sibling repo)
"""
