"""
data_generator.py
-----------------
Streamlit-cached random dataset generator for the teaching app.

Adapted from parallelization/data/create_dataset.py: instead of writing a CSV
to disk, this returns an in-memory DataFrame and is cached by Streamlit so
repeated calls with the same arguments are essentially free.
"""

import streamlit as st
import numpy as np
import pandas as pd


@st.cache_data
def generate_dataset(rows: int, cols: int = 10) -> pd.DataFrame:
    """Generate a random float dataset with columns col1..colN. Cached by Streamlit.

    Args:
        rows (int): Number of rows in the dataset.
        cols (int): Number of columns in the dataset (default 10).

    Returns:
        pd.DataFrame: A DataFrame of shape (rows, cols) filled with random
        floats in [0, 1), with columns named col1..colN.
    """
    data = np.random.rand(rows, cols)
    df = pd.DataFrame(data, columns=[f"col{i}" for i in range(1, cols + 1)])
    return df
