import pandas as pd
import numpy as np
import time

def safe_mean(series):
    return float(np.mean(series))

# Mock basic setup
BASIC_COLS = [f"col_{i}" for i in range(162)]

def extract_slow(df):
    features = []
    for col in BASIC_COLS:
        if col in df.columns:
            features.append(safe_mean(df[col]))
        else:
            features.append(0.0)
    return np.array(features)

def extract_fast(df):
    available_cols = [c for c in BASIC_COLS if c in df.columns]
    if available_cols:
        means = df[available_cols].mean(numeric_only=True).fillna(0.0).to_dict()
    else:
        means = {}
    return np.array([means.get(c, 0.0) for c in BASIC_COLS])

df = pd.DataFrame({f"col_{i}": np.random.randn(100) for i in range(150)})

t0 = time.time()
for _ in range(1000):
    extract_slow(df)
t1 = time.time()
print("Slow:", t1 - t0)

t0 = time.time()
for _ in range(1000):
    extract_fast(df)
t1 = time.time()
print("Fast:", t1 - t0)
