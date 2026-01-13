import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =========================
# CONFIG
# =========================
INPUT_CSV = "../output/file_windows_metrics_python_with_analysis.csv"
METRIC = "cc_avg"   # scegli: mi, cc_avg, cc_max, loc, sloc
AGG = "median"  # median (consigliata) oppure mean
MIN_PAIRS = 10  # soglia minima per test

# =========================
# CLIFF'S DELTA
# =========================
def cliffs_delta(x, y):
    """
    Compute Cliff's Delta for paired samples x, y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert len(x) == len(y)

    gt = lt = 0
    for xi, yi in zip(x, y):
        if yi > xi:
            gt += 1
        elif yi < xi:
            lt += 1
    return (gt - lt) / len(x)

def cliffs_magnitude(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"

# =========================
# LOAD + FILTER
# =========================
df = pd.read_csv(INPUT_CSV)

df = df[df["metrics_supported"] == True]
df = df.dropna(subset=[METRIC])

# =========================
# AGGREGATE PER EVENTO
# (file + ai_sha + window_side)
# =========================
event_cols = ["owner", "repo", "file", "ai_sha", "window_side"]

if AGG == "median":
    event_df = df.groupby(event_cols)[METRIC].median().reset_index()
else:
    event_df = df.groupby(event_cols)[METRIC].mean().reset_index()

# =========================
# PIVOT BEFORE / AFTER
# =========================
pivot = event_df.pivot_table(
    index=["owner", "repo", "file", "ai_sha"],
    columns="window_side",
    values=METRIC
).dropna()

if "BEFORE" not in pivot.columns or "AFTER" not in pivot.columns:
    raise RuntimeError("Missing BEFORE or AFTER values after pivot.")

before = pivot["BEFORE"]
after = pivot["AFTER"]

n_pairs = len(pivot)

print(f"Metric: {METRIC}")
print(f"Pairs: {n_pairs}")

if n_pairs < MIN_PAIRS:
    print("Not enough pairs for statistical testing.")
    exit(0)

# =========================
# WILCOXON SIGNED-RANK
# =========================
stat, p_value = wilcoxon(before, after, alternative="two-sided")

# =========================
# CLIFF'S DELTA
# =========================
delta = cliffs_delta(before.values, after.values)
magnitude = cliffs_magnitude(delta)

# =========================
# SUMMARY
# =========================
print("\n--- RESULTS ---")
print(f"Median BEFORE: {before.median():.4f}")
print(f"Median AFTER : {after.median():.4f}")
print(f"Median Δ     : {(after - before).median():.4f}")
print()
print(f"Wilcoxon W   : {stat:.4f}")
print(f"p-value      : {p_value:.6f}")
print()
print(f"Cliff's Δ    : {delta:.4f}")
print(f"Effect size  : {magnitude}")
