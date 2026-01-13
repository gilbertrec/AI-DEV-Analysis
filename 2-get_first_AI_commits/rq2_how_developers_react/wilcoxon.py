import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =========================
# CONFIG
# =========================
INPUT = "rq2_maintenance_activities_90d.csv"
OUTPUT = "rq2_stats_summary.csv"
MIN_PAIRS = 20

METRICS = [
    "post_commits_count",
    "post_churn",
    "ttnc_days",
]

# =========================
# CLIFF'S DELTA (PAIRED)
# =========================
def cliffs_delta_paired(before, after):
    b = np.asarray(before)
    a = np.asarray(after)
    gt = np.sum(a > b)
    lt = np.sum(a < b)
    return (gt - lt) / len(b)

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
# LOAD
# =========================
df = pd.read_csv(INPUT)

# numeric
for m in METRICS:
    df[m] = pd.to_numeric(df[m], errors="coerce")

# =========================
# BUILD PAIRS
# =========================
# pairing key: same file + AI-control linkage
key_cols = ["owner", "repo", "file"]

ai = df[df["event_type"] == "AI"].copy()
ctrl = df[df["event_type"] == "CONTROL"].copy()

# join CONTROL to AI via matched_ai_sha
pairs = ai.merge(
    ctrl,
    left_on=key_cols + ["event_sha"],
    right_on=key_cols + ["matched_ai_sha"],
    suffixes=("_AI", "_CTRL"),
)

print(f"Total AI–CONTROL pairs: {len(pairs)}")

results = []

# =========================
# STATS
# =========================
for metric in METRICS:
    a = pairs[f"{metric}_AI"]
    c = pairs[f"{metric}_CTRL"]

    mask = (~a.isna()) & (~c.isna())
    a = a[mask]
    c = c[mask]

    if len(a) < MIN_PAIRS:
        results.append({
            "metric": metric,
            "n_pairs": len(a),
            "note": f"SKIPPED (<{MIN_PAIRS})"
        })
        continue

    # Wilcoxon (paired)
    diffs = a.values - c.values
    if np.all(diffs == 0):
        w_stat, p_val = np.nan, 1.0
    else:
        w_stat, p_val = wilcoxon(a, c, alternative="two-sided")

    # Cliff's Delta
    delta = cliffs_delta_paired(c.values, a.values)  # AI vs CONTROL
    mag = cliffs_magnitude(delta)

    results.append({
        "metric": metric,
        "n_pairs": len(a),
        "median_AI": float(a.median()),
        "median_CONTROL": float(c.median()),
        "median_delta": float((a - c).median()),
        "wilcoxon_W": float(w_stat),
        "p_value": float(p_val),
        "cliffs_delta": float(delta),
        "effect_size": mag,
        "note": ""
    })

summary = pd.DataFrame(results)
summary.to_csv(OUTPUT, index=False)

print("\n=== RQ2: Wilcoxon AI vs CONTROL ===")
print(summary.to_string(index=False))
print(f"\nSaved to: {OUTPUT}")
