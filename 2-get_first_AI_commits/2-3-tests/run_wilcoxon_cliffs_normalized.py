import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =========================
# CONFIG
# =========================
INPUT_CSV = "../output/file_windows_metrics_python_with_analysis.csv"
OUTPUT_SUMMARY = "stats_summary.csv"

AGG = "median"          # "median" consigliata, oppure "mean"
MIN_PAIRS = 20          # soglia minima per test
EPS = 1e-9              # per evitare divisioni per zero

# Metriche base disponibili nel tuo CSV (se alcune mancano, verranno skippate)
BASE_METRICS = ["mi", "cc_avg", "cc_max", "loc", "sloc"]

# =========================
# Effect size: Cliff's Delta (paired)
# =========================
def cliffs_delta_paired(before, after):
    """
    Paired Cliff's Delta: confronta AFTER vs BEFORE per ciascuna coppia.
    Δ = ( #(after>before) - #(after<before) ) / N
    """
    b = np.asarray(before)
    a = np.asarray(after)
    assert len(b) == len(a)
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
# Helpers
# =========================
def safe_rel_delta(after, before):
    """(after-before)/before con protezione."""
    before = np.asarray(before)
    after = np.asarray(after)
    denom = np.where(np.abs(before) < EPS, np.nan, before)
    return (after - before) / denom

def safe_div(num, den):
    den = np.asarray(den)
    num = np.asarray(num)
    den2 = np.where(np.abs(den) < EPS, np.nan, den)
    return num / den2

def describe_pair(before, after):
    before = pd.Series(before)
    after = pd.Series(after)
    d = after - before
    return {
        "n_pairs": len(d),
        "median_before": float(before.median()),
        "median_after": float(after.median()),
        "median_delta": float(d.median()),
        "median_rel_delta": float(pd.Series(safe_rel_delta(after, before)).dropna().median()),
    }

def run_wilcoxon(before, after):
    # Wilcoxon richiede almeno una differenza non zero; gestiamo il caso
    diffs = np.asarray(after) - np.asarray(before)
    if np.all(diffs == 0):
        return np.nan, 1.0
    stat, p = wilcoxon(before, after, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p)

# =========================
# Load
# =========================
df = pd.read_csv(INPUT_CSV)

# filtri minimi
df = df[df.get("metrics_supported", False) == True].copy()

# colonne necessarie per pairing
needed = {"owner", "repo", "file", "ai_sha", "window_side"}
missing = needed - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# tieni solo BEFORE/AFTER
df = df[df["window_side"].isin(["BEFORE", "AFTER"])]

# =========================
# Aggregazione per evento e window
# (owner, repo, file, ai_sha, window_side)
# =========================
event_cols = ["owner", "repo", "file", "ai_sha", "window_side"]

# scegli metriche presenti
metrics_present = [m for m in BASE_METRICS if m in df.columns]
if not metrics_present:
    raise RuntimeError("No metrics found among: " + ", ".join(BASE_METRICS))

# converti metriche in numeric
for m in metrics_present:
    df[m] = pd.to_numeric(df[m], errors="coerce")

if AGG == "median":
    event_df = df.groupby(event_cols)[metrics_present].median(numeric_only=True).reset_index()
else:
    event_df = df.groupby(event_cols)[metrics_present].mean(numeric_only=True).reset_index()

# =========================
# Costruisci metriche normalizzate (per dimensione)
# =========================
# Usa SLOC come denominatore principale (più stabile di LOC)
if "sloc" in event_df.columns:
    if "cc_avg" in event_df.columns:
        event_df["cc_avg_per_sloc"] = safe_div(event_df["cc_avg"].values, event_df["sloc"].values)
    if "cc_max" in event_df.columns:
        event_df["cc_max_per_sloc"] = safe_div(event_df["cc_max"].values, event_df["sloc"].values)

# aggiungi alle metriche da testare
ALL_METRICS = metrics_present + [c for c in ["cc_avg_per_sloc", "cc_max_per_sloc"] if c in event_df.columns]

# =========================
# Pivot BEFORE/AFTER per ciascuna metrica
# =========================
idx = ["owner", "repo", "file", "ai_sha"]
results = []

for metric in ALL_METRICS:
    piv = event_df.pivot_table(index=idx, columns="window_side", values=metric).dropna()
    if "BEFORE" not in piv.columns or "AFTER" not in piv.columns:
        continue

    before = piv["BEFORE"].values
    after = piv["AFTER"].values

    # rimuovi NaN eventuali (divisioni per zero -> NaN)
    ok = ~np.isnan(before) & ~np.isnan(after)
    before = before[ok]
    after = after[ok]

    if len(before) < MIN_PAIRS:
        results.append({
            "metric": metric,
            "n_pairs": len(before),
            "note": f"SKIPPED (<{MIN_PAIRS} pairs)",
        })
        continue

    # stats
    desc = describe_pair(before, after)
    w_stat, p_value = run_wilcoxon(before, after)
    delta = cliffs_delta_paired(before, after)
    mag = cliffs_magnitude(delta)

    results.append({
        "metric": metric,
        "n_pairs": desc["n_pairs"],
        "median_before": desc["median_before"],
        "median_after": desc["median_after"],
        "median_delta": desc["median_delta"],
        "median_rel_delta": desc["median_rel_delta"],
        "wilcoxon_W": w_stat,
        "p_value": p_value,
        "cliffs_delta": float(delta),
        "effect_magnitude": mag,
        "note": "",
    })

summary = pd.DataFrame(results)
summary.to_csv(OUTPUT_SUMMARY, index=False)

print("Done.")
print(f"Saved summary to: {OUTPUT_SUMMARY}")
print("\nTop results (sorted by p-value):")
print(summary.sort_values(by="p_value", na_position="last").head(15).to_string(index=False))
