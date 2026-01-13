import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INPUT_CSV = "output/file_windows_metrics_python_with_analysis.csv"
METRIC = "sloc"        # scegli: mi, cc_avg, cc_max, loc, sloc
AGG = "median"       # median (consigliata) oppure mean
OUTPUT_PNG = f"plots/before_after_{METRIC}.png"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

# keep only valid metric rows
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
# AGGREGATE PER WINDOW
# =========================
summary = (
    event_df
    .groupby("window_side")[METRIC]
    .agg(["median", "quantile"])
)

# Compute IQR manually
before_vals = event_df[event_df["window_side"] == "BEFORE"][METRIC]
after_vals  = event_df[event_df["window_side"] == "AFTER"][METRIC]

stats = pd.DataFrame({
    "window": ["BEFORE", "AFTER"],
    "median": [before_vals.median(), after_vals.median()],
    "q1": [before_vals.quantile(0.25), after_vals.quantile(0.25)],
    "q3": [before_vals.quantile(0.75), after_vals.quantile(0.75)],
})

# =========================
# PLOT
# =========================
x = [0, 1]
y = stats["median"].values
yerr = [
    y - stats["q1"].values,
    stats["q3"].values - y
]

plt.figure(figsize=(5, 4))
plt.errorbar(
    x, y,
    yerr=yerr,
    fmt='-o',
    capsize=6
)

plt.xticks(x, ["BEFORE", "AFTER"])
plt.ylabel(METRIC.upper())
plt.xlabel("Window")
plt.title(f"AI-generated commit impact on {METRIC.upper()}")

plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print(f"Saved plot to {OUTPUT_PNG}")
