import pandas as pd

INPUT_CSV = "file_windows_metrics.csv"
OUTPUT_CSV = "file_windows_metrics_python.csv"

# 1) Load CSV
df = pd.read_csv(INPUT_CSV)
# 2) Filter to Python files only
df_python = df[df["file"].str.endswith(".py", na=False)].reset_index(drop=True)

# 3) Save result
df_python.to_csv(OUTPUT_CSV, index=False)