import pandas as pd

INPUT_CSV = "input/generated_by_commits.csv"
OUTPUT_CSV = "output/generated_by_commits_first_per_file.csv"

# 1) Load CSV
df = pd.read_csv(INPUT_CSV)

# Sanity checks
required_cols = {"sha", "author_date", "files_changed"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in input CSV: {missing}")

# 2) Parse date
df["author_date"] = pd.to_datetime(df["author_date"], errors="coerce")
df = df.dropna(subset=["author_date"])

# 3) Expand rows: one row per (commit, file)
rows = []
for _, r in df.iterrows():
    files = str(r["files_changed"]).split(";")
    for f in files:
        f = f.strip()
        if not f:
            continue
        rows.append({
            "file": f,
            "sha": r["sha"],
            "author_date": r["author_date"],
            "commit_url": r.get("commit_url", ""),
            "pr_url": r.get("pr_url", ""),
            "repo": r.get("repo", ""),
            "owner": r.get("owner", "")
        })

files_df = pd.DataFrame(rows)

# 4) Sort chronologically
files_df = files_df.sort_values("author_date")

# 5) Keep only FIRST AI commit per file
filtered = (
    files_df
    .drop_duplicates(subset=["file"], keep="first")
    .reset_index(drop=True)
)

# 6) Save result
filtered.to_csv(OUTPUT_CSV, index=False)

print(f"Input rows (commit-level): {len(df)}")
print(f"Expanded rows (file-level): {len(files_df)}")
print(f"Filtered rows (first AI per file): {len(filtered)}")
print(f"Saved to: {OUTPUT_CSV}")
