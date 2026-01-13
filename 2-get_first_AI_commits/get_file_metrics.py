#!/usr/bin/env python3
"""
Compute maintainability metrics for each (owner, repo, file, sha, window_side)
row from a windows CSV (BEFORE/AFTER commits touching a file).

Input (CSV): file_windows_commits.csv
Expected columns:
- owner, repo, file
- ai_sha, ai_author_date
- window_side (BEFORE/AFTER)
- window_index
- sha (the commit touching the file), author_date, commit_url

Output: file_windows_metrics.csv
Adds metrics computed from the file content at that sha:
- loc, sloc
- mi (Maintainability Index)
- functions_count, classes_count
- cc_avg, cc_max

Features:
- resume/checkpoint state (Windows-safe)
- on-disk cache of fetched file contents (optional but recommended)

Usage:
  python compute_metrics_for_windows.py --in file_windows_commits.csv --out file_windows_metrics.csv
  python compute_metrics_for_windows.py --in file_windows_commits.csv --out file_windows_metrics.csv --resume
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
from typing import Dict, Optional, Set, Tuple

import requests
import pandas as pd

from radon.raw import analyze as radon_analyze
from radon.complexity import cc_visit
from radon.metrics import mi_visit


# -----------------------------
# Windows-safe JSON save
# -----------------------------
def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"processed_items": [], "errors": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_windows_safe(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    for _ in range(8):
        try:
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    raise PermissionError(f"Could not replace file: {path}")


def ensure_csv_header(path: str, fieldnames) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_rows_csv(path: str, fieldnames, rows) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)


# -----------------------------
# GitHub access
# -----------------------------
def gh_session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "compute-metrics-for-windows",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    s.headers.update(headers)
    return s


def get_file_text_contents_api(
    s: requests.Session, owner: str, repo: str, file_path: str, sha: str
) -> Optional[str]:
    """
    Uses GitHub Contents API:
      GET /repos/{owner}/{repo}/contents/{path}?ref={sha}
    Returns decoded text, or None if not available.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    r = s.get(url, params={"ref": sha}, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub contents API error {r.status_code}: {r.text[:200]}")
    data = r.json()
    if isinstance(data, dict) and data.get("type") == "file":
        if data.get("encoding") == "base64" and "content" in data:
            raw = base64.b64decode(data["content"])
            # best-effort decode
            return raw.decode("utf-8", errors="replace")
    return None


def get_file_text_raw(
    s: requests.Session, owner: str, repo: str, file_path: str, sha: str
) -> Optional[str]:
    """
    Fallback: raw.githubusercontent.com
    """
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{file_path}"
    r = s.get(url, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub raw fetch error {r.status_code}: {r.text[:200]}")
    # raw content already text/binary; decode best-effort
    return r.content.decode("utf-8", errors="replace")


def cache_key(owner: str, repo: str, sha: str, file_path: str) -> str:
    h = hashlib.sha256(f"{owner}/{repo}@{sha}:{file_path}".encode("utf-8")).hexdigest()
    return h


def get_file_text(
    s: requests.Session,
    owner: str,
    repo: str,
    file_path: str,
    sha: str,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Fetch file content at given sha.
    STRATEGY:
    1) raw.githubusercontent.com (primary, abuse-safe)
    2) Contents API (fallback)
    """

    ck = cache_key(owner, repo, sha, file_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        p = os.path.join(cache_dir, f"{ck}.txt")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    # 1) RAW GitHub (preferred)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{file_path}"
    r = s.get(raw_url, timeout=30)
    if r.status_code == 200:
        txt = r.content.decode("utf-8", errors="replace")
        if cache_dir:
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
        return txt

    # 2) Fallback: Contents API (rare)
    try:
        txt = get_file_text_contents_api(s, owner, repo, file_path, sha)
        if txt is not None and cache_dir:
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
        return txt
    except RuntimeError as e:
        # swallow abuse errors gracefully
        if "429" in str(e):
            time.sleep(10)  # cool down
            return None
        raise



# -----------------------------
# Metrics (Python-oriented via radon)
# -----------------------------
def compute_python_metrics(code: str) -> Dict[str, object]:
    """
    Returns maintainability-related metrics for Python code.
    """
    # radon raw metrics
    raw = radon_analyze(code)
    loc = raw.loc
    sloc = raw.sloc

    # Maintainability Index (0..100-ish)
    # mi_visit signature: mi_visit(code, multi=True)
    mi = mi_visit(code, multi=True)

    # Cyclomatic complexity per block
    blocks = cc_visit(code)
    if blocks:
        cc_values = [b.complexity for b in blocks]
        cc_avg = sum(cc_values) / len(cc_values)
        cc_max = max(cc_values)
    else:
        cc_avg = 0.0
        cc_max = 0

    # crude counts from radon blocks
    # blocks contain functions/methods/classes; we can distinguish by type string
    functions_count = sum(1 for b in blocks if b.__class__.__name__ in ("Function", "Method"))
    classes_count = sum(1 for b in blocks if b.__class__.__name__ == "Class")

    return {
        "loc": loc,
        "sloc": sloc,
        "mi": float(mi),
        "cc_avg": float(cc_avg),
        "cc_max": int(cc_max),
        "functions_count": int(functions_count),
        "classes_count": int(classes_count),
    }


def is_python_file(path: str) -> bool:
    return path.lower().endswith(".py")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",  help="Input CSV (file windows)", default="./output/file_windows_metrics_python.csv")
    ap.add_argument("--out", help="Output CSV (windows + metrics)", default="./output/file_windows_metrics_python_with_analysis.csv")
    ap.add_argument("--state", default="file_windows_metrics_state.json", help="Checkpoint state JSON")
    ap.add_argument("--resume", action="store_true", help="Resume from state")
    ap.add_argument("--cache_dir", default="file_content_cache", help="Directory to cache file contents")
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token (or env GITHUB_TOKEN)")
    args = ap.parse_args()

    # Read token from git-token.txt if present (your style)
    if  os.path.exists("../git-token.txt"):
        with open("../git-token.txt", "r", encoding="utf-8") as tf:
            args.token = tf.read().strip()

    s = gh_session(args.token)

    df = pd.read_csv(args.inp)

    required = {"owner", "repo", "file", "ai_sha", "ai_author_date", "window_side", "window_index", "sha"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in input CSV: {missing}")

    out_fields = [
        # original identifiers
        "owner", "repo", "file",
        "ai_sha", "ai_author_date",
        "window_side", "window_index",
        "sha",
        # metrics
        "metrics_supported",
        "loc", "sloc", "mi", "cc_avg", "cc_max", "functions_count", "classes_count",
        # optional debug
        "error",
    ]
    ensure_csv_header(args.out, out_fields)

    state = load_state(args.state) if args.resume else {"processed_items": [], "errors": []}
    processed: Set[str] = set(state.get("processed_items", []))

    processed_now = 0

    for _, r in df.iterrows():
        owner = str(r["owner"]).strip()
        repo = str(r["repo"]).strip()
        file_path = str(r["file"]).strip()
        ai_sha = str(r["ai_sha"]).strip()
        ai_date = str(r["ai_author_date"]).strip()
        side = str(r["window_side"]).strip()
        idx = str(r["window_index"]).strip()
        sha = str(r["sha"]).strip()

        item_key = f"{owner}/{repo}::{file_path}::{sha}::{side}::{idx}"
        if args.resume and item_key in processed:
            continue

        row_out = {
            "owner": owner,
            "repo": repo,
            "file": file_path,
            "ai_sha": ai_sha,
            "ai_author_date": ai_date,
            "window_side": side,
            "window_index": idx,
            "sha": sha,
            "metrics_supported": False,
            "loc": "",
            "sloc": "",
            "mi": "",
            "cc_avg": "",
            "cc_max": "",
            "functions_count": "",
            "classes_count": "",
            "error": "",
        }

        try:
            if not is_python_file(file_path):
                # Not supported in this minimal version
                row_out["metrics_supported"] = False
            else:
                txt = get_file_text(s, owner, repo, file_path, sha, cache_dir=args.cache_dir)
                if txt is None:
                    row_out["error"] = "FILE_NOT_FOUND_AT_SHA"
                    row_out["metrics_supported"] = False
                else:
                    m = compute_python_metrics(txt)
                    row_out.update(m)
                    row_out["metrics_supported"] = True

            append_rows_csv(args.out, out_fields, [row_out])

            processed.add(item_key)
            state["processed_items"] = sorted(processed)
            save_json_windows_safe(args.state, state)

            processed_now += 1
            if processed_now % 200 == 0:
                print(f"[OK] processed {processed_now} rows")

        except Exception as e:
            err_msg = f"{item_key} | {type(e).__name__}: {e}"
            print(f"[ERR] {err_msg}", file=sys.stderr)
            row_out["error"] = f"{type(e).__name__}: {e}"
            append_rows_csv(args.out, out_fields, [row_out])

            state.setdefault("errors", []).append(err_msg)
            save_json_windows_safe(args.state, state)
            # Not marking processed -> will retry with --resume

    print(f"Done. Processed now: {processed_now}")
    print(f"State: {args.state}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
