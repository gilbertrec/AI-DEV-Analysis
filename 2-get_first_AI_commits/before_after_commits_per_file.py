#!/usr/bin/env python3
"""
Given a file-level CSV with the first AI commit per file, collect K commits
BEFORE and AFTER that AI commit that touched the same file.

Input CSV expected columns (at least):
- owner
- repo
- file
- sha            (AI commit sha)
- author_date    (ISO timestamp of AI commit)

Output:
- file_windows_commits.csv with rows for before/after commits per file.

Usage:
  python collect_before_after_commits_per_file.py \
      --in generated_by_commits_first_per_file.csv \
      --out file_windows_commits.csv \
      --k 10 \
      --state file_windows_state.json \
      --resume

Token:
- reads from git-token.txt if present, else from env GITHUB_TOKEN
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote

import requests


# -----------------------------
# Helpers: IO + Windows-safe save
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


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_rows_csv(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)


# -----------------------------
# GitHub API
# -----------------------------
def gh_session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "collect-before-after-commits-per-file",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    s.headers.update(headers)
    return s


def gh_paginate(s: requests.Session, url: str, params: Optional[dict] = None) -> Iterable[dict]:
    params = dict(params or {})
    params.setdefault("per_page", 100)

    while True:
        r = s.get(url, params=params, timeout=30)

        # basic rate-limit handling
        if r.status_code == 403 and "rate limit" in (r.text or "").lower():
            reset = r.headers.get("X-RateLimit-Reset")
            if reset and reset.isdigit():
                wait_s = max(0, int(reset) - int(time.time()) + 5)
                print(f"Rate limit hit. Sleeping {wait_s}s then retrying...", file=sys.stderr)
                time.sleep(wait_s)
                continue

        if r.status_code >= 400:
            raise RuntimeError(f"GitHub API error {r.status_code} for {url}: {r.text[:400]}")

        data = r.json()
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data

        link = r.headers.get("Link", "")
        next_url = None
        if link:
            for chunk in link.split(","):
                chunk = chunk.strip()
                if 'rel="next"' in chunk:
                    next_url = chunk.split(";", 1)[0].strip()[1:-1]
                    break
        if not next_url:
            break

        url = next_url
        params = None


def parse_iso_dt(s: str) -> datetime:
    # GitHub returns ISO like 2024-01-02T12:34:56Z
    # your CSV likely similar
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def list_commits_touching_file(
    s: requests.Session,
    owner: str,
    repo: str,
    file_path: str,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    max_needed: int = 50,
) -> List[dict]:
    """
    Returns commits touching file_path, newest->oldest, using pagination until max_needed.
    since/until are ISO timestamps.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"path": file_path, "per_page": 100}
    if since_iso:
        params["since"] = since_iso
    if until_iso:
        params["until"] = until_iso

    commits: List[dict] = []
    for item in gh_paginate(s, url, params=params):
        # item: {sha, commit:{author:{date}, message}, html_url, ...}
        commits.append(item)
        if len(commits) >= max_needed:
            break
    return commits


def normalize_commit_row(item: dict) -> dict:
    c = item.get("commit") or {}
    author = (c.get("author") or {})
    return {
        "sha": item.get("sha", ""),
        "commit_url": item.get("html_url", ""),
        "author_date": author.get("date", ""),
        "commit_message": (c.get("message") or "").replace("\n", "\\n"),
    }


# -----------------------------
# Main logic: before/after windows
# -----------------------------
def collect_windows_for_file(
    s: requests.Session,
    owner: str,
    repo: str,
    file_path: str,
    ai_sha: str,
    ai_date_iso: str,
    k: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns (before_rows, after_rows), each list length up to k.
    Rows are normalized commit rows + additional window metadata.
    """
    ai_dt = parse_iso_dt(ai_date_iso)
    ai_iso = ai_dt.isoformat().replace("+00:00", "Z")

    # BEFORE: commits up to ai_date (includes ai commit often). We want K commits strictly before ai_sha.
    before_candidates = list_commits_touching_file(
        s, owner, repo, file_path, since_iso=None, until_iso=ai_iso, max_needed=200
    )

    before_rows: List[dict] = []
    for it in before_candidates:
        sha = it.get("sha", "")
        if sha == ai_sha:
            continue
        row = normalize_commit_row(it)
        # Keep only commits strictly before AI time (defensive)
        if row["author_date"]:
            dt = parse_iso_dt(row["author_date"])
            if dt >= ai_dt:
                continue
        before_rows.append(row)
        if len(before_rows) >= k:
            break

    # AFTER: commits since ai_date. API returns newest->oldest; we want the *closest after* (earliest after).
    after_candidates = list_commits_touching_file(
        s, owner, repo, file_path, since_iso=ai_iso, until_iso=None, max_needed=500
    )

    # Filter strictly after AI commit (by sha + time), then sort ascending by time and take first k.
    tmp: List[dict] = []
    for it in after_candidates:
        sha = it.get("sha", "")
        if sha == ai_sha:
            continue
        row = normalize_commit_row(it)
        if not row["author_date"]:
            continue
        dt = parse_iso_dt(row["author_date"])
        if dt <= ai_dt:
            continue
        tmp.append(row)

    tmp.sort(key=lambda r: parse_iso_dt(r["author_date"]))
    after_rows = tmp[:k]

    return before_rows, after_rows


def read_input_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header.")
        rows = [r for r in reader]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",  help="Input CSV (file windows)", default="./output/generated_by_commits_first_per_file.csv")
    ap.add_argument("--out", help="Output CSV (windows + metrics)", default="./output/file_windows_metrics.csv")
    ap.add_argument("--k", type=int, default=10, help="Window size K (commits before and after)")
    ap.add_argument("--state", default="file_windows_state.json", help="Checkpoint state JSON")
    ap.add_argument("--resume", action="store_true", help="Resume from state")

    args = ap.parse_args()

    # Read token from git-token.txt if present (your style)
    if os.path.exists("../git-token.txt"):
        with open("../git-token.txt", "r", encoding="utf-8") as tf:
            args.token = tf.read().strip()
    s = gh_session(args.token)

    rows_in = read_input_csv(args.inp)

    # Required columns
    needed = {"owner", "repo", "file", "sha", "author_date"}
    missing = needed - set(rows_in[0].keys() if rows_in else set())
    if missing:
        raise RuntimeError(f"Missing columns in input CSV: {missing}")

    fieldnames = [
        "owner",
        "repo",
        "file",
        "ai_sha",
        "ai_author_date",
        "window_side",      # BEFORE / AFTER
        "window_index",     # 1..K (closest first)
        "sha",
        "author_date",
        "commit_url",
        "commit_message",
    ]
    ensure_csv_header(args.out, fieldnames)

    state = load_state(args.state) if args.resume else {"processed_items": [], "errors": []}
    processed: Set[str] = set(state.get("processed_items", []))

    processed_now = 0

    for r in rows_in:
        owner = (r.get("owner") or "").strip()
        repo = (r.get("repo") or "").strip()
        file_path = (r.get("file") or "").strip()
        ai_sha = (r.get("sha") or "").strip()
        ai_date = (r.get("author_date") or "").strip()

        if not (owner and repo and file_path and ai_sha and ai_date):
            continue

        item_key = f"{owner}/{repo}::{file_path}::{ai_sha}"
        if args.resume and item_key in processed:
            continue

        try:
            before, after = collect_windows_for_file(
                s=s,
                owner=owner,
                repo=repo,
                file_path=file_path,
                ai_sha=ai_sha,
                ai_date_iso=ai_date,
                k=args.k,
            )

            out_rows: List[dict] = []

            # BEFORE: closest first already (newest-before)
            for i, c in enumerate(before, start=1):
                out_rows.append({
                    "owner": owner,
                    "repo": repo,
                    "file": file_path,
                    "ai_sha": ai_sha,
                    "ai_author_date": ai_date,
                    "window_side": "BEFORE",
                    "window_index": i,
                    "sha": c["sha"],
                    "author_date": c["author_date"],
                    "commit_url": c["commit_url"],
                    "commit_message": c["commit_message"],
                })

            # AFTER: closest after first (earliest-after)
            for i, c in enumerate(after, start=1):
                out_rows.append({
                    "owner": owner,
                    "repo": repo,
                    "file": file_path,
                    "ai_sha": ai_sha,
                    "ai_author_date": ai_date,
                    "window_side": "AFTER",
                    "window_index": i,
                    "sha": c["sha"],
                    "author_date": c["author_date"],
                    "commit_url": c["commit_url"],
                    "commit_message": c["commit_message"],
                })

            append_rows_csv(args.out, fieldnames, out_rows)

            processed.add(item_key)
            state["processed_items"] = sorted(processed)
            save_json_windows_safe(args.state, state)

            processed_now += 1
            if processed_now % 50 == 0:
                print(f"[OK] processed {processed_now} items (latest: {item_key})")

        except Exception as e:
            err_msg = f"{item_key} | {type(e).__name__}: {e}"
            print(f"[ERR] {err_msg}", file=sys.stderr)
            state.setdefault("errors", []).append(err_msg)
            save_json_windows_safe(args.state, state)
            # Not marking processed -> will retry in resume.

    print(f"Done. Processed now: {processed_now}")
    print(f"State: {args.state}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
