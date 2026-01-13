#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

# ========= CONFIG =========
PER_PAGE = 100
SLEEP_BETWEEN_REQUESTS = 0.25
MAX_RETRIES = 6
MAX_COMMITS_SCAN = 2000  # per file-window (90d); alza se repo molto attivi

# ========= TIME =========
def parse_iso_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# ========= STATE (Windows-safe) =========
def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"processed": [], "errors": []}
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
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def append_rows(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)

# ========= GitHub API =========
def gh_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/vnd.github+json",
        "User-Agent": "rq2-post90d-per-file",
        "Authorization": f"Bearer {token}",
    })
    return s

def _sleep():
    time.sleep(SLEEP_BETWEEN_REQUESTS)

def gh_get_retry(s: requests.Session, url: str, params: Optional[dict] = None) -> requests.Response:
    for attempt in range(1, MAX_RETRIES + 1):
        r = s.get(url, params=params, timeout=30)
        _sleep()

        if r.status_code == 403 and "rate limit exceeded" in (r.text or "").lower():
            reset = r.headers.get("X-RateLimit-Reset")
            if reset and reset.isdigit():
                wait_s = max(0, int(reset) - int(time.time()) + 5)
                print(f"[RL] Rate limit hit. Sleeping {wait_s}s...", file=sys.stderr)
                time.sleep(wait_s)
                continue

        if r.status_code in (429, 403) and "abuse" in (r.text or "").lower():
            cool = 30 * attempt
            print(f"[ABUSE] Cooling down {cool}s...", file=sys.stderr)
            time.sleep(cool)
            continue

        if r.status_code >= 500:
            backoff = 2 ** attempt
            print(f"[5XX] {r.status_code}. Backoff {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            continue

        return r

    raise RuntimeError(f"Failed after retries: {url}")

def gh_paginate(s: requests.Session, url: str, params: dict) -> Iterable[dict]:
    params = dict(params)
    params.setdefault("per_page", PER_PAGE)
    while True:
        r = gh_get_retry(s, url, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f"GitHub API error {r.status_code} for {url}: {r.text[:300]}")
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected response (not list) for {url}: {str(data)[:200]}")
        for item in data:
            yield item

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
        params = {}

def get_commit_detail_cached(
    s: requests.Session,
    owner: str,
    repo: str,
    sha: str,
    cache: Dict[str, dict],
) -> dict:
    key = f"{owner}/{repo}@{sha}"
    if key in cache:
        return cache[key]
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    r = gh_get_retry(s, url)
    if r.status_code >= 400:
        raise RuntimeError(f"Commit detail error {r.status_code}: {r.text[:200]}")
    cache[key] = r.json()
    return cache[key]

def list_commits_touching_file(
    s: requests.Session,
    owner: str,
    repo: str,
    file_path: str,
    since_iso: str,
    until_iso: str,
    max_items: int,
) -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"path": file_path, "since": since_iso, "until": until_iso, "per_page": PER_PAGE}
    out = []
    for it in gh_paginate(s, url, params=params):
        out.append(it)
        if len(out) >= max_items:
            break
    return out

def commit_author_date_from_list_item(item: dict) -> Optional[str]:
    c = item.get("commit") or {}
    a = (c.get("author") or {})
    return a.get("date")

def extract_message_parts(message_full: str) -> Tuple[str, str, str]:
    message_full = message_full or ""
    subject = message_full.splitlines()[0] if message_full else ""
    body = "\n".join(message_full.splitlines()[1:]).strip() if "\n" in message_full else ""
    one_line = " ".join(message_full.splitlines()).strip()
    return subject, body, one_line

def file_stats_from_commit_detail(detail: dict, file_path: str) -> Tuple[int, int, int]:
    for f in (detail.get("files") or []):
        if f.get("filename") == file_path:
            add = int(f.get("additions", 0) or 0)
            dele = int(f.get("deletions", 0) or 0)
            chg = int(f.get("changes", 0) or 0)
            return add, dele, chg
    return 0, 0, 0

def compute_post_maintenance_90d_for_file(
    s: requests.Session,
    owner: str,
    repo: str,
    file_path: str,
    event_sha: str,
    t0: datetime,
    detail_cache: Dict[str, dict],
) -> Tuple[int, Optional[float], int, int, int]:
    since = to_iso_z(t0)
    until = to_iso_z(t0 + timedelta(days=90))

    commits = list_commits_touching_file(s, owner, repo, file_path, since, until, max_items=MAX_COMMITS_SCAN)

    post = []
    for it in commits:
        sha = it.get("sha", "")
        if not sha or sha == event_sha:
            continue
        d = commit_author_date_from_list_item(it)
        if not d:
            continue
        dt = parse_iso_dt(d)
        if dt <= t0:
            continue
        post.append((dt, sha))

    post.sort(key=lambda x: x[0])

    post_commits_count = len(post)
    ttnc_days = None
    if post:
        ttnc_days = (post[0][0] - t0).total_seconds() / 86400.0

    add_sum = del_sum = 0
    for _, sha in post:
        det = get_commit_detail_cached(s, owner, repo, sha, detail_cache)
        a, d, _ = file_stats_from_commit_detail(det, file_path)
        add_sum += a
        del_sum += d

    return post_commits_count, ttnc_days, add_sum, del_sum, add_sum + del_sum

# ========= CSV IO =========
def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_json_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_cache(path: str, data: dict) -> None:
    save_json_windows_safe(path, data)

# ========= MAIN =========
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../output/file_windows_metrics_python_with_analysis.csv", help="CSV with owner, repo, sha (all AI)")
    ap.add_argument("--out", default="rq2_post90d_per_file.csv", help="Output CSV (one row per ai_sha + file)")
    ap.add_argument("--state", default="rq2_post90d_per_file_state.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN"))
    ap.add_argument("--token_file", default="../../git-token.txt")
    ap.add_argument("--cache", default="commit_details_cache.json", help="On-disk cache for commit details")
    args = ap.parse_args()

    token = (args.token or "").strip()
    if not token and os.path.exists(args.token_file):
        token = open(args.token_file, "r", encoding="utf-8").read().strip()
    if not token:
        raise RuntimeError("No GitHub token found. Set GITHUB_TOKEN or provide git-token.txt")

    s = gh_session(token)

    rows = read_csv(args.inp)
    if not rows:
        raise RuntimeError("Input CSV is empty.")
    needed = {"owner", "repo", "sha"}
    missing = needed - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"Missing required columns in input CSV: {missing}")

    out_fields = [
        "owner", "repo",
        "ai_sha", "ai_author_date",
        "ai_message_subject", "ai_message_body", "ai_message_one_line",
        "file",
        "ai_file_additions", "ai_file_deletions", "ai_file_changes",
        "post_commits_count", "ttnc_days",
        "post_additions", "post_deletions", "post_churn",
        "error",
    ]
    ensure_csv_header(args.out, out_fields)

    state = load_state(args.state) if args.resume else {"processed": [], "errors": []}
    processed: Set[str] = set(state.get("processed", []))

    cache: Dict[str, dict] = load_json_cache(args.cache)

    done = 0
    for r in rows:
        owner = (r.get("owner") or "").strip()
        repo = (r.get("repo") or "").strip()
        sha = (r.get("sha") or "").strip()
        if not (owner and repo and sha):
            continue

        try:
            det = get_commit_detail_cached(s, owner, repo, sha, cache)

            ai_date = (det.get("commit", {}).get("author", {}) or {}).get("date", "")
            if not ai_date:
                raise RuntimeError("Missing author date in commit detail")
            t0 = parse_iso_dt(ai_date)

            msg_full = (det.get("commit", {}) or {}).get("message", "") or ""
            subj, body, one_line = extract_message_parts(msg_full)

            files = det.get("files") or []
            if not files:
                # still write something (rare)
                key = f"{owner}/{repo}@{sha}::NOFILES"
                if args.resume and key in processed:
                    continue
                append_rows(args.out, out_fields, [{
                    "owner": owner, "repo": repo,
                    "ai_sha": sha, "ai_author_date": ai_date,
                    "ai_message_subject": subj, "ai_message_body": body, "ai_message_one_line": one_line,
                    "file": "",
                    "ai_file_additions": "", "ai_file_deletions": "", "ai_file_changes": "",
                    "post_commits_count": "", "ttnc_days": "",
                    "post_additions": "", "post_deletions": "", "post_churn": "",
                    "error": "NO_FILES_IN_COMMIT_DETAIL",
                }])
                processed.add(key)
                state["processed"] = sorted(processed)
                save_json_windows_safe(args.state, state)
                continue

            batch_rows = []
            for f in files:
                file_path = f.get("filename", "") or ""
                if not file_path:
                    continue

                key = f"{owner}/{repo}@{sha}::{file_path}"
                if args.resume and key in processed:
                    continue

                ai_add = int(f.get("additions", 0) or 0)
                ai_del = int(f.get("deletions", 0) or 0)
                ai_chg = int(f.get("changes", 0) or 0)

                try:
                    pc, ttnc, pa, pd, pch = compute_post_maintenance_90d_for_file(
                        s, owner, repo, file_path, sha, t0, cache
                    )
                    batch_rows.append({
                        "owner": owner, "repo": repo,
                        "ai_sha": sha, "ai_author_date": ai_date,
                        "ai_message_subject": subj, "ai_message_body": body, "ai_message_one_line": one_line,
                        "file": file_path,
                        "ai_file_additions": ai_add, "ai_file_deletions": ai_del, "ai_file_changes": ai_chg,
                        "post_commits_count": pc,
                        "ttnc_days": "" if ttnc is None else f"{ttnc:.6f}",
                        "post_additions": pa, "post_deletions": pd, "post_churn": pch,
                        "error": "",
                    })
                except Exception as e_file:
                    batch_rows.append({
                        "owner": owner, "repo": repo,
                        "ai_sha": sha, "ai_author_date": ai_date,
                        "ai_message_subject": subj, "ai_message_body": body, "ai_message_one_line": one_line,
                        "file": file_path,
                        "ai_file_additions": ai_add, "ai_file_deletions": ai_del, "ai_file_changes": ai_chg,
                        "post_commits_count": "", "ttnc_days": "",
                        "post_additions": "", "post_deletions": "", "post_churn": "",
                        "error": f"{type(e_file).__name__}: {e_file}",
                    })

                processed.add(key)

            append_rows(args.out, out_fields, batch_rows)

            state["processed"] = sorted(processed)
            save_json_windows_safe(args.state, state)

            # salva cache ogni tanto
            if len(cache) % 300 == 0:
                save_json_cache(args.cache, cache)

            done += 1
            if done % 50 == 0:
                print(f"[OK] processed {done} AI commits (expanded per file)")

        except Exception as e:
            msg = f"{owner}/{repo}@{sha} | {type(e).__name__}: {e}"
            print("[ERR]", msg, file=sys.stderr)
            state.setdefault("errors", []).append(msg)
            save_json_windows_safe(args.state, state)

    save_json_cache(args.cache, cache)
    print("Done. Output:", args.out)
    print("State :", args.state)
    print("Cache :", args.cache)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
