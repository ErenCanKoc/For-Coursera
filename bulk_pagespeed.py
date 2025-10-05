
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bulk PageSpeed Insights (PSI) runner for large sites (template-first workflow).

Features
- Reads a CSV of URLs with optional "template" labels.
- Hits PSI v5 API (runPagespeed) concurrently (mobile + desktop supported).
- Extracts both Lab metrics (Lighthouse) and Field metrics (CrUX, when available).
- Writes two CSVs: per-URL raw metrics and per-template aggregates.
- Simple "template inference" fallback from URL paths if no template column provided.
- Rate limiting + robust error handling.

Requirements
- Python 3.9+
- pip install requests pandas
  (For very large runs you may also consider: pip install tenacity)
- A Google API key with PageSpeed Insights enabled:
  https://developers.google.com/speed/docs/insights/v5/get-started

Usage
    python bulk_pagespeed.py \
        --in urls.csv \
        --out outdir \
        --strategy mobile desktop \
        --concurrency 6

Environment
- GOOGLE_API_KEY can be set as an env var instead of --api-key.
- For CI, prefer: python bulk_pagespeed.py --in urls.csv --out out --strategy mobile --api-key $GOOGLE_API_KEY

CSV input format (utf-8)
url,template
https://example.com/ai/signature-generator,tool_landing
https://example.com/templates/nda,template_detail
https://example.com/blog/how-to-sign-pdf,blog_post

Outputs
- outdir/psi_results_raw.csv          (per URL x strategy)
- outdir/psi_results_by_template.csv  (aggregated by template x strategy)

Tips
- Start with 20–50 URLs per template; expand once pipeline is stable.
- Use the aggregated CSV to prioritize template-level dev tasks (max blast radius).
- Compare Lab (Lighthouse) with Field (p75 from CrUX) to validate real-user impact.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd


PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"


@dataclass
class PsiResult:
    url: str
    strategy: str
    template: str
    status: str  # ok | error
    error_message: Optional[str]

    # Lighthouse (Lab) — values are floats (ms for timing; score 0..1)
    lh_performance: Optional[float]
    lh_lcp_ms: Optional[float]
    lh_fcp_ms: Optional[float]
    lh_cls: Optional[float]
    lh_tbt_ms: Optional[float]
    lh_si_ms: Optional[float]  # speed index

    # Field (CrUX) — p75 values where available
    crux_lcp_ms_p75: Optional[float]
    crux_cls_p75: Optional[float]
    crux_inp_ms_p75: Optional[float]
    crux_fcp_ms_p75: Optional[float]

    # Page weight signals
    total_bytes: Optional[float]  # from lighthouse 'total-byte-weight' (bytes)
    req_count: Optional[float]    # from 'network-requests' (count)


def parse_args():
    p = argparse.ArgumentParser(description="Bulk PageSpeed Insights runner")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV with columns: url[,template]")
    p.add_argument("--out", dest="outdir", required=True, help="Output directory")
    p.add_argument("--api-key", dest="api_key", default=os.environ.get("GOOGLE_API_KEY"), help="Google API key")
    p.add_argument("--strategy", nargs="+", default=["mobile"], choices=["mobile", "desktop"], help="PSI strategy list")
    p.add_argument("--category", nargs="+", default=["PERFORMANCE"], help="PSI category list (default: PERFORMANCE)")
    p.add_argument("--locale", default="en_US", help="Locale for PSI (default: en_US)")
    p.add_argument("--concurrency", type=int, default=6, help="Concurrent workers")
    p.add_argument("--sleep", type=float, default=0.0, help="Optional sleep (seconds) between requests per worker")
    p.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds")
    p.add_argument("--infer-template", action="store_true", help="Infer template from URL path when not provided")
    return p.parse_args()


def read_input(infile: str) -> List[Tuple[str, Optional[str]]]:
    rows = []
    with open(infile, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "url" not in reader.fieldnames:
            print("ERROR: Input CSV must have a 'url' column.", file=sys.stderr)
            sys.exit(1)
        for r in reader:
            url = (r.get("url") or "").strip()
            template = (r.get("template") or "").strip() or None
            if not url:
                continue
            rows.append((url, template))
    return rows


def infer_template_from_url(url: str) -> str:
    """
    Very simple heuristic:
    - take first/second path segments to form a label
    - normalize some common patterns
    Customize this for your site (e.g., regex per product family)
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parts = [p for p in (parsed.path or "/").split("/") if p]
        if not parts:
            return "home"
        # Examples of custom mapping you can extend:
        first = parts[0].lower()
        if first in {"blog", "articles"}:
            return "blog_post" if len(parts) > 1 else "blog_index"
        if first in {"templates", "template"}:
            return "template_detail" if len(parts) > 1 else "template_index"
        if first in {"help", "docs"}:
            return "help_doc"
        if first in {"ai", "tools", "apps"}:
            return "tool_landing"
        # default: use first segment
        return first
    except Exception:
        return "unknown"


def safe_get(dct, path: List[str], default=None):
    cur = dct
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default


def ms(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def extract_metrics(data: Dict) -> Dict:
    """Extract a consistent set of metrics from PSI response JSON."""
    lh = data.get("lighthouseResult", {}) or {}
    audits = lh.get("audits", {}) or {}
    categories = lh.get("categories", {}) or {}
    perf_score = safe_get(categories, ["performance", "score"], None)

    # Lab metrics (ms or unitless)
    lh_lcp_ms = ms(safe_get(audits, ["largest-contentful-paint", "numericValue"], None))
    lh_fcp_ms = ms(safe_get(audits, ["first-contentful-paint", "numericValue"], None))
    lh_cls = ms(safe_get(audits, ["cumulative-layout-shift", "numericValue"], None))
    lh_tbt_ms = ms(safe_get(audits, ["total-blocking-time", "numericValue"], None))
    lh_si_ms = ms(safe_get(audits, ["speed-index", "numericValue"], None))

    # Page weight (bytes & requests)
    total_bytes = ms(safe_get(audits, ["total-byte-weight", "numericValue"], None))
    req_count = ms(safe_get(audits, ["network-requests", "details", "items"], None))
    if isinstance(req_count, list):
        req_count = float(len(req_count))
    else:
        req_count = None

    # Field (CrUX) — p75 where available
    crux = data.get("loadingExperience", {}) or {}
    metrics = crux.get("metrics", {}) or {}
    def p75(metric_key: str) -> Optional[float]:
        m = metrics.get(metric_key) or {}
        # Usually in ms for timing; CLS is unitless * 100 (or not) — PSI returns numericValue?
        # We'll try 'percentile' (p75). For CLS, PSI returns value as *100 in some docs; test defensively.
        val = m.get("percentile")
        if val is None:
            return None
        # CLS special-case: percentile is often expressed as 0–100 (e.g., 10 equals 0.10)
        if metric_key == "CUMULATIVE_LAYOUT_SHIFT_SCORE":
            # Convert 0–100 to 0–1 if value > 1
            return float(val) / 100.0 if float(val) > 1 else float(val)
        return float(val)

    crux_lcp_ms_p75 = p75("LARGEST_CONTENTFUL_PAINT_MS")
    crux_cls_p75 = p75("CUMULATIVE_LAYOUT_SHIFT_SCORE")
    crux_inp_ms_p75 = p75("INTERACTION_TO_NEXT_PAINT")
    crux_fcp_ms_p75 = p75("FIRST_CONTENTFUL_PAINT_MS")

    return {
        "lh_performance": perf_score,
        "lh_lcp_ms": lh_lcp_ms,
        "lh_fcp_ms": lh_fcp_ms,
        "lh_cls": lh_cls,
        "lh_tbt_ms": lh_tbt_ms,
        "lh_si_ms": lh_si_ms,
        "total_bytes": total_bytes,
        "req_count": req_count,
        "crux_lcp_ms_p75": crux_lcp_ms_p75,
        "crux_cls_p75": crux_cls_p75,
        "crux_inp_ms_p75": crux_inp_ms_p75,
        "crux_fcp_ms_p75": crux_fcp_ms_p75,
    }


def call_psi(url: str, strategy: str, api_key: str, categories: List[str], locale: str, timeout: float) -> Tuple[str, Optional[Dict], Optional[str]]:
    params = {
        "url": url,
        "strategy": strategy,
        "hl": locale,
    }
    # category can be repeated
    for cat in categories:
        params.setdefault("category", cat)
    if api_key:
        params["key"] = api_key
    try:
        resp = requests.get(PSI_ENDPOINT, params=params, timeout=timeout)
        if resp.status_code != 200:
            return "error", None, f"HTTP {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        # PSI sometimes returns a JSON with 'error'
        if "error" in data:
            return "error", None, json.dumps(data.get("error"))[:500]
        return "ok", data, None
    except requests.exceptions.RequestException as e:
        return "error", None, f"RequestException: {e}"
    except Exception as e:
        return "error", None, f"Exception: {e}"


def run_one(row: Tuple[str, Optional[str]], strategy: str, api_key: str, categories: List[str], locale: str, timeout: float, infer_template: bool, sleep_between: float) -> PsiResult:
    url, tpl = row
    template = tpl or (infer_template_from_url(url) if infer_template else (tpl or "unknown"))
    status, data, err = call_psi(url, strategy, api_key, categories, locale, timeout)
    if sleep_between > 0:
        time.sleep(sleep_between)

    if status != "ok" or not isinstance(data, dict):
        return PsiResult(
            url=url, strategy=strategy, template=template, status="error", error_message=err,
            lh_performance=None, lh_lcp_ms=None, lh_fcp_ms=None, lh_cls=None, lh_tbt_ms=None, lh_si_ms=None,
            crux_lcp_ms_p75=None, crux_cls_p75=None, crux_inp_ms_p75=None, crux_fcp_ms_p75=None,
            total_bytes=None, req_count=None
        )

    m = extract_metrics(data)
    return PsiResult(
        url=url, strategy=strategy, template=template, status="ok", error_message=None,
        lh_performance=m["lh_performance"],
        lh_lcp_ms=m["lh_lcp_ms"],
        lh_fcp_ms=m["lh_fcp_ms"],
        lh_cls=m["lh_cls"],
        lh_tbt_ms=m["lh_tbt_ms"],
        lh_si_ms=m["lh_si_ms"],
        crux_lcp_ms_p75=m["crux_lcp_ms_p75"],
        crux_cls_p75=m["crux_cls_p75"],
        crux_inp_ms_p75=m["crux_inp_ms_p75"],
        crux_fcp_ms_p75=m["crux_fcp_ms_p75"],
        total_bytes=m["total_bytes"],
        req_count=m["req_count"],
    )


def to_dataframe(results: List[PsiResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def aggregate_by_template(df: pd.DataFrame) -> pd.DataFrame:
    # Only aggregate successful rows
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    # Numeric columns to average
    numeric_cols = [
        "lh_performance", "lh_lcp_ms", "lh_fcp_ms", "lh_cls", "lh_tbt_ms", "lh_si_ms",
        "crux_lcp_ms_p75", "crux_cls_p75", "crux_inp_ms_p75", "crux_fcp_ms_p75",
        "total_bytes", "req_count",
    ]
    group = ok.groupby(["template", "strategy"], as_index=False)[numeric_cols].mean(numeric_only=True)

    # Add counts
    counts = ok.groupby(["template", "strategy"]).size().reset_index(name="sample_size")
    out = pd.merge(group, counts, on=["template", "strategy"], how="left")

    # Friendly sort: worst LCP first (mobile prioritized)
    out = out.sort_values(by=["strategy", "lh_lcp_ms"], ascending=[True, False])
    return out


def main():
    args = parse_args()
    if not args.api_key:
        print("ERROR: Provide an API key via --api-key or env GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)

    rows = read_input(args.infile)
    if not rows:
        print("ERROR: No rows found in input.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    jobs = []
    for s in args.strategy:
        for row in rows:
            jobs.append((row, s))

    results: List[PsiResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = []
        for row, strat in jobs:
            fut = ex.submit(
                run_one, row, strat, args.api_key, args.category, args.locale, args.timeout, args.infer_template, args.sleep
            )
            futures.append(fut)
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if i % 10 == 0:
                print(f"[{i}/{len(futures)}] processed...")

    df = to_dataframe(results)
    raw_path = os.path.join(args.outdir, "psi_results_raw.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Wrote {raw_path} ({len(df)} rows)")

    agg = aggregate_by_template(df)
    agg_path = os.path.join(args.outdir, "psi_results_by_template.csv")
    if not agg.empty:
        agg.to_csv(agg_path, index=False, encoding="utf-8")
        print(f"Wrote {agg_path} ({len(agg)} rows)")
    else:
        print("No successful rows to aggregate; check errors in raw CSV.")


if __name__ == "__main__":
    main()
