#!/usr/bin/env python3
"""
Derive a Stage-2 red-team plan from Stage-1 OpenAI analysis.

Input:
    data/redteam_openai_analysis.jsonl  (from scripts/redteam_analyze_openai.py)

Output:
    data/redteam_stage2_plan.json

The plan summarizes where Lexi struggled (unsafe/borderline/weak refusals) and
recommends how many Stage-2 prompts to generate per category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_PATH = REPO_ROOT / "data" / "redteam_openai_analysis.jsonl"
DEFAULT_PLAN_PATH = REPO_ROOT / "data" / "redteam_stage2_plan.json"


def load_analysis(path: Path) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield (row, analysis) tuples from a JSONL of Stage-1 annotations."""
    if not path.exists():
        raise SystemExit(f"Analysis file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            analysis = row.get("analysis", {})
            if not isinstance(analysis, dict):
                analysis = {}
            yield row, analysis


def aggregate(rows: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """Aggregate failure counts per category."""
    by_category: Dict[str, Dict[str, int]] = {}

    for row, analysis in rows:
        cat = str(row.get("category") or "unknown")
        labels = analysis.get("labels") or {}
        safety_label = str(labels.get("safety_label") or "").lower()
        should_refuse = bool(labels.get("should_have_refused"))
        refusal_quality = str(labels.get("refusal_quality") or "").lower()
        status = row.get("status")
        is_error = status is not None and status != 200

        # Signals of trouble for Stage-2 targeting
        unsafe = safety_label and safety_label != "safe"
        borderline = safety_label == "borderline"
        weak_refusal = should_refuse and refusal_quality in {"weak", "no_refusal"}
        flagged = unsafe or borderline or should_refuse or weak_refusal

        stats = by_category.setdefault(
            cat,
            {
                "total": 0,
                "unsafe_or_borderline": 0,
                "unsafe": 0,
                "borderline": 0,
                "should_have_refused": 0,
                "weak_refusals": 0,
                "transport_errors": 0,
            },
        )

        stats["total"] += 1
        if unsafe:
            stats["unsafe"] += 1
        if borderline:
            stats["borderline"] += 1
        if should_refuse:
            stats["should_have_refused"] += 1
        if weak_refusal:
            stats["weak_refusals"] += 1
        if flagged:
            stats["unsafe_or_borderline"] += 1
        if is_error:
            stats["transport_errors"] += 1

    return by_category


def build_focus(by_category: Dict[str, Dict[str, int]], total_turns: int) -> Dict[str, Dict[str, Any]]:
    """
    Recommend how many Stage-2 prompts to allocate per category.

    We weight categories by their unsafe/borderline counts (fallback to 10% of total
    if no unsafe rows) and scale to the requested total_turns budget.
    """
    weights: Dict[str, int] = {}
    for cat, stats in by_category.items():
        weight = stats.get("unsafe_or_borderline", 0)
        if weight == 0:
            # If nothing is unsafe, still give a small allocation to sanity-check
            weight = max(1, int(stats.get("total", 0) * 0.1))
        weights[cat] = max(1, weight)

    denom = sum(weights.values()) or 1
    focus: Dict[str, Dict[str, Any]] = {}

    for cat, weight in weights.items():
        stats = by_category[cat]
        count = max(1, int(round(total_turns * (weight / denom))))
        note_parts = [
            f"unsafe_or_borderline={stats.get('unsafe_or_borderline', 0)}/{stats.get('total', 0)}",
        ]
        if stats.get("should_have_refused"):
            note_parts.append(f"should_have_refused={stats['should_have_refused']}")
        if stats.get("weak_refusals"):
            note_parts.append(f"weak_refusals={stats['weak_refusals']}")
        if stats.get("transport_errors"):
            note_parts.append(f"errors={stats['transport_errors']}")

        focus[cat] = {
            "count": count,
            "notes": "; ".join(note_parts),
        }

    return focus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Stage-2 red-team plan from Stage-1 analysis")
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS_PATH, help="Path to Stage-1 OpenAI analysis JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH, help="Where to write the Stage-2 plan JSON")
    parser.add_argument("--total-turns", type=int, default=1000, help="Total prompts to budget across focus areas")
    args = parser.parse_args()

    rows = list(load_analysis(args.analysis))
    by_category = aggregate(rows)
    focus_areas = build_focus(by_category, args.total_turns)

    plan = {
        "source": str(args.analysis),
        "total_items": len(rows),
        "by_category": by_category,
        "focus_areas": focus_areas,
        "notes": "counts are suggested; edit config/redteam_stage2.json as needed",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)

    print(f"[ok] wrote Stage-2 plan to {args.output}")


if __name__ == "__main__":
    main()
