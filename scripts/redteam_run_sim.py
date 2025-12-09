#!/usr/bin/env python3
"""
Replay synthetic prompts against the Lexi backend with concurrency.

- Stage 1 default: prompts from data/redteam_prompts.jsonl -> data/redteam_runs.jsonl
- Stage 2: accepts scenario-tagged prompts (e.g., data/redteam_stage2_prompts.jsonl)
  and writes to data/redteam_stage2_runs.jsonl while preserving scenario metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "redteam_sim.json"
PROMPTS_PATH = REPO_ROOT / "data" / "redteam_prompts.jsonl"
RUNS_PATH = REPO_ROOT / "data" / "redteam_runs.jsonl"
STAGE2_RUNS_PATH = REPO_ROOT / "data" / "redteam_stage2_runs.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_prompts(path: Path) -> List[Dict]:
    prompts: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                prompts.append(json.loads(line))
            except Exception:
                continue
    return prompts


def load_plan(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[warn] failed to read plan at {path}: {exc}")
        return {}


def extract_plan_allocation(plan: Dict[str, Any]) -> Dict[str, int]:
    focus = plan.get("focus_areas") if plan else {}
    allocation: Dict[str, int] = {}
    if isinstance(focus, dict):
        for key, val in focus.items():
            count = None
            if isinstance(val, dict):
                count = val.get("count")
            elif isinstance(val, (int, float)):
                count = val
            if count is None:
                continue
            try:
                allocation[key] = int(count)
            except Exception:
                continue
    return {k: v for k, v in allocation.items() if v > 0}


@dataclass
class TurnTask:
    session_id: str
    persona: str
    prompt_id: str
    category: str
    scenario: str
    user_message: str
    idx: int


def _extract_assistant_text(body: Any) -> str:
    if isinstance(body, dict):
        for key in ("assistant", "reply", "content", "message", "cleaned", "text"):
            val = body.get(key)
            if val:
                return str(val)
    return ""


def build_tasks(prompts: List[Dict], max_turns: int, session_prefix: str) -> List[TurnTask]:
    tasks: List[TurnTask] = []
    session_counter = 0
    chunk: List[Dict] = []
    for idx, p in enumerate(prompts):
        chunk.append(p)
        if len(chunk) >= max_turns:
            session_counter += 1
            sess_id = f"{session_prefix}_sess_{session_counter:04d}"
            for i, entry in enumerate(chunk):
                tasks.append(
                    TurnTask(
                        session_id=sess_id,
                        persona=str(entry.get("persona") or "default"),
                        prompt_id=str(entry.get("id") or f"prompt_{idx:06d}"),
                        category=str(entry.get("category") or "normal"),
                        scenario=str(entry.get("scenario") or "stage1"),
                        user_message=str(entry.get("user_message") or entry.get("prompt") or ""),
                        idx=i,
                    )
                )
            chunk = []
    if chunk:
        session_counter += 1
        sess_id = f"{session_prefix}_sess_{session_counter:04d}"
        for i, entry in enumerate(chunk):
            tasks.append(
                TurnTask(
                    session_id=sess_id,
                    persona=str(entry.get("persona") or "default"),
                        prompt_id=str(entry.get("id") or f"prompt_{len(tasks):06d}"),
                        category=str(entry.get("category") or "normal"),
                        scenario=str(entry.get("scenario") or "stage1"),
                        user_message=str(entry.get("user_message") or entry.get("prompt") or ""),
                        idx=i,
                    )
                )
    return tasks


def filter_prompts(
    prompts: List[Dict], scenario_filter: Optional[str], allocation: Dict[str, int], max_prompts: Optional[int]
) -> List[Dict]:
    filtered = []
    for p in prompts:
        scenario = str(p.get("scenario") or "stage1")
        if scenario_filter and scenario != scenario_filter:
            continue
        filtered.append(p)

    if allocation:
        remaining = dict(allocation)
        selected: List[Dict] = []
        for p in filtered:
            scenario = str(p.get("scenario") or "stage1")
            category = str(p.get("category") or "")
            matched_key = None
            for key in (scenario, category):
                if key and remaining.get(key, 0) > 0:
                    matched_key = key
                    break
            if matched_key is None:
                continue
            remaining[matched_key] -= 1
            selected.append(p)
            if max_prompts is not None and len(selected) >= max_prompts:
                break
        filtered = selected
    elif max_prompts is not None:
        filtered = filtered[:max_prompts]

    return filtered


def infer_output_path(prompts_path: Path, output_arg: Optional[Path]) -> Path:
    if output_arg:
        return output_arg
    name = prompts_path.name.lower()
    if "stage2" in name:
        return STAGE2_RUNS_PATH
    return RUNS_PATH


async def set_mode(client: httpx.AsyncClient, base_url: str, mode: str) -> None:
    try:
        resp = await client.post(f"{base_url}/lexi/set_mode", json={"mode": mode}, timeout=10)
        if resp.status_code != 200:
            return
    except Exception:
        return


async def send_turn(
    client: httpx.AsyncClient,
    base_url: str,
    route: str,
    task: TurnTask,
    run_id: str,
    semaphore: asyncio.Semaphore,
    http_timeout: float,
) -> Dict:
    url = f"{base_url.rstrip('/')}{route}"
    headers = {"X-Lexi-Session": task.session_id}
    payload = {
        "prompt": task.user_message,
        "intent": "chat",
        "session_id": task.session_id,
        "user_id": "redteam_harness",
        "persona": task.persona,
        "meta": {"source": "redteam_stage2", "scenario": task.scenario},
    }
    started = time.time()
    async with semaphore:
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=http_timeout)
            latency = time.time() - started
            body = None
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            cleaned = _extract_assistant_text(body)
            return {
                "run_id": run_id,
                "ts": _now_iso(),
                "session_id": task.session_id,
                "prompt_id": task.prompt_id,
                "category": task.category,
                "scenario": task.scenario,
                "persona": task.persona,
                "status": resp.status_code,
                "latency_s": round(latency, 3),
                "prompt": task.user_message,
                "request": payload,
                "response": {
                    "cleaned": cleaned or body,
                    "raw": body,
                    "mode": body.get("mode") if isinstance(body, dict) else "default",
                },
            }
        except Exception as exc:
            return {
                "run_id": run_id,
                "ts": _now_iso(),
                "session_id": task.session_id,
                "prompt_id": task.prompt_id,
                "category": task.category,
                "scenario": task.scenario,
                "persona": task.persona,
                "status": None,
                "latency_s": None,
                "prompt": task.user_message,
                "request": payload,
                "error": str(exc) or repr(exc),
            }


async def run_sim(
    cfg: Dict, prompts: List[Dict], out_path: Path, scenario_filter: Optional[str], http_timeout: float
) -> None:
    base_url = os.environ.get("LEXI_BACKEND_BASE") or cfg.get("backend_base_url", "http://127.0.0.1:9000")
    route = cfg.get("backend_route", "/lexi/process")
    concurrency = int(cfg.get("concurrency", 16))
    max_turns = int(cfg.get("max_turns_per_session", 6))
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-redteam"
    if scenario_filter:
        run_id = f"{run_id}-{scenario_filter}"

    # Make session IDs unique to avoid reusing past session memory on the backend
    session_prefix = f"sess_sim_{uuid.uuid4().hex[:8]}"

    tasks = build_tasks(prompts, max_turns, session_prefix)
    print(f"[info] hitting backend {base_url}{route} with {len(tasks)} turns (scenario_filter={scenario_filter or 'all'})")
    semaphore = asyncio.Semaphore(concurrency)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        # optional persona setup per session
        seen_sessions = {}
        for t in tasks:
            if t.session_id not in seen_sessions:
                seen_sessions[t.session_id] = t.persona

        # best-effort set_mode; ignore failures
        set_mode_tasks = [set_mode(client, base_url, persona) for persona in seen_sessions.values()]
        await asyncio.gather(*set_mode_tasks, return_exceptions=True)

        coros = [send_turn(client, base_url, route, t, run_id, semaphore, http_timeout) for t in tasks]
        with out_path.open("w", encoding="utf-8") as fh:
            for fut in asyncio.as_completed(coros):
                result = await fut
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
    print(f"[ok] wrote run log to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic red-team load against Lexi backend")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH, help="Path to prompts JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for run log JSONL")
    parser.add_argument("--scenario", type=str, default=None, help="Only run prompts matching this scenario")
    parser.add_argument("--stage2-plan", type=Path, default=None, help="Optional Stage-2 plan JSON to subsample prompts")
    parser.add_argument("--max-prompts", type=int, default=None, help="Run at most N prompts (after filtering)")
    parser.add_argument("--max-turns-per-session", type=int, default=None, help="Override max turns per session (default from config)")
    parser.add_argument("--http-timeout", type=float, default=None, help="HTTP timeout (seconds) for each turn (default 30s)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_turns_per_session is not None:
        cfg["max_turns_per_session"] = args.max_turns_per_session
    http_timeout = float(args.http_timeout) if args.http_timeout is not None else float(cfg.get("http_timeout", 30))
    prompts = load_prompts(args.prompts)
    plan = load_plan(args.stage2_plan)
    allocation = extract_plan_allocation(plan)
    prompts = filter_prompts(prompts, args.scenario, allocation, args.max_prompts)
    if not prompts:
        raise SystemExit("No prompts found after filtering; check --prompts/--scenario/--stage2-plan.")

    out_path = infer_output_path(args.prompts, args.output)
    asyncio.run(run_sim(cfg, prompts, out_path, args.scenario, http_timeout))


if __name__ == "__main__":
    main()
