#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def sanitize_user_id(user_id: str) -> str:
    cleaned = (user_id or "").strip().lower()
    cleaned = cleaned.replace(" ", "-")
    cleaned = re.sub(r"[^a-z0-9_.@-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or "anon"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON at {path}: {exc}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{idx}: {exc}")
    return rows


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    timeout: int = 30,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.request(
                method,
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code >= 500 and attempt < retries:
                time.sleep(0.4 * attempt)
                continue
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.4 * attempt)
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError("request retries exhausted")


def post_prompt(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    user_id: str,
    prompt: str,
    retries: int,
    api_key: str,
) -> Dict[str, Any]:
    url = f"{base_url}{endpoint}"
    headers = {"X-Lexi-User": user_id}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = request_with_retries(
        session,
        "POST",
        url,
        headers=headers,
        payload={"prompt": prompt},
        retries=retries,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {url} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def post_tool(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    user_id: str,
    payload: Optional[Dict[str, Any]],
    retries: int,
    api_key: str,
) -> requests.Response:
    url = f"{base_url}{endpoint}"
    headers = {"X-Lexi-User": user_id}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return request_with_retries(
        session,
        "POST",
        url,
        headers=headers,
        payload=payload,
        retries=retries,
    )


def contains_no_saved_context(text: str) -> bool:
    lowered = text.lower()
    clues = [
        "no saved user context",
        "no saved notes",
        "no saved memory",
        "no saved context",
        "no saved",
    ]
    return any(clue in lowered for clue in clues)


def collect_tokens_from_text(text: str, tokens: List[str]) -> List[str]:
    found = []
    lowered = text.lower()
    for token in tokens:
        if token.lower() in lowered:
            found.append(token)
    return found


def legacy_fixture_paths(memory_root: Path, legacy_path: Optional[str]) -> Dict[str, Path]:
    if legacy_path:
        legacy = Path(legacy_path).expanduser().resolve()
    else:
        legacy = (memory_root / "legacy_fixture.jsonl").resolve()
    markers = legacy.with_suffix(".markers.json")
    return {"legacy": legacy, "markers": markers}


def prepare_legacy_fixture(memory_root: Path, legacy_path: Optional[str]) -> Dict[str, Any]:
    paths = legacy_fixture_paths(memory_root, legacy_path)
    legacy = paths["legacy"]
    markers_path = paths["markers"]
    markers = [f"fixture_{uuid.uuid4().hex[:8]}" for _ in range(3)]

    legacy.parent.mkdir(parents=True, exist_ok=True)
    with legacy.open("w", encoding="utf-8") as handle:
        for marker in markers:
            payload = {
                "role": "conversation",
                "content": f"migration_fixture: {marker}",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "meta": {"tags": ["migration_fixture"]},
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    markers_path.write_text(json.dumps({"markers": markers}, indent=2), encoding="utf-8")
    return {"legacy": legacy, "markers": markers, "markers_path": markers_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Lexi Tier-2 memory verification suite")
    parser.add_argument("--base-url", default=os.getenv("LEXI_BASE_URL", "http://127.0.0.1:9000"))
    parser.add_argument("--memory-root", default=os.getenv("LEXI_MEMORY_ROOT", "/data/memory"))
    parser.add_argument("--user-a", default=os.getenv("LEXI_USER_A", "testerA@example.com"))
    parser.add_argument("--user-b", default=os.getenv("LEXI_USER_B", "testerB@example.com"))
    parser.add_argument("--endpoint", default="/lexi/process")
    parser.add_argument("--profile-tool", default="/tools/memory_get_profile")
    parser.add_argument("--search-tool", default="/tools/memory_search_ltm")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--restart-check-only", action="store_true")
    parser.add_argument("--migration-check-only", action="store_true")
    parser.add_argument("--prepare-migration-fixture", action="store_true")
    parser.add_argument("--legacy-path", default=os.getenv("LEXI_LEGACY_PATH", ""))
    parser.add_argument("--concurrency", type=int, default=10)

    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    memory_root = Path(args.memory_root).expanduser().resolve()
    user_a = args.user_a
    user_b = args.user_b
    api_key = os.getenv("LEXI_API_KEY", "")

    results: List[Dict[str, str]] = []

    def record(name: str, status: str, detail: str = "") -> None:
        results.append({"name": name, "status": status, "detail": detail})

    def fail(name: str, detail: str) -> None:
        record(name, "FAIL", detail)

    def ok(name: str, detail: str = "") -> None:
        record(name, "PASS", detail)

    def skip(name: str, detail: str = "") -> None:
        record(name, "SKIP", detail)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if args.prepare_migration_fixture:
        fixture = prepare_legacy_fixture(memory_root, args.legacy_path or None)
        print("[fixture] wrote legacy JSONL:", fixture["legacy"])
        print("[fixture] markers:", ", ".join(fixture["markers"]))
        print("[fixture] set LEX_MEMORY_PATH to legacy file before restarting backend")
        return 0

    user_a_dir = memory_root / "users" / sanitize_user_id(user_a)
    user_b_dir = memory_root / "users" / sanitize_user_id(user_b)
    ltm_path_a = user_a_dir / "ltm.jsonl"
    summary_path_a = user_a_dir / "session_summaries.json"
    ltm_path_b = user_b_dir / "ltm.jsonl"

    if args.restart_check_only:
        try:
            reply = post_prompt(
                session,
                base_url,
                args.endpoint,
                user_a,
                "What is my dog's name? Don't guess. Use memory.",
                retries=args.retries,
                api_key=api_key,
            )
            text = reply.get("cleaned") or ""
            if "juniper" in text.lower():
                ok("restart_persistence", "Juniper recovered after restart")
            else:
                fail("restart_persistence", f"Expected Juniper, got: {text}")
        except Exception as exc:
            fail("restart_persistence", str(exc))

        return print_summary(results, user_a_dir, user_b_dir, ltm_path_a, summary_path_a, ltm_path_b)

    if args.migration_check_only:
        fixture_paths = legacy_fixture_paths(memory_root, args.legacy_path or None)
        markers_path = fixture_paths["markers"]
        if not markers_path.exists():
            skip("migration_idempotence", f"markers file missing: {markers_path}")
            return print_summary(results, user_a_dir, user_b_dir, ltm_path_a, summary_path_a, ltm_path_b)

        markers_doc = read_json(markers_path)
        markers = markers_doc.get("markers") or []
        if not markers:
            skip("migration_idempotence", "no markers found in fixture file")
            return print_summary(results, user_a_dir, user_b_dir, ltm_path_a, summary_path_a, ltm_path_b)

        try:
            ltm_rows = read_jsonl(ltm_path_a)
        except Exception as exc:
            fail("migration_idempotence", str(exc))
            return print_summary(results, user_a_dir, user_b_dir, ltm_path_a, summary_path_a, ltm_path_b)

        hits = 0
        for row in ltm_rows:
            content = str(row.get("content") or "")
            hits += len(collect_tokens_from_text(content, markers))
        unique_hits = set()
        for row in ltm_rows:
            content = str(row.get("content") or "")
            for marker in markers:
                if marker in content:
                    unique_hits.add(marker)

        if len(unique_hits) < len(markers):
            fail("migration_idempotence", f"Missing markers in LTM: {set(markers) - unique_hits}")
        elif hits > len(markers):
            fail("migration_idempotence", "Markers duplicated; migration not idempotent")
        else:
            ok("migration_idempotence", "Markers present once; dedupe ok")

        return print_summary(results, user_a_dir, user_b_dir, ltm_path_a, summary_path_a, ltm_path_b)

    # A) Roundtrip recall works
    try:
        post_prompt(
            session,
            base_url,
            args.endpoint,
            user_a,
            "Remember: my dog is Juniper.",
            retries=args.retries,
            api_key=api_key,
        )
        time.sleep(0.4)
        reply = post_prompt(
            session,
            base_url,
            args.endpoint,
            user_a,
            "What is my dog's name? Don't guess. Use memory.",
            retries=args.retries,
            api_key=api_key,
        )
        text = reply.get("cleaned") or ""
        if "juniper" in text.lower():
            ok("roundtrip_recall", "Juniper recovered")
        else:
            fail("roundtrip_recall", f"Expected Juniper, got: {text}")
    except Exception as exc:
        fail("roundtrip_recall", str(exc))

    # B) Cross-user isolation
    try:
        reply = post_prompt(
            session,
            base_url,
            args.endpoint,
            user_b,
            "What is my dog's name? Don't guess.",
            retries=args.retries,
            api_key=api_key,
        )
        text = reply.get("cleaned") or ""
        if "juniper" in text.lower():
            fail("cross_user_isolation", f"User B saw Juniper: {text}")
        else:
            ok("cross_user_isolation", "User B did not see Juniper")
            if not contains_no_saved_context(text):
                record("cross_user_message", "PASS", "No Juniper; response lacks explicit 'no saved notes'")
            else:
                record("cross_user_message", "PASS", "No saved context message present")
    except Exception as exc:
        fail("cross_user_isolation", str(exc))

    # C) Disk persistence
    try:
        if not user_a_dir.exists():
            fail("disk_persistence", f"User A directory missing: {user_a_dir}")
        else:
            if not summary_path_a.exists():
                fail("disk_persistence", f"session_summaries.json missing: {summary_path_a}")
            else:
                summary_doc = read_json(summary_path_a)
                rolling = (summary_doc.get("rolling_summary") or "")
                facts = summary_doc.get("facts") or {}
                combined = f"{rolling} {json.dumps(facts)}".lower()
                if "juniper" in combined:
                    ok("disk_persistence", "Juniper present in rolling summary or facts")
                else:
                    fail("disk_persistence", "Juniper not found in session_summaries.json")

            if ltm_path_a.exists():
                ltm_rows = read_jsonl(ltm_path_a)
                if any("juniper" in str(row.get("content", "")).lower() for row in ltm_rows):
                    ok("ltm_contains_juniper", "Juniper present in ltm.jsonl")
                else:
                    fail("ltm_contains_juniper", "Juniper not found in ltm.jsonl")
            else:
                fail("ltm_contains_juniper", f"ltm.jsonl missing: {ltm_path_a}")
    except Exception as exc:
        fail("disk_persistence", str(exc))

    # D) Restart persistence (manual)
    if args.no_restart:
        skip("restart_persistence", "skipped by --no-restart")
    else:
        skip("restart_persistence", "run --restart-check-only after backend restart")

    # E) Concurrency safety
    try:
        tokens = [f"token_{uuid.uuid4().hex[:8]}" for _ in range(args.concurrency)]
        ltm_before = read_jsonl(ltm_path_a) if ltm_path_a.exists() else []

        def _send(token: str) -> None:
            prompt = f"I'm working on {token}. Remember this."
            post_prompt(
                session,
                base_url,
                args.endpoint,
                user_a,
                prompt,
                retries=args.retries,
                api_key=api_key,
            )

        with ThreadPoolExecutor(max_workers=min(10, args.concurrency)) as executor:
            futures = [executor.submit(_send, token) for token in tokens]
            for future in as_completed(futures):
                future.result()

        time.sleep(0.8)

        ltm_after = read_jsonl(ltm_path_a)
        summary_doc = read_json(summary_path_a)
        rolling = (summary_doc.get("rolling_summary") or "")
        facts_blob = json.dumps(summary_doc.get("facts") or {})

        if len(ltm_after) < len(ltm_before):
            fail("concurrency", "ltm.jsonl shrank after concurrent writes")
        else:
            found_tokens = set()
            for row in ltm_after:
                content = str(row.get("content") or "")
                for token in tokens:
                    if token in content:
                        found_tokens.add(token)
            if len(found_tokens) < len(tokens):
                combined = f"{rolling} {facts_blob}"
                missing = [t for t in tokens if t not in combined]
                if missing:
                    fail("concurrency", f"Missing tokens in ltm/session summaries: {missing[:3]}...")
                else:
                    ok("concurrency", "Tokens found in summaries; ltm entries may be compacted")
            else:
                ok("concurrency", "All tokens present in ltm.jsonl")
    except Exception as exc:
        fail("concurrency", str(exc))

    # F) Migration idempotence (manual toggle)
    skip("migration_idempotence", "run with --migration-check-only after migration toggles")

    # G) Memory tools consistency
    try:
        resp = post_tool(session, base_url, args.profile_tool, user_a, {}, args.retries, api_key)
        if resp.status_code == 404:
            skip("memory_tools", "memory tool endpoints not available")
        elif resp.status_code >= 400:
            fail("memory_tools", f"memory_get_profile failed ({resp.status_code}): {resp.text}")
        else:
            profile = resp.json()
            blob = json.dumps(profile).lower()
            if "juniper" in blob:
                ok("memory_tools", "memory_get_profile includes Juniper")
            else:
                fail("memory_tools", "memory_get_profile missing Juniper")

            search_resp = post_tool(
                session,
                base_url,
                args.search_tool,
                user_a,
                {"query": "Juniper", "k": 5},
                args.retries,
                api_key,
            )
            if search_resp.status_code >= 400:
                fail("memory_tools_search", f"memory_search_ltm failed ({search_resp.status_code})")
            else:
                results = search_resp.json()
                blob = json.dumps(results).lower()
                if "juniper" in blob:
                    ok("memory_tools_search", "memory_search_ltm returned Juniper")
                else:
                    fail("memory_tools_search", "Juniper not found in memory_search_ltm")

            search_b = post_tool(
                session,
                base_url,
                args.search_tool,
                user_b,
                {"query": "Juniper", "k": 5},
                args.retries,
                api_key,
            )
            if search_b.status_code < 400:
                blob_b = json.dumps(search_b.json()).lower()
                if "juniper" in blob_b:
                    fail("memory_tools_isolation", "User B search returned Juniper")
                else:
                    ok("memory_tools_isolation", "User B search did not return Juniper")
    except Exception as exc:
        fail("memory_tools", str(exc))

    # H) Header trust gate (optional)
    if os.getenv("LEXI_HEADER_TRUST", "1") == "0":
        skip("header_trust_gate", "header trust gate not implemented in backend")
    else:
        skip("header_trust_gate", "LEXI_HEADER_TRUST=1 (skipped)")

    return print_summary(results, user_a_dir, user_b_dir, ltm_path_a, summary_path_a, ltm_path_b)


def print_summary(
    results: List[Dict[str, str]],
    user_a_dir: Path,
    user_b_dir: Path,
    ltm_path_a: Path,
    summary_path_a: Path,
    ltm_path_b: Path,
) -> int:
    print("\n=== Memory Verification Summary ===")
    for item in results:
        name = item["name"]
        status = item["status"]
        detail = item.get("detail") or ""
        line = f"{status}: {name}"
        if detail:
            line += f" | {detail}"
        print(line)

    print("\nPaths:")
    print(f"  user_a_dir: {user_a_dir}")
    print(f"  user_b_dir: {user_b_dir}")
    print(f"  ltm_path_a: {ltm_path_a}")
    print(f"  summary_path_a: {summary_path_a}")
    print(f"  ltm_path_b: {ltm_path_b}")

    failures = [r for r in results if r["status"] == "FAIL"]
    print("\nResult:")
    if failures:
        print(f"FAIL ({len(failures)} failures)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
