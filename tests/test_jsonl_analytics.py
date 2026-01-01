import importlib
import json
from datetime import datetime, timedelta, timezone


def _reload_analytics(monkeypatch, tmp_path):
    monkeypatch.setenv("LEXI_ANALYTICS_DIR", str(tmp_path))
    import backend.lexi.analytics.jsonl_analytics as analytics

    return importlib.reload(analytics)


def test_record_and_rollover_writes_daily(tmp_path, monkeypatch) -> None:
    analytics = _reload_analytics(monkeypatch, tmp_path)

    day1 = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    first = analytics.record_heartbeat("visitor-1", now_utc=day1)
    assert first["unique_today"] == 1
    assert first["unique_all_time"] == 1

    analytics.record_heartbeat("visitor-1", now_utc=day1 + timedelta(seconds=10))

    day2 = day1 + timedelta(days=1)
    analytics.record_heartbeat("visitor-2", now_utc=day2)

    daily_lines = (tmp_path / "daily.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(daily_lines) == 1
    daily = json.loads(daily_lines[0])
    assert daily["day"] == "2024-01-02"
    assert daily["unique_visitors"] == 1
    assert daily["heartbeats"] == 2
    assert daily["peak_concurrent"] == 1

    visitors_lines = (tmp_path / "visitors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(visitors_lines) == 2

    summary = analytics.get_summary(now_utc=day2)
    assert summary["day"] == "2024-01-03"
    assert summary["unique_today"] == 1


def test_load_visitors_skips_corrupt_lines(tmp_path, monkeypatch) -> None:
    visitors_path = tmp_path / "visitors.jsonl"
    visitors_path.write_text(
        "{\"visitor_id\": \"ok\", \"event\": \"first_seen\"}\n{bad json}\n",
        encoding="utf-8",
    )

    analytics = _reload_analytics(monkeypatch, tmp_path)
    assert analytics.state.all_time_visitor_ids == {"ok"}
