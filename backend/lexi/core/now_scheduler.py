from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..config.now import ENABLE_NOW, settings_now
from ..utils.now_utils import log_now

_scheduler: AsyncIOScheduler | None = None


def _get_refresh_job():
    from ..utils.now_ingest import refresh_now_feed

    return refresh_now_feed


def start_now_scheduler():
    global _scheduler
    if not ENABLE_NOW:
        log_now("APScheduler skipped: ENABLE_NOW disabled")
        return
    if _scheduler:
        return
    refresh_job = _get_refresh_job()
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        refresh_job,
        "interval",
        minutes=settings_now.NOW_REFRESH_MINUTES,
        id="now-refresh",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    log_now(f"APScheduler started: every {settings_now.NOW_REFRESH_MINUTES} min")
