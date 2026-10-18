import os
from dataclasses import dataclass

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Hard product requirement: upload -> final status in < 5 min. job_budget_sec is
# measured from submission (queue wait included); the remainder is upload/poll margin.
HARD_LIMIT_SEC = 300


def _i(name, default):
    return int(os.environ.get(name, default))


def _f(name, default):
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    api_key: str = os.environ.get("API_KEY", "")
    # relative paths resolve against the project root (not CWD), see __post_init__
    data_dir: str = os.environ.get("DATA_DIR", "data/jobs")
    max_upload_mb: int = _i("MAX_UPLOAD_MB", 60)
    max_duration_sec: int = _i("MAX_DURATION_SEC", 150)
    # 230 s from submission (queue wait included) + upload_timeout_sec 60 s = 290 s < 300 s
    job_budget_sec: int = _i("JOB_BUDGET_SEC", 230)
    workers: int = _i("WORKERS", 2)
    queue_limit: int = _i("QUEUE_LIMIT", 20)
    sample_fps: float = _f("SAMPLE_FPS", "4")
    onnx_model: str = os.environ.get("ONNX_MODEL", "")  # optional external detector
    ruleset: str = os.environ.get("RULESET", os.path.join(APP_DIR, "rules_default.json"))
    # --- queue / isolation (app/jobs.py, app/worker.py) ---
    # Per-job runtime estimate used by admission control (floor; the store also
    # tracks an EWMA of observed run times and uses the larger of the two).
    # Cold-start estimate; replaced by the max of recent observed run times.
    job_estimate_sec: float = _f("JOB_ESTIMATE_SEC", "60")
    # Admit a job if predicted completion <= job_budget_sec + this margin
    # (the hard deadline still applies; this only trades 429s for possible timeouts).
    admission_margin_sec: float = _f("ADMISSION_MARGIN_SEC", "20")
    # Backup watchdog margin after the deadline before a job is force-failed.
    kill_grace_sec: float = _f("KILL_GRACE_SEC", "2")
    # Finished job records (memory and data_dir/<job_id>.json) expire after this.
    job_ttl_sec: int = _i("JOB_TTL_SEC", 7200)
    # Leftover uploads / worker temp files older than this are deleted (startup + periodic).
    orphan_max_age_min: float = _f("ORPHAN_MAX_AGE_MIN", "10")
    # Optional address-space cap for the analysis child process (0 = off; Linux only).
    worker_mem_mb: int = _i("WORKER_MEM_MB", 0)
    # --- upload timeouts (read by app/main.py) ---
    receive_timeout_sec: float = _f("RECEIVE_TIMEOUT_SEC", "15")  # max stall per body chunk
    upload_timeout_sec: float = _f("UPLOAD_TIMEOUT_SEC", "60")    # whole upload

    def __post_init__(self):
        for name in ("data_dir", "ruleset"):  # CWD-independent paths
            v = getattr(self, name)
            if v and not os.path.isabs(v):
                object.__setattr__(self, name, os.path.join(PROJECT_ROOT, v))
        if self.workers < 1:
            raise ValueError("WORKERS must be >= 1")
        if self.queue_limit < 0:
            raise ValueError("QUEUE_LIMIT must be >= 0")
        if not 0 < self.job_budget_sec < HARD_LIMIT_SEC:
            raise ValueError(f"JOB_BUDGET_SEC must be in (0, {HARD_LIMIT_SEC})")
        if self.admission_margin_sec < 0:
            raise ValueError("ADMISSION_MARGIN_SEC must be >= 0")
        if self.job_estimate_sec <= 0 or self.kill_grace_sec < 0:
            raise ValueError("JOB_ESTIMATE_SEC must be > 0 and KILL_GRACE_SEC >= 0")
        if self.job_ttl_sec <= 0 or self.orphan_max_age_min <= 0:
            raise ValueError("JOB_TTL_SEC and ORPHAN_MAX_AGE_MIN must be > 0")
        if self.receive_timeout_sec <= 0 or self.upload_timeout_sec <= 0:
            raise ValueError("RECEIVE_TIMEOUT_SEC and UPLOAD_TIMEOUT_SEC must be > 0")
        if self.upload_timeout_sec + self.job_budget_sec >= HARD_LIMIT_SEC:
            raise ValueError(f"UPLOAD_TIMEOUT_SEC + JOB_BUDGET_SEC must be < {HARD_LIMIT_SEC}")
        if self.orphan_max_age_min * 60 <= self.job_budget_sec:
            raise ValueError("ORPHAN_MAX_AGE_MIN must exceed the job budget")


def get() -> Settings:
    return Settings()
