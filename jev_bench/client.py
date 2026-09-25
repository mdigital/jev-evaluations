"""Rate-limited, resumable client for the OpenRouter Decisions API.

Shared by every JEV benchmark. Results are content-addressed by a digest of the
exact request payload plus the repetition index, so a run can be interrupted and
resumed without re-spending calls, and so two benchmarks that happen to issue an
identical request never disagree about the answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def canonical(obj) -> str:
    """Order-insensitive serialisation, for comparing objects by content."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def wire(obj) -> str:
    """The exact JSON sent to the API.

    Key order is meaningful here: the order of a choice question's ``criteria``
    is the order the options are presented in, and measuring position bias
    depends on two payloads that differ only in that order staying distinct.
    So this must not sort keys, and it is used both to hash a request and to
    send it, so the two can never drift apart.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs without overwriting the existing environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class TokenBucket:
    """Thread-safe request pacer. Threads block here rather than at the socket."""

    def __init__(self, per_minute: float):
        self._interval = 60.0 / float(per_minute)
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def take(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def _retry_after_seconds(response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


class DecisionsClient:
    """Collects Decisions answers for a list of trials, with on-disk resume.

    A trial is a dict carrying at least ``trial_id`` and ``payload``. Everything
    else on the trial is analysis metadata and is never transmitted.
    """

    def __init__(self, run_dir: Path, model: str = DEFAULT_MODEL,
                 requests_per_minute: float = 600, max_workers: int = 8,
                 max_attempts: int = 5, timeout: float = 90):
        self.run_dir = Path(run_dir)
        self.records_dir = self.run_dir / "records"
        self.errors_dir = self.run_dir / "errors"
        for directory in (self.records_dir, self.errors_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.bucket = TokenBucket(requests_per_minute)
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.timeout = timeout
        self._local = threading.local()
        load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"}

    def record_path(self, trial) -> Path:
        return self.records_dir / f"{trial['record_key']}.json"

    @staticmethod
    def record_key(payload, repetition: int) -> str:
        blob = f"{repetition}|{wire(payload)}".encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=self.max_workers,
                                                    pool_maxsize=self.max_workers)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _one(self, trial) -> dict:
        path = self.record_path(trial)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        body = dict(trial["payload"])
        body["model"] = self.model
        session = self._session()
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            self.bucket.take()
            started = time.time()
            try:
                response = session.post(DECISIONS_URL, headers=self._headers,
                                        data=wire(body), timeout=self.timeout)
                elapsed = time.time() - started
                if response.status_code == 200:
                    parsed = response.json()
                    record = {
                        "record_key": trial["record_key"],
                        "status": "ok",
                        "answers": parsed.get("answers"),
                        "usage": parsed.get("usage"),
                        "provider": parsed.get("provider"),
                        "response_model": parsed.get("model"),
                        "latency_seconds": round(elapsed, 4),
                        "attempts": attempt,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                    atomic_write(path, record)
                    return record

                last_error = f"http_{response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    break
                wait = _retry_after_seconds(response)
                if wait is None:
                    wait = min(30.0, 2 ** attempt) * (0.5 + random.random())
                time.sleep(wait)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}"
                time.sleep(min(30.0, 2 ** attempt) * (0.5 + random.random()))

        record = {"record_key": trial["record_key"], "status": "failed",
                  "failure": last_error, "attempts": self.max_attempts,
                  "collected_at": datetime.now(timezone.utc).isoformat()}
        atomic_write(self.errors_dir / f"{trial['record_key']}.json", record)
        return record

    def collect(self, trials, progress_every: int = 250) -> list[dict]:
        # Blocks share a common baseline cell, so the same payload can appear
        # under several trials. Fetch each distinct request once.
        pending, seen = [], set()
        for trial in trials:
            key = trial["record_key"]
            if key in seen or self.record_path(trial).exists():
                continue
            seen.add(key)
            pending.append(trial)
        shared = len(trials) - len({t["record_key"] for t in trials})
        print(f"{len(trials):,} trials covering {len(trials) - shared:,} distinct "
              f"requests; {len(pending):,} to fetch")
        if pending:
            completed = 0
            started = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._one, t): t for t in pending}
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed % progress_every == 0 or completed == len(pending):
                        rate = completed / max(time.time() - started, 1e-9)
                        print(f"  {completed:,}/{len(pending):,} at {rate:.0f}/s")
        return [self._one(t) for t in trials]
