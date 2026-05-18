"""Streaming downloader for HuggingFace `.gguf` files.

Accepts three input shapes for the user:
1. Full file URL:   `https://huggingface.co/<repo>/resolve/main/<file>.gguf`
2. `<repo>/<file.gguf>`         (e.g. `bartowski/.../...gguf`)
3. `<owner>/<name>/<file.gguf>` (canonical)

Downloads stream to a `.gguf.part` temp file in the models folder, then atomic
rename on completion. Each job lifecycle:

    queued → downloading → done | error | cancelled

Status changes are pushed to subscribers via callbacks — `ws.py` registers
one so the renderer sees live progress.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import urlparse

import httpx

from .library import library


HF_RESOLVE_PREFIX = "https://huggingface.co/"


@dataclass
class DownloadJob:
    id: str
    repo: str             # "owner/name"
    filename: str         # file under the repo
    url: str              # resolved URL
    dest: str             # final path on disk
    status: str = "queued"   # queued | downloading | done | error | cancelled
    downloaded: int = 0
    total: int = 0
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    # Internal cancellation flag
    _cancel: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)

    def snapshot(self) -> Dict[str, Any]:
        pct = (self.downloaded / self.total * 100) if self.total else 0
        speed_mbs = 0.0
        if self.status == "downloading" and self.started_at:
            elapsed = max(0.001, time.time() - self.started_at)
            speed_mbs = (self.downloaded / (1024 * 1024)) / elapsed
        return {
            "id": self.id,
            "repo": self.repo,
            "filename": self.filename,
            "status": self.status,
            "downloaded": self.downloaded,
            "total": self.total,
            "percent": round(pct, 1),
            "speedMBs": round(speed_mbs, 1),
            "error": self.error,
        }


Listener = Callable[[Dict[str, Any]], "Awaitable[None] | None"]


def parse_hf_input(raw: str) -> tuple[str, str, str]:
    """Return (repo, filename, resolved_url) from any of the accepted shapes."""
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty input")

    # Full URL? Must point at a HF resolve link.
    if raw.startswith("http://") or raw.startswith("https://"):
        if not raw.startswith(HF_RESOLVE_PREFIX):
            raise ValueError("URL must be a huggingface.co link.")
        parsed = urlparse(raw)
        parts = parsed.path.strip("/").split("/")
        # /<owner>/<name>/resolve/<rev>/<file>
        if len(parts) < 5 or parts[2] != "resolve":
            raise ValueError(
                "Expected URL like https://huggingface.co/OWNER/REPO/resolve/main/FILE.gguf"
            )
        owner, name = parts[0], parts[1]
        filename = parts[-1]
        return f"{owner}/{name}", filename, raw

    # owner/name/file.gguf
    bits = raw.split("/")
    if len(bits) >= 3 and bits[-1].lower().endswith(".gguf"):
        owner, name = bits[0], bits[1]
        filename = "/".join(bits[2:])  # supports filenames in subfolders
        url = f"{HF_RESOLVE_PREFIX}{owner}/{name}/resolve/main/{filename}"
        return f"{owner}/{name}", filename.split("/")[-1], url

    raise ValueError(
        "Couldn't parse input. Paste either:\n"
        "  • the full huggingface.co URL ending in .gguf, or\n"
        "  • <owner>/<repo>/<file>.gguf"
    )


class Downloader:
    def __init__(self) -> None:
        self._jobs: Dict[str, DownloadJob] = {}
        self._listeners: set[Listener] = set()
        self._lock = asyncio.Lock()

    # ─── subscriptions ────────────────────────────────────────────────
    def add_listener(self, cb: Listener) -> None:
        self._listeners.add(cb)

    def remove_listener(self, cb: Listener) -> None:
        self._listeners.discard(cb)

    def _notify(self, job: DownloadJob) -> None:
        snap = job.snapshot()
        for cb in list(self._listeners):
            try:
                r = cb(snap)
                if asyncio.iscoroutine(r):
                    asyncio.create_task(r)
            except Exception:  # noqa: BLE001
                pass

    # ─── job control ──────────────────────────────────────────────────
    def list(self) -> list[Dict[str, Any]]:
        return [j.snapshot() for j in self._jobs.values()]

    def get(self, job_id: str) -> Optional[DownloadJob]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status not in ("queued", "downloading"):
            return False
        job._cancel.set()
        return True

    async def start(self, raw_input: str) -> DownloadJob:
        repo, filename, url = parse_hf_input(raw_input)
        dest = os.path.join(library.root, filename)
        if os.path.exists(dest):
            raise FileExistsError(f"Model '{filename}' already exists in the library.")

        job = DownloadJob(
            id=str(uuid.uuid4()),
            repo=repo,
            filename=filename,
            url=url,
            dest=dest,
        )
        self._jobs[job.id] = job
        self._notify(job)
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: DownloadJob) -> None:
        tmp = job.dest + ".part"
        try:
            job.status = "downloading"
            job.started_at = time.time()
            self._notify(job)

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, read=300.0),
                follow_redirects=True,
            ) as client:
                async with client.stream("GET", job.url) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        raise RuntimeError(
                            f"HTTP {r.status_code}: "
                            f"{body.decode(errors='replace')[:300]}"
                        )
                    total = int(r.headers.get("content-length") or 0)
                    job.total = total
                    self._notify(job)

                    last_notify = 0.0
                    with open(tmp, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                            if job._cancel.is_set():
                                raise asyncio.CancelledError()
                            f.write(chunk)
                            job.downloaded += len(chunk)
                            # throttle notifications to ~4 Hz
                            now = time.time()
                            if now - last_notify > 0.25:
                                last_notify = now
                                self._notify(job)

            # Atomic rename on success
            os.replace(tmp, job.dest)
            job.status = "done"
            job.finished_at = time.time()
            self._notify(job)

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "Cancelled by user"
            self._safe_unlink(tmp)
            self._notify(job)
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            self._safe_unlink(tmp)
            self._notify(job)

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


downloader = Downloader()
