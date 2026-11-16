"""The monitor page's server: read-only JSON over a training run, token-protected, plus the page itself.

    MONITOR_TOKEN=... uvicorn avazmon.app:app_from_env --factory --host 0.0.0.0 --port 8000

Settings (environment): MONITOR_RUNS (/workspace/runs), MONITOR_PRICE (GPU $/hour), MONITOR_TOKEN.
``<runs>/balance.json`` is written by the watcher on the user's machine (the RunPod key never reaches the pod).
Adapted from the model-tr monitor (türkcə damitma/tr-train) for Avaz: one voice, and a "listen" section that
plays fixed sentences read by every checkpoint next to the old Avaz.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .reader import MetricsReader, downsample, listen_versions, progress, validation_extras, validation_steps

STATIC = Path(__file__).with_name("static")
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_LISTEN_FOLDER = re.compile(r"^(base|step-\d{8})$")


@dataclass(frozen=True)
class Settings:
    runs: Path
    price_per_hour: float
    token: str | None


def settings_from_env() -> Settings:
    return Settings(runs=Path(os.environ.get("MONITOR_RUNS", "/workspace/runs")),
                    price_per_hour=float(os.environ.get("MONITOR_PRICE", "0.59")),
                    token=os.environ.get("MONITOR_TOKEN") or None)


def _ok(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def _safe_name(value: str) -> str:
    if not _NAME.match(value) or value.startswith("."):
        raise HTTPException(status_code=400, detail="Yanlış ad.")
    return value


def _alive(pid_file: Path, platform: str = os.name) -> bool:
    """Whether the trainer in the PID file still runs. Never asked on Windows: there os.kill(pid, 0) sends Ctrl+C
    (signal 0 is CTRL_C_EVENT) instead of probing, and a run downloaded from the pod holds the pod's PID anyway."""
    if platform == "nt":
        return False
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


class _Gpu:
    """nvidia-smi, at most every 10 s."""

    def __init__(self) -> None:
        self._at, self._value = 0.0, None

    def read(self) -> dict[str, Any] | None:
        if time.time() - self._at > 10:
            self._at = time.time()
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                                      "temperature.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
                self._value = {"name": out[0], "utilization": int(out[1]), "memory_used_mb": int(out[2]),
                               "memory_total_mb": int(out[3]), "temperature": int(out[4])}
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                self._value = None
        return self._value


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Avaz monitor", docs_url=None, redoc_url=None, openapi_url=None)
    readers: dict[str, MetricsReader] = {}
    gpu = _Gpu()

    @app.middleware("http")
    async def revalidate_page_files(request: Request, call_next: Any) -> Any:
        """The page's scripts change when the monitor is updated; make browsers (a phone above all) check for a
        new version instead of running a cached one. Unchanged files still come back as 304."""
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def authorized(request: Request) -> None:
        if settings.token is None:
            return
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), settings.token.encode()):
            raise HTTPException(status_code=401, detail="Token yanlışdır.")

    protected = [Depends(authorized)]

    def runs() -> list[str]:
        found = [p.parent.name for p in settings.runs.glob("*/metrics.jsonl")]
        return sorted(found, key=lambda name: (settings.runs / name / "metrics.jsonl").stat().st_mtime, reverse=True)

    def chosen(run: str | None) -> str:
        """The asked-for run if it exists (names come from the disk, so they are safe), else the newest one."""
        names = runs()
        if run is not None and run not in names:
            raise HTTPException(status_code=404, detail="Belə təlim yoxdur.")
        if not names:
            raise HTTPException(status_code=404, detail="Hələ təlim yoxdur.")
        return run or names[0]

    def rows_of(run: str) -> list[dict[str, Any]]:
        reader = readers.setdefault(run, MetricsReader(settings.runs / run / "metrics.jsonl"))
        return reader.rows()

    @app.get("/api/status", dependencies=protected)
    def status(run: str | None = None) -> dict[str, Any]:
        name = chosen(run)
        options_path = settings.runs / name / "training-options.json"
        options = json.loads(options_path.read_text()) if options_path.is_file() else {}
        balance_path = settings.runs / "balance.json"
        balance = json.loads(balance_path.read_text()) if balance_path.is_file() else None
        return _ok({"run": name, "runs": runs(), "alive": _alive(settings.runs / f"{name}.pid"),
                    **progress(rows_of(name), int(options.get("max_steps", 0)) or 1, settings.price_per_hour),
                    "gpu": gpu.read(), "balance": balance, "price_per_hour": settings.price_per_hour,
                    "now": time.time()})

    @app.get("/api/history", dependencies=protected)
    def history(run: str | None = None, points: int = Query(300, ge=10, le=1000),
                from_step: int = Query(0, ge=0)) -> dict[str, Any]:
        return _ok(downsample(rows_of(chosen(run)), points, from_step))

    @app.get("/api/validation", dependencies=protected)
    def validation(run: str | None = None, limit: int = Query(3, ge=1, le=1000)) -> dict[str, Any]:
        folder = settings.runs / chosen(run) / "validation"
        return _ok({"steps": validation_steps(folder, limit), **validation_extras(folder)})

    @app.get("/api/listen", dependencies=protected)
    def listen(run: str | None = None) -> dict[str, Any]:
        return _ok(listen_versions(settings.runs / chosen(run) / "listen"))

    @app.get("/api/audio", dependencies=protected)
    def audio(file: str, run: str | None = None, listen: str | None = None) -> FileResponse:
        name = _safe_name(file)
        if not name.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Yalnız .wav.")
        if listen is not None and not _LISTEN_FOLDER.match(listen):
            raise HTTPException(status_code=400, detail="Yanlış ad.")
        run_folder = settings.runs / chosen(run)
        folder = run_folder / "listen" / listen if listen else run_folder / "validation"
        path = folder / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Səs tapılmadı.")
        return FileResponse(path, media_type="audio/wav")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def app_from_env() -> FastAPI:
    return create_app(settings_from_env())
