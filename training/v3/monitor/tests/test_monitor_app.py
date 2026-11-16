import json

import pytest
from fastapi.testclient import TestClient

from avazmon import app as monitor_app
from avazmon.app import Settings, _alive, create_app

TOKEN = "secret-token"


@pytest.fixture
def client(tmp_path):
    run = tmp_path / "runs" / "main"
    (run / "validation").mkdir(parents=True)
    rows = [{"step": i, "epoch": 1, "stage": "posterior_warmup", "loss_g": 2.0, "loss_d": 1.0, "loss_mel": 1.0,
             "loss_duration": 0.5, "loss_kl": 1.5, "peak_reserved_gb": 30.0, "wall_time": 100.0 + i} for i in range(1, 21)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "training-options.json").write_text(json.dumps({"max_steps": 100}))
    (run / "validation" / "step-00002500.wav").write_bytes(b"RIFF....WAVE")
    (run / "listen" / "base").mkdir(parents=True)
    (run / "listen" / "base" / "00.wav").write_bytes(b"RIFF....WAVE")
    (run / "listen" / "sentences.json").write_text(json.dumps(["Salam."]), encoding="utf-8")
    (tmp_path / "runs" / "balance.json").write_text(json.dumps({"balance": 31.5, "updated": 1.0}))
    settings = Settings(runs=tmp_path / "runs", price_per_hour=0.59, token=TOKEN)
    return TestClient(create_app(settings))


def auth(client, path):
    return client.get(path, headers={"Authorization": f"Bearer {TOKEN}"})


def test_api_needs_the_token(client):
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/listen").status_code == 401
    assert client.get("/api/audio?file=step-00002500.wav").status_code == 401
    assert auth(client, "/api/status").json()["data"]["step"] == 20


def test_status_reports_the_newest_run_its_progress_and_the_balance(client):
    data = auth(client, "/api/status").json()["data"]
    assert data["run"] == "main" and data["max_steps"] == 100 and data["alive"] is False
    assert data["seconds_per_step"] == pytest.approx(1.0)
    assert data["balance"]["balance"] == 31.5


def test_unknown_run_and_unsafe_file_names_are_refused(client):
    assert auth(client, "/api/status?run=other").status_code == 404
    assert auth(client, "/api/audio?file=..%2Fmetrics.jsonl").status_code == 400
    assert auth(client, "/api/audio?file=step-00002500.wav").status_code == 200


def test_listening_clips_are_served_only_from_known_folders(client):
    data = auth(client, "/api/listen").json()["data"]
    assert data["sentences"] == ["Salam."] and [v["name"] for v in data["versions"]] == ["base"]
    assert auth(client, "/api/audio?listen=base&file=00.wav").status_code == 200
    assert auth(client, "/api/audio?listen=step-00005000&file=00.wav").status_code == 404
    assert auth(client, "/api/audio?listen=..&file=00.wav").status_code == 400
    assert auth(client, "/api/audio?listen=validation&file=step-00002500.wav").status_code == 400


def test_page_files_are_revalidated_so_an_update_reaches_the_phone(client):
    assert client.get("/").headers["cache-control"] == "no-cache"


def test_alive_never_signals_on_windows(tmp_path, monkeypatch):
    """A downloaded run's PID belongs to the pod, and on Windows os.kill(pid, 0) sends Ctrl+C instead of probing."""
    pid_file = tmp_path / "main.pid"
    pid_file.write_text("1156")
    calls = []
    monkeypatch.setattr(monitor_app.os, "kill", lambda *args: calls.append(args))
    assert _alive(pid_file, platform="nt") is False and calls == []


def test_alive_probes_the_pid_on_linux(tmp_path, monkeypatch):
    pid_file = tmp_path / "main.pid"
    pid_file.write_text("1156\n")
    calls = []
    monkeypatch.setattr(monitor_app.os, "kill", lambda *args: calls.append(args))
    assert _alive(pid_file, platform="posix") is True and calls == [(1156, 0)]

    def gone(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(monitor_app.os, "kill", gone)
    assert _alive(pid_file, platform="posix") is False


def test_validation_lists_steps_and_the_before_clip(client, tmp_path):
    folder = tmp_path / "runs" / "main" / "validation"
    (folder / "before.wav").write_bytes(b"RIFF....WAVE")
    (folder / "sentence.json").write_text(json.dumps({"text": "Salam."}), encoding="utf-8")
    data = auth(client, "/api/validation").json()["data"]
    assert [s["step"] for s in data["steps"]] == [2500]
    assert data["text"] == "Salam." and data["before"] == "before.wav" and data["target"] is None
    assert auth(client, "/api/audio?file=before.wav").status_code == 200
