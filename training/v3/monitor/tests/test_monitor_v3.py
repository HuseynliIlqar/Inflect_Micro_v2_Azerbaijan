"""v3 additions to the monitor: phases that log different losses, folder-per-step validation and its audio."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from avazmon.app import Settings, create_app
from avazmon.reader import progress, validation_steps


def test_progress_survives_phases_without_some_losses():
    rows = [{"step": 1, "stage": "prosody", "wall_time": 0.0, "loss_mel": 0.0, "loss_g": 5.0, "loss_d": 0.0,
             "loss_kl": 1.3, "loss_duration": 0.4, "loss_pitch_l1": 2.5},
            {"step": 2, "stage": "disc_warmup", "wall_time": 1.0, "loss_mel": 0.0, "loss_g": 0.0, "loss_d": 4.5}]
    latest = progress(rows, 10, 0.59)["latest"]
    assert latest["loss_kl"] == 1.3 and latest["loss_pitch_l1"] == 2.5 and latest["loss_d"] == 2.25


def _v3_validation(folder):
    step = folder / "step-00000500"
    step.mkdir(parents=True)
    for i in range(2):
        (step / f"{i:02d}.wav").write_bytes(b"RIFF")
    (step / "summary.json").write_text(json.dumps([{"clip": 0, "seconds": 2.0, "range_st_pred": 11.8,
                                                    "range_st_true": 12.4}, {"clip": 1, "seconds": 3.0}]))


def test_v3_validation_folders_are_listed_with_pitch_ranges(tmp_path):
    _v3_validation(tmp_path)
    steps = validation_steps(tmp_path, limit=5)
    assert steps[0]["step"] == 500
    first = steps[0]["clips"][0]
    assert first["folder"] == "step-00000500" and first["range_pred"] == 11.8 and first["range_true"] == 12.4


def test_v3_validation_audio_is_served_from_its_folder_only(tmp_path):
    run = tmp_path / "v3_trial"
    _v3_validation(run / "validation")
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1, "stage": "prosody", "loss_mel": 0, "loss_g": 1,
                                                   "loss_d": 0, "loss_kl": 1, "loss_duration": 1}) + "\n")
    client = TestClient(create_app(Settings(runs=tmp_path, price_per_hour=0.59, token=None)))
    assert client.get("/api/audio", params={"file": "00.wav", "vstep": "step-00000500"}).status_code == 200
    assert client.get("/api/audio", params={"file": "00.wav", "vstep": "../x"}).status_code == 400
