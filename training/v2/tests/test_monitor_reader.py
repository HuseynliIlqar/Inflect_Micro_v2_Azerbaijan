import json

import pytest

from avazmon.reader import MetricsReader, downsample, listen_versions, progress, validation_steps


def row(step, wall, mel=1.0, stage="linguistic_adaptation"):
    return {"step": step, "epoch": 1, "stage": stage, "loss_g": 2.0, "loss_d": 1.0, "loss_mel": mel,
            "loss_duration": 0.5, "loss_kl": 1.5, "lr": {"linguistic": 2e-4}, "peak_memory_gb": 30.0,
            "peak_reserved_gb": 32.0, "wall_time": wall}


def write(path, rows, mode="w"):
    with path.open(mode, encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)


def test_reader_reads_new_lines_only_and_skips_a_half_written_one(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write(path, [row(1, 100.0), row(2, 101.0)])
    reader = MetricsReader(path)
    assert [r["step"] for r in reader.rows()] == [1, 2]
    write(path, [row(3, 102.0)], mode="a")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 4, "wall')  # the trainer is mid-write
    assert [r["step"] for r in reader.rows()] == [1, 2, 3]


def test_reader_starts_over_when_the_file_is_replaced(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write(path, [row(1, 100.0), row(2, 101.0), row(3, 102.0)])
    reader = MetricsReader(path)
    reader.rows()
    write(path, [row(1, 200.0)])
    assert [r["wall_time"] for r in reader.rows()] == [200.0]


def test_reader_is_safe_when_two_requests_read_at_once(tmp_path):
    import threading

    path = tmp_path / "metrics.jsonl"
    write(path, [row(i, float(i)) for i in range(1, 2001)])
    reader = MetricsReader(path)
    threads = [threading.Thread(target=reader.rows) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write(path, [row(2001, 2001.0)], mode="a")
    assert [r["step"] for r in reader.rows()] == list(range(1, 2002))


def test_reader_skips_a_broken_line_and_waits_for_a_tail_still_landing(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(json.dumps(row(1, 1.0)).encode() + b"\n" + b"garbage\n" + json.dumps(row(2, 2.0)).encode()
                     + b"\n" + b"\x00" * 30 + b"\n")
    reader = MetricsReader(path)
    assert [r["step"] for r in reader.rows()] == [1, 2]  # the zeros are a network-disk tail not written yet
    path.write_bytes(path.read_bytes().replace(b"\x00" * 30, json.dumps(row(3, 3.0)).encode()))
    assert [r["step"] for r in reader.rows()] == [1, 2, 3]  # the same bytes, once they have landed


def test_downsample_averages_losses_into_at_most_n_points():
    rows = [row(i, float(i), mel=float(i)) for i in range(1, 101)]
    points = downsample(rows, 10)
    assert len(points) == 10
    assert points[0]["step"] == 10 and points[0]["loss_mel"] == pytest.approx(5.5)
    assert downsample(rows[:5], 10) == downsample(rows[:5], 5)


def test_downsample_keeps_the_spread_speed_and_a_step_window():
    rows = [row(i, 2.0 * i, mel=float(i % 4)) for i in range(1, 101)]
    points = downsample(rows, 10)
    assert points[0]["loss_mel_min"] == 0.0 and points[0]["loss_mel_max"] == 3.0
    assert points[0]["seconds_per_step"] == pytest.approx(2.0)
    recent = downsample(rows, 10, from_step=51)
    assert recent[0]["step"] > 51 and recent[-1]["step"] == 100 and len(recent) == 10


def test_progress_reports_speed_eta_and_cost():
    rows = [row(i, 1000.0 + 1.25 * i) for i in range(1, 301)]
    status = progress(rows, max_steps=1300, price_per_hour=0.6)
    assert status["step"] == 300 and status["percent"] == pytest.approx(23.08, abs=0.01)
    assert status["seconds_per_step"] == pytest.approx(1.25)
    assert status["eta_seconds"] == pytest.approx(1000 * 1.25)
    assert status["cost_so_far"] == pytest.approx(299 * 1.25 / 3600 * 0.6, abs=0.001)
    assert progress([], max_steps=10, price_per_hour=0.6)["step"] == 0


def test_progress_counts_epochs_from_the_logged_epoch_numbers():
    rows = [{**row(step, float(step)), "epoch": (step - 1) // 680 + 1} for step in range(1, 2501)]
    status = progress(rows, max_steps=68000, price_per_hour=0.6)
    assert status["epoch"] == 4
    assert status["steps_per_epoch"] == 680
    assert status["total_epochs"] == pytest.approx(100.0)
    early = progress(rows[:600], max_steps=68000, price_per_hour=0.6)
    assert early["epoch"] == 1 and early["steps_per_epoch"] is None and early["total_epochs"] is None


def test_validation_steps_list_one_clip_per_step_newest_first(tmp_path):
    folder = tmp_path / "validation"
    folder.mkdir()
    for step in (2500, 5000, 7500):
        (folder / f"step-{step:08d}.wav").write_bytes(b"RIFF")
        (folder / f"step-{step:08d}.json").write_text(json.dumps({"seconds": step / 1000}))
    (folder / "notes.wav").write_bytes(b"RIFF")
    steps = validation_steps(folder, limit=2)
    assert [s["step"] for s in steps] == [7500, 5000]
    assert steps[0]["clips"] == [{"speaker": "Avaz", "file": "step-00007500.wav", "seconds": 7.5}]


def test_validation_steps_of_a_missing_folder_are_empty(tmp_path):
    assert validation_steps(tmp_path / "nothing") == []


def test_listen_versions_put_the_newest_checkpoint_first_and_the_old_avaz_last(tmp_path):
    folder = tmp_path / "listen"
    for name in ("base", "step-00005000", "step-00010000", "step-00015000.partial"):
        (folder / name).mkdir(parents=True)
        (folder / name / "00.wav").write_bytes(b"RIFF")
    (folder / "step-00010000" / "01.wav").write_bytes(b"RIFF")
    (folder / "sentences.json").write_text(json.dumps(["Bir.", "İki."]), encoding="utf-8")
    data = listen_versions(folder)
    assert data["sentences"] == ["Bir.", "İki."]
    assert [v["name"] for v in data["versions"]] == ["step-00010000", "step-00005000", "base"]
    assert data["versions"][0] == {"name": "step-00010000", "step": 10000, "files": ["00.wav", "01.wav"], "metrics": None}
    assert data["versions"][-1]["step"] is None


def test_listen_versions_before_any_sample(tmp_path):
    assert listen_versions(tmp_path / "listen") == {"sentences": [], "versions": [], "reference": None}


def test_validation_extras_name_the_sentence_and_the_comparison_clips(tmp_path):
    from avazmon.reader import validation_extras

    folder = tmp_path / "validation"
    folder.mkdir()
    assert validation_extras(folder) == {"text": None, "before": None, "target": None}
    (folder / "before.wav").write_bytes(b"RIFF")
    (folder / "target.wav").write_bytes(b"RIFF")
    (folder / "sentence.json").write_text(json.dumps({"text": "Salam."}), encoding="utf-8")
    assert validation_extras(folder) == {"text": "Salam.", "before": "before.wav", "target": "target.wav"}
    assert [s["step"] for s in validation_steps(folder)] == []  # before/target are not validation steps


def test_listen_versions_carry_measurements_and_the_dataset_reference(tmp_path):
    folder = tmp_path / "listen"
    (folder / "step-00025000").mkdir(parents=True)
    (folder / "step-00025000" / "00.wav").write_bytes(b"RIFF")
    (folder / "step-00025000" / "metrics.json").write_text(json.dumps({"h2_4k": 6.9, "ring_share": 0.54}))
    (folder / "dataset_metrics.json").write_text(json.dumps({"h2_4k": 10.6, "ring_share": 0.13}))
    data = listen_versions(folder)
    assert data["versions"][0]["metrics"] == {"h2_4k": 6.9, "ring_share": 0.54}
    assert data["reference"]["h2_4k"] == 10.6


def test_downsample_passes_decoder_run_losses_only_where_logged():
    rows = [row(1, 100.0), {**row(2, 101.0), "d_mrd": 1.4}, {**row(3, 102.0), "d_mrd": 1.2}]
    points = downsample(rows, points=1)
    assert points[0]["d_mrd"] == pytest.approx(1.3) and "d_cqtd" not in points[0]
