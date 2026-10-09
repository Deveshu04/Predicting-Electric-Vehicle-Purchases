import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("run_notebook", ROOT / "scripts" / "run_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_folder(tmp_path):
    folder = tmp_path / "notebooks" / "01-eda"
    folder.mkdir(parents=True)
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "someone/ev-purchase-01-eda"}), encoding="utf-8")
    return folder


def make_outputs(tmp_path, before, after):
    dest = tmp_path / "kaggle_out" / "02-models"
    dest.mkdir(parents=True)
    (dest / "stdout_prev.txt").write_text(before, encoding="utf-8")
    (dest / "stdout.txt").write_text(after, encoding="utf-8")


def test_build_skips_when_source_is_absent(tmp_path, monkeypatch):
    runner = load_runner()
    folder = make_folder(tmp_path)
    (folder / "01-eda.ipynb").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))
    runner.build("01-eda")
    assert calls == []


def test_build_runs_jupytext_when_source_exists(tmp_path, monkeypatch):
    runner = load_runner()
    folder = make_folder(tmp_path)
    (folder / "01-eda.py").write_text("", encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))
    runner.build("01-eda")
    assert len(calls) == 1 and "jupytext" in calls[0][0]


def test_smoke_push_restores_notebook_without_source(tmp_path, monkeypatch):
    runner = load_runner()
    folder = make_folder(tmp_path)
    notebook = folder / "01-eda.ipynb"
    original = b'{"cells": [{"source": ["SMOKE = False"]}]}\n'
    notebook.write_bytes(original)
    pushed = []

    def fake_run(command, **kwargs):
        pushed.append(notebook.read_text(encoding="utf-8"))
        return type("Result", (), {"stdout": "Kernel version 1 successfully pushed.", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.push("01-eda", smoke=True)
    assert "SMOKE = True" in pushed[0]
    assert notebook.read_bytes() == original


def test_push_error_text_fails_even_with_exit_zero(tmp_path, monkeypatch):
    runner = load_runner()
    folder = make_folder(tmp_path)
    (folder / "01-eda.ipynb").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: type("Result", (), {"stdout": "Kernel push error: invalid source", "stderr": "", "returncode": 0})())
    try:
        runner.push("01-eda", smoke=False)
    except SystemExit as stop:
        assert "failed" in str(stop)
    else:
        raise AssertionError("push should have stopped")


def test_compare_flags_changed_numbers_but_not_timing(tmp_path, monkeypatch, capsys):
    runner = load_runner()
    make_outputs(
        tmp_path,
        "best fold-0 AUC 0.94511\ntuning elapsed: 650 seconds\ntrials completed: 14 of 20\n",
        "best fold-0 AUC 0.94511\ntuning elapsed: 702 seconds\ntrials completed: 15 of 20\n",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.compare("02-models")
    out = capsys.readouterr().out
    assert "+ trials completed: 15 of 20" in out
    assert "1 lines differ" in out


def test_compare_identical_when_only_timing_differs(tmp_path, monkeypatch, capsys):
    runner = load_runner()
    make_outputs(
        tmp_path,
        "LightGBM seed 42: OOF AUC 0.94617\nLightGBM elapsed: 1280 seconds\n",
        "LightGBM seed 42: OOF AUC 0.94617\nLightGBM elapsed: 1311 seconds\n",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.compare("02-models")
    assert capsys.readouterr().out.strip().endswith("identical")


def status_sequence(runner, monkeypatch, tmp_path, statuses):
    make_folder(tmp_path)
    pending = iter(statuses)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Result", (), {"stdout": next(pending), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    return calls


def test_wait_ignores_a_finished_status_until_the_new_run_starts(tmp_path, monkeypatch):
    runner = load_runner()
    calls = status_sequence(runner, monkeypatch, tmp_path, [
        'has status "KernelWorkerStatus.COMPLETE"',
        'has status "KernelWorkerStatus.QUEUED"',
        'has status "KernelWorkerStatus.RUNNING"',
        'has status "KernelWorkerStatus.COMPLETE"',
    ])
    runner.wait("01-eda", 1)
    assert len(calls) == 4


def test_wait_keeps_waiting_through_a_transient_cli_error(tmp_path, monkeypatch):
    runner = load_runner()
    calls = status_sequence(runner, monkeypatch, tmp_path, [
        'has status "KernelWorkerStatus.RUNNING"',
        "SSL error: connection reset by peer",
        'has status "KernelWorkerStatus.RUNNING"',
        'has status "KernelWorkerStatus.COMPLETE"',
    ])
    runner.wait("01-eda", 1)
    assert len(calls) == 4


def test_wait_stops_when_authentication_expires(tmp_path, monkeypatch):
    runner = load_runner()
    status_sequence(runner, monkeypatch, tmp_path, [
        'has status "KernelWorkerStatus.RUNNING"',
        "Authentication required to call the Kaggle API.",
    ])
    try:
        runner.wait("01-eda", 1)
    except SystemExit as stop:
        assert "kaggle auth login" in str(stop)
    else:
        raise AssertionError("wait should have stopped")


def test_fetch_reads_only_the_kernel_log(tmp_path, monkeypatch):
    runner = load_runner()
    make_folder(tmp_path)
    dest = tmp_path / "kaggle_out" / "01-eda"
    dest.mkdir(parents=True)
    (dest / "zz-older-run.log").write_text(json.dumps([{"stream_name": "stdout", "data": "old run\n"}]), encoding="utf-8")

    def fake_run(command, **kwargs):
        (dest / "ev-purchase-01-eda.log").write_text(json.dumps([{"stream_name": "stdout", "data": "new run\n"}]), encoding="utf-8")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.fetch("01-eda", None)
    assert (dest / "stdout.txt").read_text(encoding="utf-8") == "new run\n"
