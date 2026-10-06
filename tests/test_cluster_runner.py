"""Cluster runner preflight tests; no network or GPU."""

from __future__ import annotations

import importlib

import pytest

import run_cluster_benchmark
from cluster import check_vllm_runtime


def test_gpu_entrypoints_are_safe_to_import_for_spawn_workers(monkeypatch):
    monkeypatch.setattr("sys.argv", ["spawn-worker"])
    importlib.import_module("cluster.smoke_judge")
    importlib.import_module("open_cascade.__main__")


def test_runtime_preflight_reports_missing_ninja(monkeypatch):
    monkeypatch.setattr(check_vllm_runtime.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="pip install --no-cache-dir ninja"):
        check_vllm_runtime.main()


def test_runner_exposes_tools_installed_beside_python(monkeypatch):
    monkeypatch.setattr(run_cluster_benchmark.sys, "executable", "/env/bin/python")
    monkeypatch.setenv("PATH", "/usr/bin")
    run_cluster_benchmark.ensure_python_bin_on_path()
    assert run_cluster_benchmark.os.environ["PATH"].split(
        run_cluster_benchmark.os.pathsep
    )[0] == "/env/bin"


def test_visionarena_preflight_checks_a_real_shard(monkeypatch):
    import huggingface_hub

    observed = {}
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-token")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_url",
        lambda repo, filename, repo_type: f"https://hub/{repo}/{repo_type}/{filename}",
    )

    def metadata(url, token):
        observed.update(url=url, token=token)

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    run_cluster_benchmark.require_visionarena_access()

    assert observed["token"] == "test-token"
    assert observed["url"].endswith("/data/train-00000-of-00031.parquet")


def test_visionarena_preflight_reports_unaccepted_gate(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-token")
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda *args, **kwargs: "url")
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("403 gated")),
    )

    with pytest.raises(SystemExit, match="cannot read lmarena-ai/VisionArena-Battle"):
        run_cluster_benchmark.require_visionarena_access()
