"""Cluster runner preflight tests; no network or GPU."""

from __future__ import annotations

import pytest

import run_cluster_benchmark


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
