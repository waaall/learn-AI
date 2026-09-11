"""模型来源的替身回归测试；不调用网络或 GPU，供目标环境手动运行。"""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_model
from model_files import validate_model_directory


@pytest.fixture
def model_directory(tmp_path):
    root = tmp_path / "model"
    codec = root / "speech_tokenizer"
    codec.mkdir(parents=True)
    (root / "config.json").write_text('{"tts_model_type":"custom_voice"}')
    for name in ("generation_config.json", "tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
        (root / name).write_text('{}')
    for name in ("config.json", "preprocessor_config.json"):
        (codec / name).write_text('{}')
    for directory in (root, codec):
        (directory / "model.safetensors").write_bytes(b"placeholder-not-real-weights")
    return root


@pytest.fixture
def source_env(monkeypatch, tmp_path):
    for name in ("MODEL_LOCAL_PATH", "HF_TOKEN_FILE", "HF_HUB_OFFLINE", "HF_HUB_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    monkeypatch.setenv("HF_FALLBACK_TO_OFFICIAL", "1")
    monkeypatch.setenv("MODEL_DOWNLOAD_ATTEMPTS", "1")
    monkeypatch.setenv("MODEL_REVISION", "main")
    monkeypatch.setattr(prepare_model, "cached_model", lambda *args: None)
    hub, errors = ModuleType("huggingface_hub"), ModuleType("huggingface_hub.errors")

    class Missing(FileNotFoundError):
        pass

    def missing(*args, **kwargs):
        raise Missing()

    hub.snapshot_download = missing
    errors.LocalEntryNotFoundError = Missing
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


def test_local_priority(source_env, monkeypatch, model_directory):
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(model_directory))
    monkeypatch.setattr(prepare_model.subprocess, "run", lambda *a, **k: pytest.fail("不应联网"))
    assert prepare_model.prepare() == model_directory


def test_missing_local_never_falls_back(source_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="不存在"):
        prepare_model.prepare()


def test_lfs_pointer_rejected(model_directory):
    (model_directory / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS"):
        validate_model_directory(str(model_directory))


def test_missing_codec_rejected(model_directory):
    (model_directory / "speech_tokenizer" / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="权重"):
        validate_model_directory(str(model_directory))


def test_offline_does_not_download(source_env, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="离线"):
        prepare_model.prepare()


def test_fallback_and_token_isolation(source_env, monkeypatch, model_directory, tmp_path):
    token_file = tmp_path / "secret"
    token_file.write_text("test-only-not-a-real-token")
    monkeypatch.setenv("HF_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("HF_TOKEN", "must-not-be-in-worker-env")
    calls = []

    def run(command, *, input, env, **kwargs):
        config = json.loads(input)
        calls.append(config)
        assert "HF_TOKEN" not in env
        assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        if len(calls) == 1:
            assert config["token_file"] == ""
            Path(config["revision_file"]).write_text("a" * 40)
            return SimpleNamespace(returncode=prepare_model.DOWNLOAD_FAILURE)
        assert config["endpoint"] == prepare_model.OFFICIAL_ENDPOINT
        assert config["token_file"] == str(token_file)
        assert Path(config["revision_file"]).read_text() == "a" * 40
        Path(config["output"]).write_text(str(model_directory))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare_model.subprocess, "run", run)
    assert prepare_model.prepare() == model_directory
    assert len(calls) == 2
    # 再次启动直接复用已完成目录，不再访问下载源。
    assert prepare_model.prepare() == model_directory
    assert len(calls) == 2


def test_non_download_error_does_not_fallback(source_env, monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=22)
    monkeypatch.setattr(prepare_model.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="非下载"):
        prepare_model.prepare()
    assert len(calls) == 1


def test_timeout_exhausts_sources(source_env, monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        raise prepare_model.subprocess.TimeoutExpired("worker", 1)
    monkeypatch.setattr(prepare_model.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="所有下载源"):
        prepare_model.prepare()
    assert len(calls) == 2


def test_reject_credentials_in_endpoint(source_env, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://user:password@mirror.example")
    with pytest.raises(ValueError, match="无凭据"):
        prepare_model.endpoint_sources()
