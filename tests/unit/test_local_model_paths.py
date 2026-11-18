from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ncm_lab.teacher import model as teachers


@pytest.mark.parametrize("absolute", [False, True])
def test_model_and_tokenizer_use_repository_asset_paths_from_another_directory(tmp_path, monkeypatch, absolute):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".ncm-lab").touch()
    cached = project / "runs/assets/target/revision"
    cached.mkdir(parents=True)
    outside = tmp_path / "terminal directory"
    outside.mkdir()
    monkeypatch.setenv("NCM_REPO_ROOT", str(project))
    monkeypatch.chdir(outside)
    spec = dict(
        model_id="Qwen/Qwen3-8B", revision="revision", tokenizer_revision="revision",
        path=str(cached) if absolute else "runs/assets/target/revision",
    )
    original = Mock()
    original.config = SimpleNamespace(use_cache=True)
    backbone_loader = Mock(return_value=original)
    tokenizer_loader = Mock(return_value=object())
    verified = []
    monkeypatch.setattr(teachers, "verify_cached_assets", lambda value: verified.append(teachers.cached_path(value)))
    monkeypatch.setattr(teachers, "validate_model_spec", lambda *_: None)
    monkeypatch.setattr(teachers.AutoModelForCausalLM, "from_pretrained", backbone_loader)
    monkeypatch.setattr(teachers.AutoTokenizer, "from_pretrained", tokenizer_loader)

    assert teachers.load_original(spec, frozen=True) is original
    teachers.load_tokenizer(spec)

    assert verified == [cached]
    for loader in (backbone_loader, tokenizer_loader):
        args, kwargs = loader.call_args
        assert args == (cached,)
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == "revision"
    assert original.config.use_cache is False
    original.requires_grad_.assert_called_once_with(False)
