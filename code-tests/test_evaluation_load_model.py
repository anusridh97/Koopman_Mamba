"""Characterization tests for evaluation's `load_model`.

WHY THESE EXIST. `load_model` was duplicated: evaluate.py:104 and
evaluate_retrieval.py:83, byte-equivalent after stripping the docstring. The
duplication is worth removing, but evaluate.py's version produced the numbers on
record for this repo (a held-out ppl of 214.6 among them), so the extraction has
to be provably behaviour-preserving rather than argued to be. These tests were
written and run against BOTH copies first, then re-run unchanged against the
extracted one. If they ever need editing to accommodate a refactor of
`load_model`, the refactor changed behaviour.

WHAT IS PINNED: the 4-tuple return, the 5-way model_type dispatch and its error,
where model_type and cfg come from, the tokenizer resolution order, that
vocab_size is taken from the tokenizer, and that the model comes back on the CPU
-- moving it to a device is the caller's job, and a load_model that quietly did
it would OOM a multi-checkpoint comparison.

HOW, WITHOUT A GPU. Every one of the five builders builds a Mamba backbone, and
`from mamba_ssm import Mamba2` needs CUDA, so no real model can be constructed
here. The dispatch is therefore observed by substituting the builders -- but
substituted in `load_model.__globals__`, i.e. in whatever module the function
actually lives in, looked up at call time. That is what makes the same test text
valid before and after the extraction: it patches where the function looks,
not where the function used to be.

Everything is offline; the tokenizer is built from `tokenizers` primitives.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast

pytestmark = pytest.mark.correctness

_BUILDERS = {
    "mamba_attn": "build_mamba_attention",
    "mamba_only": "build_mamba_only",
    "mamba_ska_swiglu": "build_mamba_ska_swiglu",
    "mamba_ska_koopman": "build_mamba_ska_koopman",
    "koopman": "KoopmanLM",
}


def _implementations():
    """Every `load_model` in the evaluation package, by import path.

    Parametrizing over this is what makes these tests a *characterization* of
    the duplication rather than of one copy: while both existed, each assertion
    ran twice. After the extraction they resolve to the same function, and the
    duplicate ids are the record that they used not to.
    """
    from experimentation.evaluation import evaluate, evaluate_retrieval

    return [pytest.param(evaluate.load_model, id="evaluate"),
            pytest.param(evaluate_retrieval.load_model, id="evaluate_retrieval")]


implementations = pytest.mark.parametrize("load_model", _implementations())


class _Stub(nn.Module):
    """Stands in for a backbone. One parameter, so load_state_dict and the
    parameter count both have something real to work on."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.zeros(3, 4))


@pytest.fixture
def tokenizer_dir(tmp_path_factory):
    """A real, tiny, offline tokenizer directory, 128 tokens wide to match the
    1m config's vocab_size."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "</s>": 1}
    vocab.update({f"tok{i}": i for i in range(2, 128)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=backend,
                                   unk_token="<unk>", eos_token="</s>")
    out = tmp_path_factory.mktemp("tok")
    fast.save_pretrained(str(out))
    return out


@pytest.fixture
def stub_builders(monkeypatch):
    """Replace all five builders in the module `load_model` actually lives in,
    and record which name got called.

    Patching `load_model.__globals__` rather than a named module is deliberate:
    it follows the function through the extraction, so these tests do not have
    to be rewritten to keep testing the same thing.
    """
    calls = []

    def install(load_model):
        for model_type, builder_name in _BUILDERS.items():
            def make(cfg, _name=builder_name):
                calls.append(_name)
                return _Stub(cfg)
            monkeypatch.setitem(load_model.__globals__, builder_name, make)
        return calls

    return install


def _checkpoint(tmp_path, tokenizer_dir=None, meta=None, name="model.pt"):
    """A loadable checkpoint directory: a state dict matching _Stub, optionally
    a meta.pt, optionally a tokenizer saved alongside it."""
    ckpt = tmp_path / name
    torch.save(_Stub(None).state_dict(), ckpt)
    if meta is not None:
        torch.save(meta, tmp_path / name.replace("model.pt", "meta.pt"))
    if tokenizer_dir is not None:
        for f in tokenizer_dir.iterdir():
            (tmp_path / f.name).write_bytes(f.read_bytes())
    return str(ckpt)


# ------------------------------------------------------------------ contract

@implementations
def test_returns_model_cfg_tokenizer_and_model_type(load_model, tmp_path,
                                                    tokenizer_dir,
                                                    stub_builders):
    from koopman_lm.config import KoopmanLMConfig

    stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir)

    out = load_model(ckpt, model_size="1m")

    assert isinstance(out, tuple) and len(out) == 4, (
        "three callers unpack this into exactly four names")
    model, cfg, tokenizer, model_type = out
    assert isinstance(model, nn.Module)
    assert isinstance(cfg, KoopmanLMConfig)
    assert hasattr(tokenizer, "encode")
    assert model_type == "koopman"


@implementations
def test_the_returned_model_is_on_the_cpu(load_model, tmp_path, tokenizer_dir,
                                          stub_builders):
    """load_model must not move the model to a device. Placement is the
    caller's: evaluate.py compares up to three checkpoints in one process, and a
    load_model that took a device would hold all of them on it at once."""
    stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir)

    model, _, _, _ = load_model(ckpt, model_size="1m")

    assert all(p.device.type == "cpu" for p in model.parameters())


@implementations
def test_the_state_dict_is_actually_loaded(load_model, tmp_path, tokenizer_dir,
                                           stub_builders):
    """Not just built: the checkpoint's weights must be in the returned model."""
    stub_builders(load_model)
    ckpt = tmp_path / "model.pt"
    saved = _Stub(None).state_dict()
    saved["w"] = torch.full((3, 4), 7.0)
    torch.save(saved, ckpt)
    for f in tokenizer_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())

    model, _, _, _ = load_model(str(ckpt), model_size="1m")

    assert torch.equal(model.w.detach(), torch.full((3, 4), 7.0))


# ------------------------------------------------------------------ dispatch

@implementations
@pytest.mark.parametrize("model_type,builder", sorted(_BUILDERS.items()))
def test_meta_model_type_selects_its_builder(load_model, model_type, builder,
                                             tmp_path, tokenizer_dir,
                                             stub_builders):
    calls = stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir, meta={"model_type": model_type})

    _, _, _, resolved = load_model(ckpt, model_size="1m")

    assert calls == [builder]
    assert resolved == model_type


@implementations
@pytest.mark.parametrize("model_type,builder", sorted(_BUILDERS.items()))
def test_the_model_type_argument_overrides_meta(load_model, model_type, builder,
                                                tmp_path, tokenizer_dir,
                                                stub_builders):
    """An explicit --model_type wins; meta is only the fallback."""
    calls = stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir,
                       meta={"model_type": "mamba_only"})

    _, _, _, resolved = load_model(ckpt, model_size="1m",
                                   model_type=model_type)

    assert calls == [builder]
    assert resolved == model_type


@implementations
def test_model_type_defaults_to_koopman_when_meta_is_silent(load_model, tmp_path,
                                                            tokenizer_dir,
                                                            stub_builders):
    """Legacy checkpoints have no meta.pt at all, and pre-date the baselines."""
    calls = stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir)

    _, _, _, resolved = load_model(ckpt, model_size="1m")

    assert calls == ["KoopmanLM"] and resolved == "koopman"


@implementations
def test_an_unknown_model_type_raises_rather_than_guessing(load_model, tmp_path,
                                                          tokenizer_dir,
                                                          stub_builders):
    stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir, meta={"model_type": "gpt2"})

    with pytest.raises(ValueError, match="model_type='gpt2'"):
        load_model(ckpt, model_size="1m")


# ----------------------------------------------------------------------- cfg

@implementations
def test_the_checkpoints_embedded_cfg_beats_the_model_size_argument(
        load_model, tmp_path, tokenizer_dir, stub_builders):
    """A checkpoint's own cfg auto-detects its scale; model_size is the fallback
    for legacy checkpoints that have none. Getting this backwards would evaluate
    a 180m checkpoint under a 50m architecture."""
    from koopman_lm.config import build_config

    stub_builders(load_model)
    embedded = dataclasses.replace(build_config("1m"), d_model=64)
    ckpt = _checkpoint(tmp_path, tokenizer_dir, meta={"cfg": embedded})

    _, cfg, _, _ = load_model(ckpt, model_size="440m")

    assert cfg.d_model == 64


@implementations
def test_model_size_is_used_when_the_checkpoint_has_no_cfg(load_model, tmp_path,
                                                           tokenizer_dir,
                                                           stub_builders):
    from koopman_lm.config import build_config

    stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir, meta={"model_type": "koopman"})

    _, cfg, _, _ = load_model(ckpt, model_size="1m")

    assert cfg.d_model == build_config("1m").d_model


@implementations
def test_vocab_size_comes_from_the_tokenizer_not_the_config(load_model, tmp_path,
                                                            tokenizer_dir,
                                                            stub_builders):
    """The embedding table has to match the tokenizer that produced the ids, so
    the config's own vocab_size is overwritten, not trusted."""
    from koopman_lm.config import build_config

    stub_builders(load_model)
    embedded = dataclasses.replace(build_config("1m"), vocab_size=999)
    ckpt = _checkpoint(tmp_path, tokenizer_dir, meta={"cfg": embedded})

    _, cfg, tokenizer, _ = load_model(ckpt, model_size="1m")

    assert cfg.vocab_size == len(tokenizer) == 128


# ----------------------------------------------------------------- tokenizer

@implementations
def test_the_checkpoint_directory_tokenizer_is_preferred(load_model, tmp_path,
                                                         tokenizer_dir,
                                                         stub_builders,
                                                         monkeypatch):
    """train.py saves the tokenizer that produced the checkpoint's vocab next to
    it. Both it and tokenizer_name's default are 32k-wide, so choosing wrong
    mismaps every token id to the wrong embedding silently instead of erroring.
    """
    stub_builders(load_model)
    seen = []
    real = AutoTokenizer.from_pretrained
    monkeypatch.setitem(
        load_model.__globals__, "AutoTokenizer",
        type("Spy", (), {"from_pretrained": staticmethod(
            lambda p, *a, **k: (seen.append(str(p)), real(p, *a, **k))[1])}))
    ckpt = _checkpoint(tmp_path, tokenizer_dir)

    load_model(ckpt, model_size="1m",
               tokenizer_name="mistralai/Mistral-7B-v0.1")

    assert seen == [str(tmp_path)], (
        f"the argument must not even be consulted; loads were {seen}")


@implementations
def test_tokenizer_name_is_the_fallback_and_only_the_fallback(
        load_model, tmp_path, tokenizer_dir, stub_builders, monkeypatch):
    """Checkpoints written before train.py saved a tokenizer have none
    alongside, so tokenizer_name is tried second."""
    stub_builders(load_model)
    seen = []
    real = AutoTokenizer.from_pretrained
    monkeypatch.setitem(
        load_model.__globals__, "AutoTokenizer",
        type("Spy", (), {"from_pretrained": staticmethod(
            lambda p, *a, **k: (seen.append(str(p)), real(p, *a, **k))[1])}))
    ckpt = _checkpoint(tmp_path)          # no tokenizer next to the checkpoint

    load_model(ckpt, model_size="1m", tokenizer_name=str(tokenizer_dir))

    assert seen == [str(tmp_path), str(tokenizer_dir)]


@implementations
def test_a_failing_fallback_tokenizer_is_not_swallowed(load_model, tmp_path,
                                                       stub_builders):
    """Only the first load is wrapped in try/except; if the fallback fails too,
    the caller must hear about it rather than get a model with no tokenizer."""
    stub_builders(load_model)
    ckpt = _checkpoint(tmp_path)

    with pytest.raises(Exception):
        load_model(ckpt, model_size="1m",
                   tokenizer_name=str(tmp_path / "nonexistent"))


@implementations
def test_pad_token_falls_back_to_eos(load_model, tmp_path, tokenizer_dir,
                                     stub_builders):
    """Batched eval needs a pad id; the tiny tokenizer here has none, so the
    eos-token fallback is what supplies it."""
    stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir)

    _, _, tokenizer, _ = load_model(ckpt, model_size="1m")

    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token == tokenizer.eos_token


# ---------------------------------------------------------------- meta path

@implementations
def test_meta_is_found_by_substituting_model_pt_in_the_checkpoint_path(
        load_model, tmp_path, tokenizer_dir, stub_builders):
    """meta.pt is located by string-replacing "model.pt" in the checkpoint path.
    A checkpoint not named model.pt therefore has no discoverable meta, and
    falls back to model_size and the koopman default."""
    calls = stub_builders(load_model)
    ckpt = _checkpoint(tmp_path, tokenizer_dir, name="weights.pt")
    # meta.pt exists in the directory, but the path substitution cannot find it:
    # "weights.pt".replace("model.pt", "meta.pt") is still "weights.pt".
    torch.save({"model_type": "mamba_only"}, tmp_path / "meta.pt")

    _, _, _, resolved = load_model(ckpt, model_size="1m")

    assert (calls, resolved) == (["KoopmanLM"], "koopman")
