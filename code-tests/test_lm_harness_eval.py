"""What can be executed of the lm-eval wrapper without a GPU.

Background: commit e2dc98a renamed KoopmanEvalWrapper.__init__'s `dtype`
parameter and left `self._dtype = dtype` behind, so the constructor raised
NameError on every call; a0cf125 fixed it by reading the dtype off the model's
parameters. Neither the break nor the fix was ever *run* -- `lm_eval` is absent
from the CPU environment, so nothing could even import the module. With lm_eval
on the path the import gate in test_module_import_health.py covers module scope,
and these tests cover the part of the constructor that a CPU can reach.

HOW FAR THE CONSTRUCTOR GETS ON A CPU. In order, __init__ does: LM.__init__ ->
plain attribute assignment -> build_config(model_size) -> load meta.pt and check
model_type -> load the tokenizer -> MODEL_BUILDERS[...](cfg) -> load_state_dict
-> .to(device) -> autocast + recurrent wrapper. The builder is a hard stop: all
five MODEL_BUILDERS entries build a Mamba backbone, and
koopman_lm/modules/seq/mamba.py:10 does `from mamba_ssm import Mamba2`, a
CUDA-only package that is not installed here.

So the model-type guard and the tokenizer resolution ARE testable; the autocast,
the recurrent/attention split, _model_call and _model_generate are NOT, and no
test here pretends otherwise by stubbing the backbone -- a stub would exercise
the stub. Those paths stay unverified until someone runs this on a GPU.

Everything here is offline: no test reaches the network or the HF hub.
"""


import pytest
import torch

pytest.importorskip("lm_eval", reason="lm-eval is an optional heavy dependency")

from lm_eval.api.registry import get_model                        # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast   # noqa: E402

from experimentation.evaluation import lm_harness_eval as lh      # noqa: E402


@pytest.fixture(scope="module")
def local_tokenizer(tmp_path_factory):
    """A real, tiny, offline tokenizer directory AutoTokenizer can load.

    Built from `tokenizers` primitives rather than downloaded, so these tests
    do not depend on the network or on a warm HF cache.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "</s>": 1}
    vocab.update({f"tok{i}": i for i in range(2, 128)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=backend,
                                   unk_token="<unk>", eos_token="</s>")
    out = tmp_path_factory.mktemp("tokenizer")
    fast.save_pretrained(str(out))
    return out


@pytest.fixture
def record_tokenizer_loads(monkeypatch):
    """Record every AutoTokenizer.from_pretrained argument, then delegate to the
    real loader. The fallback ORDER is the behaviour under test, so the loads
    themselves stay real -- only the call sites are observed."""
    calls = []
    real = AutoTokenizer.from_pretrained

    def spy(name_or_path, *args, **kwargs):
        calls.append(str(name_or_path))
        return real(name_or_path, *args, **kwargs)

    monkeypatch.setattr(lh, "AutoTokenizer",
                        type("Spy", (), {"from_pretrained": staticmethod(spy)}))
    return calls


def _construct(tmp_path, **kwargs):
    """Construct the wrapper, returning the exception it dies on.

    It always dies: the Mamba backbone needs CUDA (see the module docstring).
    The point is how far it got first.
    """
    kwargs.setdefault("model_size", "1m")
    kwargs.setdefault("device", "cpu")
    with pytest.raises(Exception) as exc:
        lh.KoopmanEvalWrapper(checkpoint=str(tmp_path / "model.pt"), **kwargs)
    return exc.value


# ---------------------------------------------------------------- model type

def test_wrapper_rejects_a_checkpoint_trained_as_a_different_model_type(tmp_path):
    """meta.pt's model_type must match the --model the harness was invoked with.

    Every variant is 32k-vocab, so without this guard a mamba_only checkpoint
    scored under --model koopman would produce plausible numbers for the wrong
    model rather than an error.
    """
    torch.save({"model_type": "mamba_only"}, tmp_path / "meta.pt")

    with pytest.raises(ValueError, match="model_type='mamba_only'"):
        lh.KoopmanEvalWrapper(checkpoint=str(tmp_path / "model.pt"),
                              model_size="1m", device="cpu")


def test_a_matching_model_type_gets_past_the_guard(tmp_path, local_tokenizer):
    """The guard is not vacuous: a matching model_type reaches the tokenizer."""
    torch.save({"model_type": "koopman"}, tmp_path / "meta.pt")
    exc = _construct(tmp_path, tokenizer=str(local_tokenizer))
    assert "model_type" not in str(exc), str(exc)
    assert isinstance(exc, ModuleNotFoundError) and exc.name == "mamba_ssm"


def test_a_checkpoint_with_no_meta_is_not_read_as_a_mismatch(tmp_path,
                                                             local_tokenizer):
    """meta.pt is optional; `meta.get("model_type")` is None, not a mismatch."""
    exc = _construct(tmp_path, tokenizer=str(local_tokenizer))
    assert "model_type" not in str(exc), str(exc)


def test_meta_without_a_model_type_key_is_not_read_as_a_mismatch(tmp_path,
                                                                 local_tokenizer):
    """Older checkpoints wrote a meta.pt with no model_type at all."""
    torch.save({"cfg_hash": "abc"}, tmp_path / "meta.pt")
    exc = _construct(tmp_path, tokenizer=str(local_tokenizer))
    assert "model_type" not in str(exc), str(exc)


# --------------------------------------------------------------- tokenizer

def test_the_checkpoint_directory_tokenizer_wins_over_the_argument(
        tmp_path, local_tokenizer, record_tokenizer_loads):
    """train.py saves the tokenizer that produced the checkpoint's vocab next to
    it. That one must be preferred: the `tokenizer` argument's default differs
    from train.py's own default and both are 32k-vocab, so the wrong choice
    mismaps every token id to the wrong embedding silently instead of erroring.
    """
    for f in local_tokenizer.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())

    _construct(tmp_path, tokenizer="mistralai/Mistral-7B-v0.1")

    assert record_tokenizer_loads == [str(tmp_path)], (
        "the checkpoint directory's tokenizer must be the only one loaded; "
        f"loads were {record_tokenizer_loads}")


def test_the_argument_is_the_fallback_when_the_checkpoint_has_no_tokenizer(
        tmp_path, local_tokenizer, record_tokenizer_loads):
    """Checkpoints from before train.py saved a tokenizer have none alongside,
    so the argument is tried second -- and only second."""
    _construct(tmp_path, tokenizer=str(local_tokenizer))

    assert record_tokenizer_loads == [str(tmp_path), str(local_tokenizer)], (
        f"expected ckpt-dir-then-argument; loads were {record_tokenizer_loads}")


def test_a_bad_fallback_tokenizer_is_not_swallowed(tmp_path):
    """Only the FIRST load is wrapped in try/except. If the fallback also fails
    the constructor must raise, not continue with no tokenizer."""
    exc = _construct(tmp_path, tokenizer=str(tmp_path / "nonexistent-tokenizer"))
    assert not isinstance(exc, ModuleNotFoundError), (
        "reached the backbone with an unresolved tokenizer")


# ---------------------------------------------------------------- registry

def test_every_lm_eval_name_resolves_to_a_wrapper_with_a_matching_model_type():
    """@register_model's name and the class's MODEL_TYPE are declared twice. The
    builder lookup uses MODEL_TYPE, so a drift between them means `--model X`
    silently builds a different architecture than X."""
    for name in lh.MODEL_BUILDERS:
        cls = get_model(name)
        assert cls.MODEL_TYPE == name, (
            f"lm-eval name {name!r} resolves to {cls.__name__}, whose "
            f"MODEL_TYPE is {cls.MODEL_TYPE!r}")


def test_every_wrapper_subclass_has_a_builder():
    """The other direction: a wrapper whose MODEL_TYPE has no MODEL_BUILDERS
    entry is a --model value lm-eval accepts and then fails on, after loading a
    tokenizer."""
    for cls in (lh.KoopmanEvalWrapper, *lh.KoopmanEvalWrapper.__subclasses__()):
        assert cls.MODEL_TYPE in lh.MODEL_BUILDERS, (
            f"{cls.__name__}.MODEL_TYPE={cls.MODEL_TYPE!r} has no builder")


def test_the_usage_example_does_not_advertise_the_removed_precision_knob():
    """a0cf125 removed weight_dtype because its only correct value was derivable
    from the file. The docstring is this entry point's only documentation."""
    assert "weight_dtype" not in lh.__doc__
    for flag in ("--model", "--model_args", "--tasks", "--batch_size", "--device"):
        assert flag in lh.__doc__, f"{flag} vanished from the usage example"
