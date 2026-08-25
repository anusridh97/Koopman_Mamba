"""The retrieval arm of the write-gate comparison: `--ska_beta_policy` on
`mqar_finetune`.

## Why this arm exists, and why it is the one that was owed

The v1.1 sqrt-beta convention was landed on `main` with its own acceptance spec
declaring, in writing, that a contractivity proof is "necessary but NOT
sufficient" and that the binding criterion is a **retrained MQAR/retrieval + LM
eval**. That spec (`ACCEPTANCE_v1_1_sqrt_beta.md`) and its harness exist only on
an unmerged branch; the numerics reached `main` by a different route, and the A/B
was never run. The failure mode the spec singled out is the one no residual or
contractivity test can see:

    alpha == 1 AND retrieval regressed

`code-tests/test_ska_contractivity_contract.py` establishes the first half of
that conjunction as a THEOREM -- alpha is identically 1 under any single-stream
weighting -- which makes the second half the only remaining question, and makes
it unanswerable from the math side by construction.

LM loss on a 25M model over 98M tokens is a poor instrument for it. The write
gate exists to decide what enters the associative memory, so the objective that
can see it is associative recall. MQAR is that objective, it is the one the
acceptance spec named, and this flag is what makes the four-way comparison
runnable on it.

## Why an explicit flag rather than a generic override

`mqar_finetune.py` builds its config from `build_config(args.model_size)` and a
single `dataclasses.replace`. A generic `--model_override KEY=VALUE` would have
been fewer lines and is the wrong shape for the same reason
`space.restrict_space` refuses one: an unvalidated passthrough turns a typo into
a failed run after a GPU has been claimed. `ska_beta_policy` is validated
against `BETA_POLICIES` at parse time, on a login node.
"""
import dataclasses

import pytest

from koopman_lm.config import BETA_POLICIES, build_config

pytestmark = pytest.mark.correctness


def _parse(argv):
    from experimentation.experiments.mqar_finetune import parse_args
    return parse_args(argv)


def _cfg(argv):
    from experimentation.experiments.mqar_finetune import model_config_for
    return model_config_for(_parse(argv))


BASE = ["--model_size", "1m", "--num_kv_pairs", "8", "--distractor_gap", "64"]


def test_omitting_the_flag_inherits_the_config_default():
    """No flag must mean no change, or every archived MQAR cell silently
    changes meaning."""
    cfg = _cfg(BASE)
    assert cfg.ska_beta_policy == build_config("1m").ska_beta_policy == "learned"


@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_the_flag_reaches_the_model_config(policy):
    cfg = _cfg(BASE + ["--ska_beta_policy", policy])
    assert cfg.ska_beta_policy == policy


def test_the_seq_len_derivation_is_unchanged_by_the_flag():
    """`max_seq_len` is derived as 4*M + gap and the flag must not disturb it --
    a retrieval comparison in which the cells had different context lengths
    would be measuring context length."""
    plain = _cfg(BASE)
    gated = _cfg(BASE + ["--ska_beta_policy", "one"])
    assert plain.max_seq_len == gated.max_seq_len == 4 * 8 + 64
    # And nothing ELSE moved either: the two configs must differ in exactly one
    # field, or the arm is not a controlled comparison.
    differing = {
        f.name for f in dataclasses.fields(plain)
        if getattr(plain, f.name) != getattr(gated, f.name)}
    assert differing == {"ska_beta_policy"}, differing


def test_an_unknown_policy_is_refused_at_parse_time():
    """On a login node, before a GPU is claimed. argparse `choices` gives the
    valid set in the error, which a post-hoc check inside the run would not."""
    with pytest.raises(SystemExit):
        _parse(BASE + ["--ska_beta_policy", "sqrt"])


def test_the_policy_changes_the_config_hash():
    """Each cell must be a distinct recorded artefact. `mqar_finetune` prints
    and stores `config_hash(cfg)`, and two policies sharing a hash would make
    the four cells indistinguishable in their own saved results."""
    from koopman_lm.config import config_hash

    hashes = {p: config_hash(_cfg(BASE + ["--ska_beta_policy", p]))
              for p in sorted(BETA_POLICIES)}
    assert len(set(hashes.values())) == len(BETA_POLICIES), hashes
    # ... and `learned` still hashes as the pre-flag config did, because it is
    # an identity-transparent default. So an archived MQAR result keeps its hash.
    assert hashes["learned"] == config_hash(
        dataclasses.replace(build_config("1m"), max_seq_len=4 * 8 + 64))
