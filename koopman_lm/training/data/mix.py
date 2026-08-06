"""mix.py -- corpus source registry + mix resolution for pretokenize.

Pure Python (NO numpy / torch / datasets imports) so the mix arithmetic is
unit-testable without the training stack. ``pretokenize.py`` imports
``SOURCE_SPECS`` and the resolver helpers from here; the actual streaming /
tokenization lives there.

A "source" is one streamable corpus. Each entry in ``SOURCE_SPECS`` says how to
load it (``load_dataset`` coordinates) and how to read text from it:

  path / name / split / data_dir : passed straight to ``datasets.load_dataset``
  text_field                     : the column holding raw text (``kind='plain'``)
  kind                           : 'plain'   (LM text; read ``text_field``)
                                   'scrolls' (ctx->query->answer, answer span
                                              up-weighted in the recall stream)
                                   'qa_context' (evidence-bearing QA example ->
                                              question + flattened evidence/
                                              distractor paragraphs as LM text;
                                              the retrieval-oriented LM bucket)
  trust_remote_code              : forwarded to ``load_dataset`` when True
  subsets                        : (scrolls only) the sub-configs to chain

The 4-bucket continued-pretraining mix maps onto these sources as:
  40% FineWeb-Edu        -> fineweb
  25% code / math        -> code (StarCoder) + math (OpenWebMath)   (12.5% each)
  20% structured QA/reas -> cosmopedia
  15% retrieval-oriented -> wikipedia + hotpotqa + musique (evidence text; the
                            same corpora Phase-2 retrieval adaptation trains on,
                            but here consumed as plain causal-LM text)
"""

from copy import deepcopy

DEFAULT_SCROLLS_SUBSETS = [
    "gov_report", "summ_screen_fd", "qasper",
    "narrative_qa", "quality", "contract_nli",
]

# Default HF coordinates for every known source. Override at the CLI (see
# pretokenize.py) for auth-gated paths or language/subset selection.
SOURCE_SPECS = {
    # --- general web / long-form (present in the original pipeline) ---
    "fineweb": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                    split="train", text_field="text", kind="plain"),
    "pg19": dict(path="pg19", name=None, split="train", text_field="text",
                 kind="plain", trust_remote_code=True),
    "scrolls": dict(path="tau/scrolls", name=None, split="train",
                    kind="scrolls", subsets=list(DEFAULT_SCROLLS_SUBSETS)),
    # --- code / math bucket ---
    "code": dict(path="bigcode/starcoderdata", name=None, split="train",
                 text_field="content", kind="plain", trust_remote_code=True,
                 data_dir=None),
    "math": dict(path="open-web-math/open-web-math", name=None, split="train",
                 text_field="text", kind="plain"),
    # --- structured QA / reasoning bucket ---
    "cosmopedia": dict(path="HuggingFaceTB/cosmopedia", name="web_samples_v2",
                       split="train", text_field="text", kind="plain"),
    # --- retrieval-oriented LM bucket (evidence-bearing corpora as plain LM) ---
    "wikipedia": dict(path="wikimedia/wikipedia", name="20231101.en",
                      split="train", text_field="text", kind="plain"),
    "hotpotqa": dict(path="hotpot_qa", name="distractor", split="train",
                     kind="qa_context", trust_remote_code=True),
    "musique": dict(path="dgslibisey/MuSiQue", name=None, split="train",
                    kind="qa_context"),
    # NQ full (`natural_questions`) is very heavy to stream and its evidence is
    # Wikipedia-derived (covered by `wikipedia`). Kept here for completeness /
    # explicit --sources use, but off the default continued-pretraining mix.
    "nq": dict(path="google-research-datasets/natural_questions", name="default",
               split="train", kind="qa_context", trust_remote_code=True),
}

# Sources whose examples are evidence-bearing QA records (``kind='qa_context'``);
# pretokenize.py flattens them to "question + evidence paragraphs" LM text.
QA_CONTEXT_SOURCES = ("hotpotqa", "musique", "nq")


def parse_sources(pairs):
    """``['fineweb=0.40', 'code=0.125']`` -> ``{'fineweb':0.40,'code':0.125}``.

    Order is preserved (round-robin interleave order in pretokenize follows it).
    Duplicate names accumulate. Unknown names / malformed entries raise.
    """
    mix = {}
    for entry in pairs:
        if "=" not in entry:
            raise ValueError(f"--sources entry {entry!r} must be name=frac")
        name, frac = entry.split("=", 1)
        name = name.strip()
        if name not in SOURCE_SPECS:
            raise ValueError(
                f"unknown source {name!r}; known: {sorted(SOURCE_SPECS)}")
        f = float(frac)
        if f < 0:
            raise ValueError(f"fraction for {name!r} must be >= 0 (got {f})")
        mix[name] = mix.get(name, 0.0) + f
    if not mix:
        raise ValueError("--sources given but empty")
    return mix


def normalize_mix(mix, tol=1e-3):
    """Return a copy whose positive fractions sum to exactly 1.0.

    Fractions are renormalized proportionally, so ``fineweb=2 code=1 math=1``
    is equivalent to ``fineweb=0.5 code=0.25 math=0.25``. Raises if nothing has
    positive weight. ``tol`` is only used by callers that want to warn on a
    large pre-normalization drift; the renorm itself is unconditional.
    """
    total = sum(v for v in mix.values() if v > 0)
    if total <= 0:
        raise ValueError("mix has no positive-weighted source")
    return {k: (v / total if v > 0 else 0.0) for k, v in mix.items()}


def resolve_specs(mix, overrides=None):
    """Deep-copy the ``SOURCE_SPECS`` entries for the sources in ``mix``.

    ``overrides`` is ``{source_name: {spec_key: value}}``; a value of ``None``
    leaves the default in place (so unset CLI flags don't clobber the registry).
    """
    overrides = overrides or {}
    out = {}
    for name in mix:
        spec = deepcopy(SOURCE_SPECS[name])
        for k, v in (overrides.get(name) or {}).items():
            if v is not None:
                spec[k] = v
        out[name] = spec
    return out


def interleave_quota(mix, chunk_tokens):
    """Fractional mix -> integer token quota per source per round-robin cycle.

    A source with weight exactly 0 gets quota 0 (not the ``max(1, ...)`` floor)
    so a pure single-source mix doesn't still pull a whole document from the
    zero-weighted sources every shard.
    """
    return {k: (max(1, int(round(v * chunk_tokens))) if v > 0 else 0)
            for k, v in mix.items()}
