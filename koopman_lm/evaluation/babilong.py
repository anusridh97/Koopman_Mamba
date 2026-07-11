"""BABILong subset (Kuratov et al. 2024): QA1 (single supporting fact) and QA2
(two supporting facts) at 4K and 8K context.

Loads the HF dataset lazily (so importing this module needs no network / datasets
install). Data-dependent -> runs on a GPU box with internet, not in CPU CI.
"""
import torch


_HF_NAME = "RMT-team/babilong"
# BABILong ships per-length configs; map a token budget to the dataset split key.
_LEN_TO_SPLIT = {4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k"}


@torch.no_grad()
def eval_babilong_subset(model, tokenizer, device, context_lens=(4096, 8192),
                         tasks=("qa1", "qa2"), n_examples=20, max_new_tokens=8,
                         seed=0):
    """Return {task: {ctx_len: accuracy}}. Accuracy = fraction of examples whose
    greedy continuation contains the gold answer (case-insensitive).

    Raises a clear error if `datasets` isn't installed or the data can't be
    fetched (this eval needs network + the HF datasets package).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "BABILong eval needs the `datasets` package: pip install datasets"
        ) from e
    from koopman_lm.evaluation.ruler import _greedy_generate
    from koopman_lm.globals.modules.recurrent import RecurrentKoopmanLM

    gen = model
    if hasattr(model, "cfg"):
        try:
            gen = RecurrentKoopmanLM(model)
        except Exception:
            gen = model

    results = {t: {} for t in tasks}
    for task in tasks:
        for ctx in context_lens:
            split = _LEN_TO_SPLIT.get(ctx)
            if split is None:
                continue
            try:
                ds = load_dataset(_HF_NAME, split, split=task)
            except Exception as e:
                raise RuntimeError(
                    f"could not load {_HF_NAME} config={split} split={task}: {e}"
                ) from e
            n = min(n_examples, len(ds))
            hits = 0
            for i in range(n):
                ex = ds[i]
                prompt = f"{ex['input']}\nQuestion: {ex['question']}\nAnswer:"
                text = _greedy_generate(gen, tokenizer, device, prompt, max_new_tokens)
                hits += int(str(ex["target"]).strip().lower() in text.lower())
            results[task][ctx] = hits / max(n, 1)
    return results
