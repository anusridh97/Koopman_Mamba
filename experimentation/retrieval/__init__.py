"""Phase-2 retrieval adaptation: turn the pretrained Koopman LM into a dense
dual-encoder without long contexts.

Modules:
  encoder  -- RetrievalEncoder (backbone.encode -> projection head -> L2 norm)
              and the InfoNCE loss.
  data     -- contrastive (query, positive, hard_negatives) builders for
              HotpotQA / MuSiQue / self-supervised Wikipedia, mixed by fraction.
  adapt    -- the adaptation training loop (InfoNCE + LM anchor, 4:1 batch
              schedule, backbone/projection param groups) + Recall@k eval.

There is no launcher script and no RETRIEVAL.md; adapt.py's own docstring is
the reference, and it is run as `python -m experimentation.retrieval.adapt`.
"""
