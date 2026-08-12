"""
prefix_bench -- fair, prefix-mask-aware associative-recall benchmarks.

Modules:
  prefix_masks : the mask taxonomy (none/prefix/soft/causal, prefix_lm) + helpers.
  mqar_data    : MQAR with an in-sequence SEP delimiter.
  toolcall_data: long-context tool-call trace with overwrites (recency).
  models       : mask-aware Mamba / Mamba+Attn / Mamba+SKA.
  train_eval   : shared online-training harness.
  run_mqar     : MQAR mask-taxonomy sweep.
  run_toolcall : long-context tool-call sweep.
"""
