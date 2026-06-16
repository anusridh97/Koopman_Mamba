"""Evaluation task builders + scorers used by eval_harness.py.

Each module exposes pure generator functions (CPU-testable, no model) and a
scorer that runs a model. The harness wires them into one JSON output.
"""
from koopman_lm.evals.mqar import make_mqar, eval_mqar, eval_mqar_grid
from koopman_lm.evals.ruler import (
    build_multikey_niah,
    build_variable_tracking,
    build_common_word_extraction,
    eval_ruler_subset,
)
from koopman_lm.evals.babilong import eval_babilong_subset

__all__ = [
    "make_mqar",
    "eval_mqar",
    "eval_mqar_grid",
    "build_multikey_niah",
    "build_variable_tracking",
    "build_common_word_extraction",
    "eval_ruler_subset",
    "eval_babilong_subset",
]
