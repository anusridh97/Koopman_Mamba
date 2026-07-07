"""
Optuna baseline HPO at ~1M parameter scale.

This script runs small, aggressively pruned sweeps over three baseline families:

  - transformer: pure Hugging Face GPT-style causal Transformer
  - mamba_only:  all Mamba-2 sequence layers
  - mamba_attn:  Mamba-2 + causal attention hybrid

The default task trains on a mixed WikiText + MQAR objective and evaluates both
WikiText perplexity and MQAR accuracy. At small scale, the parameter target is
applied to non-embedding parameters by default because real LM vocabularies make
token embeddings larger than 1M parameters by themselves.

Outputs:
  - SQLite Optuna study
  - trials.csv with all trial params/metrics
  - pareto_front.csv from final MQAR accuracy vs WikiText PPL
  - plots/*.html and, when dependencies allow, plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import optuna
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from koopman_lm.config import KoopmanLMConfig, _evenly_spaced_indices
from koopman_lm.baselines import (
    build_mamba_attention,
    build_mamba_only,
    build_transformer,
)


MODEL_TYPES = ("transformer", "mamba_only", "mamba_attn")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention_indices(n_layers: int, fraction: float, placement: str) -> list[int]:
    n_special = max(1, round(n_layers * fraction))
    if placement == "uniform":
        return _evenly_spaced_indices(n_layers, n_special)
    if placement == "early":
        return list(range(n_special))
    if placement == "late":
        return list(range(n_layers - n_special, n_layers))
    if placement == "middle":
        center = (n_layers - 1) / 2
        ordered = sorted(range(n_layers), key=lambda i: (abs(i - center), i))
        return sorted(ordered[:n_special])
    raise ValueError(f"Unknown placement: {placement}")


def valid_num_heads(d_model: int, head_dim: int) -> int:
    if d_model % head_dim != 0:
        raise optuna.TrialPruned(f"d_model={d_model} not divisible by head_dim={head_dim}")
    return d_model // head_dim


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_non_embedding_params(model: torch.nn.Module) -> int:
    total = 0
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        lowered = name.lower()
        if any(key in lowered for key in ("embed", "wte", "lm_head")):
            continue
        total += param.numel()
    return total


def mamba_available() -> bool:
    return importlib.util.find_spec("mamba_ssm") is not None


def available_model_types(requested: list[str], require_all: bool) -> list[str]:
    requested = list(dict.fromkeys(requested))
    missing = []
    if not mamba_available():
        missing = [m for m in requested if m in {"mamba_only", "mamba_attn"}]

    if missing and require_all:
        raise RuntimeError(
            "Requested Mamba baselines but mamba_ssm is not installed. "
            "Install mamba_ssm in the GPU environment or rerun with "
            "--model_types transformer for a local smoke test."
        )

    selected = [m for m in requested if m not in missing]
    if missing:
        print(
            "WARNING: skipping unavailable model types "
            f"{missing}; mamba_ssm is not installed in this environment."
        )
    if not selected:
        raise RuntimeError("No runnable model types selected.")
    return selected


def build_trial_config(trial: optuna.Trial, args: argparse.Namespace) -> tuple[str, KoopmanLMConfig, dict]:
    model_type = trial.suggest_categorical("model_type", args.model_types)

    # Broad for tiny scale, while still respecting tensor-core/config constraints.
    d_model = trial.suggest_categorical("d_model", [64, 128, 192, 256, 384])
    n_layers = trial.suggest_categorical("n_layers", [2, 4, 6, 8, 12])
    head_dim = trial.suggest_categorical("head_dim", [32, 64])
    n_heads = valid_num_heads(d_model, head_dim)

    mlp_expand = trial.suggest_categorical("mlp_expand", [2.0, 2.667, 3.0, 4.0])

    # Mamba-specific knobs are still sampled for transformer trials as None in
    # user attrs, which keeps feature importance easier to interpret by family.
    d_state = None
    d_conv = None
    mamba_expand = None
    if model_type in {"mamba_only", "mamba_attn"}:
        d_state = trial.suggest_categorical("d_state", [16, 32, 64, 128])
        d_conv = trial.suggest_categorical("d_conv", [2, 3, 4])
        mamba_expand = trial.suggest_categorical("mamba_expand", [1, 2, 3, 4])
    else:
        d_state = 32
        d_conv = 4
        mamba_expand = 2

    attention_fraction = 0.0
    attention_placement = "none"
    if model_type == "mamba_attn":
        attention_fraction = trial.suggest_categorical(
            "attention_fraction", [0.125, 0.25, 0.333, 0.5]
        )
        attention_placement = trial.suggest_categorical(
            "attention_placement", ["uniform", "early", "middle", "late"]
        )
        special_layers = attention_indices(n_layers, attention_fraction, attention_placement)
    else:
        special_layers = []

    cfg = KoopmanLMConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=args.vocab_size,
        d_state=d_state,
        d_conv=d_conv,
        mamba_expand=mamba_expand,
        ska_n_heads=n_heads,
        ska_rank=32,
        ska_layer_indices=special_layers,
        mlp_expand=mlp_expand,
        max_seq_len=args.seq_len,
        tie_embeddings=True,
    )

    family_params = {
        "head_dim": head_dim,
        "attention_fraction": attention_fraction,
        "attention_placement": attention_placement,
        "n_special_layers": len(special_layers),
    }
    return model_type, cfg, family_params


def build_model(model_type: str, cfg: KoopmanLMConfig, dropout: float) -> torch.nn.Module:
    if model_type == "transformer":
        return build_transformer(cfg, dropout=dropout)
    if model_type == "mamba_only":
        return build_mamba_only(cfg)
    if model_type == "mamba_attn":
        return build_mamba_attention(cfg)
    raise ValueError(f"Unknown model_type: {model_type}")


class MQARBatcher:
    """
    Generates associative recall batches.

    A sequence contains key/value pairs followed by query_marker + queried_key.
    The training/eval target is the value associated with the queried key.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_pairs: int,
        batch_size: int,
        device: torch.device,
        seed: int,
    ):
        min_vocab = 2 + 2 * n_pairs + 32
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size={vocab_size} too small for n_pairs={n_pairs}; "
                f"need at least {min_vocab}"
            )
        if seq_len < 2 * n_pairs + 2:
            raise ValueError(
                f"seq_len={seq_len} too short for n_pairs={n_pairs}; "
                f"need at least {2 * n_pairs + 2}"
            )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_pairs = n_pairs
        self.batch_size = batch_size
        self.device = device
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)

        self.pad_token = 0
        self.query_token = 1
        self.key_start = 2
        self.value_start = 2 + n_pairs + 16

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        y = torch.zeros((self.batch_size,), dtype=torch.long)

        for b in range(self.batch_size):
            keys = torch.randperm(self.n_pairs, generator=self.rng) + self.key_start
            values = torch.randperm(self.n_pairs, generator=self.rng) + self.value_start
            query_idx = int(torch.randint(0, self.n_pairs, (1,), generator=self.rng))

            pos = 0
            for k, v in zip(keys, values):
                x[b, pos] = k
                x[b, pos + 1] = v
                pos += 2

            # Fill the gap with harmless distractors so sequence length matters.
            while pos < self.seq_len - 2:
                x[b, pos] = int(torch.randint(
                    self.value_start + self.n_pairs,
                    self.vocab_size,
                    (1,),
                    generator=self.rng,
                ))
                pos += 1

            x[b, self.seq_len - 2] = self.query_token
            x[b, self.seq_len - 1] = keys[query_idx]
            y[b] = values[query_idx]

        return x.to(self.device), y.to(self.device)


class TokenBatcher:
    """Random fixed-length next-token batches from a token tensor."""

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seed: int,
        random_sample: bool = True,
    ):
        if tokens.numel() < seq_len + 2:
            raise ValueError(
                f"Need at least {seq_len + 2} tokens, got {tokens.numel()}"
            )
        self.tokens = tokens.cpu().long()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.random_sample = random_sample
        self.position = 0
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - self.seq_len - 1
        if self.random_sample:
            starts = torch.randint(
                0, max_start + 1, (self.batch_size,), generator=self.rng
            )
        else:
            starts = torch.arange(
                self.position,
                self.position + self.batch_size * self.seq_len,
                self.seq_len,
            ) % max_start
            self.position = int((self.position + self.batch_size * self.seq_len) % max_start)

        x = torch.stack([self.tokens[s:s + self.seq_len] for s in starts])
        y = torch.stack([self.tokens[s + 1:s + self.seq_len + 1] for s in starts])
        return x.to(self.device), y.to(self.device)


def tokenize_wikitext(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_split(split: str, max_tokens: int) -> torch.Tensor:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=split)
        token_ids: list[int] = []
        eos = tokenizer.eos_token_id
        for example in ds:
            text = example.get("text", "")
            if not text or not text.strip():
                continue
            token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if eos is not None:
                token_ids.append(eos)
            if max_tokens and len(token_ids) >= max_tokens:
                token_ids = token_ids[:max_tokens]
                break
        return torch.tensor(token_ids, dtype=torch.long)

    train_tokens = encode_split("train", args.max_train_tokens)
    val_tokens = encode_split("validation", args.max_eval_tokens)
    return train_tokens, val_tokens, len(tokenizer)


@torch.no_grad()
def evaluate_wikitext_ppl(
    model: torch.nn.Module,
    batcher: TokenBatcher,
    n_batches: int,
    amp_enabled: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    for _ in range(n_batches):
        input_ids, labels = batcher.next()
        with autocast_ctx:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
        total_loss += float(loss.detach().cpu())

    avg_loss = total_loss / max(n_batches, 1)
    return math.exp(min(avg_loss, 20.0))


@torch.no_grad()
def evaluate_mqar(
    model: torch.nn.Module,
    batcher: MQARBatcher,
    n_batches: int,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    for _ in range(n_batches):
        input_ids, targets = batcher.next()
        with autocast_ctx:
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"][:, -1, :]
            loss = F.cross_entropy(logits, targets)
        total_loss += float(loss.detach().cpu())
        total_correct += int((logits.argmax(dim=-1) == targets).sum().detach().cpu())
        total += targets.numel()

    avg_loss = total_loss / max(n_batches, 1)
    acc = total_correct / max(total, 1)
    return acc, math.exp(min(avg_loss, 20.0))


def train_trial(
    trial: optuna.Trial,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, float]:
    lr = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0])
    dropout = trial.params.get("transformer_dropout", 0.0)
    _ = dropout  # recorded through params; model already built with it

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    warmup_steps = max(1, int(args.steps * warmup_ratio))
    amp_enabled = args.bf16 and device.type == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    train_batcher = TokenBatcher(
        args.train_tokens, args.seq_len, args.batch_size,
        device, args.seed + trial.number * 1009, random_sample=True,
    )
    val_batcher = TokenBatcher(
        args.val_tokens, args.seq_len, args.eval_batch_size,
        device, args.seed + 5557, random_sample=False,
    )
    train_mqar_batcher = MQARBatcher(
        args.vocab_size, args.seq_len, args.n_pairs, args.batch_size,
        device, args.seed + trial.number * 2003,
    )
    eval_batcher = MQARBatcher(
        args.vocab_size, args.seq_len, args.n_pairs, args.eval_batch_size,
        device, args.seed + 99991,
    )

    model.train()
    last_loss = float("inf")
    last_lm_loss = float("inf")
    last_mqar_loss = float("inf")
    for step in range(1, args.steps + 1):
        if step <= warmup_steps:
            step_lr = lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
            step_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        input_ids, labels = train_batcher.next()
        mqar_input_ids, mqar_targets = train_mqar_batcher.next()
        with autocast_ctx:
            outputs = model(input_ids=input_ids, labels=labels)
            lm_loss = outputs["loss"]
            mqar_outputs = model(input_ids=mqar_input_ids)
            mqar_logits = mqar_outputs["logits"][:, -1, :]
            mqar_loss = F.cross_entropy(mqar_logits, mqar_targets)
            loss = lm_loss + args.mqar_train_weight * mqar_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        last_lm_loss = float(lm_loss.detach().cpu())
        last_mqar_loss = float(mqar_loss.detach().cpu())

        if step % args.eval_every == 0 or step == args.steps:
            wikitext_ppl = evaluate_wikitext_ppl(
                model, val_batcher, args.eval_batches, amp_enabled
            )
            mqar_acc, proxy_ppl = evaluate_mqar(
                model, eval_batcher, args.eval_batches, amp_enabled
            )
            prune_metric = mqar_acc if args.prune_metric == "mqar" else -math.log(wikitext_ppl)
            trial.report(prune_metric, step=step)
            trial.set_user_attr(f"train_lm_loss_step_{step}", last_lm_loss)
            trial.set_user_attr(f"train_mqar_loss_step_{step}", last_mqar_loss)
            trial.set_user_attr(f"train_total_loss_step_{step}", last_loss)
            trial.set_user_attr(f"mqar_acc_step_{step}", mqar_acc)
            trial.set_user_attr(f"wikitext_ppl_step_{step}", wikitext_ppl)
            trial.set_user_attr(f"mqar_proxy_ppl_step_{step}", proxy_ppl)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at step {step}: MQAR acc={mqar_acc:.4f}, "
                    f"WikiText PPL={wikitext_ppl:.2f}"
                )
            model.train()

    final_wikitext_ppl = evaluate_wikitext_ppl(
        model, val_batcher, args.final_eval_batches, amp_enabled
    )
    final_acc, final_ppl = evaluate_mqar(
        model, eval_batcher, args.final_eval_batches, amp_enabled
    )
    return final_acc, final_wikitext_ppl, final_ppl, last_loss


def objective(args: argparse.Namespace, device: torch.device):
    def _objective(trial: optuna.Trial) -> float:
        set_seed(args.seed + trial.number)
        model_type, cfg, family_params = build_trial_config(trial, args)

        dropout = 0.0
        if model_type == "transformer":
            dropout = trial.suggest_categorical("transformer_dropout", [0.0, 0.05, 0.1])

        model = build_model(model_type, cfg, dropout=dropout).to(device)
        n_params = count_params(model)
        n_non_embedding_params = count_non_embedding_params(model)
        target_count = n_non_embedding_params if args.param_target == "non_embedding" else n_params
        lower = max(1, int(args.target_params * (1.0 - args.param_tolerance)))
        upper = int(args.target_params * (1.0 + args.param_tolerance))
        if not lower <= target_count <= upper:
            raise optuna.TrialPruned(
                f"{args.param_target}_params={target_count:,} outside target "
                f"window [{lower:,}, {upper:,}]"
            )

        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("n_non_embedding_params", n_non_embedding_params)
        trial.set_user_attr("cfg", asdict(cfg))
        for key, value in family_params.items():
            trial.set_user_attr(key, value)

        final_acc, final_wikitext_ppl, final_mqar_proxy_ppl, last_train_loss = train_trial(
            trial, model, args, device)

        trial.set_user_attr("final_mqar_acc", final_acc)
        trial.set_user_attr("final_wikitext_ppl", final_wikitext_ppl)
        trial.set_user_attr("final_mqar_proxy_ppl", final_mqar_proxy_ppl)
        trial.set_user_attr("last_train_loss", last_train_loss)

        # Scalar objective for pruning compatibility. Pareto analysis is produced
        # after the study from final_mqar_acc and final_wikitext_ppl.
        score = final_acc - args.ppl_penalty * math.log(final_wikitext_ppl)
        return score

    return _objective


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def completed_trials_below_ppl(
    study: optuna.Study,
    ppl_max: float | None,
) -> list[optuna.trial.FrozenTrial]:
    trials = completed_trials(study)
    if ppl_max is None:
        return trials
    return [
        t for t in trials
        if t.user_attrs.get("final_wikitext_ppl") is not None
        and float(t.user_attrs["final_wikitext_ppl"]) <= ppl_max
    ]


def is_pareto_efficient(trial: optuna.trial.FrozenTrial, trials: Iterable[optuna.trial.FrozenTrial]) -> bool:
    acc = trial.user_attrs.get("final_mqar_acc")
    ppl = trial.user_attrs.get("final_wikitext_ppl")
    if acc is None or ppl is None:
        return False
    for other in trials:
        if other.number == trial.number:
            continue
        other_acc = other.user_attrs.get("final_mqar_acc")
        other_ppl = other.user_attrs.get("final_wikitext_ppl")
        if other_acc is None or other_ppl is None:
            continue
        if other_acc >= acc and other_ppl <= ppl and (other_acc > acc or other_ppl < ppl):
            return False
    return True


def write_trials_csv(study: optuna.Study, output_dir: Path) -> None:
    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": t.state.name,
            "score": t.value,
            **t.params,
            **t.user_attrs,
        }
        if "cfg" in row:
            row["cfg"] = json.dumps(row["cfg"], sort_keys=True)
        rows.append(row)

    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_pareto_csv(
    study: optuna.Study,
    output_dir: Path,
    ppl_max: float | None = None,
    filename: str = "pareto_front.csv",
) -> None:
    trials = completed_trials_below_ppl(study, ppl_max)
    pareto = [t for t in trials if is_pareto_efficient(t, trials)]
    pareto = sorted(
        pareto,
        key=lambda t: (
            -t.user_attrs.get("final_mqar_acc", -1.0),
            t.user_attrs.get("final_wikitext_ppl", float("inf")),
        ),
    )

    keys = [
        "number", "score", "model_type", "n_params", "final_mqar_acc",
        "n_non_embedding_params", "final_wikitext_ppl", "final_mqar_proxy_ppl",
        "d_model", "n_layers", "head_dim", "mlp_expand",
        "d_state", "d_conv", "mamba_expand", "attention_fraction",
        "attention_placement", "learning_rate", "weight_decay", "warmup_ratio",
        "max_grad_norm", "transformer_dropout",
    ]
    with (output_dir / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for t in pareto:
            row = {
                "number": t.number,
                "score": t.value,
                "n_params": t.user_attrs.get("n_params"),
                "n_non_embedding_params": t.user_attrs.get("n_non_embedding_params"),
                "final_mqar_acc": t.user_attrs.get("final_mqar_acc"),
                "final_wikitext_ppl": t.user_attrs.get("final_wikitext_ppl"),
                "final_mqar_proxy_ppl": t.user_attrs.get("final_mqar_proxy_ppl"),
            }
            row.update(t.params)
            writer.writerow({key: row.get(key) for key in keys})


def write_filtered_trials_csv(study: optuna.Study, output_dir: Path, ppl_max: float) -> None:
    rows = []
    for t in completed_trials_below_ppl(study, ppl_max):
        row = {
            "number": t.number,
            "state": t.state.name,
            "score": t.value,
            **t.params,
            **t.user_attrs,
        }
        if "cfg" in row:
            row["cfg"] = json.dumps(row["cfg"], sort_keys=True)
        rows.append(row)

    if not rows:
        return

    keys = sorted({key for row in rows for key in row.keys()})
    path = output_dir / f"trials_completed_ppl_le_{int(ppl_max)}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plotly(fig, path: Path) -> None:
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")))
    except Exception:
        pass


def write_plots(study: optuna.Study, output_dir: Path, plot_ppl_max: float | None) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
    except Exception as exc:
        (plots_dir / "plot_error.txt").write_text(f"Optuna visualization unavailable: {exc}\n")
        return

    plot_specs = [
        ("optimization_history", plot_optimization_history),
        ("param_importances_fanova", plot_param_importances),
        ("parallel_coordinate", plot_parallel_coordinate),
        ("slice", plot_slice),
        ("contour", plot_contour),
    ]
    for name, fn in plot_specs:
        try:
            fig = fn(study)
            save_plotly(fig, plots_dir / name)
        except Exception as exc:
            (plots_dir / f"{name}_error.txt").write_text(str(exc) + "\n")

    try:
        import plotly.express as px

        def build_rows(trials: list[optuna.trial.FrozenTrial]) -> list[dict]:
            rows = []
            for t in trials:
                rows.append({
                    "trial": t.number,
                    "score": t.value,
                    "model_type": t.params.get("model_type"),
                    "n_params": t.user_attrs.get("n_params"),
                    "n_non_embedding_params": t.user_attrs.get("n_non_embedding_params"),
                    "final_mqar_acc": t.user_attrs.get("final_mqar_acc"),
                    "final_wikitext_ppl": t.user_attrs.get("final_wikitext_ppl"),
                })
            return rows

        rows = build_rows(completed_trials(study))
        if rows:
            fig = px.scatter(
                rows,
                x="final_wikitext_ppl",
                y="final_mqar_acc",
                color="model_type",
                size="n_non_embedding_params",
                hover_data=["trial", "score", "n_params", "n_non_embedding_params"],
                title="Pareto view: MQAR accuracy vs WikiText PPL",
            )
            fig.update_xaxes(title="WikiText PPL (lower is better)")
            fig.update_yaxes(title="MQAR accuracy (higher is better)")
            save_plotly(fig, plots_dir / "pareto_mqar_vs_ppl")

        filtered_rows = build_rows(completed_trials_below_ppl(study, plot_ppl_max))
        if plot_ppl_max is not None and filtered_rows:
            fig = px.scatter(
                filtered_rows,
                x="final_wikitext_ppl",
                y="final_mqar_acc",
                color="model_type",
                size="n_non_embedding_params",
                hover_data=["trial", "score", "n_params", "n_non_embedding_params"],
                title=(
                    "Pareto view: MQAR accuracy vs WikiText PPL "
                    f"(PPL <= {plot_ppl_max:g})"
                ),
            )
            fig.update_xaxes(title="WikiText PPL (lower is better)")
            fig.update_yaxes(title="MQAR accuracy (higher is better)")
            save_plotly(
                fig,
                plots_dir / f"pareto_mqar_vs_ppl_filtered_ppl_le_{int(plot_ppl_max)}",
            )
    except Exception as exc:
        (plots_dir / "pareto_plot_error.txt").write_text(str(exc) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna HPO for 1M-scale baselines")
    p.add_argument("--study_name", type=str, default="baseline_1m_hpo")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--output_dir", type=Path, default=Path("optuna_baseline_1m"))
    p.add_argument("--n_trials", type=int, default=60)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_types", nargs="+", default=list(MODEL_TYPES),
                   choices=list(MODEL_TYPES))
    p.add_argument("--require_all_model_types", action="store_true")

    p.add_argument("--target_params", type=float, default=1_000_000)
    p.add_argument("--param_tolerance", type=float, default=0.65)
    p.add_argument("--param_target", type=str, default="non_embedding",
                   choices=["non_embedding", "total"])
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--max_train_tokens", type=int, default=1_000_000)
    p.add_argument("--max_eval_tokens", type=int, default=200_000)
    p.add_argument("--vocab_size", type=int, default=None,
                   help="Set automatically from tokenizer; override only for debugging.")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_pairs", type=int, default=16)

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--eval_batches", type=int, default=4)
    p.add_argument("--final_eval_batches", type=int, default=16)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--ppl_penalty", type=float, default=0.05)
    p.add_argument("--mqar_train_weight", type=float, default=1.0)
    p.add_argument("--plot_ppl_max", type=float, default=5000.0,
                   help="Also write report-friendly Pareto/trial outputs excluding completed trials above this WikiText PPL.")
    p.add_argument("--prune_metric", type=str, default="mqar",
                   choices=["mqar", "wikitext"])

    p.add_argument("--pruner_startup_trials", type=int, default=10)
    p.add_argument("--pruner_warmup_steps", type=int, default=100)
    p.add_argument("--pruner_interval_steps", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_types = available_model_types(
        args.model_types, args.require_all_model_types)
    train_tokens, val_tokens, tokenizer_vocab_size = tokenize_wikitext(args)
    args.train_tokens = train_tokens
    args.val_tokens = val_tokens
    if args.vocab_size is None:
        args.vocab_size = tokenizer_vocab_size

    if args.storage is None:
        args.storage = f"sqlite:///{args.output_dir / 'study.db'}"

    set_seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")
    print(f"Study storage: {args.storage}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model types: {args.model_types}")
    print(
        f"Dataset: {args.dataset_name}/{args.dataset_config}, "
        f"tokenizer={args.tokenizer}, vocab_size={args.vocab_size:,}"
    )
    print(
        f"Loaded tokens: train={args.train_tokens.numel():,}, "
        f"validation={args.val_tokens.numel():,}"
    )
    print(f"Parameter target: {args.param_target}")
    lower = max(1, args.target_params * (1 - args.param_tolerance))
    upper = args.target_params * (1 + args.param_tolerance)
    print(
        "Parameter target window: "
        f"{lower:,.0f} - {upper:,.0f}"
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_steps,
        interval_steps=args.pruner_interval_steps,
    )
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective(args, device), n_trials=args.n_trials, timeout=args.timeout)

    write_trials_csv(study, args.output_dir)
    write_pareto_csv(study, args.output_dir)
    if args.plot_ppl_max is not None:
        write_filtered_trials_csv(study, args.output_dir, args.plot_ppl_max)
        write_pareto_csv(
            study,
            args.output_dir,
            ppl_max=args.plot_ppl_max,
            filename=f"pareto_front_ppl_le_{int(args.plot_ppl_max)}.csv",
        )
    write_plots(study, args.output_dir, args.plot_ppl_max)

    if completed_trials(study):
        print("\nBest scalar objective trial:")
        print(f"  number: {study.best_trial.number}")
        print(f"  score:  {study.best_value:.6f}")
        print(f"  params: {study.best_trial.params}")
        print(f"  attrs:  {study.best_trial.user_attrs}")
    else:
        print("\nNo completed trials. Check pruned/failed trial messages.")
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
