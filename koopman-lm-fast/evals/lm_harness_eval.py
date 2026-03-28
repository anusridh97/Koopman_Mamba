"""
lm_harness_eval.py -- lm-evaluation-harness wrapper for Koopman LM.

Follows the Mamba eval pattern:
  https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py

Supports both loglikelihood-based tasks (HellaSwag, PIQA, ARC, etc.)
and generation-based tasks (LAMBADA openai, gsm8k, etc.) via the
O(1) recurrent generation wrapper.

Usage:
  python evals/lm_harness_eval.py \
      --model koopman \
      --model_args checkpoint=./koopman-180m-output/step_5000/model.pt,model_size=180m,tokenizer=mistralai/Mistral-7B-v0.1,max_length=2048 \
      --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
      --batch_size 16 \
      --device cuda
"""

import torch
import transformers
from transformers import AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from koopman_lm.config import config_180m, config_180m_gated, config_370m
from koopman_lm.model import KoopmanLM
from koopman_lm.recurrent import RecurrentKoopmanLM


@register_model("koopman")
class KoopmanEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        checkpoint="./koopman-180m-output/final/model.pt",
        model_size="180m",
        tokenizer="mistralai/Mistral-7B-v0.1",
        max_length=2048,
        batch_size=None,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        # Skip HFLM.__init__ (it tries to load a HF model), but we need
        # LM.__init__ for the base harness plumbing.
        LM.__init__(self)

        # ------------------------------------------------------------------
        # Attributes that HFLM.__init__ normally sets and that lm-eval
        # internals (tok_encode, loglikelihood_rolling, _model_call, etc.)
        # access directly. Missing any of these causes AttributeError.
        # ------------------------------------------------------------------
        self.add_bos_token = False
        self.custom_prefix_token_id = None
        self.logits_cache = True
        self.truncation = False          # don't silently truncate inside tok_encode
        self._rank = 0
        self._world_size = 1
        self.mixed_precision_dtype = None
        self.softmax_dtype = None
        self.revision = "N/A"
        self.pretrained = checkpoint
        self.delta = None
        self.peft = None
        self.backend = "causal"
        # ------------------------------------------------------------------

        if model_size == "180m":
            cfg = config_180m()
        elif model_size == "180m_gated":
            cfg = config_180m_gated()
        elif model_size == "370m":
            cfg = config_370m()
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        meta_path = checkpoint.replace("model.pt", "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if "cfg" in meta:
                cfg = meta["cfg"]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg.vocab_size = len(self.tokenizer)
        cfg.max_seq_len = int(max_length)

        self._model = KoopmanLM(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)

        self._device = torch.device(device)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self._model = self._model.to(device=self._device, dtype=dtype)
        self._model.eval()

        # Build recurrent wrapper for generation
        self._recurrent = RecurrentKoopmanLM(self._model)

        self.vocab_size = len(self.tokenizer)
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = int(max_length)
        self._dtype = dtype

        self._model.param_summary()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    def _model_call(self, inps, attn_mask=None, labels=None):
        """Override HFLM._model_call: KoopmanLM.forward returns a dict, not a
        namedtuple with .logits."""
        with torch.no_grad():
            out = self._model(inps)
            return out["logits"]

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Autoregressive generation using the O(1) recurrent wrapper.
        """
        # Guard: if context already exceeds max_length, return as-is
        if context.shape[1] >= max_length:
            return context

        max_new = max_length - context.shape[1]

        # Build stop token set
        eos_id = self.tokenizer.eos_token_id
        stop_ids = set()
        if eos_id is not None:
            stop_ids.add(eos_id)
        if stop is not None:
            for s in stop:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    stop_ids.add(ids[0])

        # Truncate context to max_length if needed (keep the END, not beginning)
        if context.shape[1] > self._max_length:
            context = context[:, -self._max_length:]

        # Prefill prompt (parallel forward)
        self._recurrent.reset()
        logits = self._recurrent.prefill(context)

        B = context.shape[0]
        device = context.device
        generated = context.clone()
        active = torch.ones(B, dtype=torch.bool, device=device)

        next_logits = logits[:, -1, :]

        with torch.no_grad():
            for _ in range(max_new):
                # Greedy decode
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

                # Mask finished sequences: replace their token with pad
                pad_id = self.tokenizer.pad_token_id or 0
                next_token = torch.where(
                    active.unsqueeze(1), next_token,
                    torch.full_like(next_token, pad_id))

                generated = torch.cat([generated, next_token], dim=1)

                # Check stop
                new_tokens = next_token.squeeze(-1)
                for sid in stop_ids:
                    active &= (new_tokens != sid)
                if not active.any():
                    break

                # O(1) step
                step_logits = self._recurrent.step(next_token)
                next_logits = step_logits[:, 0, :]

        return generated


if __name__ == "__main__":
    cli_evaluate()
