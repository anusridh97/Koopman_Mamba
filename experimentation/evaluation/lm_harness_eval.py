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

import dataclasses
from koopman_lm.config import build_config
from koopman_lm.precision import as_dtype
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.models.baselines import (
    build_mamba_only,
    build_mamba_attention,
    build_mamba_ska_swiglu,
    build_mamba_ska_koopman,
)
from koopman_lm.modules.seq.attention import CausalAttentionBlock
from koopman_lm.models.recurrent import RecurrentKoopmanLM


MODEL_BUILDERS = {
    "koopman": KoopmanLM,
    "mamba_only": build_mamba_only,
    "mamba_attn": build_mamba_attention,
    "mamba_ska_swiglu": build_mamba_ska_swiglu,
    "mamba_ska_koopman": build_mamba_ska_koopman,
}


@register_model("koopman")
class KoopmanEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    MODEL_TYPE = "koopman"

    def __init__(
        self,
        checkpoint="./koopman-180m-output/final/model.pt",
        model_size="180m",
        tokenizer="mistralai/Mistral-7B-v0.1",
        max_length=2048,
        batch_size=None,
        device="cuda",
        # Serving precision: a per-invocation caller choice, not a config field
        # (precision design 5). It casts WEIGHTS -- the checkpoint declares
        # compute, the caller chooses storage, and naming this `dtype` invited
        # exactly the confusion between the two that the design dissolves. The
        # bf16 default is preserved deliberately: changing it would silently move
        # every lm-eval-harness number this repo has reported.
        weight_dtype="bf16",
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

        cfg = build_config(model_size)

        meta_path = checkpoint.replace("model.pt", "meta.pt")
        meta = {}
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if "cfg" in meta:
                cfg = meta["cfg"]
        checkpoint_type = meta.get("model_type")
        if checkpoint_type is not None and checkpoint_type != self.MODEL_TYPE:
            raise ValueError(
                f"checkpoint contains model_type={checkpoint_type!r}, but lm-eval "
                f"was invoked with --model {self.MODEL_TYPE!r}")

        # The tokenizer that actually produced this checkpoint's vocab is saved
        # alongside it by train.py's tokenizer.save_pretrained(ckpt_dir) -- load
        # THAT rather than the `tokenizer` arg's default, which differs from
        # train.py's own CLI default. Both are 32k-vocab, so a mismatch would
        # silently mismap every token id to the wrong embedding instead of
        # erroring.
        ckpt_dir = os.path.dirname(checkpoint)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = dataclasses.replace(cfg, vocab_size=len(self.tokenizer),
                                  max_seq_len=int(max_length))   # frozen: use replace

        try:
            builder = MODEL_BUILDERS[self.MODEL_TYPE]
        except KeyError as exc:
            raise ValueError(f"unsupported lm-eval model type {self.MODEL_TYPE!r}") from exc
        self._model = builder(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)

        self._device = torch.device(device)
        # as_dtype rather than getattr(torch, ...): the old form silently accepted
        # any torch attribute name, so a typo became an AttributeError deep in
        # .to() rather than a named rejection here.
        self._model = self._model.to(device=self._device,
                                     dtype=as_dtype(weight_dtype))
        self._model.eval()

        # Mamba/SKA variants have an O(1)-state recurrent implementation.
        # Attention blocks do not, so they use a correct full-prefix fallback.
        has_attention = any(isinstance(m, CausalAttentionBlock)
                            for m in self._model.modules())
        self._recurrent = None if has_attention else RecurrentKoopmanLM(self._model)

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
        """Greedy generation, recurrent when supported and full-prefix otherwise."""
        if context.shape[1] >= max_length:
            return context

        # Keep the end if a caller supplies a context beyond the configured window.
        if context.shape[1] > self._max_length:
            context = context[:, -self._max_length:]
        max_new = max_length - context.shape[1]

        eos_id = self.tokenizer.eos_token_id
        stop_ids = set()
        if eos_id is not None:
            stop_ids.add(eos_id)
        if stop is not None:
            for text in stop:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    stop_ids.add(ids[0])

        generated = context.clone()
        B = generated.shape[0]
        active = torch.ones(B, dtype=torch.bool, device=generated.device)
        if self._recurrent is not None:
            self._recurrent.reset()
            next_logits = self._recurrent.prefill(generated)[:, -1, :]
        else:
            next_logits = self._model(generated)["logits"][:, -1, :]

        with torch.no_grad():
            for _ in range(max_new):
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                pad_id = self.tokenizer.pad_token_id
                pad_id = 0 if pad_id is None else pad_id
                next_token = torch.where(
                    active.unsqueeze(1), next_token,
                    torch.full_like(next_token, pad_id))
                generated = torch.cat([generated, next_token], dim=1)

                new_tokens = next_token.squeeze(-1)
                for sid in stop_ids:
                    active &= (new_tokens != sid)
                if not active.any():
                    break

                if self._recurrent is not None:
                    next_logits = self._recurrent.step(next_token)[:, 0, :]
                else:
                    window = generated[:, -self._max_length:]
                    next_logits = self._model(window)["logits"][:, -1, :]

        return generated


@register_model("mamba_only")
class MambaOnlyEvalWrapper(KoopmanEvalWrapper):
    MODEL_TYPE = "mamba_only"


@register_model("mamba_attn")
class MambaAttentionEvalWrapper(KoopmanEvalWrapper):
    MODEL_TYPE = "mamba_attn"


@register_model("mamba_ska_swiglu")
class MambaSKASwiGLUEvalWrapper(KoopmanEvalWrapper):
    MODEL_TYPE = "mamba_ska_swiglu"


@register_model("mamba_ska_koopman")
class MambaSKAKoopmanEvalWrapper(KoopmanEvalWrapper):
    MODEL_TYPE = "mamba_ska_koopman"


if __name__ == "__main__":
    cli_evaluate()
