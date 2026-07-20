"""Shared greedy-decode helper, used by ruler.py and babilong.py.

Uses the O(1) recurrent prefill/step interface when the passed-in `gen`
supports it (RecurrentKoopmanLM), otherwise falls back to a parallel
re-forward loop (works for any model type).
"""
import torch


def _greedy_generate(gen, tokenizer, device, prompt, max_new_tokens):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out_ids = []
    if hasattr(gen, "prefill") and hasattr(gen, "step"):
        logits = gen.prefill(ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out_ids.append(nxt.item())
        for _ in range(max_new_tokens - 1):
            logits = gen.step(nxt)
            logits = logits[:, -1] if logits.dim() == 3 else logits
            nxt = logits.argmax(-1, keepdim=True)
            out_ids.append(nxt.item())
    else:                                     # fallback: parallel re-forward
        cur = ids
        for _ in range(max_new_tokens):
            out = gen(input_ids=cur)
            logits = out["logits"] if isinstance(out, dict) else out
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out_ids.append(nxt.item())
            cur = torch.cat([cur, nxt], dim=1)
    return tokenizer.decode(out_ids)
