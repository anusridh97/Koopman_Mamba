"""_torch_data.py -- torch IterableDataset for contrastive retrieval pairs.

Kept separate from data.py so the pure extraction layer (data.py) imports no
torch/datasets and stays unit-testable. This wires the extractors to streaming
HF datasets, tokenizes to fixed short lengths, and yields stackable tensors.
"""

import random

import torch
from torch.utils.data import IterableDataset

from experimentation.retrieval.data import (
    RETRIEVAL_SOURCE_SPECS, extract_pairs, sample_source,
)


def _make_stream(spec, seed, buffer_size=10000):
    from datasets import load_dataset
    kw = dict(split=spec["split"], streaming=True)
    if spec.get("name"):
        kw["name"] = spec["name"]
    if spec.get("trust_remote_code"):
        kw["trust_remote_code"] = True
    ds = load_dataset(spec["path"], **kw)
    return iter(ds.shuffle(seed=seed, buffer_size=buffer_size))


def _cycle(spec, seed):
    """Infinite stream: re-create + reshuffle the HF stream when exhausted."""
    epoch = 0
    while True:
        it = _make_stream(spec, seed + epoch)
        for ex in it:
            yield ex
        epoch += 1


class ContrastivePairDataset(IterableDataset):
    def __init__(self, mix, tokenizer, q_len=64, p_len=256, n_hard=2,
                 seed=42, source_specs=None):
        """
        mix:        {source_name: fraction} over RETRIEVAL_SOURCE_SPECS keys.
        tokenizer:  HF tokenizer (pad_token must be set).
        q_len/p_len: fixed query / passage token lengths (no long contexts).
        n_hard:     hard negatives per query (padded by resampling to this many).
        """
        super().__init__()
        self.mix = {k: v for k, v in mix.items() if v > 0}
        assert self.mix, "empty retrieval mix"
        self.tok = tokenizer
        self.q_len, self.p_len, self.n_hard = q_len, p_len, n_hard
        self.seed = seed
        self.specs = dict(RETRIEVAL_SOURCE_SPECS)
        if source_specs:
            self.specs.update(source_specs)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
            else (tokenizer.eos_token_id or 0)

    def _tok(self, text, length):
        ids = self.tok(text or "", add_special_tokens=False,
                       truncation=True, max_length=length)["input_ids"]
        mask = [1] * len(ids)
        if len(ids) < length:                       # right-pad
            pad = length - len(ids)
            ids = ids + [self.pad_id] * pad
            mask = mask + [0] * pad
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random(self.seed + 1000 * wid)
        # per-source infinite cycles (distinct seeds per worker)
        streams = {name: _cycle(self.specs[name], self.seed + 7 * wid + i)
                   for i, name in enumerate(self.mix)}
        while True:
            name = sample_source(self.mix, rng)
            try:
                ex = next(streams[name])
            except StopIteration:
                continue
            pairs = extract_pairs(name, ex, rng)
            for pair in pairs:
                negs = list(pair.hard_negatives)
                if not negs:
                    continue                        # need >=1 for a fixed K tensor
                while len(negs) < self.n_hard:      # pad by resampling
                    negs.append(rng.choice(pair.hard_negatives))
                negs = negs[:self.n_hard]
                q_ids, q_mask = self._tok(pair.query, self.q_len)
                p_ids, p_mask = self._tok(pair.positive, self.p_len)
                n_ids, n_mask = zip(*(self._tok(t, self.p_len) for t in negs))
                yield {
                    "q_ids": q_ids, "q_mask": q_mask,
                    "p_ids": p_ids, "p_mask": p_mask,
                    "n_ids": torch.stack(n_ids),        # (K, p_len)
                    "n_mask": torch.stack(n_mask),      # (K, p_len)
                }
