"""data.py -- contrastive (query, positive, hard-negatives) data for Phase-2
retrieval adaptation.

Two layers:

  * Pure-Python extraction (no torch / datasets): ``hotpot_to_pair``,
    ``musique_to_pair``, ``wiki_to_pairs`` turn one raw dataset record into
    ``ContrastivePair(query, positive, hard_negatives)`` text triples. These are
    unit-testable on toy dicts.

  * ``ContrastivePairDataset`` (torch IterableDataset): mixes the sources by
    fraction, tokenizes to fixed short lengths (query 64 / passage 256 -- NO long
    concatenated contexts, per the plan), right-pads, and yields tensors ready to
    stack with the default collate.

Positives / hard negatives come straight from the datasets' structure:
  HotpotQA (distractor): supporting paragraph = positive; the distractor
      paragraphs (same question, wrong) = same-topic hard negatives.
  MuSiQue: is_supporting paragraph = positive; the rest = hard negatives
      (compositional multi-hop, so the distractors are genuinely hard).
  Wikipedia (self-supervised): title->section, section->nearby-section,
      first-sentence->paragraph, heading->subsection; other sections of the same
      article are hard negatives (same topic, different content).

In-batch negatives (every other query's positive) are added by the InfoNCE loss;
this module only supplies the explicit per-query hard negatives.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ContrastivePair:
    query: str
    positive: str
    hard_negatives: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure-Python extraction (torch-free, unit-testable)
# ---------------------------------------------------------------------------

def _hotpot_paragraphs(ex):
    """-> (list[(title, paragraph_text)], set[supporting_title])."""
    ctx = ex.get("context") or {}
    titles = ctx.get("title") or []
    sents = ctx.get("sentences") or []
    paras = [(t, " ".join(ss or [])) for t, ss in zip(titles, sents)]
    sup = set((ex.get("supporting_facts") or {}).get("title") or [])
    return paras, sup


def hotpot_to_pair(ex, max_negs=8):
    """HotpotQA distractor record -> ContrastivePair, or None if unusable."""
    q = (ex.get("question") or "").strip()
    if not q:
        return None
    paras, sup = _hotpot_paragraphs(ex)
    pos = [p for (t, p) in paras if t in sup and p.strip()]
    neg = [p for (t, p) in paras if t not in sup and p.strip()]
    if not pos:
        return None
    return ContrastivePair(q, pos[0], neg[:max_negs])


def musique_to_pair(ex, max_negs=8):
    """MuSiQue record -> ContrastivePair, or None if unusable."""
    q = (ex.get("question") or "").strip()
    paras = ex.get("paragraphs") or []
    if not q or not paras:
        return None
    pos, neg = [], []
    for p in paras:
        text = (p.get("paragraph_text") or "").strip()
        if not text:
            continue
        (pos if p.get("is_supporting") else neg).append(text)
    if not pos:
        return None
    return ContrastivePair(q, pos[0], neg[:max_negs])


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^\s*=+\s*(.+?)\s*=+\s*$")  # wiki-style "== Heading =="


def _wiki_sections(text):
    """Split a Wikipedia article body into (heading, paragraph) chunks.

    Handles both wiki-markup headings ('== X ==') and plain double-newline
    paragraph breaks. Returns a list of (heading_or_None, paragraph_text).
    """
    chunks = []
    heading = None
    for block in re.split(r"\n\s*\n", text or ""):
        block = block.strip()
        if not block:
            continue
        m = _HEADING.match(block)
        if m:
            heading = m.group(1).strip()
            continue
        chunks.append((heading, block))
    return chunks


def wiki_to_pairs(ex, rng, max_pairs=3):
    """Self-supervised Wikipedia -> list[ContrastivePair].

    Builds several positive kinds from one article; other sections of the SAME
    article are hard negatives. ``rng`` is a random.Random for reproducible,
    stream-position-independent sampling (do not use module-level random).
    """
    title = (ex.get("title") or "").strip()
    body = ex.get("text") or ""
    secs = _wiki_sections(body)
    secs = [(h, p) for (h, p) in secs if len(p) > 60]     # drop stubs
    if len(secs) < 2:
        return []
    out = []
    n = len(secs)
    for _ in range(min(max_pairs, n)):
        i = rng.randrange(n)
        h_i, p_i = secs[i]
        kind = rng.choice(["title_section", "section_nearby",
                           "firstsent_para", "heading_sub"])
        if kind == "title_section" and title:
            query = title
        elif kind == "heading_sub" and h_i:
            query = h_i
        elif kind == "firstsent_para":
            sents = _SENT_SPLIT.split(p_i)
            query = sents[0] if sents else p_i[:120]
        else:  # section_nearby: a neighboring section is the query
            j = i - 1 if i > 0 else i + 1
            query = secs[j][1][:200]
        query = (query or "").strip()
        if not query:
            continue
        # hard negatives: other sections (same article, different content)
        negs = [p for k, (h, p) in enumerate(secs) if k != i]
        rng.shuffle(negs)
        out.append(ContrastivePair(query, p_i, negs[:4]))
    return out


# name -> pure extractor. Wikipedia returns a list; the others a single pair.
def extract_pairs(name, ex, rng):
    """Dispatch to the right extractor; always returns a list[ContrastivePair]."""
    if name == "hotpotqa":
        p = hotpot_to_pair(ex)
        return [p] if p else []
    if name == "musique":
        p = musique_to_pair(ex)
        return [p] if p else []
    if name == "wikipedia":
        return wiki_to_pairs(ex, rng)
    raise ValueError(f"no contrastive extractor for source {name!r}")


# Default HF coordinates for the contrastive sources (overridable at the CLI).
RETRIEVAL_SOURCE_SPECS = {
    "hotpotqa": dict(path="hotpot_qa", name="distractor", split="train",
                     trust_remote_code=True),
    "musique": dict(path="dgslibisey/MuSiQue", name=None, split="train"),
    "wikipedia": dict(path="wikimedia/wikipedia", name="20231101.en",
                      split="train"),
}


def sample_source(fracs, rng):
    """Weighted categorical draw of a source name from {name: fraction}."""
    r = rng.random() * sum(fracs.values())
    upto = 0.0
    for name, f in fracs.items():
        upto += f
        if r <= upto:
            return name
    return next(iter(fracs))  # fp fallback


# ---------------------------------------------------------------------------
# torch IterableDataset (imported lazily so the pure layer stays torch-free)
# ---------------------------------------------------------------------------

def build_iterable_dataset(*args, **kwargs):
    """Factory that imports torch lazily and returns a ContrastivePairDataset."""
    from experimentation.retrieval._torch_data import ContrastivePairDataset
    return ContrastivePairDataset(*args, **kwargs)
