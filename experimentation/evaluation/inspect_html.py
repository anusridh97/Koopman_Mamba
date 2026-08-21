"""Render a token_trace.json as a page you can read. Pure; no torch, no GPU.

    python -m experimentation.evaluation.token_trace  --checkpoint ... --data_dir ...
    python -m experimentation.evaluation.inspect_html <run_dir>/eval/final/token_trace.json

Two views over the same tokens:

  * **log probability** -- one hue, light to dark, because magnitude is one
    direction. Dark means the model was surprised.
  * **SKA ablation delta** -- two hues with a NEUTRAL GRAY midpoint, because
    polarity needs a middle that reads as "nothing". Blue where zeroing the SKA
    branch made that token worse (SKA helped), red where it made it better.

The second view is why this exists. `ska_ablation.loss_delta` collapses the whole
question to one scalar, and on a lightly-trained model that scalar is 1.45e-05,
which reads as "SKA does nothing". Per token it becomes answerable: nothing at
all, or a little everywhere, or a lot on exactly the positions needing recall.

## Colour, computed rather than chosen

Ramps are the design system's blue (sequential) and blue-to-red with a gray
midpoint (diverging). One step, `#2a78d6`, is deliberately ABSENT from the
sequential ramp: measured against both inks it scores 4.46 (dark) and 4.42
(white), so neither clears WCAG 4.5:1 and any token landing there would be
unreadable. Every remaining step clears 4.5 with the ink chosen per cell by
luminance -- worst case 5.39 sequential, 4.98 diverging.

Text sits ON the colour here, which is what makes per-cell ink selection
necessary rather than decorative: a fixed ink is exactly what makes the middle of
a 12-step ramp illegible.

Dark mode is a separate selection, not an inversion: the surfaces come from the
palette's own dark values while the cell colours stay put, because the ink is
already chosen per cell and so reads on either surface.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["render_html", "main"]

SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec",
             "#f9c9c8", "#ef8f8e", "#e34948", "#a81f1e"]
INK_DARK, INK_LIGHT = "#0b0b0b", "#ffffff"


def _luminance(hex_color: str) -> float:
    def lin(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def ink_for(background: str) -> str:
    """Whichever ink contrasts more. Computed per cell, not picked once."""
    return (INK_DARK if _contrast(background, INK_DARK) >= _contrast(background, INK_LIGHT)
            else INK_LIGHT)


def _bucket(value: float, lo: float, hi: float, ramp: List[str]) -> str:
    if hi <= lo:
        return ramp[len(ramp) // 2]
    t = (value - lo) / (hi - lo)
    return ramp[max(0, min(len(ramp) - 1, int(t * len(ramp))))]


def render_html(trace: Dict[str, Any], *,
                scale_override: Optional[Dict[str, float]] = None) -> str:
    """A token_trace envelope (or its bare `metrics`) -> one self-contained page.

    `scale_override` lets two traces be rendered on ONE scale by passing the
    union of their bounds, so "step 200 vs step 400" is a fair comparison rather
    than two pages each normalised to itself. Without it the trace's own summary
    is used.
    """
    metrics = trace.get("metrics", trace)
    summary = dict(metrics["summary"])
    if scale_override:
        summary.update(scale_override)

    lo, hi = summary["logprob_p05"], summary["logprob_p95"]
    dmax = summary.get("ska_delta_absmax") or 0.0
    seq_ramp = list(reversed(SEQUENTIAL))     # dark = surprising
    div_ramp = list(reversed(DIVERGING))      # blue = SKA helped

    blocks = []
    for si, seq in enumerate(metrics["sequences"]):
        lp_spans, d_spans = [], []
        for tok in seq["tokens"]:
            label = html.escape(tok["token"]).replace(" ", "&nbsp;") or "&#9251;"
            tip = html.escape(json.dumps(tok))
            bg = _bucket(tok["logprob"], lo, hi, seq_ramp)
            lp_spans.append(f'<span class="tk" style="background:{bg};'
                            f'color:{ink_for(bg)}" data-t="{tip}">{label}</span>')
            if tok.get("ska_delta") is not None and dmax > 0:
                dbg = _bucket(tok["ska_delta"], -dmax, dmax, div_ramp)
                d_spans.append(f'<span class="tk" style="background:{dbg};'
                               f'color:{ink_for(dbg)}" data-t="{tip}">{label}</span>')
        blocks.append(
            f'<h3>sequence {si} &middot; {len(seq["tokens"])} tokens</h3>'
            f'<div class="seq" data-view="logprob">{"".join(lp_spans)}</div>'
            + (f'<div class="seq hidden" data-view="ska">{"".join(d_spans)}</div>'
               if d_spans else ""))

    meta = {k: v for k, v in trace.items() if k != "metrics"}
    meta_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in list(meta.items()) + [
            ("tokens", summary["n_tokens"]),
            ("mean logprob", summary["mean_logprob"]),
            ("source", summary.get("source", "")),
        ])

    return _PAGE.format(
        meta_rows=meta_rows, blocks="".join(blocks),
        legend_lp="".join(f'<span class="sw" style="background:{c}"></span>'
                          for c in seq_ramp),
        legend_d="".join(f'<span class="sw" style="background:{c}"></span>'
                         for c in div_ramp),
        lo=f"{lo:.2f}", hi=f"{hi:.2f}", dmax=f"{dmax:.2e}",
        has_ska="true" if (summary.get("has_ablation") and dmax > 0) else "false")


_PAGE = """<!DOCTYPE html>
<meta charset="utf-8"><title>token inspector</title>
<style>
:root {{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --rule:#e1e0d9; --ring:rgba(11,11,11,.10);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --rule:#2c2c2a; --ring:rgba(255,255,255,.10);
  }}
}}
body {{ margin:0; padding:24px; background:var(--plane); color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
h1 {{ font-size:18px; margin:0 0 8px; }}
h3 {{ font-size:11px; color:var(--muted); font-weight:600; margin:20px 0 6px;
  text-transform:uppercase; letter-spacing:.06em; }}
.card {{ background:var(--surface); border:1px solid var(--ring);
  border-radius:10px; padding:18px 20px; max-width:1100px; }}
table.meta {{ border-collapse:collapse; margin:0 0 6px; font-size:12px; }}
table.meta th {{ text-align:left; color:var(--muted); font-weight:500;
  padding:2px 14px 2px 0; white-space:nowrap; }}
table.meta td {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--ink2); }}
.seq {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px;
  line-height:2.1; word-break:break-word; }}
.seq.hidden {{ display:none; }}
.tk {{ padding:2px 1px; border-radius:3px; margin:0 1px; cursor:default;
  box-shadow:0 0 0 2px var(--surface); }}
.tk:hover {{ outline:2px solid var(--ink); outline-offset:1px; }}
.controls {{ display:flex; gap:8px; align-items:center; margin:14px 0 4px; }}
button {{ font:inherit; padding:5px 11px; border-radius:7px; cursor:pointer;
  border:1px solid var(--ring); background:var(--plane); color:var(--ink); }}
button[aria-pressed="true"] {{ background:var(--ink); color:var(--surface); }}
button:disabled {{ opacity:.45; cursor:not-allowed; }}
.legend {{ display:flex; align-items:center; gap:7px; font-size:11px;
  color:var(--muted); margin:8px 0 2px; }}
.legend.hidden {{ display:none; }}
.sw {{ width:15px; height:11px; border-radius:2px;
  box-shadow:0 0 0 1px var(--ring) inset; }}
#tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--surface); color:var(--ink); border:1px solid var(--ring);
  border-radius:8px; padding:9px 11px; font-size:12px; max-width:300px;
  box-shadow:0 6px 20px rgba(0,0,0,.16); z-index:9; }}
#tip table {{ border-collapse:collapse; margin-top:5px; width:100%; }}
#tip td {{ padding:1px 0; font-family:ui-monospace,Menlo,monospace; }}
#tip td.p {{ text-align:right; color:var(--ink2); padding-left:12px; }}
.note {{ font-size:12px; color:var(--muted); margin-top:14px;
  border-top:1px solid var(--rule); padding-top:10px; max-width:72ch; }}
</style>
<div class="card">
<h1>Token inspector</h1>
<table class="meta">{meta_rows}</table>

<div class="controls">
  <button id="b-lp" aria-pressed="true">log probability</button>
  <button id="b-ska" aria-pressed="false">SKA ablation delta</button>
</div>
<div class="legend" id="lg-lp">
  <span>surprising ({lo})</span>{legend_lp}<span>confident ({hi})</span>
</div>
<div class="legend hidden" id="lg-ska">
  <span>SKA hurt (&minus;{dmax})</span>{legend_d}<span>SKA helped (+{dmax})</span>
</div>

{blocks}

<p class="note">Log probability is a magnitude, so it gets one hue
light&#8594;dark. The ablation delta is a polarity, so it gets two hues with a
<em>neutral gray</em> midpoint &mdash; gray means SKA changed nothing at that
token, which is most of them on a lightly-trained model. That scale is symmetric
about zero so colour cannot misreport sign. Ink is chosen per cell by luminance,
and the one ramp step where neither ink reached WCAG&nbsp;4.5:1 was dropped rather
than shipped unreadable. Range is the 5th&ndash;95th percentile, so one
pathological token cannot flatten the ramp. Hover a token for the top-k the model
actually predicted and where the true token ranked.</p>
</div>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.tk').forEach(el => {{
  el.addEventListener('mouseenter', () => {{
    const d = JSON.parse(el.dataset.t);
    let h = `<b>pos ${{d.pos}}</b> &middot; logprob ${{d.logprob}} &middot; rank ${{d.rank}}`;
    if (d.ska_delta !== undefined && d.ska_delta !== null)
      h += `<br><b>SKA delta</b> ${{d.ska_delta}}`;
    h += '<table>' + (d.top || []).map(([t, p]) =>
      `<tr><td>${{String(t).replace(/</g,'&lt;')}}</td><td class="p">${{p}}</td></tr>`
    ).join('') + '</table>';
    tip.innerHTML = h; tip.style.opacity = 1;
  }});
  el.addEventListener('mousemove', e => {{
    const pad = 14;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + tip.offsetWidth > innerWidth) x = e.clientX - tip.offsetWidth - pad;
    if (y + tip.offsetHeight > innerHeight) y = e.clientY - tip.offsetHeight - pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }});
  el.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; }});
}});
const hasSka = {has_ska};
function show(view) {{
  document.querySelectorAll('.seq').forEach(s =>
    s.classList.toggle('hidden', s.dataset.view !== view));
  document.getElementById('lg-lp').classList.toggle('hidden', view !== 'logprob');
  document.getElementById('lg-ska').classList.toggle('hidden', view !== 'ska');
  document.getElementById('b-lp').setAttribute('aria-pressed', view === 'logprob');
  document.getElementById('b-ska').setAttribute('aria-pressed', view === 'ska');
}}
document.getElementById('b-lp').onclick = () => show('logprob');
document.getElementById('b-ska').onclick = () => hasSka && show('ska');
if (!hasSka) document.getElementById('b-ska').disabled = true;
</script>
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experimentation.evaluation.inspect_html",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="a token_trace.json written by token_trace.py")
    p.add_argument("--out", default=None,
                   help="default: the trace's own name with .html")
    p.add_argument("--also", default=None,
                   help="a second trace; both are rendered on ONE scale so the "
                        "comparison is fair rather than each self-normalised")
    args = p.parse_args(argv)

    trace = json.loads(Path(args.trace).read_text())
    override = None
    if args.also:
        other = json.loads(Path(args.also).read_text())
        a = trace.get("metrics", trace)["summary"]
        b = other.get("metrics", other)["summary"]
        override = {
            "logprob_p05": min(a["logprob_p05"], b["logprob_p05"]),
            "logprob_p95": max(a["logprob_p95"], b["logprob_p95"]),
            "ska_delta_absmax": max(a.get("ska_delta_absmax") or 0.0,
                                    b.get("ska_delta_absmax") or 0.0),
        }

    out = Path(args.out or Path(args.trace).with_suffix(".html"))
    out.write_text(render_html(trace, scale_override=override))
    print(f"wrote {out}")
    if args.also:
        other_out = out.with_name(out.stem + "-b.html")
        other_out.write_text(render_html(other, scale_override=override))
        print(f"wrote {other_out}  (same scale, so the two are comparable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
