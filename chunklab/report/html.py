"""Standalone HTML report (Jinja2, inline CSS, no external assets)."""

from pathlib import Path

from jinja2 import BaseLoader, Environment

from chunklab.models import EvalReport

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChunkLab report</title>
<style>
  :root { --bg:#fff; --fg:#1a1a2e; --muted:#667; --line:#e2e2ea; --accent:#2563eb;
          --win:#ecfdf5; --winb:#10b981; --bad:#ef4444; --warn:#f59e0b; --ok:#10b981; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
         color:var(--fg); background:var(--bg); padding:2rem 1rem; }
  main { max-width:60rem; margin:0 auto; }
  h1 { font-size:1.6rem; margin:0 0 .25rem; } h2 { font-size:1.15rem; margin:2.2rem 0 .6rem; }
  .sub { color:var(--muted); margin-bottom:1.5rem; }
  .rec { background:var(--win); border-left:4px solid var(--winb);
         padding:.9rem 1.1rem; border-radius:0 8px 8px 0; margin:1.2rem 0; }
  .warn { background:#fffbeb; border-left:4px solid var(--warn);
          padding:.6rem 1rem; border-radius:0 8px 8px 0; margin:.5rem 0; font-size:.9rem; }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:.9rem; }
  th, td { text-align:right; padding:.45rem .7rem; border-bottom:1px solid var(--line);
           white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; }
  tr.winner { background:var(--win); font-weight:600; }
  .hist { display:flex; align-items:flex-end; gap:2px; height:56px; margin:.3rem 0 .1rem; }
  .hist div { background:var(--accent); min-width:8px; border-radius:2px 2px 0 0; }
  .histlbl { font-size:.72rem; color:var(--muted); }
  details { border:1px solid var(--line); border-radius:8px; margin:.5rem 0;
            padding:.5rem .9rem; }
  summary { cursor:pointer; font-weight:600; }
  .chip { display:inline-block; font-size:.75rem; padding:.05rem .5rem; border-radius:99px;
          color:#fff; margin-left:.4rem; }
  .chip.hit { background:var(--ok); } .chip.miss { background:var(--bad); }
  .chip.split { background:var(--warn); }
  .chunktext { font-size:.82rem; background:#f8f8fb; border:1px solid var(--line);
               border-radius:6px; padding:.5rem .7rem; margin:.35rem 0;
               white-space:pre-wrap; max-height:9rem; overflow-y:auto; }
  mark { background:#fde68a; }
  .track { position:relative; height:22px; background:#f1f1f6; border-radius:4px;
           margin:.25rem 0 .9rem; }
  .bound { position:absolute; top:0; bottom:0; width:1px; background:#99a; }
  .gold { position:absolute; top:4px; bottom:4px; border-radius:2px; min-width:3px; }
  .gold.clean { background:var(--ok); } .gold.split { background:var(--bad); }
  .lgnd { font-size:.8rem; color:var(--muted); }
  .lgnd i { display:inline-block; width:10px; height:10px; border-radius:2px;
            vertical-align:-1px; margin:0 .25rem 0 .8rem; }
</style>
</head>
<body>
<main>
<h1>ChunkLab report</h1>
<p class="sub">{{ cs.num_documents }} document(s) · {{ cs.num_scored_questions }} scored
questions · top_k={{ k }} · model={{ cs.embedding_model }} · generated
{{ report.generated_at }}</p>

<div class="rec"><strong>Recommendation.</strong> {{ report.recommendation }}</div>
{% for w in report.warnings %}<div class="warn">⚠ {{ w }}</div>{% endfor %}

<h2>Ranked comparison</h2>
<div class="tablewrap">
<table>
<tr><th>Strategy</th><th>recall@{{ k }}</th><th>hit@{{ k }}</th><th>MRR</th>
<th>prec@{{ k }}</th><th>#chunks</th><th>med tok</th><th>%tiny</th>
<th>boundary</th><th>tables intact</th></tr>
{% for r in report.strategy_results %}
<tr {% if loop.first %}class="winner"{% endif %}>
<td>{{ '▶ ' if loop.first }}{{ r.strategy }}</td>
<td>{{ '%.2f'|format(r.recall_at_k) }}</td>
<td>{{ '%.2f'|format(r.hit_rate_at_k) }}</td>
<td>{{ '%.2f'|format(r.mrr) }}</td>
<td>{{ '%.2f'|format(r.precision_at_k) }}</td>
<td>{{ r.chunk_health.num_chunks }}</td>
<td>{{ '%.0f'|format(r.chunk_health.tokens_median) }}</td>
<td>{{ '%.0f%%'|format(r.chunk_health.pct_tiny * 100) }}</td>
<td>{{ '%.0f%%'|format(r.chunk_health.boundary_health * 100) }}</td>
<td>{% if r.chunk_health.table_integrity is not none %}{{ '%.0f%%'|format(r.chunk_health.table_integrity * 100) }}{% else %}–{% endif %}</td>
</tr>
{% endfor %}
</table>
</div>

<h2>Per-strategy diagnostics</h2>
{% for r in report.strategy_results %}
<details {% if loop.first %}open{% endif %}>
<summary>{{ r.strategy }} — token size distribution</summary>
{% set maxcount = r.chunk_health.token_histogram | map(attribute=1) | max %}
<div class="hist">
{% for bucket, count in r.chunk_health.token_histogram %}
<div style="height:{{ (count / maxcount * 100) | round }}%"
     title="{{ bucket }}–{{ bucket + 99 }} tok: {{ count }} chunks"></div>
{% endfor %}
</div>
<p class="histlbl">{{ r.chunk_health.tokens_min }}–{{ r.chunk_health.tokens_max }} tokens
(median {{ '%.0f'|format(r.chunk_health.tokens_median) }}) ·
{{ '%.0f%%'|format(r.chunk_health.pct_tiny * 100) }} tiny ·
{{ '%.0f%%'|format(r.chunk_health.pct_oversized * 100) }} oversized</p>
</details>
{% endfor %}

<h2>Per-question drill-down</h2>
{% for qid in question_ids %}
<details>
<summary>{{ qid }} — {{ queries[qid] }}
{% for r in report.strategy_results %}{% set qr = per_q[r.strategy][qid] %}
<span class="chip {{ 'hit' if qr.hit else ('split' if qr.split_across_chunks else 'miss') }}">
{{ r.strategy }}{{ ' #%d'|format(qr.first_hit_rank) if qr.first_hit_rank else '' }}</span>
{% endfor %}</summary>
{% for r in report.strategy_results %}{% set qr = per_q[r.strategy][qid] %}
<p><strong>{{ r.strategy }}</strong> —
{% if qr.hit %}hit at rank {{ qr.first_hit_rank }}
({{ qr.gold_found_count }}/{{ qr.gold_total }} gold found){% elif qr.split_across_chunks %}
answer <strong>split across two adjacent chunks</strong>{% else %}not retrieved in
top-{{ k }}{% endif %}</p>
{% for rc in qr.retrieved[:3] %}
<div class="chunktext">[#{{ rc.rank }} · {{ '%.3f'|format(rc.score) }}
{% if rc.is_hit %}· <strong>contains gold</strong>{% endif %}]
{{ rc.chunk.text[:600] }}{{ '…' if rc.chunk.text|length > 600 }}</div>
{% endfor %}
{% endfor %}
</details>
{% endfor %}

{% if report.viz %}
<h2>Chunk boundaries — sample document “{{ report.viz.doc_id }}”</h2>
<p class="lgnd">Vertical lines are chunk boundaries.
<i style="background:var(--ok)"></i>gold snippet inside one chunk
<i style="background:var(--bad)"></i>gold snippet split across chunks</p>
{% set L = report.viz.doc_length %}
{% for sv in report.viz.strategies %}
<p style="margin:.6rem 0 0"><strong>{{ sv.strategy }}</strong></p>
<div class="track">
{% for b in sv.boundaries %}
<span class="bound" style="left:{{ '%.2f'|format(b / L * 100) }}%"></span>
{% endfor %}
{% for g in sv.gold_markers %}
<span class="gold {{ g.status }}"
      style="left:{{ '%.2f'|format(g.start / L * 100) }}%;
             width:{{ '%.2f'|format((g.end - g.start) / L * 100) }}%"
      title="{{ g.question_id }} ({{ g.status }})"></span>
{% endfor %}
</div>
{% endfor %}
{% endif %}

</main>
</body>
</html>
"""


def write_html_report(report: EvalReport, path: str | Path) -> Path:
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(_TEMPLATE)

    question_ids: list[str] = []
    queries: dict[str, str] = {}
    per_q: dict[str, dict[str, object]] = {}
    for r in report.strategy_results:
        per_q[r.strategy] = {}
        for qr in r.per_question:
            per_q[r.strategy][qr.question_id] = qr
            if qr.question_id not in question_ids:
                question_ids.append(qr.question_id)

    stored = report.corpus_summary.get("queries", {})
    for qid in question_ids:
        queries[qid] = stored.get(qid, qid)

    html = template.render(
        report=report,
        cs=report.corpus_summary,
        k=report.corpus_summary.get("top_k", 5),
        question_ids=question_ids,
        queries=queries,
        per_q=per_q,
    )
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
