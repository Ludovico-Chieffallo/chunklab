"""Gradio demo for chunklab (local `chunklab demo` and Hugging Face Spaces entrypoint).

Questions are entered one per line as:  question :: gold snippet
"""

import tempfile
from pathlib import Path

from chunklab.chunkers.registry import available_strategies
from chunklab.config import default_config
from chunklab.loaders.registry import get_loader
from chunklab.models import Question
from chunklab.report.html import write_html_report
from chunklab.runner import evaluate

#: `{k}` is filled in with the retrieval cutoff once the report is available.
TABLE_HEADERS = [
    "strategy",
    "recall@{k}",
    "hit@{k}",
    "MRR",
    "#chunks",
    "%tiny",
    "boundary",
]

EXAMPLE_QUESTIONS = (
    "What is the termination notice period? :: "
    "written notice at least 30 days prior to termination\n"
    "How is overtime compensated? :: Overtime is paid at 1.5x the regular hourly rate"
)


def parse_questions(raw: str) -> list[Question]:
    questions = []
    for i, line in enumerate(raw.strip().splitlines(), 1):
        if not line.strip():
            continue
        if "::" in line:
            query, gold = line.split("::", 1)
            questions.append(
                Question(id=f"q{i}", query=query.strip(), gold_snippets=[gold.strip()])
            )
        else:
            questions.append(Question(id=f"q{i}", query=line.strip()))
    return questions


def run_eval(file, questions_text: str, strategies: list[str]):
    if file is None:
        raise ValueError("Upload a document first (PDF, DOCX, TXT, or MD).")
    questions = parse_questions(questions_text)
    if not any(q.gold_snippets for q in questions):
        raise ValueError("Add at least one question with a gold snippet (question :: gold).")

    config = default_config()
    if strategies:
        config.strategies = [s for s in config.strategies if s.name in strategies]

    doc = get_loader(Path(file).suffix).load(Path(file))
    report = evaluate(docs=[doc], questions=questions, config=config)

    k = report.corpus_summary["top_k"]
    rows = [
        [
            r.strategy,
            round(r.recall_at_k, 2),
            round(r.hit_rate_at_k, 2),
            round(r.mrr, 2),
            r.chunk_health.num_chunks,
            f"{r.chunk_health.pct_tiny:.0%}",
            f"{r.chunk_health.boundary_health:.0%}",
        ]
        for r in report.strategy_results
    ]
    headers = [h.format(k=k) for h in TABLE_HEADERS]

    html_path = Path(tempfile.mkdtemp()) / "report.html"
    write_html_report(report, html_path)
    return report.recommendation, (headers, rows), str(html_path)


def build_app():
    import gradio as gr

    with gr.Blocks(title="ChunkLab") as demo:
        gr.Markdown(
            "# ChunkLab\n"
            "Find out whether the difference between chunking strategies is **real** on "
            "your own document — and be told when it is not. Everything runs locally."
        )
        with gr.Row():
            with gr.Column():
                file = gr.File(
                    label="Document (PDF, DOCX, TXT, MD)",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    type="filepath",
                )
                questions = gr.Textbox(
                    label="Questions (one per line: question :: gold snippet)",
                    value=EXAMPLE_QUESTIONS,
                    lines=6,
                )
                strategies = gr.CheckboxGroup(
                    choices=list(available_strategies()),
                    value=list(available_strategies()),
                    label="Strategies to compare",
                )
                run_btn = gr.Button("Run evaluation", variant="primary")
            with gr.Column():
                recommendation = gr.Textbox(label="Recommendation", lines=4)
                table = gr.Dataframe(label="Ranked comparison", headers=TABLE_HEADERS)
                report_file = gr.File(label="Full HTML report")

        def _run(file, questions_text, strategies):
            # Rows go straight to the Dataframe: building a pandas frame here made
            # pandas an undeclared dependency that only happened to arrive via gradio.
            rec, (_headers, rows), path = run_eval(file, questions_text, strategies)
            return rec, rows, path

        run_btn.click(_run, [file, questions, strategies], [recommendation, table, report_file])
    return demo


if __name__ == "__main__":
    build_app().launch()
