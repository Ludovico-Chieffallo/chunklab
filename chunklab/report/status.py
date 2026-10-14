"""What each row of the ranked table is, according to the verdict.

The table used to mark its first row with ▶, which reads as "the winner", on every
run - including the bundled example's, whose verdict is "No single winner", and a
26-paper matrix whose marked cell held a strategy the text said could be ruled out.
A row is now marked only for what the verdict decided.
"""

from typing import Literal

from chunklab.models import EvalReport, StrategyResult

Status = Literal["winner", "ruled out"]

MARKS: dict[Status | None, str] = {"winner": "✓ ", "ruled out": "✗ ", None: "  "}


def row_status(report: EvalReport, result: StrategyResult) -> Status | None:
    """'winner', 'ruled out', or None for a row still in play.

    In a strategy x retriever matrix a row is a combination: it is the winner only
    when both of its values are named winners, and ruled out when either is.
    """
    verdict = report.verdict
    if verdict is None:
        return None
    axes = [(verdict.chunking, result.strategy)]
    if verdict.retrieval is not None:
        axes.append((verdict.retrieval, result.retriever))
    if any(name in axis.ruled_out for axis, name in axes):
        return "ruled out"
    if all(axis.decided and axis.winner == name for axis, name in axes):
        return "winner"
    return None
