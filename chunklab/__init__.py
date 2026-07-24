"""chunklab — find which chunking strategy retrieves your answers best."""

__version__ = "0.2.0"


def evaluate(*args, **kwargs):
    """Run a full chunking evaluation. See chunklab.runner.evaluate."""
    from chunklab.runner import evaluate as _evaluate

    return _evaluate(*args, **kwargs)
