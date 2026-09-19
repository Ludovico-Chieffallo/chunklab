"""Entry point for `python -m chunklab`.

The console script installed by the package is the documented way in, but
`python -m` is what people reach for when the script is not on PATH — inside a
virtualenv they have not activated, or in CI. Without this module it printed
nothing and exited 0, which reads as a broken install.
"""

from chunklab.cli import app

app()
