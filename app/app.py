"""Hugging Face Spaces entrypoint (`app_file` in app/README.md).

The demo itself is `chunklab.demo`, inside the package, so that a PyPI install
ships it; this file only launches it.
"""

from chunklab.demo import build_app

if __name__ == "__main__":
    build_app().launch()
