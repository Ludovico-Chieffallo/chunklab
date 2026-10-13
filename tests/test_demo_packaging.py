"""`chunklab demo` must work from an installed package, not only from a checkout.

The demo lived in `app/app.py`, at the repository root, and the wheel packages only
`chunklab/`. On 0.4.0 installed from PyPI, `chunklab demo` therefore could never
import it, and reported the failure as a missing gradio. Every test passed anyway,
because they ran from the checkout with `app/` put on `sys.path` by hand.
"""

import importlib.util
import sys
from types import SimpleNamespace

import pytest


def test_the_demo_is_part_of_the_package():
    assert importlib.util.find_spec("chunklab.demo") is not None


def test_chunklab_demo_does_not_import_a_top_level_app_module(cli, monkeypatch):
    """Outside a checkout there is no `app` module - or worse, an unrelated one
    from some other package. Blocking it must not matter."""
    pytest.importorskip("gradio")
    monkeypatch.setitem(sys.modules, "app", None)
    monkeypatch.setitem(sys.modules, "app.app", None)
    import chunklab.demo

    launched = []
    monkeypatch.setattr(
        chunklab.demo, "build_app", lambda: SimpleNamespace(launch=lambda: launched.append(True))
    )

    result = cli("demo")

    assert result.exit_code == 0, result.output
    assert launched == [True]


def test_missing_gradio_is_reported_with_the_extra_to_install(cli, monkeypatch):
    """The import of gradio happens inside build_app(), so it escaped the old
    try/except around the module import and surfaced as a traceback."""
    monkeypatch.setitem(sys.modules, "gradio", None)

    result = cli("demo")

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "pip install 'chunklab[demo]'" in result.output
