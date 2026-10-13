"""The README must not call the install permissive while it pulls in AGPL code.

chunklab's own code is MIT and the README said so - "permissive on purpose" - while
`pip install chunklab` also installed PyMuPDF and pymupdf4llm, which are licensed
under the AGPL-3.0 (or a commercial licence from Artifex). An organisation that does
not allow AGPL dependencies could only find that out by reading the metadata of
what pip had already installed.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

#: Distributions known to be AGPL-licensed, as they appear in pyproject.toml.
AGPL_DISTRIBUTIONS = {"pymupdf", "pymupdf4llm", "pymupdf-layout"}


def _declared() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    specs = list(project["dependencies"])
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    return {re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].strip().lower() for spec in specs}


def _license_section() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split("\n## License", 1)[1]


def test_an_agpl_dependency_is_named_where_the_licence_is_stated():
    agpl = sorted(_declared() & AGPL_DISTRIBUTIONS)
    if not agpl:
        return  # nothing to disclose

    section = _license_section()
    assert "AGPL" in section, "the License section must say an AGPL dependency is installed"
    assert "PyMuPDF" in section, "and name it, so a policy check can find it"
