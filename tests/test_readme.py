"""What the README and the CHANGELOG tell the user, where their words carry a decision (read as text, nothing run):
what only a real run shows is listed; what leaves this machine is said as the page says it, the Hub asked at each
load of a downloaded model among it; the departures from the plan are named; and every line of the package's Python
is 120 characters at most."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def words(name: str) -> str:
    return re.sub(r"\s+", " ", (HERE / name).read_text(encoding="utf-8"))


def test_the_readme_lists_what_only_a_real_run_shows():
    readme = words("README.md")
    assert "## Not yet confirmed by a real run" in readme
    for part in ("release-assets.githubusercontent.com", "whether the CUDA 13.5 build sees this PC's RTX 5090",
                 "--query-gpu=index,compute_cap", "Visual C++ runtime", "`download_failed` with `{\"stopped\": true}`",
                 "The Hub's two answers"):
        assert part.lower() in readme.lower(), part


def test_the_readme_says_what_leaves_this_machine_as_the_page_does():
    readme = words("README.md")
    assert "llama.cpp has **no telemetry and no update check**" in readme
    assert "**Loading a model it downloaded asks huggingface.co first**" in readme
    assert "`--offline` stops its router's downloads too" in readme
    assert "never switched on" in readme and "**No other site's page can read it**" in readme


def test_the_readme_names_its_departures_from_the_plan():
    readme = words("README.md")
    assert "## Decisions and departures" in readme
    for part in ("A-revisions M11", "finding 11", "finding 12", "finding 13", "finding 31", "A-inference 6", "M5"):
        assert part in readme, part


def test_the_changelog_says_its_release_and_its_floor():
    changelog = words("CHANGELOG.md")
    assert "## 0.1.0 (2026-11-18)" in changelog and changelog.rstrip().endswith("Needs liminal-libs>=1.62.0.")


def test_every_line_of_its_python_is_120_characters_at_most():
    for path in sorted([*HERE.glob("src/**/*.py"), *HERE.glob("tests/*.py")]):
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert len(line) <= 120, f"{path.name}:{k} is {len(line)} characters"
