"""What the README and the CHANGELOG tell the user, where their words carry a decision (read as text, nothing run):
what only a real run shows is listed; what leaves this machine is said as the page says it (offline unless its
setting is off, and what a load does online); the offline trade-off is put to the user; the departures from the plan
are named, the figures review's findings numbered from 1; and every line of the package's Python is 120 characters
at most."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def words(name: str) -> str:
    return re.sub(r"\s+", " ", (HERE / name).read_text(encoding="utf-8"))


def test_the_readme_lists_what_only_a_real_run_shows():
    readme = words("README.md")
    assert "## Not yet confirmed by a real run" in readme
    unconfirmed = readme.split("## Not yet confirmed by a real run", 1)[1].split(" ## ", 1)[0]
    for part in ("release-assets.githubusercontent.com", "whether the CUDA 13.5 build sees this PC's RTX 5090",
                 "--query-gpu=index,compute_cap", "Visual C++ runtime", "`download_failed` with `{\"stopped\": true}`",
                 "The Hub's two answers", "`LLAMA_ARG_OFFLINE=1` in the router's environment",
                 "downloads the newer files by itself", "moves its folder's `refs/main`", "`--no-agent`",
                 "Llama Chat", "`memory.effective.n_ctx`"):
        assert part.lower() in unconfirmed.lower(), part
    assert "memory.n_ctx`" not in unconfirmed.replace("memory.effective.n_ctx`", "")  # (the review: not b12359's)


def test_the_readme_says_what_leaves_this_machine_as_the_page_does():
    readme = words("README.md")
    assert "llama.cpp has **no telemetry and no update check**" in readme
    assert "**A server started here is offline unless you turn its Offline setting off**" in readme
    assert "downloads the newer files by itself before it loads the model" in readme
    assert "`--no-agent`" in readme and "**No other site's page can read it**" in readme
    assert "`AIP_*`" in readme and "`MTMD_*`" in readme
    # the one storage host the figures saw is the one allowed (the review): any other is refused in words
    assert "from `release-assets.githubusercontent.com` alone" in readme and "is refused in words" in readme
    assert "is allowed though not seen" not in readme


def test_the_readme_puts_the_offline_trade_off_to_the_user():
    readme = words("README.md")
    assert "## Open for the user" in readme
    open_ = readme.split("## Open for the user", 1)[1].split(" ## ", 1)[0]
    assert "Offline" in open_ and "into llama.cpp's own cache" in open_


def test_the_readme_names_its_departures_from_the_plan():
    readme = words("README.md")
    assert "## Decisions and departures" in readme
    # the review of the figures' findings, numbered from 1 as fix-figures.json lists them (the review: not from 0)
    for part in ("A-revisions M11", "finding 10", "finding 12", "finding 13", "finding 14", "finding 32",
                 "numbered from 1", "A-inference 6", "M5"):
        assert part in readme, part
    for wrong in ("finding 9 ", "finding 11", "finding 31", "findings 11"):
        assert wrong not in readme, wrong


def test_the_changelog_says_its_release_and_its_floor():
    changelog = words("CHANGELOG.md")
    assert "## 0.1.0 (2026-11-18)" in changelog and changelog.rstrip().endswith("Needs liminal-libs>=1.62.0.")


def test_every_line_of_its_python_is_120_characters_at_most():
    for path in sorted([*HERE.glob("src/**/*.py"), *HERE.glob("tests/*.py")]):
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert len(line) <= 120, f"{path.name}:{k} is {len(line)} characters"
