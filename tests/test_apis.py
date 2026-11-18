"""What the plugin attaches to (plugin API 2.2): its six cards are sound (problems() is empty); the server's card
names exactly the calls the plugin makes of a server (server.CALLS, models.CALLS) and the Hub's those it makes of
the Hub; the releases' card names every archive it pins; the cards hold nothing of one machine; every outside host
the package names is on a card or among release.py's pins, and no address is written outside apis.py and
release.py; and, with LIMINAL_VENDOR_PAGES=1 alone, every page on the cards answers (the only test that reaches out).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from liminal_libs.sources.apis import dead_links, problems, urls

from plugin_platform_llamacpp import apis as cards
from plugin_platform_llamacpp import models, release, server

SRC = Path(cards.__file__).resolve().parent
ADDRESS = re.compile(r"[a-z][a-z0-9+.-]*://[A-Za-z0-9\[][^\s\"')]*")


def test_the_cards_are_sound_and_each_says_where_its_documentation_is():
    assert problems(cards.APIS) == []
    assert [a.name for a in cards.APIS] == ["llama-server's API", "llama-server, the command", "llama.cpp's releases",
                                            "Hugging Face Hub API", "nvidia-smi", "Start at sign-in"]
    assert all(a.verified == cards.VERIFIED and a.docs.startswith("https://") for a in cards.APIS)
    assert not any(c.costs for a in cards.APIS for c in a.calls)  # nothing here costs money


def test_the_servers_card_names_exactly_the_calls_made_of_a_server():
    used = {c.name for c in cards.SERVER.used}
    assert used == set(server.CALLS) | set(models.CALLS)
    writes = {c.name for c in cards.SERVER.calls if c.writes}
    assert writes == {"POST /models", "POST /models/load", "POST /models/unload", "DELETE /models"}
    assert {c.name for c in cards.SERVER.calls if not c.used} >= {"GET /slots", "GET /metrics", "POST /v1/embeddings"}


def test_the_hubs_card_names_the_two_asks_of_a_lookup():
    assert {c.name for c in cards.HUB_API.used} == set(models.HUB_CALLS)
    assert cards.HUB_API.base == release.HUB


def test_the_releases_card_names_every_archive_pinned():
    pinned = {f"GET {part.name}" for parts in release.BUILDS.values() for part in parts}
    assert {c.name for c in cards.RELEASES.used} == pinned and len(pinned) == 12
    assert cards.RELEASES.base == release.RELEASE


def test_the_cards_say_what_holds_on_any_machine():
    for api in cards.APIS:
        words = [api.about, api.base, api.version, api.auth, *(a.about for a in api.areas),
                 *(c.about for c in api.calls)]
        assert not any(w and re.search(r"595\.71|RTX 5090 Laptop|this PC", w) for w in words), api.name


def _addresses(path: Path) -> list[str]:
    return ADDRESS.findall(path.read_text(encoding="utf-8"))


def test_no_address_is_written_outside_apis_and_release():
    for path in SRC.glob("*.py"):
        if path.name in ("apis.py", "release.py"):
            continue
        assert _addresses(path) == [], path.name


def test_every_host_the_package_names_is_on_a_card_or_pinned():
    on_cards = {urlsplit(u).hostname for u in urls(cards.APIS)} | {urlsplit(a.base).hostname for a in cards.APIS
                                                                    if a.base.startswith("https://")}
    pinned = {*release.HOSTS, *release.ASSET_HOSTS, urlsplit(release.HUB).hostname}
    for path in (SRC / "apis.py", SRC / "release.py"):
        for address in _addresses(path):
            host = urlsplit(address).hostname
            assert host in on_cards | pinned | {"127.0.0.1"}, (path.name, address)
    assert {"github.com", "release-assets.githubusercontent.com", "huggingface.co"} <= on_cards | pinned


@pytest.mark.skipif(os.environ.get("LIMINAL_VENDOR_PAGES") != "1",
                    reason="reaches the makers' pages: LIMINAL_VENDOR_PAGES=1")
def test_the_cards_pages_answer(monkeypatch):
    monkeypatch.undo()  # this test alone reaches the network (the autouse guard refuses it to every other)
    assert dead_links(cards.APIS) == []
