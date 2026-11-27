"""Who may ask the plugin's own changes (0.1.1; the sweep after the review's finding PLT-H1, 2026-11-25). The platform
hands an ask from any program on this PC to a plugin's hook, so the plugin says which of the hooks it adds only its
page asks (PAGE_ONLY_HOOKS, as liminal-libs 1.66.0's apps/asking.py and liminal-platform 0.34.0 read them): a model
downloaded, deleted, loaded or unloaded by an instance's router, and the build Install brings. Another plugin's ask of
one is refused, and so is an ask that names no one: nothing is asked of the router or the Hub, and nothing is kept.
The page's ask reaches the router as before. The model server's services are answered to anyone. Its lookups (find,
clear_found, details) are the page's alone too (0.1.2; the review of item 14, P3-1; the user, 2026-11-26, of reads that
reach outside or with a key: "Page-only"): find asks the Hub about a name of the asker's choosing and keeps it,
clear_found empties what the Models tab keeps, and details asks an instance, another machine's with its key. It names
its own alone: the model server's base says its own (liminal-libs 1.99.0), and the names add up along the class line.

Asked through the library's own App (its ask names who asks as the platform's does), over this suite's stand-ins
(conftest.py): no server runs and nothing is downloaded, and its guard fails a test that reaches out."""
from __future__ import annotations

from urllib.parse import urlsplit

import pytest

pytest.importorskip("liminal_libs.apps.asking", reason="no hook is asked only from its page before liminal-libs 1.66.0")

from liminal_libs import internal  # noqa: E402 - after the skip, as each of these
from liminal_libs.apps import App, asking  # noqa: E402
from liminal_libs.machines.modelserver import ModelServer  # noqa: E402
from liminal_libs.machines.plugin import MachinePlugin  # noqa: E402

from plugin_platform_llamacpp import MACHINE, LlamaCppPlugin  # noqa: E402
from conftest import local, router, running, sse  # noqa: E402
from test_models import REPO, TARGET, Panel, _events, hub  # noqa: E402

QWEN = "ggml-org/Qwen3-8B-GGUF:Q4_K_M"  # one the stand-in router lists, downloaded by it and not loaded
CHANGES = ("pull", "delete_model", "load", "unload", "build")  # what it adds to the model server's changes
PAGE_READS = ("find", "clear_found", "details")  # its lookups: the page's since 0.1.2
BASES = tuple(getattr(ModelServer, "PAGE_ONLY_HOOKS", ()))  # the base's own (liminal-libs 1.99.0's; none before)
ASKS = {  # each change as an ask that would be done, a download's licence accepted: an argument the asker sets
    "pull": ("default", REPO, True),
    "delete_model": ("default", QWEN),
    "load": ("default", QWEN),
    "unload": ("default", TARGET),
    "build": ("vulkan",),
}


class Neighbour(MachinePlugin):
    """Another plugin of the platform, asking llama.cpp's plugin through its app as any plugin may (a harness, or the
    MCP plugin for a local agent)."""

    name, title = "neighbour", "Neighbour"

    def wants(self, hook, *args):
        """Not a hook: one of llama.cpp's plugin's asked through the app."""
        return self.app.ask("llamacpp", hook, *args)


class Platform(App):
    """The library's App asking as the platform does: from its page's route, the page named in so many words, or from
    its own code, no one named."""

    NAME, TITLE, ICON = "platform", "Platform", "chip"

    def from_page(self, hook, *args, **more):
        """As the page's route asks a hook (a job's with its panel as `log`)."""
        with asking.asking(asking.PAGE):
            return self.ask("llamacpp", hook, *args, **more)

    def on_its_own(self, hook, *args):
        """As the app's own code asks a hook."""
        return self.ask("llamacpp", hook, *args)


@pytest.fixture
def attached(world):
    """The plugin on the library's App beside another plugin, with its install, a router run here that answers, and a
    model looked up at the Hub; let go after."""
    p = world.plugin("windows", attach=False)
    pairs = p.login
    app, neighbour = Platform(log=lambda m: None, login=pairs), Neighbour()
    app.attach(p)
    app.attach(neighbour)
    world.installed(p)
    local(p, offline="off")  # online, so a download can go through its router
    running(p, world)
    router(world)
    p.models_of("default")
    hub(world)
    p.find(REPO)
    yield app, neighbour, p, pairs
    app.detach(neighbour.name)
    app.detach(p.name)


def test_its_own_hooks_are_said_to_be_asked_only_from_its_page():
    assert LlamaCppPlugin.PAGE_ONLY_HOOKS == (*CHANGES, *PAGE_READS) and set(ASKS) == set(CHANGES)  # its own alone
    assert set(internal.page_only(MACHINE)) == {*BASES, *CHANGES, *PAGE_READS}  # what the platform reads
    assert internal.plugin_only(MACHINE) == ()
    assert [p for p in internal.problems(MACHINE) if "_ONLY_HOOKS" in p] == []  # each a hook of its own
    assert set(MACHINE.HOOKS) >= {*CHANGES, *PAGE_READS}
    opened = {*getattr(MACHINE, "SERVICE_HOOKS", ()), *internal.declared(MACHINE, "READ_HOOKS")}
    assert not {*CHANGES, *PAGE_READS} & opened
    for hook in (*CHANGES, *PAGE_READS):
        assert asking.refusal(MACHINE, hook, asking.PAGE) == ""
        assert asking.refusal(MACHINE, hook, "neighbour") == (f"llamacpp's {hook} is asked only from its page, never "
                                                              "by another plugin: neighbour asked it.")
        assert asking.refusal(MACHINE, hook, None).endswith("This ask names no asker: neither the page nor a plugin.")
    for hook in ("status", "serving", "models_of"):  # as before, and what a harness asks of it
        assert {asking.refusal(MACHINE, hook, who) for who in (asking.PAGE, "neighbour", None)} == {""}


def test_another_plugin_or_no_one_asking_a_change_asks_its_router_nothing_and_its_page_does(attached, world):
    app, neighbour, p, pairs = attached
    web = world.transport
    web.on("POST", "/models", (200, {"success": True}))
    web.on("GET", "/models/sse", lambda *a: sse(*_events("progress", "finished")))
    web.on("DELETE", "/models", (200, {"success": True}))
    web.on("POST", "/models/load", (200, {"success": True}))
    web.on("POST", "/models/unload", (200, {"success": True}))
    sent, kept = len(web.asked), dict(pairs.kept)
    for hook in CHANGES:  # each as a program would send it, and each would be done
        for ask in (lambda: neighbour.wants(hook, *ASKS[hook]), lambda: app.on_its_own(hook, *ASKS[hook])):
            with pytest.raises(asking.AskRefused, match=f"llamacpp's {hook} is asked only from its page") as refused:
                ask()
            assert refused.value.status == 403 and refused.value.hook == hook
    assert len(web.asked) == sent and pairs.kept == kept  # nothing asked of its router or the Hub, no build kept
    # its page's asks reach the router, and its build is kept through the login, as before
    assert app.from_page("load", *ASKS["load"])["ok"] and app.from_page("unload", *ASKS["unload"])["ok"]
    assert app.from_page("delete_model", *ASKS["delete_model"])["ok"]
    panel = Panel()
    assert app.from_page("pull", *ASKS["pull"], log=panel)["ok"] and panel.lines
    asked = {f"{method} {urlsplit(url).path}" for method, url, _, _ in web.asked[sent:]}
    assert {"POST /models/load", "POST /models/unload", "DELETE /models", "POST /models"} <= asked
    assert app.from_page("build", *ASKS["build"])["ok"] and pairs.kept[p.variable("build")] == "vulkan"
    # what a harness asks of it, by another plugin as before
    assert isinstance(neighbour.wants("status"), dict)
    assert [row["name"] for row in neighbour.wants("serving")["instances"]] == ["default"]


def test_another_plugin_or_no_one_asking_a_lookup_reaches_nothing_and_forgets_nothing(attached, world):
    """find, clear_found and details (0.1.2): refused to another plugin and to an ask that names no one before the
    Hub or an instance is asked, the lookup kept stays; the page asks all three as before."""
    app, neighbour, p, _ = attached
    web = world.transport
    asks = {"find": ("ggml-org/gemma-3-4b-it-GGUF",), "clear_found": (), "details": ("default", TARGET)}
    sent = len(web.asked)
    for hook, args in asks.items():
        for ask in (lambda: neighbour.wants(hook, *args), lambda: app.on_its_own(hook, *args)):
            with pytest.raises(asking.AskRefused, match=f"llamacpp's {hook} is asked only from its page") as refused:
                ask()
            assert refused.value.status == 403 and refused.value.hook == hook
    assert len(web.asked) == sent and p.found_of(REPO) is not None  # nothing asked, the fixture's lookup kept
    assert app.from_page("details", *asks["details"])["ok"] and len(web.asked) > sent  # the page's asks its router
    assert app.from_page("find", REPO)["ok"] and p.found_of(REPO) is not None
    assert app.from_page("clear_found")["ok"] and p.found_of(REPO) is None
