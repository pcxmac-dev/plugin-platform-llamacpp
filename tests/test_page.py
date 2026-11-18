"""What the platform's page draws: Main (status and main_more) asks no server, connects to no port, reads no boot
registration and runs nothing; its tables What leaves this machine, On this system (each row on its stand-in system:
Windows without its C runtime or nvidia-smi, Linux, WSL, another processor) and What Install brings with its build
form; every form's fields are its hook's arguments in their order and its words name only its fields; a key kept is
in no answer, page, line or error; the command line prints the status, an instance's lines to paste, and the
install's size, hosts and licence before its refusal, without running anything; and a detach pauses a download
under way through its router."""
from __future__ import annotations

import inspect
import json
import re
import threading

import pytest

from liminal_libs.machines.endpoint import Reply

import plugin_platform_llamacpp as plugin
from plugin_platform_llamacpp import __main__ as cli
from plugin_platform_llamacpp import release
from conftest import KEY, load, local, router, running


def _tables(out: dict) -> dict:
    return {t["title"]: t for t in out.get("tables", [])}


def test_main_asks_no_server_and_runs_nothing(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p)
    p.add_instance("wsl", "found", None, 9940, False)
    p.add_instance("gpu", "remote", "https://gpu.example.org", None, False)
    world.transport.asked.clear()
    world.runner.ran.clear()
    heard, calls = list(world.heard), list(world.boot.calls)  # (adding asked this machine's ports: a click's)
    out = p.status()
    assert world.transport.asked == [] and world.runner.ran == []
    assert world.heard == heard and world.boot.calls == calls  # no port connected to, no boot registration read
    assert out["valid"] and out["summary"].startswith("llama.cpp 1.0.0 (b12359), its CUDA 13 build; 0 of 1 running")
    tables = _tables(out)
    assert {"Instances", "What leaves this machine", "What Install brings"} <= set(tables)
    assert "On this system" not in tables  # nothing lacks here (the C runtime and nvidia-smi are there)
    brings = {r[0]: r for r in tables["What Install brings"]["rows"]}
    assert brings["In use"][1] == "1.0.0 (b12359), its CUDA 13 build" and brings["Build"][1] == "Automatic"
    assert any(f["id"] == "ai-build" for f in out["forms"])


@pytest.mark.parametrize("system, wsl, arch, setup, rows", [
    ("windows", "", "AMD64", "no-runtime", {"Runtime"}),
    ("windows", "", "AMD64", "no-nvidia-smi", {"GPU"}),
    ("linux", "", "x86_64", "", {"System"}),
    ("linux", "Ubuntu", "x86_64", "no-nvidia-smi", {"GPU", "System", "This Linux"}),
    ("linux", "", "aarch64", "", {"Processor"}),
])
def test_on_this_system_says_what_applies_here(world, system, wsl, arch, setup, rows):
    if setup == "no-runtime":
        (world.tmp / "Windows" / "System32" / "vcruntime140.dll").unlink()
    if setup == "no-nvidia-smi":
        world.found.pop("nvidia-smi")
    p = world.plugin(system, wsl=wsl, arch=arch)
    table = release.lacks(p)
    assert {r[0] for r in table["rows"]} == rows
    said = {r[0]: r[2] for r in table["rows"]}
    if "GPU" in said:
        assert ("processor's build" in said["GPU"]) == bool(wsl)
    if "System" in said:
        assert "glibc 2.39" in {r[0]: r[1] for r in table["rows"]}["System"]


def test_at_boot_under_wsl_without_systemd_is_disabled_in_its_words(world, monkeypatch):
    p = world.plugin("linux", wsl="Ubuntu")
    world.installed(p)
    local(p)
    monkeypatch.setattr(p, "systemd_runs", lambda: False)
    buttons = {b["title"]: b for b in p.status()["tables"][0]["tasks"]["actions"]}
    assert buttons["At boot…"]["disabled"] and "systemd" in buttons["At boot…"]["note"]


def test_wsls_relay_on_llama_cpps_port_is_offered_as_a_found_instance(world):
    p = world.plugin("windows")
    world.probe.listen(5120, 9931, "wslrelay.exe")
    answering = _tables(p.status())["Answering on this machine"]
    [row] = answering["rows"]
    assert "WSL's relay (wslrelay.exe, pid 5120)" in row[1] and row[2]["values"] == {"name": "wsl", "where": "found",
                                                                                    "port": 9931}
    with pytest.raises(ValueError, match="held by WSL's relay .* Add it as a found instance, or choose another port"):
        local(p, "default")  # a server run here on the same port: the relay's conflict is said as it is added


def test_the_build_form_is_one_choice_confirmed(world):
    p = world.plugin("linux", wsl="Ubuntu")
    form = release.build_form(p)
    assert [c["value"] for c in form["fields"][0]["choices"]] == ["automatic", "cuda13", "cuda12", "vulkan", "cpu"]
    assert form["confirm"]["words"] == "The build is kept through your login and used by the next Install."
    assert form["fields"][0]["value"] == "automatic"


def _forms(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p)
    running(p, world)
    router(world)
    p.find("ggml-org/gemma-3-4b-it-GGUF")
    return p, [*p.status()["forms"], *p.models()["forms"]]


def test_every_forms_fields_are_its_hooks_arguments_in_order(world):
    p, forms = _forms(world)
    assert {"ai-build", "model-find", "model-pull", "model-delete", "model-load", "model-unload",
            "model-details"} <= {f["id"] for f in forms}
    for form in forms:
        hook = getattr(p, form["hook"])
        params = [x for x in inspect.signature(hook).parameters.values() if x.name != "log"]
        names = [f["name"] for f in form["fields"]]
        for k, name in enumerate(names):
            if k < len(params) and params[k].kind is inspect.Parameter.VAR_POSITIONAL:
                break  # the settings after a hook's *values
            assert k < len(params) and params[k].name == name, (form["id"], names, [x.name for x in params])
        placeholders = set(re.findall(r"\{(\w+)\}", json.dumps(form.get("confirm") or {})))
        assert placeholders <= set(names), (form["id"], placeholders)
        if form.get("job"):
            assert form["hook"] in p.JOB_HOOKS
    pull = next(f for f in forms if f["id"] == "model-pull")
    assert [f["name"] for f in pull["fields"]] == ["name", "model", "accept_licence"] and pull["job"] is True


def test_a_key_kept_is_in_no_answer_page_or_line(world, capsys):
    p = world.plugin("windows")
    world.installed(p)
    local(p)
    p.set_key("default", KEY)
    p.add_instance("gpu", "remote", "https://gpu.example.org", None, False)
    p.set_key("gpu", KEY)
    router(world, host="gpu.example.org")
    pages = [p.status(), p.models_of("gpu"), p.serving(), p.models(), p.api_facts()]
    p.choose("gpu")
    pages.append(p.models())
    assert not any(KEY in json.dumps(x, default=str) for x in pages)
    monkey = cli.MACHINE
    cli.MACHINE = p
    try:
        assert cli.main(["launch", "default"]) == 0
    finally:
        cli.MACHINE = monkey
    printed = capsys.readouterr().out
    assert "<the key kept for it>" in printed and KEY not in printed and "llama-server.exe" in printed


def test_the_command_line_prints_the_status_and_says_its_usage(world, capsys):
    p = world.plugin("windows")
    keep = cli.MACHINE
    cli.MACHINE = p
    try:
        assert cli.main(["status"]) == 0
        assert json.loads(capsys.readouterr().out)["valid"] is True
        assert cli.main(["nothing"]) == 2 and "python -m plugin_platform_llamacpp" in capsys.readouterr().err
        assert cli.main(["install"]) == 1  # its licence first: nothing is downloaded without --accept-licence
        assert "--accept-licence accepts it" in capsys.readouterr().err and world.opener.asked == []
    finally:
        cli.MACHINE = keep
    assert world.runner.ran == [] and world.transport.asked == []


def test_the_command_lines_install_says_its_size_hosts_and_licence_before_its_accept(world, capsys):
    """The review: `install` without --accept-licence printed the refusal alone, so `install --accept-licence` was
    accepted unseen. It prints what the page's confirmation says first: what is downloaded, from where, its size,
    what is checked and the licence; then the refusal. Nothing is downloaded or run."""
    p = world.plugin("windows")
    keep = cli.MACHINE
    cli.MACHINE = p
    try:
        assert cli.main(["install"]) == 1
    finally:
        cli.MACHINE = keep
    out = capsys.readouterr()
    shown = " ".join(out.out.split())
    pkg = p.package()
    for part in ("Install llama.cpp 1.0.0 (b12359)?", "github.com and release-assets.githubusercontent.com",
                 "about 678 MB", "checked against the size and sha256 pinned in this plugin", "MIT, llama.cpp's own",
                 "NVIDIA's CUDA runtime, under NVIDIA's terms", "Each model carries its own licence"):
        assert part in shown, part
    assert " ".join(pkg.parts.split()) in shown and "refused:" not in out.out
    assert out.err.startswith("refused: the licence of llama.cpp 1.0.0 (b12359) is not accepted yet")
    assert world.opener.asked == [] and world.runner.ran == [] and world.transport.asked == []


def test_detach_pauses_a_download_under_way(world):
    """The review: this test checked the panel's flag alone. A download is driven in a thread through a router whose
    event stream holds; a detach cancels it, and the router is asked to pause it (POST /models/unload), its part
    kept."""
    p = world.plugin("windows")
    world.installed(p)
    local(p, offline="off")
    running(p, world)
    router(world)
    p.models_of("default")
    model = "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
    world.transport.on("GET", "/api/models/ggml-org/gemma-3-4b-it-GGUF", (200, load("hub-model.json")),
                       "huggingface.co")
    world.transport.on("GET", "/api/models/ggml-org/gemma-3-4b-it-GGUF/tree/main", (200, load("hub-tree.json")),
                       "huggingface.co")
    p.find(model)
    under_way, done = threading.Event(), threading.Event()

    def held(*a):  # a stream that says one line of progress, then nothing until the download is over
        def chunks():
            yield ("data: " + json.dumps({"model": model, "event": "download_progress",
                                          "data": {"progress": {"a.gguf": {"done": 1, "total": 9}}}}) + "\n\n").encode()
            under_way.set()
            while not done.wait(0.01):
                yield b""
        return Reply(200, {"content-type": "text/event-stream"}, chunks())
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("POST", "/models/unload", (200, {"success": True}))
    world.transport.on("GET", "/models/sse", held)

    class Panel(list):
        def __init__(self):
            super().__init__()
            self._stop = threading.Event()

        def __call__(self, line):
            self.append(line)

        def cancel(self):
            self._stop.set()

        @property
        def cancelled(self):
            return self._stop.is_set()

        def wait_cancelled(self, timeout):
            return self._stop.wait(timeout)

    panel, answers = Panel(), []
    pulling = threading.Thread(target=lambda: answers.append(p.pull("default", model, True, log=panel)),
                               name="test-pull")
    pulling.start()
    try:
        assert under_way.wait(5), "the download did not get under way"
        p.detach()
        pulling.join(5)
    finally:
        done.set()
        pulling.join(5)
    assert not pulling.is_alive() and panel.cancelled and p.app is None
    [answer] = answers
    assert not answer["ok"] and answer["said"].startswith(f"cancelled: what is here of {model} is kept")
    paused = [json.loads(b) for m, u, _, b in world.transport.asked if m == "POST" and u.endswith("/models/unload")]
    assert paused == [{"model": model}]


def test_the_module_says_its_users_words_and_its_design():
    doc = plugin.__doc__
    assert "lama.cpp, ollama and vLLM" in doc and "all plugins are middleware" in doc
    assert "python -m plugin_platform_llamacpp status" in doc
