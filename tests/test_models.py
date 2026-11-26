"""Models (models.py): a repository's files picked by llama.cpp's own rules (the quantization named, its defaults, a
split file's parts, its projector, no importance matrix); Find a model at the Hub (found, unknown, no answer, a
licence of its own, gated, paged through its Link header and never to another host, kept an hour and 12 at most,
cleared; the quantization typed kept; nothing offered where the Hub says there is nothing, a download unseen where
it did not answer); a download as a job (each refusal before its first line, nothing asked for them, a server
started offline among them; its size and licence asked twice from what is kept alone; its progress; finished,
failed, paused from elsewhere, cancelled and paused, a stream gone quiet, a router that stops answering); delete,
load, unload and details with their refusals (the context the harnesses are told); each model's cells and the Find
table, a button that does not apply disabled with its note."""
from __future__ import annotations

import json
import threading

import pytest

from liminal_libs.instances import Refusal
from liminal_libs.machines.ai import ACCEPT_LICENCE

from plugin_platform_llamacpp import models
from plugin_platform_llamacpp.models import File, pick
from conftest import load, local, reply, router, running, sse

REPO = "ggml-org/gemma-3-4b-it-GGUF"
TARGET = REPO + ":Q4_K_M"
HUB = "huggingface.co"


class Panel:
    def __init__(self, cancel_on: str = ""):
        self.lines: list[str] = []
        self._cancel = threading.Event()
        self.cancel_on = cancel_on

    def __call__(self, line):
        self.lines.append(str(line))
        if self.cancel_on and self.cancel_on in str(line):
            self._cancel.set()

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def wait_cancelled(self, timeout):
        return self.cancelled


def refusal(call) -> Refusal:
    with pytest.raises(Refusal) as caught:
        call()
    return caught.value


def hub(world, repo: str = REPO, card: dict | None = None, tree: list | None = None, headers: dict | None = None):
    world.transport.on("GET", f"/api/models/{repo}", (200, card if card is not None else load("hub-model.json")), HUB)
    world.transport.on("GET", f"/api/models/{repo}/tree/main",
                       (200, tree if tree is not None else load("hub-tree.json"), headers or {}), HUB)


def serving_here(world, offline="off", **settings):
    """A router run here and running, asked once by its Models tab (its kind and build kept): online unless asked,
    so a download can go through it."""
    p = world.plugin("windows")
    world.installed(p)
    local(p, offline=offline, **settings)
    running(p, world)
    router(world)
    p.models_of("default")
    return p


def _download_bodies(world) -> list[dict]:
    return [json.loads(b) for m, u, _, b in world.transport.asked if m == "POST" and u.endswith("/models")]


def _find_row(p) -> list:
    table = next(t for t in p.models()["tables"] if t["title"] == "Find a model")
    [row] = table["rows"]
    return row


# -- llama.cpp's own way of picking files -----------------------------------------------------------------------------
def _files(name: str) -> list[File]:
    return [File(r["path"], r["lfs"]["size"] if "lfs" in r else r["size"]) for r in load(name) if r["type"] == "file"]


def test_the_files_are_picked_as_llama_cpp_picks_them():
    files = _files("hub-tree.json")
    chosen, mmproj, quant = pick(files, "")
    assert [f.path for f in chosen] == ["gemma-3-4b-it-Q4_K_M.gguf"] and quant == "Q4_K_M"
    assert mmproj.path == "mmproj-model-f16.gguf"
    assert [f.path for f in pick(files, "q8_0")[0]] == ["gemma-3-4b-it-Q8_0.gguf"]
    assert pick(files, "F16")[2] == "F16" and pick(files, "Q2_K") == ([], None, "")
    assert pick([f for f in files if "Q4" not in f.path and "Q8" not in f.path], "")[0][0].path.endswith("f16.gguf")
    assert not any("imatrix" in f.path for f in pick(files, "")[0])


def test_a_split_file_brings_every_part_and_the_nearest_projector():
    files = _files("hub-tree-split.json")
    chosen, mmproj, quant = pick(files, "Q4_K_M")
    assert [f.path for f in chosen] == ["Q4_K_M/big-Q4_K_M-00001-of-00002.gguf",
                                       "Q4_K_M/big-Q4_K_M-00002-of-00002.gguf"]
    assert mmproj.path == "mmproj-Q8_0.gguf" and quant == "Q4_K_M"  # the projector nearest Q4 of the two


@pytest.mark.parametrize("typed", ["", "gemma", "a/b/c", "x/../y", "user/repo:Q4 K", "https://huggingface.co/a/b"])
def test_a_name_that_is_no_model_is_refused_in_words(typed):
    assert "user/" in refusal(lambda: models.clean_model(typed)).args[0]


# -- Find a model -----------------------------------------------------------------------------------------------------
def test_a_lookup_reads_the_card_and_the_files_and_is_kept(world):
    p = world.plugin("windows")
    hub(world)
    said = p.find(REPO)
    assert said["ok"] and said["said"] == f"{TARGET}: 3.1 GB to download in 2 files; licence: gemma"
    assert said["lines"][0].startswith("fits: 3.1 GB of the NVIDIA GeForce RTX 5090 Laptop GPU's")
    assert "llama.cpp fits a model by itself" in said["lines"][0] and "read at huggingface.co" in said["lines"][1]
    found = p.found_of(REPO)
    assert (found.model, found.size, found.files, found.mmproj, found.gated) == (TARGET, 3341008960, 2,
                                                                                "mmproj-model-f16.gguf", False)
    assert found.licence_page == "https://ai.google.dev/gemma/terms" and p.found_of(TARGET) is found
    _, url, headers, _ = world.transport.asked[0]
    assert url == f"https://{HUB}/api/models/{REPO}" and "Authorization" not in headers
    assert headers["User-Agent"] == "plugin-platform-llamacpp/0.1.1"
    assert world.transport.asked[1][1] == f"https://{HUB}/api/models/{REPO}/tree/main?recursive=true"


def test_an_unknown_model_or_no_answer_is_kept_with_why(world):
    p = world.plugin("windows")
    said = p.find("someone/nothing-GGUF")
    assert not said["ok"] and "knows no such model, or it is private" in said["said"]
    assert p.found_of("someone/nothing-GGUF").why and p.found_of("someone/nothing-GGUF").absent
    world.transport.on("GET", f"/api/models/{REPO}", ConnectionRefusedError("refused"), HUB)
    said = p.find(REPO)
    assert not said["ok"] and "it did not answer" in said["said"] and not p.found_of(REPO).absent


@pytest.mark.parametrize("answer", [(429, {"error": "Too many requests"}), ConnectionRefusedError("refused")])
def test_a_hub_that_did_not_answer_keeps_the_quantization_typed_for_a_download_unseen(world, answer):
    """The review: a lookup the Hub did not answer kept the repository alone, so a download unseen fetched llama.cpp's
    default quantization instead of the one typed, and without its projector. Now it keeps what was typed, says it
    would be downloaded unseen, and leaves the sidecars to the router (its default brings the projector)."""
    p = serving_here(world)
    world.transport.on("GET", "/api/models/bartowski/X-GGUF", answer, HUB)
    said = p.find("bartowski/X-GGUF:q8_0")
    found = p.found_of("bartowski/X-GGUF:Q8_0")
    assert not said["ok"] and (found.model, found.asked) == ("bartowski/X-GGUF:Q8_0", "bartowski/X-GGUF:q8_0")
    button = _find_row(p)[4]
    assert button["values"]["model"] == "bartowski/X-GGUF:Q8_0" and not button.get("disabled")
    asked = refusal(lambda: p.pull("default", button["values"]["model"]))
    assert asked.accept == ACCEPT_LICENCE and "bartowski/X-GGUF:Q8_0's size or licence" in asked.args[0]
    assert "it would be downloaded unseen" in asked.args[0]
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("GET", "/models/sse",
                       lambda *a: sse({"model": "bartowski/X-GGUF:Q8_0", "event": "download_finished"}))
    assert p.pull("default", button["values"]["model"], True, log=Panel())["ok"]
    assert _download_bodies(world) == [{"model": "bartowski/X-GGUF:Q8_0"}]  # the router's own sidecars


@pytest.mark.parametrize("typed, kept, why", [
    ("bartowski/X-GGUF:Q8_0", "bartowski/X-GGUF:Q8_0", "knows no such model, or it is private"),
    (REPO + ":Q2_K", REPO + ":Q2_K", f"{REPO} holds no GGUF file llama.cpp takes for Q2_K"),
])
def test_nothing_is_offered_for_download_where_the_hub_says_there_is_nothing(world, typed, kept, why):
    """The review: a lookup the Hub answered with no such model (404), or a repository without the quantization
    typed, still offered Download… "unseen", which fetched another file. Now nothing that was asked can be
    downloaded: Download… is off with its why, and a download is refused in words, nothing asked."""
    p = serving_here(world)
    hub(world)  # (the repository llama.cpp's own picks are read from; another one is not known)
    said = p.find(typed)
    found = p.found_of(typed)
    assert not said["ok"] and found.absent and found.model == kept and why in found.why
    assert "Download… is off for it" in said["lines"][0]
    button = _find_row(p)[4]
    assert button["disabled"] and why in button["note"] and "nothing to download" in button["note"]
    world.transport.asked.clear()
    words = refusal(lambda: p.pull("default", kept, True)).args[0]
    assert why in words and "nothing that was asked can be downloaded" in words and world.transport.asked == []


def test_a_licence_of_its_own_and_a_gated_model_are_said(world):
    p = world.plugin("windows")
    card = load("hub-model.json")
    card["cardData"] = {"license": "other", "license_name": "llama3.1"}
    card["gated"] = "manual"
    hub(world, card=card)
    said = p.find(REPO)
    assert said["ok"] and "licence: llama3.1 (its own)" in said["said"] and "gated:" in said["lines"][0]


def test_the_files_are_read_page_by_page_and_never_from_another_host(world):
    p = world.plugin("windows")
    first = load("hub-tree.json")[:2]
    rest = load("hub-tree.json")[2:]
    link = f'<https://{HUB}/api/models/{REPO}/tree/main?recursive=true&cursor=Mg>; rel="next"'
    pages = [reply(200, first, {"link": link}), reply(200, rest)]
    world.transport.on("GET", f"/api/models/{REPO}", (200, load("hub-model.json")), HUB)
    world.transport.on("GET", f"/api/models/{REPO}/tree/main", pages, HUB)
    assert p.find(REPO)["ok"] and p.found_of(REPO).files == 2
    assert world.transport.asked[-1][1].endswith("cursor=Mg")
    world.transport.asked.clear()
    elsewhere = '<https://elsewhere.example/api/models/x/tree/main?cursor=Mg>; rel="next"'
    world.transport.on("GET", f"/api/models/{REPO}/tree/main", (200, first, {"link": elsewhere}), HUB)
    p.find(REPO)
    assert all("elsewhere" not in url for _, url, _, _ in world.transport.asked)


def test_lookups_are_kept_an_hour_twelve_at_most_and_cleared(world):
    p = world.plugin("windows")
    for k in range(13):
        hub(world, repo=f"ggml-org/m{k}-GGUF")
        p.find(f"ggml-org/m{k}-GGUF")
    kept = models.lookups(p)
    assert len(kept) == 12 and kept[0].asked == "ggml-org/m12-GGUF" and p.found_of("ggml-org/m0-GGUF") is None
    world.clock.now += 3601
    assert models.lookups(p) == []
    hub(world)
    p.find(REPO)
    assert p.clear_found()["said"] == "1 lookup forgotten" and models.lookups(p) == []


# -- the download -----------------------------------------------------------------------------------------------------
def _events(*kinds, model=TARGET):
    out = []
    for kind in kinds:
        if kind == "progress":
            out.append({"model": model, "event": "download_progress",
                        "data": {"progress": {"https://huggingface.co/a.gguf": {"done": 1 << 30, "total": 2489757856},
                                              "https://huggingface.co/m.gguf": {"done": 0, "total": 851251104}}}})
        elif kind == "other":
            out.append({"model": "someone/else:Q8_0", "event": "download_finished"})
        elif kind == "finished":
            out.append({"model": model, "event": "download_finished"})
        elif kind == "failed":
            out.append({"model": model, "event": "download_failed"})
        elif kind == "stopped":
            out.append({"model": model, "event": "download_failed", "data": {"stopped": True}})
    return out


def test_a_download_is_refused_while_its_server_is_started_offline(world):
    """The review's high finding, its Download… side: a router started offline (Offline on, the default) downloads
    nothing, so Download… is off with its why, and a download is refused in words before its first line, nothing
    asked; with Offline off (Change…, then started again), it goes through."""
    p = world.plugin("windows")
    world.installed(p)
    local(p)
    running(p, world)
    router(world)
    p.models_of("default")
    hub(world)
    p.find(REPO)
    world.transport.asked.clear()
    words = refusal(lambda: p.pull("default", REPO, True)).args[0]
    assert words.startswith("default is started offline (Offline is on in its settings)")
    assert "turn Offline off with Change… while it is stopped, and start it again" in words
    assert world.transport.asked == []
    button = _find_row(p)[4]
    assert button["disabled"] and button["note"] == words
    p.stop("default")
    world.probe.conns.clear()  # (it no longer listens)
    world.clock.now += 6
    p.change_instance("default", None, None, False, *[("off" if s.name == "offline" else None) for s in p.SETTINGS])
    assert p.instance("default").settings["offline"] == "off"
    assert "LLAMA_ARG_OFFLINE" not in p.prepared(p.instance("default")).env
    running(p, world)
    p.models_of("default")
    assert not _find_row(p)[4].get("disabled")


def test_a_download_is_refused_before_its_first_line_from_what_is_kept_alone(world, monkeypatch):
    p = world.plugin("windows")
    world.installed(p)
    local(p, offline="off")
    hub(world)
    assert "start default first: llama.cpp downloads a model through its running server" in refusal(
        lambda: p.pull("default", REPO)).args[0]
    running(p, world)
    router(world)
    words = refusal(lambda: p.pull("default", REPO)).args[0]
    assert "look ggml-org/gemma-3-4b-it-GGUF up first (Find a model)" in words
    p.find(REPO)
    world.transport.asked.clear()
    asked = refusal(lambda: p.pull("default", REPO))
    assert asked.accept == ACCEPT_LICENCE and world.transport.asked == []  # its words from the lookup, nothing asked
    words = asked.args[0]
    for part in (f"{TARGET}: 3.1 GB in 2 files from huggingface.co", f"into {p.home / 'models'} (500 GB free)",
                 "Licence: gemma; its words are at https://ai.google.dev/gemma/terms"):
        assert part in words, words
    monkeypatch.setattr(models.release, "disk_free", lambda path: 1 << 30)
    assert "needs 3.1 GB" in refusal(lambda: p.pull("default", REPO)).args[0]


def test_a_found_instance_not_asked_yet_one_models_server_and_an_old_build_are_refused(world):
    p = world.plugin("windows")
    hub(world)
    p.find(REPO)
    p.add_instance("wsl", "found", None, 9940, False)
    assert "wsl has not been asked yet" in refusal(lambda: p.pull("wsl", REPO)).args[0]
    world.transport.on("GET", "/health", (200, {"status": "ok"}, {"server": "llama.cpp"}))
    world.transport.on("GET", "/props", (200, load("one-props.json")))
    world.transport.on("GET", "/v1/models", (200, load("one-models.json")))
    p.models_of("wsl")
    assert "wsl serves one model" in refusal(lambda: p.pull("wsl", REPO)).args[0]
    props = load("router-props.json")
    props["build_info"] = "b9000-abc1234"
    router(world)
    world.transport.on("GET", "/props", (200, props, {"server": "llama.cpp"}))
    p.models_of("wsl")
    assert "(b9000) is older than b9688" in refusal(lambda: p.pull("wsl", REPO)).args[0]


def test_a_gated_model_is_never_downloaded(world):
    p = serving_here(world)
    card = load("hub-model.json")
    card["gated"] = "auto"
    hub(world, card=card)
    p.find(REPO)
    assert "gated" in refusal(lambda: p.pull("default", REPO, True)).args[0]


def test_a_download_follows_the_routers_events_to_its_end(world):
    p = serving_here(world)
    hub(world)
    p.find(REPO)
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("GET", "/models/sse", lambda *a: sse(*_events("progress", "other", "finished")))
    panel = Panel()
    said = p.pull("default", REPO, True, log=panel)
    assert said["ok"] and said["said"] == f"{TARGET} is here (3.1 GB)"
    body = next(json.loads(b) for m, u, _, b in world.transport.asked if m == "POST" and u.endswith("/models"))
    assert body == {"model": TARGET, "sidecars": ["mmproj"]}
    assert any(line.startswith(f"{TARGET}: 1.0 GB of 3.1 GB (32%)") for line in panel.lines)
    assert "No Hugging Face token is given to it" in panel.lines[0]
    assert p.seen("default") is None  # forgotten: the next look asks again


@pytest.mark.parametrize("kind, said", [("failed", "says its download failed"),
                                        ("stopped", "was paused from elsewhere")])
def test_a_download_that_fails_or_is_paused_elsewhere_is_said(world, kind, said):
    p = serving_here(world)
    hub(world)
    p.find(REPO)
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("GET", "/models/sse", lambda *a: sse(*_events("progress", kind)))
    got = p.pull("default", REPO, True, log=Panel())
    assert not got["ok"] and said in got["said"]


def test_a_cancel_pauses_the_download_and_keeps_its_part(world):
    p = serving_here(world)
    hub(world)
    p.find(REPO)
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("POST", "/models/unload", (200, {"success": True}))
    world.transport.on("GET", "/models/sse", lambda *a: sse(*_events("progress", "progress", "finished")))
    got = p.pull("default", REPO, True, log=Panel(cancel_on="%"))
    assert not got["ok"] and got["said"] == (f"cancelled: what is here of {TARGET} is kept, and the next download goes "
                                             "on from it")
    unload = [json.loads(b) for m, u, _, b in world.transport.asked if m == "POST" and u.endswith("/models/unload")]
    assert unload == [{"model": TARGET}]


def test_a_quiet_stream_is_followed_by_asking_the_router(world):
    p = serving_here(world)
    hub(world)
    p.find(REPO)
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("GET", "/models/sse", lambda *a: sse(*_events("progress"), then=TimeoutError("timed out")))
    assert p.pull("default", REPO, True, log=Panel())["ok"]  # the router lists it whole (router-models.json)
    assert world.transport.paths()[-1] == "GET /models"


def test_a_router_that_stops_answering_ends_the_download_in_words(world):
    p = serving_here(world)
    hub(world)
    p.find(REPO)
    world.transport.on("POST", "/models", (200, {"success": True}))
    world.transport.on("GET", "/models/sse", lambda *a: ConnectionRefusedError("refused"))
    world.transport.on("GET", "/models", ConnectionRefusedError("refused"))
    got = p.pull("default", REPO, True, log=Panel())
    assert not got["ok"] and "stopped answering while it downloaded" in got["said"]


def test_a_remote_instances_download_is_said_on_its_host(world):
    p = world.plugin("windows")
    p.add_instance("gpu", "remote", "https://gpu.example.org", None, False)
    router(world, host="gpu.example.org")
    p.models_of("gpu")
    hub(world)
    p.find(REPO)
    words = refusal(lambda: p.pull("gpu", REPO)).args[0]
    assert "into gpu's models folder on gpu.example.org" in words and "free" not in words


# -- delete, load, unload, details -----------------------------------------------------------------------------------
def test_delete_is_refused_for_what_llama_cpp_did_not_download_and_what_is_loaded(world):
    p = serving_here(world)
    assert "in your GGUF folder: llama.cpp deletes only what it downloaded" in refusal(
        lambda: p.delete_model("default", "my-coder-Q5_K_M")).args[0]
    assert "unload it first" in refusal(lambda: p.delete_model("default", TARGET)).args[0]
    assert "Cancel its download first" in refusal(
        lambda: p.delete_model("default", "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M")).args[0]
    assert "is not listed on default" in refusal(lambda: p.delete_model("default", "x/y:Q4_K_M")).args[0]
    world.transport.on("DELETE", "/models", (200, {"success": True}))
    said = p.delete_model("default", "ggml-org/qwen3-8b-gguf:q4_k_m")
    assert said["ok"] and said["said"] == "ggml-org/Qwen3-8B-GGUF:Q4_K_M is deleted from default"
    _, url, _, _ = next(a for a in world.transport.asked if a[0] == "DELETE")
    assert url.endswith("/models?model=ggml-org%2FQwen3-8B-GGUF%3AQ4_K_M")


def test_load_and_unload_ask_the_router(world):
    p = serving_here(world)
    world.transport.on("POST", "/models/load", (200, {"success": True}))
    world.transport.on("POST", "/models/unload", (200, {"success": True}))
    assert p.load("default", "ggml-org/Qwen3-8B-GGUF:Q4_K_M")["said"].endswith("is loading: Models shows when it is "
                                                                              "loaded")
    assert p.load("default", TARGET)["said"] == f"{TARGET} is loaded on default already"
    assert "its download is under way" in refusal(
        lambda: p.load("default", "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M")).args[0]
    assert p.unload("default", TARGET)["said"] == f"{TARGET} is unloaded from default"
    posted = [(u.rsplit("/", 1)[-1], json.loads(b)) for m, u, _, b in world.transport.asked if m == "POST"]
    assert posted == [("load", {"model": "ggml-org/Qwen3-8B-GGUF:Q4_K_M"}), ("unload", {"model": TARGET})]


def test_details_say_the_listing_and_the_loaded_models_props(world):
    p = serving_here(world)
    props = {"model_path": "C:/models/gemma-3-4b-it-Q4_K_M.gguf", "total_slots": 2,
             "default_generation_settings": {"n_ctx": 32768}, "modalities": {"vision": True}}
    world.transport.on("GET", "/props", (200, props, {"server": "llama.cpp"}))
    said = p.details("default", TARGET)
    assert said["ok"] and "file: C:/models/gemma-3-4b-it-Q4_K_M.gguf" in said["lines"]
    assert "context: 32,768 tokens" in said["lines"] and "takes in: text, vision" in said["lines"]
    _, url, _, _ = world.transport.asked[-1]
    assert "autoload=false" in url and "model=ggml-org%2Fgemma-3-4b-it-GGUF%3AQ4_K_M" in url
    said = p.details("default", "ggml-org/Qwen3-8B-GGUF:Q4_K_M")
    assert any("is not loaded: its file, its context and its slots are said once it is" in x for x in said["lines"])


def test_details_say_what_context_the_harnesses_are_told(world):
    """The review: a router started here reports no loaded model's context, so the harnesses are told one only where
    Context size is set; Details says which."""
    p = serving_here(world)
    props = {**load("router-props.json"), "default_generation_settings": {"n_ctx": 32768}}  # (the loaded model's)
    world.transport.on("GET", "/props", (200, props, {"server": "llama.cpp"}))
    lines = p.details("default", TARGET)["lines"]
    assert "context: 32,768 tokens" in lines
    told = next(x for x in lines if x.startswith("the harnesses are told"))
    assert "no context for it" in told and "Context size" in told and "Change…" in told
    p.stop("default")
    world.probe.conns.clear()
    world.clock.now += 6
    p.change_instance("default", None, None, False,
                      *[{"context": 16384, "offline": "off"}.get(s.name) for s in p.SETTINGS])
    running(p, world)
    p.models_of("default")
    told = next(x for x in p.details("default", TARGET)["lines"] if x.startswith("the harnesses are told"))
    assert told == "the harnesses are told its context is 16,384 tokens (its Context size)"


# -- what the Models tab draws ---------------------------------------------------------------------------------------
def test_each_models_cells_say_what_does_not_apply(world):
    p = serving_here(world)
    page = p.models()
    rows = {r[0]["text"] if isinstance(r[0], dict) else r[0]: r for r in page["tables"][0]["rows"]}
    details, load_, delete = rows[TARGET][-3:]
    assert details["form"] == "model-details" and load_["title"] == "Unload…" and delete["note"] == "unload it first"
    assert rows["my-coder-Q5_K_M"][-1]["note"].startswith("in your GGUF folder")
    downloading = rows["unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"]
    assert downloading[-2]["note"] == "its download is under way" and "Cancel it first" in downloading[-1]["note"]
    assert "Download… again goes on from it" in rows["bartowski/Mistral-7B-GGUF:Q8_0"][-2]["note"]
    qwen = rows["ggml-org/Qwen3-8B-GGUF:Q4_K_M"]
    assert qwen[-2]["title"] == "Load…" and "disabled" not in qwen[-2] and "disabled" not in qwen[-1]


def test_the_find_table_is_drawn_whether_or_not_an_instance_answers(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p, offline="off")
    hub(world)
    p.find(REPO)
    page = p.models()
    table = next(t for t in page["tables"] if t["title"] == "Find a model")
    assert table["search"] == {"form": "model-find", "field": "model", "value": REPO,
                               "placeholder": "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"}
    [row] = table["rows"]
    assert row[1] == "3.1 GB" and row[2] == "gemma" and row[3]["text"].startswith("fits: 3.1 GB")
    assert row[4]["disabled"] and row[4]["note"].startswith("start default first")
    assert {f["id"] for f in page["forms"]} >= {"model-find", "model-pull", "model-delete", "model-load",
                                                "model-unload", "model-details"}
    # a stopped one is not asked
    assert not any(u.startswith("http://127.0.0.1") for _, u, _, _ in world.transport.asked)
