"""A server run here and how one is recognised (server.py): the launch line and its environment for each setting,
the bind address shared and not, the key in the environment alone and in no line to paste, what is dropped (the
tool's own variables, ggml's, Hugging Face's, every pair of Liminal's), every refusal; a start through the base;
look() over a router, one model's server, something that is not llama.cpp, one behind a proxy, one that wants its
key, one loading, one whose remote access is on (its share code never kept); the start and share words; what leaves
this machine, a config.ini of yours named where one exists; and what the harnesses are told (serving)."""
from __future__ import annotations

import pytest

from liminal_libs.instances import Refusal
from liminal_libs.machines import children
from liminal_libs.machines.endpoint import Endpoint

from plugin_platform_llamacpp import release, server
from conftest import KEY, load, local, router, running


def refusal(call) -> Refusal:
    with pytest.raises(Refusal) as caught:
        call()
    return caught.value


# -- the launch -------------------------------------------------------------------------------------------------------
def test_a_server_run_here_is_a_router_on_this_machine_alone(world):
    p = world.plugin("windows")
    folder = world.installed(p)
    local(p)
    launch = p.prepared(p.instance("default"))
    assert launch.argv == (str(folder / "llama-server.exe"), "--host", "127.0.0.1", "--port", "9931", "--models-max",
                           "4", "--sleep-idle-seconds", "300")
    assert launch.env == {"LLAMA_CACHE": str(p.home / "models"), "LLAMA_ARG_CORS_ORIGINS": "http://127.0.0.1:9931"}
    assert launch.cwd == str(p.home) and launch.secret == ()
    for word in ("--connect", "--tools", "--agent", "--metrics", "--log-prompts-dir", "--slot-save-path", "-m"):
        assert word not in launch.argv
    env = children.environment(launch, base=world.env("windows"), system="windows")
    for gone in ("LLAMA_ARG_CONNECT", "LLAMA_API_KEY", "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "HF_TOKEN", "HF_ENDPOINT",
                 "MASSIVE_API_KEY", "LIMINAL_AI_LLAMACPP_INSTANCES", "LIMINAL_PLATFORM_DIR"):
        assert gone not in env, gone
    assert env["LLAMA_CACHE"] == str(p.home / "models") and env["CUDA_VISIBLE_DEVICES"] == "0"  # the user's card kept


def test_each_setting_becomes_its_switch(world, tmp_path):
    p = world.plugin("linux")
    world.installed(p)
    gguf, models = tmp_path / "gguf", tmp_path / "store"
    local(p, "coder", context=32768, layers="ALL", most=2, idle=0, gguf=str(gguf), models=str(models))
    launch = p.prepared(p.instance("coder"))
    assert launch.argv[1:] == ("--host", "127.0.0.1", "--port", "9931", "--models-max", "2", "-c", "32768", "-ngl",
                               "all", "--models-dir", str(gguf))
    assert "--sleep-idle-seconds" not in launch.argv  # 0 is never: llama-server refuses the switch at 0
    assert launch.env["LLAMA_CACHE"] == str(models)


def test_shared_it_listens_on_every_address_and_its_key_is_in_its_environment_alone(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p)
    p.set_key("default", KEY)
    assert p.share("default", "yes", True)["ok"]
    launch = p.prepared(p.instance("default"))
    assert launch.argv[1:3] == ("--host", "0.0.0.0") and launch.env["LLAMA_API_KEY"] == KEY
    assert launch.secret == ("LLAMA_API_KEY",) and not any(KEY in a for a in launch.argv)
    lines = children.shown(launch, "windows", base={"LLAMA_ARG_CONNECT": "1"})
    assert "Remove-Item Env:LLAMA_ARG_CONNECT -ErrorAction SilentlyContinue" in lines
    assert "$env:LLAMA_API_KEY = '<the key kept for it>'" in lines and not any(KEY in x for x in lines)
    assert "llama.cpp's key guards its API" in p.share_words(p.instance("default"))


def test_a_setting_typed_that_cannot_be_one_is_refused_as_it_is_added(world):
    p = world.plugin("windows")
    world.installed(p)
    assert "auto, all or a whole number" in refusal(lambda: local(p, layers="most")).args[0]
    assert "whole path" in refusal(lambda: local(p, gguf="models/mine")).args[0]
    assert "from 1 to 16" in refusal(lambda: local(p, most=40)).args[0]


def test_the_launch_refuses_what_cannot_be_started(world, monkeypatch):
    p = world.plugin("windows")
    local(p)
    assert "Install… first" in refusal(lambda: p.start("default")).args[0]
    world.found["llama-server"] = "C:/old/llama-server.exe"
    world.runner.version = (0, "", "version: 6900 (1a2b3c4)\nbuilt with cc\n")  # an old build, as it said it then
    p.refresh()
    words = refusal(lambda: p.prepared(p.instance("default"))).args[0]
    assert "this llama.cpp (b6900) cannot serve without a model" in words and "b7218" in words
    world.found.pop("llama-server")
    world.installed(p)
    monkeypatch.setattr(server.mounts, "foreign", lambda path, *a, **k: "elsewhere" in str(path))
    p.change_instance("default", None, None, False, str(world.tmp / "elsewhere"))
    assert "on another system's drive" in refusal(lambda: p.prepared(p.instance("default"))).args[0]


def test_a_found_llama_is_started_with_serve(world):
    p = world.plugin("linux")
    world.found["llama"] = "/home/me/.local/bin/llama"
    local(p)
    assert p.prepared(p.instance("default")).argv[:3] == ("/home/me/.local/bin/llama", "serve", "--host")


def test_a_start_goes_through_the_base_as_a_windowless_child(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p)
    said = p.start("default")
    assert said["ok"] and "started (pid" in said["said"]
    child = world.kids.made[-1]
    assert child.launch.argv[0].endswith("llama-server.exe") and "LLAMA_*" in child.launch.drop
    assert "MASSIVE_API_KEY" in child.launch.drop and "LIMINAL_*" in child.launch.drop


def test_the_start_words_say_the_models_ports_and_what_reaches_the_network(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p, most=2, idle=600)
    out = p.status()
    start = next(f for f in out["forms"] if f["id"] == "ai-start")["confirm"]["words"]
    for words in ("starts as a router, without a model", "at most 2 at once", "unloaded after 600 s",
                  "a port of this machine alone (127.0.0.1) that llama.cpp chooses itself: the tool's doing, not a "
                  "setting", "No other site's page can read it", "to ask huggingface.co for that model's list"):
        assert words in start, words


# -- recognising a server ---------------------------------------------------------------------------------------------
def _look(p, world, host="127.0.0.1", port=9931):
    return server.look(p, Endpoint(f"http://{host}:{port}", transport=world.transport), p.instance("default"))


def test_a_router_is_recognised_with_its_models(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p, context=16384)
    router(world)
    seen = _look(p, world)
    assert seen.ok and seen.version == "b12359" and dict(seen.facts) == {"router": "yes", "build": "b12359",
                                                                        "connect": "off"}
    assert "a router of build 12359, 5 models listed" in seen.said
    models = {m["id"]: m for m in seen.models}
    gemma = models["ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"]
    assert (gemma["loaded"], gemma["size"], gemma["context"], gemma["train"]) == (True, 2489757856, 32768, 131072)
    assert gemma["takes"] == ["text", "image"] and gemma["sidecars"] == ["mmproj"]
    qwen = models["ggml-org/Qwen3-8B-GGUF:Q4_K_M"]
    assert (qwen["loaded"], qwen["size"], qwen["context"]) == (False, None, 16384)  # the context it is run with
    mine = models["my-coder-Q5_K_M"]
    assert mine["detail"] == "from your GGUF folder" and not mine["can_remove"]
    assert models["unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"]["detail"] == "downloading 50%"
    assert models["bartowski/Mistral-7B-GGUF:Q8_0"]["detail"] == "incomplete"
    assert world.transport.paths() == ["GET /health", "GET /props", "GET /models"]


def test_one_models_server_is_recognised_by_its_v1_models(world):
    p = world.plugin("windows")
    local(p)
    world.transport.on("GET", "/health", (200, {"status": "ok"}, {"server": "llama.cpp"}))
    world.transport.on("GET", "/props", (200, load("one-props.json")))
    world.transport.on("GET", "/v1/models", (200, load("one-models.json")))
    seen = _look(p, world)
    assert seen.ok and seen.facts["router"] == "no: one model" and seen.version == "b9000"
    [model] = seen.models
    assert (model["id"], model["size"], model["context"], model["loaded"]) == ("coder", 4920000000, 8192, True)


@pytest.mark.parametrize("health, props, said", [
    ((200, {"status": "ok"}, {"server": "nginx/1.27"}), (404, {"error": "no"}),
     "is not llama.cpp (it says it is nginx"),
    ((404, {"error": "no"}), None, "is not llama.cpp (its /health says HTTP 404)"),
    ((200, {"status": "ok"}, {"server": "llama.cpp"}), (401, {"error": {"message": "Invalid API Key"}}),
     "it wants its key (Key…)"),
])
def test_what_is_not_llama_cpp_or_wants_a_key_is_said(world, health, props, said):
    p = world.plugin("windows")
    local(p)
    world.transport.on("GET", "/health", health)
    if props is not None:
        world.transport.on("GET", "/props", props)
    seen = _look(p, world)
    assert not seen.ok and said in seen.said, seen.said


def test_a_server_behind_a_proxy_is_recognised_by_its_props(world):
    p = world.plugin("windows")
    local(p)
    router(world)
    world.transport.on("GET", "/health", (200, {"status": "ok"}, {"server": "Caddy"}))
    seen = _look(p, world)
    assert seen.ok and seen.facts["router"] == "yes"


def test_a_server_loading_its_model_is_said(world):
    p = world.plugin("windows")
    local(p)
    world.transport.on("GET", "/health", (503, {"error": {"message": "Loading model"}}, {"server": "llama.cpp"}))
    world.transport.on("GET", "/props", (503, {"error": {"message": "Loading model"}}, {"server": "llama.cpp"}))
    seen = _look(p, world)
    assert seen.ok and "it is loading a model" in seen.said


def test_a_share_code_is_never_kept_only_said_as_remote_access_on(world):
    p = world.plugin("windows")
    local(p)
    router(world)
    props = load("router-props.json")
    props["connect_code"] = "0123456789abcdef0123456789abcdef01234567"
    world.transport.on("GET", "/props", (200, props, {"server": "llama.cpp"}))
    seen = _look(p, world)
    assert seen.facts["connect"] == "on" and "remote access (--connect) is on" in seen.said
    assert "0123456789abcdef" not in repr(seen)


def test_a_remote_instance_is_asked_at_its_own_address_and_key(world):
    p = world.plugin("windows")
    p.add_instance("gpu", "remote", "https://gpu.example.org:8443/llama", None, False)
    p.set_key("gpu", KEY)
    router(world, host="gpu.example.org", prefix="/llama")
    got = p.models_of("gpu")
    assert got["ok"] and {m["id"] for m in got["models"]} >= {"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"}
    method, url, headers, _ = world.transport.asked[0]
    assert url == "https://gpu.example.org:8443/llama/health" and headers["Authorization"] == f"Bearer {KEY}"
    assert KEY not in repr(got)


# -- what leaves this machine, and what the harnesses are told ------------------------------------------------------
def test_what_leaves_this_machine_names_a_config_ini_of_yours_where_one_exists(world):
    p = world.plugin("windows")
    rows = {r[0]: r for r in server.leaves(p)["rows"]}
    assert rows["Telemetry and update checks"][1] == "none" and "A config.ini of yours" not in rows
    assert "asks huggingface.co" in rows["Loading a downloaded model"][1]
    mine = world.tmp / "AppData" / "Roaming" / "llama.cpp" / "config.ini"
    mine.parent.mkdir(parents=True)
    mine.write_text("[*]\nconnect = true\n", encoding="utf-8")
    rows = {r[0]: r for r in server.leaves(p)["rows"]}
    yours = rows["A config.ini of yours"]
    assert yours[1] == str(mine) and "reads it at every start" in yours[2]


def test_the_harnesses_are_told_its_instances_the_context_run_with_and_no_key(world):
    p = world.plugin("windows")
    world.installed(p)
    local(p, context=32768)
    p.set_key("default", KEY)
    running(p, world)
    router(world)
    p.models_of("default")
    told = p.serving()
    assert told["native"] == "llamacpp" and told["plugin"] == "llamacpp"
    [row] = told["instances"]
    assert row["openai"] == "http://127.0.0.1:9931/v1" and row["context"] == 32768 and row["state"] == "running"
    assert row["key"] == "LIMINAL_AI_LLAMACPP_DEFAULT_KEY" and KEY not in repr(told)
    contexts = {m["id"]: m["context"] for m in row["models"]}
    assert contexts["ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"] == 32768 and contexts["my-coder-Q5_K_M"] == 32768
    assert all(m["tools"] is None for m in row["models"])
    # a model downloading, or whose download is not whole, is never offered to a harness
    assert set(contexts) == {"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", "ggml-org/Qwen3-8B-GGUF:Q4_K_M", "my-coder-Q5_K_M"}
    assert {m["id"] for m in p.models_of("default")["models"]} == set(contexts)
    assert len(p.models()["tables"][0]["rows"]) == 5  # the Models tab shows them all


def test_build_numbers_are_read_from_build_info():
    assert server.build_number("b12359-b1b770d9") == 12359 and server.build_number("") is None
    assert release.ROUTER_FROM == 7218 and release.MODELS_FROM == 9688
