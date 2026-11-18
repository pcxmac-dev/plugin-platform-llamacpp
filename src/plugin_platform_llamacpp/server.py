"""A llama.cpp server run here, and how one is recognised (the user, 2026-11-16: "the plugins manage the services, the
app can be started @ boot like a service, ports can be configured via the plugin, the plugin should report
conflicts"; "also remote connection to instances if possible").

A server run here is always a router: llama-server given no model, which loads a model when it is first asked (at
most `most` at once, the one used longest ago unloaded for the next), unloads it after `idle` seconds without a
request, and runs each loaded model as a program of its own on a loopback port it chooses (the tool's doing, not a
setting: A-revisions B1). Its command line carries its bind address (127.0.0.1, or every address once shared), its
port, its models' limits, its context and GPU layers where set, and a folder of the user's GGUF files where named;
its environment carries the folder its downloads go to (LLAMA_CACHE: the plugin's own models folder, in Hugging
Face's layout), the one origin a browser page may read it from (its own address, so no other site's page can), and
its key where one is kept (LLAMA_API_KEY, never on the command line). Every llama.cpp, ggml and Hugging Face variable
of the platform's environment is dropped (DROP), so nothing set elsewhere reconfigures it; never --connect, --tools,
--agent, an MCP file, --metrics, --log-prompts-dir or --slot-save-path.

A server is recognised by asking it (`look`, for the Models tab and a user's click, never for a status): /health and
its `Server: llama.cpp` header, /props (its build, whether it is a router, whether its remote access is on: a share
code it holds is never kept, only said as on), then its models: a router's GET /models, one model's /v1/models."""
from __future__ import annotations

import re
from pathlib import Path

from liminal_libs import mounts
from liminal_libs.instances import Refusal
from liminal_libs.machines.children import Launch
from liminal_libs.machines.endpoint import EndpointError
from liminal_libs.machines.modelserver import LOOPBACK, Seen

from . import release

# llama.cpp's own variables (LLAMA_ARG_*, LLAMA_API_KEY, LLAMA_CACHE, LLAMA_SERVER_ROUTER_PORT, LLAMA_APP_CMD), ggml's
# (GGML_CUDA_ENABLE_UNIFIED_MEMORY and the like), and the Hugging Face ones it reads (RL 14.9: a token, another hub,
# another cache): dropped from a server started here (the base adds every pair of Liminal's)
DROP = ("LLAMA_*", "GGML_*", "HF_TOKEN", "HF_ENDPOINT", "MODEL_ENDPOINT", "HF_HOME", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE")
KEY_VAR = "LLAMA_API_KEY"
LAYERS = re.compile(r"^(?:auto|all|\d{1,4})$")
# what the server is asked, by look() (the Models tab) and by the base's Try it
CALLS = ("GET /health", "GET /props", "GET /models", "GET /v1/models", "POST /v1/chat/completions")
STATES = {"loaded": "", "loading": "loading", "downloading": "downloading", "incomplete": "incomplete",
          "sleeping": "sleeping", "downloaded": "downloaded", "unloaded": ""}
SOURCES = {"cache": "", "models_dir": "from your GGUF folder", "preset": "a preset's"}
UNSERVED = ("downloading", "incomplete", "downloaded")  # a router's models not whole yet: never offered to a harness


# -- the settings ---------------------------------------------------------------------------------------------------
def _number(value, low: int, high: int, default: int, label: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise Refusal(f"{label} is kept as {value!r}, which is no whole number from {low:,} to {high:,}: Change… it")
    return value


def models_folder(p, instance) -> Path:
    """The folder a local server's downloads go to: its Models folder setting, else the plugin's own home/models."""
    named = instance.settings.get("models")
    return Path(named) if isinstance(named, str) and named.strip() else p.home / "models"


def _folder(p, text, label: str) -> str:
    """A folder of a setting as a server is given it: a whole path of this system's own disk (else a Refusal)."""
    path = Path(str(text))
    if not path.is_absolute():
        raise Refusal(f"{label} is {text!r}, which is not a whole path: give it from its drive or its root")
    try:
        foreign = mounts.foreign(path)
    except Exception:  # noqa: BLE001 - a path that cannot be judged is this system's
        foreign = False
    if foreign:
        raise Refusal(f"{label} {text} is on another system's drive: a folder of this system's own disk is given here")
    return str(path)


def launch(p, instance, key: str = "") -> Launch:
    """A local instance's server: llama-server as a router (no model), its bind address and port, its limits and
    settings on its command line, its downloads' folder, its one origin and its key by environment. Refused: not
    installed, a found program older than b7218 (it cannot serve without a model), a setting that cannot be one."""
    have = p.installed()
    head = release.program(p, have)
    if not head:
        raise Refusal(f"{p.title} is not installed here: Install… first")
    build = release.found_build(p, have)
    if build is not None and build < release.ROUTER_FROM:
        raise Refusal(f"this llama.cpp (b{build}) cannot serve without a model: a server run here is a router, which "
                      f"llama.cpp has from b{release.ROUTER_FROM} on")
    settings = instance.settings
    most = _number(settings.get("most"), 1, 16, 4, "Models loaded at once")
    idle = _number(settings.get("idle"), 0, 86400, 300, "Unload after idle (seconds)")
    context = settings.get("context")
    layers = str(settings.get("layers") or "auto").strip().lower()
    if not LAYERS.fullmatch(layers):
        raise Refusal(f"Layers on the GPU is kept as {layers[:20]!r}: auto, all or a whole number. Change… it")
    port = p.port_of(instance)
    host = "0.0.0.0" if instance.shared else LOOPBACK
    argv = [*head, "--host", host, "--port", str(port), "--models-max", str(most)]
    if idle:  # (llama-server refuses 0: without the switch it never unloads a model for idleness)
        argv += ["--sleep-idle-seconds", str(idle)]
    if context is not None:
        argv += ["-c", str(_number(context, 0, 1_048_576, 0, "Context size (tokens)"))]
    if layers != "auto":
        argv += ["-ngl", layers]
    gguf = settings.get("gguf")
    if isinstance(gguf, str) and gguf.strip():
        argv += ["--models-dir", _folder(p, gguf, "A folder of GGUF files of yours")]
    models = _folder(p, str(models_folder(p, instance)), "Models folder")
    env = {"LLAMA_CACHE": models, "LLAMA_ARG_CORS_ORIGINS": f"http://{LOOPBACK}:{port}"}
    if key:
        env[KEY_VAR] = key
    return Launch(tuple(argv), env=env, cwd=str(p.home), drop=DROP, secret=(KEY_VAR,) if key else ())


# -- the words --------------------------------------------------------------------------------------------------------
def start_words(p, instance) -> str:
    """What a start does, after the base's words (ai-start's confirmation)."""
    settings = instance.settings
    most = settings.get("most") if isinstance(settings.get("most"), int) else 4
    idle = settings.get("idle") if isinstance(settings.get("idle"), int) else 300
    unload = f"unloaded after {idle:,} s without a request" if idle else "kept loaded until another needs its place"
    return (f"It starts as a router, without a model: one is loaded when it is first asked (at most {most} at once) "
            f"and {unload}. Each loaded model is a program of its own on a port of this machine alone (127.0.0.1) that "
            "llama.cpp chooses itself: the tool's doing, not a setting. Its models are in "
            f"{models_folder(p, instance)}. No other site's page can read it. It reaches the network for a download "
            "you ask for, and, each time it loads a model it downloaded, to ask huggingface.co for that model's list "
            "of files (llama.cpp's own way: no switch keeps that off while its downloads work).")


def share_words(p, instance) -> str:
    """What sharing means beyond the base's words."""
    return ("llama.cpp's key guards its API; /health and its chat page stay open, and its makers advise against "
            "serving it to a network you do not trust.")


def leaves(p) -> dict:
    """The table "What leaves this machine": files and words only, nothing asked."""
    rows = [
        ["Telemetry and update checks", "none", "llama.cpp has neither: nothing is sent of its own accord"],
        ["Downloads", "on your click only", "from huggingface.co, by the server itself (its router's own download), "
                                            "without a token: public models only"],
        ["Loading a downloaded model", "asks huggingface.co first",
         "each time a model it downloaded is loaded, llama.cpp asks huggingface.co for that repository's list of files "
         "(two requests, no token, its own User-Agent) before it loads it from its folder; without a network it loads "
         "it all the same. llama.cpp has no switch that keeps this off while its downloads work. A model of your GGUF "
         "folder is loaded from its file alone."],
        ["Find a model", "on your click only",
         "a model's files, size and licence asked of the Hugging Face Hub's API by this plugin, without a token"],
        ["Remote access, tools, agent, MCP", "never switched on here",
         "a found or remote server with remote access on says so in its facts"],
        ["Other sites' pages", "cannot read a server started here",
         "its one origin is its own address; llama.cpp's own default lets any page in your browser read it"],
        ["Its chat page", "at its own address",
         "Llama Chat keeps its conversations in your browser and asks the Hugging Face Hub for a model's details only "
         "once you turn that on in its settings (off by default)"],
    ]
    for path in config_files(p):
        rows.append(["A config.ini of yours", str(path),
                     "llama.cpp reads it at every start, under what is set here: a switch in it this plugin does not "
                     "set holds for a server started here too"])
    return {"title": "What leaves this machine", "note": "For a server started here.",
            "columns": [{"title": "What"}, {"title": "Here"}, {"title": "What it means"}], "rows": rows}


def config_files(p) -> list[Path]:
    """The config.ini files llama.cpp reads at every start that exist here (looked at only): machine-wide, then
    yours."""
    env = release.environment(p)
    if p.system == "windows":
        places = [Path(v) / "llama.cpp" / "config.ini" for v in (release.env_value(env, "PROGRAMDATA"),
                                                                  release.env_value(env, "APPDATA")) if v]
    else:
        config = release.env_value(env, "XDG_CONFIG_HOME")
        home = release.env_value(env, "HOME")
        mine = Path(config) if config else (Path(home) / ".config" if home else None)
        places = [Path("/etc/llama.cpp/config.ini"), *([mine / "llama.cpp" / "config.ini"] if mine else [])]
    out = []
    for path in places:
        try:
            if path.is_file():
                out.append(path)
        except OSError:
            continue
    return out


# -- recognising a server ---------------------------------------------------------------------------------------------
def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _context_of(row: dict) -> int | None:
    """The context a loaded model of a router runs with, as its child reported it (`memory.n_ctx`, or its measured
    `memory.effective.n_ctx`); None when it does not say."""
    memory = row.get("memory") if isinstance(row.get("memory"), dict) else {}
    effective = memory.get("effective") if isinstance(memory.get("effective"), dict) else {}
    return _count(memory.get("n_ctx")) or _count(effective.get("n_ctx"))


def router_model(p, instance, row: dict) -> dict | None:
    """One model of a router's GET /models as the page and the harnesses read it."""
    ident = row.get("id")
    if not isinstance(ident, str) or not ident.strip():
        return None
    status = row.get("status") if isinstance(row.get("status"), dict) else {}
    value = str(status.get("value") or "")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    kept = p.found_of(ident)
    running = value in ("loaded", "sleeping")
    context = _context_of(row) if running else None
    if context is None and instance.where == "local" and isinstance(instance.settings.get("context"), int):
        context = instance.settings["context"] or None
    detail = [STATES.get(value, value)]
    if value in ("downloading", "loading"):
        detail[0] = f"{value} {_progress(status)}".strip()
    if status.get("failed"):
        detail.append(f"failed (exit {status.get('exit_code')})")
    detail.append(SOURCES.get(str(row.get("source") or ""), ""))
    train = _count(row.get("context_length")) or _count(meta.get("n_ctx_train"))
    modalities = (row.get("architecture") or {}).get("input_modalities") if isinstance(row.get("architecture"),
                                                                                       dict) else None
    return {"id": ident.strip(), "size": _count(meta.get("size")) or (kept.size if kept else None), "context": context,
            "loaded": running, "tools": None, "detail": " · ".join(d for d in detail if d),
            "status": value, "source": str(row.get("source") or ""), "can_remove": row.get("can_remove") is True,
            "train": train, "takes": [m for m in modalities if isinstance(m, str)] if isinstance(modalities, list)
            else [], "sidecars": [s for s in row.get("sidecars") or [] if isinstance(s, str)]}


def _progress(status: dict) -> str:
    """How far a router says a download or a load is: "40%"."""
    progress = status.get("progress")
    if isinstance(progress, dict):
        if isinstance(progress.get("value"), (int, float)):
            return f"{int(progress['value'] * 100)}%"
        done = sum(int(v.get("done") or 0) for v in progress.values() if isinstance(v, dict))
        total = sum(int(v.get("total") or 0) for v in progress.values() if isinstance(v, dict))
        if total > 0:
            return f"{min(100, done * 100 // total)}%"
    return ""


def build_number(build_info) -> int | None:
    """A server's build from its /props' build_info ("b12359-b1b770d9")."""
    found = re.match(r"^b(\d{3,7})\b", str(build_info or "").strip())
    return int(found.group(1)) if found else None


def look(p, at, instance) -> Seen:
    """What an instance answers now: llama.cpp or not (/health and its Server header; behind a proxy that names
    itself there, /props' own fields), its build and kind (/props), its models (a router's /models, else /v1/models).
    A few asks of the endpoint's short wait each."""
    address = p.address(instance)
    try:
        health = at.answer("GET", "/health", ok=(200, 503))
    except EndpointError as e:
        if e.refused:  # (llama.cpp's own /health is open: a proxy in front wants a key)
            return Seen(False, f"what answers at {address} wants a key before it says anything (Key…)")
        if e.status:
            return Seen(False, f"what answers at {address} is not llama.cpp (its /health says HTTP {e.status})")
        raise
    named = str(health.headers.get("server", "")).strip()
    llama = named.lower() == "llama.cpp"
    other = f" (it says it is {named[:40]})" if named and not llama else ""
    try:
        props = at.get("/props")
    except EndpointError as e:
        if e.refused:
            who = "llama.cpp answers" if llama else "what answers"
            return Seen(False, f"{who} at {address}, and it wants its key (Key…)")
        if llama and (e.status == 503 or health.status == 503):
            return Seen(True, f"llama.cpp answers at {address}: it is loading a model", facts={"loading": "yes"})
        if not llama:
            return Seen(False, f"what answers at {address} is not llama.cpp{other}")
        raise
    props = props if isinstance(props, dict) else {}
    if not llama and "build_info" not in props:
        return Seen(False, f"what answers at {address} is not llama.cpp{other}")
    build = build_number(props.get("build_info"))
    version = f"b{build}" if build else ""
    router = props.get("role") == "router"
    facts = {"router": "yes" if router else "no: one model", "build": version or "not said",
             "connect": "on" if props.get("connect_code") else "off"}  # the share code itself is never kept
    if router:
        listed = at.get("/models")
        rows = listed.get("data") if isinstance(listed, dict) else None
        models = [m for m in (router_model(p, instance, r) for r in rows or [] if isinstance(r, dict)) if m]
        kind = f"a router of build {build}" if build else "a router"
    else:
        listed = at.get("/v1/models")
        rows = listed.get("data") if isinstance(listed, dict) else None
        settings = props.get("default_generation_settings")
        n_ctx = _count(settings.get("n_ctx")) if isinstance(settings, dict) else None
        models = []
        for row in rows or []:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                train = _count(meta.get("n_ctx_train"))
                models.append({"id": row["id"], "size": _count(meta.get("size")), "context": n_ctx, "loaded": True,
                               "tools": None, "detail": f"trained for {train:,} tokens" if train else "",
                               "status": "loaded", "source": "", "can_remove": False, "train": train, "takes": [],
                               "sidecars": []})
        kind = "one model's server" + (f" of build {build}" if build else "")
    many = f"{len(models)} model{'' if len(models) == 1 else 's'} listed"
    remote = (" Its remote access (--connect) is on: its share code lets anyone use it."
              if facts["connect"] == "on" else "")
    return Seen(True, f"llama.cpp answers at {address}: {kind}, {many}.{remote}", version=version,
                models=tuple(models), facts=facts)
