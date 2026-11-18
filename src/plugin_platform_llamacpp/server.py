"""A llama.cpp server run here, and how one is recognised (the user, 2026-11-16: "the plugins manage the services, the
app can be started @ boot like a service, ports can be configured via the plugin, the plugin should report
conflicts"; "also remote connection to instances if possible").

A server run here is always a router: llama-server given no model, which loads a model when it is first asked (at
most `most` at once, the one used longest ago unloaded for the next), unloads it after `idle` seconds without a
request, and runs each loaded model as a program of its own on a loopback port it chooses (the tool's doing, not a
setting: A-revisions B1). Its command line carries its bind address (127.0.0.1, or every address once shared), its
port, its models' limits, --no-agent (no tools and no MCP proxy, whatever a config.ini says), its context and GPU
layers where set, and a folder of the user's GGUF files where named; its environment carries the folder its
downloads go to (LLAMA_CACHE: the plugin's own models folder, in Hugging Face's layout), the one origin a browser page
may read it from (its own address, so no other site's page can), LLAMA_ARG_OFFLINE=1 unless its Offline setting is
off, and its key where one is kept (LLAMA_API_KEY, never on the command line). The router hands its arguments and its
environment on to every model program it starts (tools/server/server-models.cpp at b12359), so they hold for those
too. Every llama.cpp, ggml and Hugging Face variable of the platform's environment is dropped (DROP), and the others
llama-server reads (Vertex AI's AIP_*, which can move it off its port; MTMD_*, the projector's device), so nothing
set elsewhere reconfigures it; never --connect, --tools, --agent, an MCP file, --metrics, --log-prompts-dir or
--slot-save-path, and a start is refused while a config.ini of the user's turns on remote access or MCP servers
(no switch turns those off).

Offline (the review of 0.1.0, its high finding): started online, a router's load of a model it downloaded asks the
Hub for that repository's refs and files first (common/download.cpp, common/hf-cache.cpp at b12359), moves its
cache's refs/main to the newest revision, and downloads changed files by itself before it loads them: no size, no
licence, no Cancel. So a server run here is started offline unless its Offline setting is off; offline it asks the Hub
nothing and its router downloads nothing either (the same switch reaches its download programs), so Download… is
off for it until Offline is turned off (Change…, then started again). The other way, the plugin downloading itself
into llama.cpp's cache with the router always offline, is the user's to choose (README, Open for the user).

A server is recognised by asking it (`look`, for the Models tab and a user's click, never for a status): /health and
its `Server: llama.cpp` header, /props (its build, whether it is a router, whether its remote access is on: a share
code it holds is never kept, only said as on), then its models: a router's GET /models, one model's /v1/models."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from liminal_libs import mounts
from liminal_libs.instances import Refusal
from liminal_libs.machines.children import Launch
from liminal_libs.machines.endpoint import EndpointError
from liminal_libs.machines.modelserver import LOOPBACK, Seen

from . import release

# llama.cpp's own variables (LLAMA_ARG_*, LLAMA_API_KEY, LLAMA_CACHE, LLAMA_SERVER_ROUTER_PORT, LLAMA_APP_CMD), ggml's
# (GGML_CUDA_ENABLE_UNIFIED_MEMORY and the like), the Hugging Face ones it reads (RL 14.9: a token, another hub,
# another cache), Vertex AI's that llama-server reads (AIP_MODE, AIP_HTTP_PORT, AIP_HEALTH_ROUTE, AIP_PREDICT_ROUTE:
# tools/server/server-http.cpp at b12359; AIP_HTTP_PORT moves it off its port) and the projector's (MTMD_BACKEND_DEVICE,
# --mmproj-device's variable): dropped from a server started here (the base adds every pair of Liminal's)
DROP = ("LLAMA_*", "GGML_*", "HF_TOKEN", "HF_ENDPOINT", "MODEL_ENDPOINT", "HF_HOME", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "AIP_*", "MTMD_*")
KEY_VAR = "LLAMA_API_KEY"
OFFLINE_VAR = "LLAMA_ARG_OFFLINE"  # --offline's variable: "forces use of cache, prevents network access"
NO_AGENT = "--no-agent"  # its tools and the chat page's MCP proxy cleared, whatever a config.ini sets (common/arg.cpp)
LAYERS = re.compile(r"^(?:auto|all|\d{1,4})$")
# What a config.ini may turn on that a server started here never has and no switch of llama-server turns off (by its
# keys: an argument without its dashes, or its variable's name; common/arg.cpp and the server's README at b12359)
REMOTE, MCP = "remote access (llama-connect's tunnel)", "MCP servers"
CONFIG_ON = {"connect": REMOTE, "llama_arg_connect": REMOTE, "mcp-servers-config": MCP,
             "llama_arg_mcp_servers_config": MCP, "mcp-servers-json": MCP, "llama_arg_mcp_servers_json": MCP}
CONFIG_FLAGS = ("connect", "llama_arg_connect")  # a switch: on unless its value says off
# what it may turn on that --no-agent turns off for a server started here
CONFIG_CLEARED = ("tools", "llama_arg_tools", "agent", "ag", "llama_arg_agent", "ui-mcp-proxy", "webui-mcp-proxy",
                  "llama_arg_ui_mcp_proxy")
OFF = ("off", "disabled", "false", "0", "")  # a value that leaves a switch off (common/arg.cpp's falsey): else it is on
CONFIG_MOST = 256 << 10  # bytes of a config.ini read at the most
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


def offline(instance) -> bool:
    """Whether a server run here is started offline: its Offline setting, on unless it says off (left empty, or kept
    before the setting was there, it is on)."""
    return str(instance.settings.get("offline") or "on").strip().lower() != "off"


def offline_words(instance) -> str:
    """Why a server run here that is started offline downloads nothing, and how that changes."""
    return (f"{instance.name} is started offline (Offline is on in its settings), and llama.cpp downloads nothing "
            "then: to download through it, turn Offline off with Change… while it is stopped, and start it again")


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
    """A local instance's server: llama-server as a router (no model), its bind address and port, its limits, no
    tools (--no-agent) and its settings on its command line, its downloads' folder, its one origin, offline unless its
    Offline setting is off, and its key by environment. Refused: not installed, a found program older than b7218 (it
    cannot serve without a model), a setting that cannot be one, a config.ini of the user's that turns on what a
    server started here never has (remote access, MCP servers)."""
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
    turned = [c for c in configs(p) if c.on]
    if turned:
        c = turned[0]
        raise Refusal(f"{c.path} turns on {_and(sorted({what for what, _ in c.on}))} ({', '.join(k for _, k in c.on)}) "
                      "for every llama-server that reads it, a server started here too, and no switch turns it off: "
                      "this plugin never starts one so. Take it out of that file, or start llama.cpp yourself")
    port = p.port_of(instance)
    host = "0.0.0.0" if instance.shared else LOOPBACK
    argv = [*head, "--host", host, "--port", str(port), "--models-max", str(most), NO_AGENT]
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
    if offline(instance):
        env[OFFLINE_VAR] = "1"
    if key:
        env[KEY_VAR] = key
    return Launch(tuple(argv), env=env, cwd=str(p.home), drop=DROP, secret=(KEY_VAR,) if key else ())


def _and(names) -> str:
    """Names in words: "a", "a and b", "a, b and c"."""
    shown = [str(n) for n in names if str(n)]
    if len(shown) < 2:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " and " + shown[-1]


# -- the words --------------------------------------------------------------------------------------------------------
def online_words(when: str) -> str:
    """What a router started online does at each load of a model it downloaded (common/download.cpp,
    common/hf-cache.cpp and tools/server/server.cpp at b12359), after `when`."""
    return (f"{when} asks huggingface.co for that repository's list of files (two requests, no token, its own "
            "User-Agent) and moves its folder's refs/main to the newest revision; when the repository has moved on, "
            "it downloads the newer files by itself before it loads the model, gigabytes perhaps, with no size, "
            "licence or Cancel shown here")


def start_words(p, instance) -> str:
    """What a start does, after the base's words (ai-start's confirmation): offline or online, as its setting says."""
    settings = instance.settings
    most = settings.get("most") if isinstance(settings.get("most"), int) else 4
    idle = settings.get("idle") if isinstance(settings.get("idle"), int) else 300
    unload = f"unloaded after {idle:,} s without a request" if idle else "kept loaded until another needs its place"
    words = (f"It starts as a router, without a model: one is loaded when it is first asked (at most {most} at once) "
             f"and {unload}. Each loaded model is a program of its own on a port of this machine alone (127.0.0.1) "
             "that llama.cpp chooses itself: the tool's doing, not a setting. Its models are in "
             f"{models_folder(p, instance)}. No other site's page can read it, and it is started with {NO_AGENT}: no "
             "tools and no MCP proxy, whatever a config.ini of yours says. ")
    if offline(instance):
        words += (f"It is started offline (Offline is on: {OFFLINE_VAR}=1): it asks huggingface.co nothing, loads "
                  "each model it downloaded from its folder as it is, and downloads nothing; to download a model "
                  "through it, turn Offline off with Change… and start it again.")
    else:
        words += (f"It is started online (Offline is off): it reaches huggingface.co for a download you ask for, and "
                  f"{online_words('each time it loads a model it downloaded it first')}. Without a network it loads "
                  "what it has.")
    have = p.installed()
    if have is not None and have.how == "found":
        build = release.found_build(p, have)
        if build is None or build < release.BUILD_NUMBER:
            said = f"b{build}" if build else "its build not read yet"
            words += (f" It is your own llama.cpp ({said}): whether a build older than {release.BUILD} knows "
                      f"{OFFLINE_VAR} and {NO_AGENT} is not known here (one that does not know {NO_AGENT} refuses to "
                      "start, its log says so).")
    return words


def share_words(p, instance) -> str:
    """What sharing means beyond the base's words."""
    return ("llama.cpp's key guards its API; /health and its chat page stay open, and its makers advise against "
            "serving it to a network you do not trust.")


def leaves(p) -> dict:
    """The table "What leaves this machine": files and words only, nothing asked."""
    rows = [
        ["Telemetry and update checks", "none", "llama.cpp has neither: nothing is sent of its own accord"],
        ["Downloads", "on your click only, with Offline off",
         "from huggingface.co, by the server itself (its router's own download), without a token: public models only. "
         "A server started with Offline on (the default) downloads nothing"],
        ["Loading a downloaded model", "nothing is asked with Offline on (the default)",
         f"with Offline on ({OFFLINE_VAR}=1), a model it downloaded loads from its folder as it is. With Offline off, "
         f"{online_words('each load first')}. A model of your GGUF folder is loaded from its file alone; a found or "
         "remote server does as it was started"],
        ["Find a model", "on your click only",
         "a model's files, size and licence asked of the Hugging Face Hub's API by this plugin, without a token"],
        ["Remote access, tools, agent, MCP", "off for a server started here",
         f"it is started with {NO_AGENT} (no tools, no MCP proxy, whatever a config.ini says), and a start is refused "
         "while a config.ini of yours turns on remote access (connect) or MCP servers, which no switch turns off; a "
         "found or remote server with remote access on says so in its facts"],
        ["Other sites' pages", "cannot read a server started here",
         "its one origin is its own address; llama.cpp's own default lets any page in your browser read it"],
        ["Its chat page", "at its own address",
         "Llama Chat runs in your browser from the server's own address and keeps its conversations there; what it "
         "asks of other sites was not read from its source here"],
    ]
    for c in configs(p):
        said = ["llama.cpp reads it at every start, under what is set here"]
        if c.on:
            said.append(f"it turns on {_and(sorted({what for what, _ in c.on}))} ({', '.join(k for _, k in c.on)}): "
                        "a start is refused while it does")
        if c.cleared:
            said.append(f"its {', '.join(c.cleared)}: {NO_AGENT} turns them off for a server started here")
        said.append("any other switch in it that this plugin does not set holds for a server started here too")
        rows.append(["A config.ini of yours", str(c.path), "; ".join(said)])
    return {"title": "What leaves this machine", "note": "For a server started here.",
            "columns": [{"title": "What"}, {"title": "Here"}, {"title": "What it means"}], "rows": rows}


@dataclass(frozen=True)
class Config:
    """A config.ini llama.cpp reads at every start, as this plugin read it (files only): its path; what in it turns
    on what a server started here never has, by (what, its key); the keys in it that --no-agent turns off."""

    path: Path
    on: tuple = ()
    cleared: tuple = ()


def read_config(path: Path) -> Config:
    """What a config.ini turns on (CONFIG_ON) or sets that --no-agent turns off (CONFIG_CLEARED), its keys read in
    every section (llama.cpp applies its top and its global sections; a key of another section is counted too: in
    doubt, a start is refused rather than run with it); CONFIG_MOST bytes at the most, nothing run."""
    try:
        with open(path, "rb") as f:
            text = f.read(CONFIG_MOST).decode("utf-8", "replace").lstrip("\ufeff")
    except OSError:
        return Config(Path(path))
    on: list = []
    cleared: list = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in ";#[" or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().lstrip("-"), value.strip().strip("\"'").strip()
        word = key.lower()
        if word in CONFIG_ON:
            if (value.lower() not in OFF) if word in CONFIG_FLAGS else bool(value):
                on.append((CONFIG_ON[word], key[:60]))
        elif word in CONFIG_CLEARED and value.lower() not in OFF and key[:60] not in cleared:
            cleared.append(key[:60])
    return Config(Path(path), tuple(dict.fromkeys(on)), tuple(cleared))


def configs(p) -> list[Config]:
    """The config.ini files llama.cpp reads at every start that exist here, each as it was read (files only)."""
    return [read_config(path) for path in config_files(p)]


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
    """The context a loaded model of a router runs with, where its router says it: `memory.effective.n_ctx`, which
    b12359's GET /models lists only with its memory accounting on (--models-memory-margin: never given to a server
    started here; tools/server/server-models.cpp, its README's Memory-based unloading); None otherwise."""
    memory = row.get("memory") if isinstance(row.get("memory"), dict) else {}
    effective = memory.get("effective") if isinstance(memory.get("effective"), dict) else {}
    return _count(effective.get("n_ctx"))


def router_model(p, instance, row: dict) -> dict | None:
    """One model of a router's GET /models as the page and the harnesses read it. Its `context`, what a harness is
    told: what its router says of a loaded one (`_context_of`), else a server run here's Context size (the router
    hands it every model it loads), else None: not known."""
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
