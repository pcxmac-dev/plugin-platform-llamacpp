"""What the llama.cpp plugin attaches to (plugin API 2.2, liminal_libs.sources.apis; the user, 2026-11-11: "each plugin
should reflect the the API its attached to, and present links that provide documentation ... it should also expose
more of what the API is about"): llama-server's own HTTP API (a router's model calls among it), llama-server as a
command, llama.cpp's release files, the Hugging Face Hub's API (a model's files, size and licence before a download),
nvidia-smi (this machine's GPU) and the system's start at sign-in (a scheduled task, a systemd user unit). Every
address was read on the makers' own pages on 2026-11-18 (`VERIFIED`); llama.cpp's own pages are given at the pinned
build's tag, b12359, so they say what that build does. The cards say what holds on any machine; what one machine's
servers said is api_facts', from what was kept, asking nothing."""
from __future__ import annotations

from liminal_libs.sources.apis import Api, Area, Call, Link

from . import release

VERIFIED = "2026-11-18"

# -- the makers' pages, as prefixes (every outside host the plugin names lives here or in release.py) ----------------
REPO = "https://github.com/ggml-org/llama.cpp"
TAG = REPO + "/blob/" + release.BUILD + "/"  # llama.cpp's files at the pinned build
SERVER_DOCS = TAG + "tools/server/README.md"
APP = "https://llama.app/docs/"
HF = "https://huggingface.co/"
HF_DOCS = HF + "docs/hub/"
HF_CLIENT = HF + "docs/huggingface_hub/v2.4.0/en/package_reference/hf_api"
NVIDIA = "https://docs.nvidia.com/"
LEARN = "https://learn.microsoft.com/en-us/"
TASKS = LEARN + "powershell/module/scheduledtasks/"
WS = "?view=windowsserver2025-ps"
SYSTEMD = "https://www.freedesktop.org/software/systemd/man/latest/"

SERVER = Api(
    name="llama-server's API",
    maker="the ggml-org project (llama.cpp)",
    about=("llama.cpp's HTTP server: its native routes, OpenAI-compatible ones under /v1 and an Anthropic-compatible "
           "one. Started without a model it is a router, which lists its models, loads each one asked as a program "
           "of its own and, from build b9688, downloads, deletes and tells of them over its API. The plugin "
           "recognises a server by its /health and its Server header, reads its build and kind from /props, lists "
           "its models, downloads, loads, unloads and deletes them on your confirmed click, and Try it asks it a "
           "prompt; it is asked only while its page is shown, and the harnesses reach it at its /v1."),
    kind="rest",
    base="http://127.0.0.1:9931 for a server run here (its port is its instance's), a found or remote one at its own",
    version=(f"llama.cpp {release.VERSION}, build {release.BUILD}: a router from build {release.ROUTER_FROM}, its "
             f"model calls from {release.MODELS_FROM}, port 9931 by default from {release.PORT_FROM} (8080 before)"),
    auth=("its key as a bearer, where one is set (LLAMA_API_KEY; kept through your login as its own pair, given to "
          "a server run here by its environment); none by default. /health and its chat page stay open"),
    limits="",
    costs="",
    links=(
        Link("llama-server: its arguments, variables and endpoints", SERVER_DOCS, "docs"),
        Link("Using multiple models: the router, presets and its model endpoints",
             SERVER_DOCS + "#using-multiple-models", "reference"),
        Link("API overview", APP + "api", "guide"),
        Link("Security policy", TAG + "SECURITY.md", "guide"),
    ),
    areas=(
        Area("Health and properties", "whether it answers and is loading; its build, kind and settings",
             SERVER_DOCS + "#get-health-returns-health-check-result",
             covered="recognised (its Server header), its build and kind read, a model's details"),
        Area("Models of a router", "its models listed, loaded, unloaded, downloaded and deleted, their events",
             SERVER_DOCS + "#get-models-list-available-models",
             covered="listed on the Models tab; Download… (a job), Load…, Unload…, Delete…, Details"),
        Area("OpenAI-compatible", "/v1/models, /v1/chat/completions, /v1/completions, /v1/responses, /v1/embeddings",
             SERVER_DOCS + "#openai-compatible-api-endpoints",
             covered="one model's server listed; Try it; the harnesses' address"),
        Area("Anthropic-compatible", "/v1/messages and its token count",
             SERVER_DOCS + "#anthropic-compatible-api-endpoints"),
        Area("Native completion", "/completion, /tokenize, /detokenize, /apply-template, /infill, /embedding",
             SERVER_DOCS + "#api-endpoints"),
        Area("Slots and metrics", "the slots' state, Prometheus metrics (off unless --metrics), saved prompt caches",
             SERVER_DOCS + "#get-slots-returns-the-current-slots-processing-state"),
        Area("LoRA adapters", "the adapters loaded and their scales",
             SERVER_DOCS + "#get-lora-adapters-get-list-of-all-lora-adapters"),
        Area("Its chat page", "Llama Chat at its own address, in your browser", SERVER_DOCS,
             covered="left on: it is the server's own page, read only from its own address"),
    ),
    calls=(
        Call("GET /health", "200 when it answers, 503 while it loads a model; its Server header names llama.cpp",
             "Health and properties", used="the Models tab: whether what answers is llama.cpp"),
        Call("GET /props", "its build, whether it is a router; with ?model= a loaded model's file, context, slots",
             "Health and properties", used="the Models tab (build, kind, remote access on or off) and Details"),
        Call("GET /models", "a router's models: id, status, source, whether it can be deleted, its sidecars",
             "Models of a router", used="the Models tab, Details, Load…, Unload…, Delete…, a download's end"),
        Call("GET /v1/models", "the model one model's server serves, with its size", "OpenAI-compatible",
             used="the Models tab of a server that is no router"),
        Call("GET /models/sse", "a router's events: download_progress, download_finished and download_failed",
             "Models of a router", used="Download…: its progress in the panel"),
        Call("POST /models", "a download started (the model and its sidecars), answered at once", "Models of a router",
             used="Download… (a job, asked twice with its size and licence)", writes=True),
        Call("POST /models/load", "a model loaded now as a program of its own", "Models of a router", used="Load…",
             writes=True),
        Call("POST /models/unload", "a model unloaded, or a download paused (its part kept)", "Models of a router",
             used="Unload…, and Cancel of a download", writes=True),
        Call("DELETE /models", "a model of its cache deleted (?model=)", "Models of a router", used="Delete…",
             writes=True),
        Call("POST /v1/chat/completions", "a prompt answered the OpenAI way", "OpenAI-compatible",
             used="Try it (not streamed, 512 tokens and 4,000 characters at most)"),
        Call("GET /slots", "each slot's state", "Slots and metrics"),
        Call("GET /metrics", "Prometheus metrics, off unless --metrics", "Slots and metrics"),
        Call("POST /completion", "llama.cpp's own completion", "Native completion"),
        Call("POST /v1/completions", "an OpenAI completion", "OpenAI-compatible"),
        Call("POST /v1/responses", "OpenAI's Responses API", "OpenAI-compatible"),
        Call("POST /v1/embeddings", "embeddings", "OpenAI-compatible"),
        Call("POST /v1/messages", "Anthropic's Messages API", "Anthropic-compatible"),
        Call("POST /tokenize", "a text's tokens", "Native completion"),
        Call("GET /lora-adapters", "the LoRA adapters", "LoRA adapters"),
    ),
    verified=VERIFIED,
)

COMMAND = Api(
    name="llama-server, the command",
    maker="the ggml-org project (llama.cpp)",
    about=("llama-server started as a program: every setting an argument, most with an LLAMA_ARG_ variable beside "
           "it, a machine's or a user's config.ini read first. The plugin starts a server of an instance run here as "
           "a router (no model), windowless, as a child it owns (Ctrl+C ends it on Windows, SIGTERM on Linux), with "
           "its bind address, port, models' limits, --no-agent, context, GPU layers and GGUF folder on its command "
           "line and its downloads' folder, one origin, LLAMA_ARG_OFFLINE=1 (unless the instance's Offline setting "
           "is off) and its key in its environment; the router hands them on to the model programs it starts. It "
           "reads an install's build and devices by running it once. --connect, --tools, --agent, MCP, --metrics "
           "and the prompt logs are never given, and a start is refused while a config.ini turns on remote access "
           "or MCP servers."),
    kind="command",
    base="llama-server (this plugin's install in app/<pin>, or yours on PATH: llama-server, or llama serve)",
    version=f"llama.cpp {release.VERSION}, build {release.BUILD}",
    auth="runs as you: no administrator, no root",
    limits="",
    costs="",
    links=(
        Link("llama-server's arguments", SERVER_DOCS + "#usage", "reference"),
        Link("Running a server", APP + "serve", "guide"),
        Link("Presets and config.ini", TAG + "docs/preset.md", "reference"),
        Link("Build from source: CUDA, Vulkan and the other backends", TAG + "docs/build.md", "docs"),
    ),
    areas=(
        Area("Serving", "a model or, without one, a router; its address, port and limits", SERVER_DOCS + "#usage",
             covered="a router started for an instance run here"),
        Area("Its version and devices", "--version (its build), --list-devices (the GPUs it sees)",
             SERVER_DOCS + "#common-params", covered="an install's check, and Read again of a found one"),
        Area("Configuration files", "a machine's and a user's config.ini, read at every start",
             TAG + "docs/preset.md", covered="read and said where one exists (What leaves this machine), a start "
                                             "refused while one turns on remote access or MCP servers; never written"),
        Area("Its own downloads and updates", "llama download, llama update, -hf at the command line",
             TAG + "docs/install.md"),
        Area("Remote access and tools", "--connect (a peer-to-peer tunnel), --tools, --agent, MCP servers",
             SERVER_DOCS + "#server-specific-params"),
    ),
    calls=(
        Call("llama-server --host <address> --port <port> --models-max <n> --no-agent [--sleep-idle-seconds <s>] "
             "[-c <n>] [-ngl <n>] [--models-dir <folder>]", "a router, its environment LLAMA_CACHE, "
             "LLAMA_ARG_CORS_ORIGINS, LLAMA_ARG_OFFLINE (Offline on) and LLAMA_API_KEY", "Serving",
             used="Start (a click's, or its task's or unit's at sign-in)"),
        Call("llama serve <the same arguments>", "the same router from llama.cpp's one program", "Serving",
             used="Start, where your install is the one program `llama`"),
        Call("llama-server --version", "its build, on its standard error", "Its version and devices",
             used="the install's check; Read again of a found install"),
        Call("llama version", "the one program's build, on its standard output", "Its version and devices",
             used="Read again of a found `llama`"),
        Call("llama-server --list-devices", "the devices it sees (CUDA0: ...), or (none)", "Its version and devices",
             used="the install's check; Read again of a found install"),
        Call("llama download -hf <repository>", "a model downloaded at the command line",
             "Its own downloads and updates"),
        Call("llama update", "llama.cpp's install script run again", "Its own downloads and updates"),
        Call("llama-server --cache-list", "the models of its cache", "Its own downloads and updates"),
        Call("llama-server --connect", "remote access through llama-connect's tunnel", "Remote access and tools"),
        Call("llama-server --tools / --agent", "tools that read and write files and run commands",
             "Remote access and tools"),
    ),
    verified=VERIFIED,
)

RELEASES = Api(
    name="llama.cpp's releases",
    maker="the ggml-org project (llama.cpp)",
    about=("The release files of llama.cpp's builds on GitHub: for each system and backend an archive of its "
           "programs, and NVIDIA's CUDA runtime beside a CUDA build. Install downloads the archives of the build "
           "chosen (automatic: settled by what nvidia-smi reads as it installs) from github.com, which sends each on "
           "to its storage host, checks each against the size and sha256 pinned in this plugin (the release's "
           "metadata and its attestation give them; it publishes no checksum file), and unpacks them into the "
           "plugin's own folder. Nothing looks for a newer release: a newer pin comes with a newer plugin."),
    kind="file",
    base=release.RELEASE,
    version=(f"v{release.VERSION} = build {release.BUILD} ({release.TAG_DAY}): a pre-release that is not immutable, "
             "so its files could be replaced, which the pinned sums notice"),
    auth="none: public files, downloaded on your click only",
    limits="",
    costs="",
    links=(
        Link("Build b12359, its archives", REPO + "/releases/tag/" + release.BUILD, "docs"),
        Link("v1.0.0, which names that build", REPO + "/releases/tag/v" + release.VERSION, "changelog"),
        Link("Every release", REPO + "/releases", "changelog"),
        Link("The build's attestation (each file's sha256)", REPO + "/attestations/67120335", "spec"),
        Link("Install", TAG + "docs/install.md", "guide"),
        Link("llama.cpp's licence (MIT)", TAG + "LICENSE", "terms"),
        Link("NVIDIA's CUDA licence, for the runtime archives", NVIDIA + "cuda/eula/index.html", "terms"),
    ),
    areas=(
        Area("Windows x64", "CPU, CUDA 12.4, CUDA 13.5, Vulkan; and OpenVINO, SYCL, ROCm, arm64",
             REPO + "/releases/tag/" + release.BUILD,
             covered="CUDA 13.5 and 12.4 (each with NVIDIA's runtime), Vulkan, the processor's"),
        Area("Linux x86-64", "CPU, CUDA 12.8, CUDA 13.5, Vulkan; and OpenVINO, SYCL, ROCm, arm64",
             REPO + "/releases/tag/" + release.BUILD,
             covered="CUDA 13.5 and 12.8 (each with NVIDIA's runtime), Vulkan, the processor's"),
        Area("macOS and other processors", "Apple's builds, arm64", REPO + "/releases/tag/" + release.BUILD),
        Area("The stable pointer", "v1.0.0 holds nightly-tag.txt alone, naming its build",
             REPO + "/releases/tag/v" + release.VERSION),
    ),
    calls=(
        *(Call(f"GET {part.name}", f"{part.what}, {part.size:,} bytes, sha256 {part.sha256[:12]}…",
               "Windows x64" if system == "windows" else "Linux x86-64", used=f"Install of the {release.TITLES[which]} "
               "build", docs=REPO + "/releases/tag/" + release.BUILD)
          for (system, which), parts in release.BUILDS.items() for part in parts),
        Call("GET nightly-tag.txt", "the build v1.0.0 stands for", "The stable pointer"),
    ),
    verified=VERIFIED,
)

HUB_API = Api(
    name="Hugging Face Hub API",
    maker="Hugging Face",
    about=("The Hub's API, where llama.cpp's GGUF models are published. The plugin asks it, on your click alone and "
           "without a token, for a model's card (whether it is gated, its licence) and its files with their sizes, "
           "picks the files llama.cpp would download by its own rules, and says their size, licence and what fits "
           "this machine's GPU before any download. A download itself is the server's: its router asks the Hub and "
           "fetches the files. Started online (an instance's Offline setting off), it also asks the Hub for a "
           "repository's list of files each time it loads a model it downloaded, and fetches a newer revision by "
           "itself; started offline, the default here, it asks the Hub nothing."),
    kind="rest",
    base=release.HUB,
    version="the Hub's API as its OpenAPI document describes it (the model card's address is its own client's)",
    auth=("none: no token is sent from here or given to llama.cpp (public models only; a gated one is said, not "
          "fetched)"),
    limits="the Hub limits how often one address may ask it",
    costs="",
    links=(
        Link("The Hub's API", HF_DOCS + "api", "docs"),
        Link("Its endpoints (OpenAPI playground)", HF + "spaces/huggingface/openapi", "reference"),
        Link("Its OpenAPI document, as text", HF + ".well-known/openapi.md", "spec"),
        Link("Model cards: where the licence is said", HF_DOCS + "model-cards", "guide"),
        Link("Licences: the identifiers", HF_DOCS + "repositories-licenses", "reference"),
        Link("Gated models", HF_DOCS + "models-gated", "guide"),
        Link("The Hub's own client: model_info and the repository tree", HF_CLIENT, "sdk"),
    ),
    areas=(
        Area("A model's card", "its id, gated or not, its licence (cardData.license, or its license: tag)",
             HF_CLIENT, covered="Find a model: gated, licence, its licence's page"),
        Area("A repository's files", "its tree, each file's size and LFS hash, a page of 1,000 at a time (Link)",
             HF + ".well-known/openapi.md", covered="Find a model: the files llama.cpp would take, their sizes"),
        Area("Sizes and paths", "a folder's size, several paths' details", HF + ".well-known/openapi.md"),
        Area("Downloads", "a file resolved to its storage", HF + ".well-known/openapi.md",
             covered="the server's, never this plugin's"),
        Area("Accounts and tokens", "gated and private models need a token", HF_DOCS + "models-gated"),
    ),
    calls=(
        Call("GET /api/models/{repo}", "a model's card: gated, cardData, tags", "A model's card",
             used="Find a model", docs=HF_CLIENT),
        Call("GET /api/models/{repo}/tree/main", "its files (?recursive=true), paged by its Link header",
             "A repository's files", used="Find a model", docs=HF + ".well-known/openapi.md"),
        Call("GET /api/models/{repo}/treesize/main", "the size of every file of a repository", "Sizes and paths",
             docs=HF + ".well-known/openapi.md"),
        Call("POST /api/models/{repo}/paths-info/main", "named paths' details", "Sizes and paths",
             docs=HF + ".well-known/openapi.md"),
        Call("GET /{repo}/resolve/main/{file}", "a file itself, sent on to its storage", "Downloads",
             docs=HF + ".well-known/openapi.md"),
    ),
    verified=VERIFIED,
)

NVIDIA_SMI = Api(
    name="nvidia-smi",
    maker="NVIDIA",
    about=("NVIDIA's System Management Interface, installed with its driver: the plugin asks it for the cards' "
           "names, memory and driver, windowless, for the Models tab (what fits) and inside an install's job (which "
           "build automatic brings), never for a status. Inside WSL it is Windows' driver's own, at "
           "/usr/lib/wsl/lib when it is not on PATH. No other maker's card is read."),
    kind="command",
    base="nvidia-smi on PATH, else /usr/lib/wsl/lib/nvidia-smi inside WSL",
    version="the NVIDIA driver's own tool",
    auth="runs as you",
    limits="",
    costs="",
    links=(
        Link("nvidia-smi", NVIDIA + "deploy/nvidia-smi/index.html", "docs"),
        Link("CUDA on WSL", NVIDIA + "cuda/wsl-user-guide/index.html", "guide"),
        Link("CUDA's minor version compatibility: which drivers a toolkit runs on",
             NVIDIA + "deploy/cuda-compatibility/latest/minor-version-compatibility.html", "reference"),
        Link("CUDA Toolkit release notes: each toolkit's driver", NVIDIA + "cuda/cuda-toolkit-release-notes/index.html",
             "reference"),
    ),
    areas=(
        Area("Queries", "--query-gpu fields as CSV (nvidia-smi --help-query-gpu names them)",
             NVIDIA + "deploy/nvidia-smi/index.html", covered="memory, name and driver; a card's compute capability"),
        Area("Monitoring and settings", "nvidia-smi -q, pmon, clocks and limits",
             NVIDIA + "deploy/nvidia-smi/index.html"),
    ),
    calls=(
        Call("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version "
             "--format=csv,noheader,nounits", "each card's memory and driver", "Queries",
             used="the Models tab's GPU and what fits; the build automatic brings"),
        Call("nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader", "each card's compute capability",
             "Queries", used="the build automatic brings on Windows (an RTX 50 is 12.0)"),
        Call("nvidia-smi -q", "everything it knows of each card", "Monitoring and settings"),
    ),
    verified=VERIFIED,
)

BOOT = Api(
    name="Start at sign-in",
    maker="Microsoft (Task Scheduler) and the systemd project",
    about=("An instance run here, started by the system when you sign in, on your confirmed click (At boot…): on "
           "Windows a scheduled task at your logon (your rights, never the highest), on Linux a systemd user unit. "
           "Each runs the platform's launcher (python -m liminal_libs.machines.launch serve), which reads the "
           "instance through your login at each start: no key is in a task, a unit or a command line."),
    kind="system",
    base="a task LiminalAI-llamacpp-<instance>; a unit ~/.config/systemd/user/liminal-ai-llamacpp-<instance>.service",
    version="Windows 11's ScheduledTasks module; systemd's user manager",
    auth="as you: no administrator, no root",
    limits="",
    costs="",
    links=(
        Link("Register-ScheduledTask", TASKS + "register-scheduledtask" + WS, "reference"),
        Link("Task Scheduler", LEARN + "windows/win32/taskschd/task-scheduler-start-page", "docs"),
        Link("systemd.service(5)", SYSTEMD + "systemd.service.html", "reference"),
        Link("systemctl(1)", SYSTEMD + "systemctl.html", "reference"),
        Link("systemd.unit(5): user units", SYSTEMD + "systemd.unit.html", "docs"),
    ),
    areas=(
        Area("Windows: a task at logon", "registered, read, started, stopped and removed",
             TASKS + "register-scheduledtask" + WS, covered="At boot… on Windows"),
        Area("Linux: a user unit", "written, enabled, started, stopped, disabled", SYSTEMD + "systemctl.html",
             covered="At boot… on Linux (systemd only: said where it does not run)"),
    ),
    calls=(
        Call("Register-ScheduledTask -AtLogOn", "a task at your logon, through the launcher",
             "Windows: a task at logon",
             used="At boot… on", writes=True, docs=TASKS + "register-scheduledtask" + WS),
        Call("Get-ScheduledTask | Get-ScheduledTaskInfo", "its state and last result", "Windows: a task at logon",
             used="the instance's state at boot", docs=TASKS + "get-scheduledtask" + WS),
        Call("Start-ScheduledTask / Stop-ScheduledTask", "the task run or ended now", "Windows: a task at logon",
             used="Start and Stop of an instance that starts at boot", writes=True,
             docs=TASKS + "start-scheduledtask" + WS),
        Call("Unregister-ScheduledTask", "the task removed", "Windows: a task at logon",
             used="At boot… off, Forget…",
             writes=True, docs=TASKS + "unregister-scheduledtask" + WS),
        Call("systemctl --user daemon-reload / enable", "the unit read and enabled for default.target",
             "Linux: a user unit", used="At boot… on", writes=True, docs=SYSTEMD + "systemctl.html"),
        Call("systemctl --user show", "the unit's state", "Linux: a user unit", used="the instance's state at boot",
             docs=SYSTEMD + "systemctl.html"),
        Call("systemctl --user start / stop", "the unit started or stopped now", "Linux: a user unit",
             used="Start and Stop of an instance that starts at boot", writes=True, docs=SYSTEMD + "systemctl.html"),
        Call("systemctl --user disable --now", "the unit stopped and disabled", "Linux: a user unit",
             used="At boot… off, Forget…", writes=True, docs=SYSTEMD + "systemctl.html"),
    ),
    verified=VERIFIED,
)

APIS = (SERVER, COMMAND, RELEASES, HUB_API, NVIDIA_SMI, BOOT)
