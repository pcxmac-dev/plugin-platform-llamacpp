# plugin-platform-llamacpp

llama.cpp for Liminal: llama.cpp's server installed, started and stopped on this machine as a router, found here or
reached on another machine, its GGUF models looked up at the Hugging Face Hub, downloaded, loaded, unloaded, deleted
and asked a prompt, every change confirmed on the platform's page.

The user's words (2026-11-16): "claude i need 3 plugins for inference serving local models on the platform, :: lama.cpp,
ollama and vLLM, i also need 3 harness plugins for the platform :: OpenCode, Aider, and Continue"; "they will be
classified as AI plugins"; "all three plugins need to allow installation and configuration of their respective code";
"also remote connection to instances if possible"; "the plugins manage the services, the app can be started @ boot
like a service, ports can be configured via the plugin, the plugin should report conflicts". Then (2026-11-17): "all
plugins are middleware", "even for other plugins", "and then the app object code".

## What it does

A machine plugin of the platform (plugin API 2.2) on liminal-libs 1.62.0's model server base
(`liminal_libs.machines.modelserver.ModelServer`), drawn under the AI heading's Inference group (`SECTION = "ai"`,
`GROUP = "inference"`, `order = 510`), its icon llama.cpp's mark (`ICON = "llamacpp"`, drawn anew by the library).
It is middleware: below it llama.cpp (its server's HTTP API, its command, its release files) and the Hugging Face
Hub; above it the platform's object code and the other plugins, the harnesses asking its `serving` through the
platform (`NATIVE = "llamacpp"`: OpenAI-compatible at `<address>/v1`).

- **Install, update, uninstall** (`release.py`). The pinned llama.cpp 1.0.0, whose binaries are build b12359, in the
  build chosen with **The build…** (kept through your login): Automatic, CUDA 13, CUDA 12, Vulkan or the processor's.
  Automatic is settled as it installs, by what nvidia-smi reads then: no NVIDIA GPU, the Vulkan build (under WSL the
  processor's); a driver of CUDA 13.5's own branch, R620, or newer, the CUDA 13 build; below it, on Windows, a card of
  compute capability 12 (an RTX 50) gets the CUDA 13 build from driver 580 on (the CUDA 12.4 build carries no code of
  its own for it), else Vulkan; otherwise the CUDA 12 build from driver 525 on, else Vulkan (under WSL the
  processor's). The job's first line says which and why (a CUDA 12 build by NVIDIA's minor version compatibility with
  its caveat: what is new in the build's CUDA 12 release may be refused by a driver older than that release's own).
  Each archive is pinned here by name, size and sha256 as the release's metadata gives them (its attestation repeats
  every sum; llama.cpp publishes no checksum file), downloaded from github.com, which sends it on to its storage,
  taken from `release-assets.githubusercontent.com` alone (a redirect to any other host is refused in words, nothing
  asked of it), checked against the pin, unpacked into the plugin's own folder (`app/b12359-<build>/`,
  `app/CURRENT`), and `llama-server --version` must then say build 12359 before it is put in use; its
  `--list-devices` lines are kept with the install (`liminal-install.json`), and a CUDA build that lists no CUDA
  device is kept and said; a Cancel pressed while that check runs is honoured (nothing put in use). For you alone: no
  administrator, no root. Uninstall is asked twice; your models, logs and instances stay. An install found on PATH
  (llama-server, or llama.cpp's one program `llama`, started as `llama serve`) is your own: used as it is, never
  replaced or removed from here, its build read by **Read again** alone.
- **Instances** (the base's, `server.py` for llama.cpp's part). A server run here, one found on this machine, or
  another machine's by its address and an optional key (plain http to another machine asked twice). A server run
  here is always a **router**: llama-server given no model, which loads a model when it is first asked (at most
  **Models loaded at once**), unloads it after **Unload after idle**, and runs each loaded model as a program of its
  own on a loopback port it chooses itself (the tool's doing, not a setting). Its port is its instance's own (9931 by
  default) and its conflicts are said before a start and in the status: a port of Liminal's apps, another instance's
  (of this plugin or another inference plugin), a program holding it by name and pid, WSL's relay. It binds to
  127.0.0.1 unless shared (asked twice, and only with a key). Start and Stop by a click, as a windowless child the
  plugin owns (Ctrl+C on Windows, SIGTERM on Linux, then ended with what it started); **At boot…** as a scheduled task
  at your logon or a systemd user unit. Its settings: **Models folder** (`LLAMA_CACHE`; empty: the plugin's own),
  **A folder of GGUF files of yours** (`--models-dir`), **Context size** (`-c`), **Layers on the GPU** (`-ngl`),
  **Models loaded at once** (`--models-max`, 4), **Unload after idle** (`--sleep-idle-seconds`, 300 here; 0: never),
  **Offline** (`LLAMA_ARG_OFFLINE=1`, on by default: see What leaves this machine). Every start carries
  `--no-agent`, and is refused while a config.ini of yours turns on remote access or MCP servers.
- **Models** (`models.py`, the Models tab). **Find a model**: `user/repository[:quantization]` asked of the Hub's API
  without a token: its card (gated or not, its licence) and its files, page by page, the ones llama.cpp would take
  picked by its own rules (the quantization named, else Q4_K_M, else Q8_0, else the first; a split file's parts; its
  projector), their size and what fits this machine's GPU said before anything is downloaded; kept an hour, the
  quantization typed kept whatever the Hub answers. Where the Hub says there is nothing to download (no such
  repository, or none of its files for the quantization typed), Download… is off for it; where the Hub does not
  answer, it can be downloaded unseen, asked twice, the quantization as typed and the projector as the router picks
  it. **Download…** is a job through the router's own API (b9688 or newer), refused before its first line from what
  is kept alone, then asked twice with its size, files, folder, free space and licence; its progress in the panel;
  **Cancel** pauses it, llama.cpp keeping what came for the next download; it stops by itself after 6 hours. A server
  run here that is started offline (the default) downloads nothing: Download… is off for it, with how to turn
  Offline off (Change… while it is stopped, then Start).
  **Delete…** (never a model of your GGUF folder: llama.cpp deletes only what it downloaded), **Load…**, **Unload…**,
  **Details**, and the base's **Try it** (a prompt answered through `/v1/chat/completions`, bounded).

Nothing is asked of a server while its page is not shown; nothing runs by itself; nothing is sent anywhere unasked.

### What leaves this machine

For a server started here (the page's table of the same name):

- llama.cpp has **no telemetry and no update check**.
- **A server started here is offline unless you turn its Offline setting off** (`LLAMA_ARG_OFFLINE=1`, which the
  router hands on to the model programs it starts): it asks huggingface.co nothing and loads each model it downloaded
  from its folder as it is. The reason is what llama.cpp does online (its common/download.cpp, common/hf-cache.cpp and
  tools/server/server.cpp at b12359): each time it loads a model it downloaded, it first asks huggingface.co for that
  repository's list of files (two requests, no token, its own User-Agent), moves its folder's `refs/main` to the newest
  revision, and, when the repository has moved on, downloads the newer files by itself before it loads the model,
  gigabytes perhaps, with no size, licence or Cancel shown here. Offline also stops its router's own downloads, so
  **Download…** is off for a server started offline: turn Offline off with Change… (while it is stopped), start it,
  download, and turn Offline on again if you like. The start's confirmation says which it will be. The other way is
  under Open for the user. A model of your GGUF folder is loaded from its file alone.
- A **download** happens on your click alone, by the server itself (with Offline off), from huggingface.co, without a
  token: public models only (a gated one is said and not downloaded; no Hugging Face token is given to llama.cpp in
  this version).
- **Find a model** asks the Hub's API, on your click, with this plugin's name and version as its User-Agent.
- Its remote access (`--connect`), tools, agent and MCP are **never switched on** here: every start carries
  `--no-agent` (no tools, no MCP proxy, whatever a config.ini says), and a start is refused while a config.ini of
  yours turns on remote access (`connect`) or MCP servers (`mcp-servers-config`, `mcp-servers-json`), which no switch
  turns off. A found or remote server with remote access on says so on its Models tab (its share code is never kept,
  only said as on).
- **No other site's page can read it**: its one allowed origin is its own address (`LLAMA_ARG_CORS_ORIGINS`), where
  llama.cpp's own default lets any page in your browser read it. Its chat page (Llama Chat) stays at its own address;
  what it asks of other sites was not read from its source (Not yet confirmed, below).
- A **config.ini of yours** (`%APPDATA%\llama.cpp\config.ini`, `%PROGRAMDATA%\llama.cpp\config.ini`,
  `~/.config/llama.cpp/config.ini`, `/etc/llama.cpp/config.ini`) is read by llama.cpp at every start, before its
  variables and arguments: the page names one where it exists, and what in it turns on what a server started here
  never has (a start is then refused) or what `--no-agent` turns off; it is read, never written.
- Every llama.cpp, ggml and Hugging Face variable of the platform's environment is dropped from the server's
  (`LLAMA_*`, `GGML_*`, `HF_TOKEN`, `HF_ENDPOINT`, `MODEL_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`,
  `HUGGINGFACE_HUB_CACHE`), and the others llama-server reads: Vertex AI's `AIP_*` (`AIP_HTTP_PORT` would move it off
  its port) and `MTMD_*` (the projector's device); with every pair of Liminal's. `CUDA_VISIBLE_DEVICES` and your
  proxy variables stay (your choice of card, your network). Its key travels by its environment alone, never on its
  command line.

## Its API

What it attaches to, as its page shows it (`apis.py`, read on the makers' pages on 2026-11-18; llama.cpp's own pages
at the pinned build's tag):

- **llama-server's API** (the ggml-org project): `GET /health`, `/props`, `/models`, `/v1/models`, `/models/sse`,
  `POST /models`, `/models/load`, `/models/unload`, `DELETE /models` and `POST /v1/chat/completions` used; the rest
  named. The [server's README](https://github.com/ggml-org/llama.cpp/blob/b12359/tools/server/README.md) and its
  [router section](https://github.com/ggml-org/llama.cpp/blob/b12359/tools/server/README.md#using-multiple-models),
  the [API overview](https://llama.app/docs/api), the
  [security policy](https://github.com/ggml-org/llama.cpp/blob/b12359/SECURITY.md).
- **llama-server, the command**: the router's line (`--no-agent` among it, `LLAMA_ARG_OFFLINE` in its environment),
  `--version`, `--list-devices`, `llama serve` and `llama version`.
  [Running a server](https://llama.app/docs/serve), [presets and config.ini](https://github.com/ggml-org/llama.cpp/blob/b12359/docs/preset.md),
  [the backends](https://github.com/ggml-org/llama.cpp/blob/b12359/docs/build.md).
- **llama.cpp's releases**: the twelve archives of [b12359](https://github.com/ggml-org/llama.cpp/releases/tag/b12359)
  (named by [v1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0)), each pinned; its
  [attestation](https://github.com/ggml-org/llama.cpp/attestations/67120335), llama.cpp's MIT licence and
  [NVIDIA's CUDA licence](https://docs.nvidia.com/cuda/eula/index.html) for the runtime archives.
- **Hugging Face Hub API**: `GET /api/models/{repo}` (its card: what the Hub's own client asks) and
  `GET /api/models/{repo}/tree/main` (its files, paged by its Link header). [The Hub's API](https://huggingface.co/docs/hub/api),
  [its OpenAPI document](https://huggingface.co/.well-known/openapi.md), [model cards](https://huggingface.co/docs/hub/model-cards),
  [licences](https://huggingface.co/docs/hub/repositories-licenses), [gated models](https://huggingface.co/docs/hub/models-gated).
- **nvidia-smi** (NVIDIA): the cards' memory and driver (the Models tab, and the build Automatic settles) and, inside
  the install's job on Windows, each card's compute capability. [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
  [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
  [minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).
- **Start at sign-in**: the task and the user unit At boot… makes (the library's `machines/autostart.py`).

`api_facts()` says what it kept, asking nothing: each instance's last answer, the Hub lookups kept, the pin.
`LIMINAL_VENDOR_PAGES=1 pytest tests/test_apis.py` checks that every address on the cards still answers (that test
alone lifts the tests' guard, as the Docker plugin's does: it reaches the makers' pages).

## Its hooks

The base's (`machines/modelserver.py`): `present`, `status`, `refresh`, `install(accept_licence, log)` (a job, Cancel),
`uninstall(accept_removal)`, `choose`, `add_instance`, `change_instance`, `remove_instance`, `set_key`, `forget_key`,
`share`, `start`, `stop`, `boot`, `log_of`, `models` (the Models tab), `try_it`, and the two services other plugins
ask through the platform, `serving(name="")` and `models_of(name)` (the base's answers, a model still downloading or
not whole left out: a harness is never offered one the router would refuse). Its own: `find(model)`, `clear_found()`,
`pull(name, model, accept_licence, log)` (a job, Cancel), `delete_model(name, model)`, `load(name, model)`,
`unload(name, model)`, `details(name, model)`, `build(which)`. For the Internal API tab it declares no reads of its own
(the base's `READ_HOOKS` alone) and what it keeps through the login (`KEEPS`: its instances,
each instance's key as its own pair, the build). A refusal that a second confirmation accepts names its argument
(`accept`), and the page asks again before sending it.

Who may ask them (0.1.1; the sweep after the independent review's finding PLT-H1, 2026-11-25): its own changes, `pull`,
`delete_model`, `load`, `unload` and `build`, are the plugin's `PAGE_ONLY_HOOKS` (liminal-libs 1.66.0's word). Another
plugin's ask of one is refused, and liminal-platform 0.34.0 refuses one over HTTP unless its login shows that you are
at the page. Before, any program on this PC could ask them with two headers: a model deleted from a router, or one
downloaded with its licence accepted for you. The base's changes are liminal-libs 1.99.0's to say (on `ModelServer`);
the base's reads and the two services are asked by anyone, as before. Its lookups are the page's too (0.1.2; the
review of item 14, P3-1, as vLLM's and LMDeploy's are): `find` asks the Hub about a name of the asker's choosing,
`clear_found` empties the Models tab's lookups, and `details` asks an instance, another machine's with its key, so
another plugin's ask of one, or one that names no one, is refused. The platform's check rests on its login's word that
your browser is signed in, and a local login signs `secret` and `default` in with no password: it does not stop a
program on this PC that signs itself in.

## From the command line

```
python -m plugin_platform_llamacpp status
python -m plugin_platform_llamacpp serving
python -m plugin_platform_llamacpp models default
python -m plugin_platform_llamacpp launch default          # the lines to paste: its key the placeholder you replace
python -m plugin_platform_llamacpp find ggml-org/gemma-3-4b-it-GGUF:Q4_K_M
python -m plugin_platform_llamacpp install [--accept-licence]
```

## Decisions and departures

Built to DECISIONS.md part A, A-inference.md 4.0 and 4.2, A-foundation.md 4, A-revisions.md 3 to 5 and the review of
the figures (review/fix-figures.json; its findings numbered from 1 as that file lists them: its llama.cpp findings
12, 13, 14 and 32, taken as binding), then the review of this plugin's 0.1.0; where this plugin differs from its
plan:

- One layout for every install (A-revisions M11): `app/<pin>/` and `app/CURRENT`, with `liminal-install.json` beside
  the programs, in place of `app.new` and `app/install.json`. The pin is `b12359-<build>`; Automatic accepts any
  build of b12359 in use, so choosing a build of its own offers **Update…** (a new folder, put in use once whole).
- Each archive pinned by name, size and sha256, no size asked of the server (finding 12, "every archive has a
  published sha256"); the words say "checked against the size and sha256 pinned in this plugin". Downloads are kept
  under `downloads/b12359/` (the Windows runtime archives carry no build in their names: finding 32, the smaller
  refinements) and removed once the install is in use. A release file comes from `release-assets.githubusercontent.com`
  alone, the one storage host the figures saw (the plan's `objects.githubusercontent.com`, seen for none, is not
  allowed: a redirect there is refused in words, as A-inference 10 asks).
- CUDA 13 from driver 620 (R620, A-revisions M9; finding 13, the driver thresholds), CUDA 12 from 525 on both systems
  (finding 13 asked for one meaning: NVIDIA's minor version compatibility, its caveat said, against 551.78 on Windows
  and 570.124.06 on Linux, the builds' own CUDA 12 releases' drivers), and on Windows a card of compute capability 12
  gets the CUDA 13 build from 580 on, else Vulkan (finding 14, RTX 50 cards, which changes M9: on this PC's RTX 5090
  Laptop GPU with driver 595.71, M9 alone would bring the CUDA 12.4 build, which has no code for that card). The
  capability is read with `nvidia-smi --query-gpu=index,compute_cap` inside the job; where it cannot be read, M9's
  rule stands. Under WSL a driver older than 525 brings the processor's build, as no card does (no Vulkan case there).
- Both Linux tarballs are unpacked without their top folder into one folder (finding 32: the runtime's tarball has one
  too); the build is read by "build 12359" (its version text is `1.0.0-dev`).
- Windows' Visual C++ runtime and the disk's free space are refused before the job's first line (an override of the
  base's `install`), not inside the job.
- A server run here is started offline by default (its **Offline** setting, `LLAMA_ARG_OFFLINE=1`), where the plan
  had it online "reaching the network only for a download you ask for": llama.cpp's source at b12359 asks the Hub
  for a downloaded model's files at every load unless offline, and downloads a newer revision by itself when the
  repository has moved on (the review of 0.1.0, its high finding). Offline also stops the router's downloads, so
  Download… is off for such a server until Offline is turned off; the other way is under Open for the user.
- Every start carries `--no-agent`, and is refused while a config.ini of yours turns on remote access or MCP servers:
  the plan's "never `--connect`, `--tools`, `--agent`, an MCP file" held against a config.ini too, which llama.cpp
  applies at every start before its variables and arguments.
- A model's `context`, as the harnesses read it, is what the server runs it with (A-inference 6): the instance's
  Context size (the router hands it every model it loads), else what a router with its memory accounting on says
  of a loaded one (`memory.effective.n_ctx`; never on for a server started here), else not known, and Details says
  which; the context a model was trained for is in its Details (4.2's table had `context_length` there).
- `GGML_*` is dropped too (A-inference 4.0's rule; the figures review's finding 10 for Ollama, whose runner is
  llama-server), and `AIP_*` and `MTMD_*`, which llama-server reads too.
- A server behind a proxy that names itself in the `Server` header is still recognised by its `/props`.
- A lookup picks the files by llama.cpp's own rules, ported from its source, and keeps what is downloaded as llama.cpp
  names it (`user/repo:QUANTIZATION`, the quantization typed kept whatever the Hub answers); a licence `other` is said
  by its `license_name` (finding 32). Where the Hub says there is nothing to download, nothing is offered; a download
  unseen (the Hub did not answer) leaves the sidecars to the router.
- A download's event stream that goes quiet for 30 s, or breaks off, is followed again once the router says how the
  download is; a router that stops answering ends it in words. A model downloading cannot be deleted, and a model
  downloading or not whole is left out of what the harnesses are told (`serving`, `models_of`).
- The command line has `find` and `install [--accept-licence]` (A-critique M5) besides the plan's reads; `install`
  prints what the page's confirmation says (what is downloaded, from where, its size, what is checked, the licence)
  before anything else.

## Open for the user

- **Offline by default, or downloads that need no restart.** As built, a server started here is offline unless its
  Offline setting is off: nothing is asked of huggingface.co at a load, and its router downloads nothing, so a
  download means Change… (Offline off), Start, Download…, and Offline on again if you like. The other way: this
  plugin downloads the files a lookup picked itself, as a job with their size, licence, sha256 (the Hub's own) and
  Cancel, into llama.cpp's own cache (its Hugging Face layout of `models--`, `blobs`, `snapshots` and `refs`), with the
  router always offline, so Download… works without a restart; at the price of writing that layout the way llama.cpp
  would, which only a real run confirms. Your word decides; until then, the first.

## Not yet confirmed by a real run

Nothing was installed, started or downloaded while this plugin was built (A-revisions 3): the user's first click after
landing tries each of these.

- GitHub's redirect of each pinned archive to `release-assets.githubusercontent.com` (a header-only read saw it for all
  twelve; a redirect to any other host is refused in words), a download resumed.
- The archives' layout as the release workflows make them: the Windows zips flat (the runtime's three DLLs beside
  llama-server.exe), the Linux tarballs each with one top folder and links kept inside; b12359's version line on
  standard error; what `--list-devices` prints here.
- Whether the CUDA 13.5 build sees this PC's RTX 5090 Laptop GPU on driver 595.71 (NVIDIA's minor version
  compatibility; what is new in 13.5 needs R620), and that `nvidia-smi --query-gpu=index,compute_cap` answers on this
  driver (the field is named by nvidia-smi's own `--help-query-gpu`, not on NVIDIA's page).
- Whether llama.cpp's Windows zips need the system's Visual C++ runtime (the refusal rests on its winget package's
  dependency); the install's check says when the program does not start.
- The router as read from its source and README at b12359: `GET /models`' fields (no `memory` without its memory
  accounting; with it, a loaded model's `memory.effective.n_ctx`), its event stream's `data:` lines and
  `download_progress`' data, `download_failed` with `{"stopped": true}` after a pause, the pause by `POST
  /models/unload`, `POST /models` asking the Hub before it answers.
- Offline: that `LLAMA_ARG_OFFLINE=1` in the router's environment keeps the router and the model programs it starts
  offline (they inherit its environment: tools/server/server-models.cpp), that a model it downloaded loads from its
  folder so, and that its `POST /models` then refuses a model not in its folder ("resolves no file to download"),
  which is why Download… is off for it. Online: that a load of a model it downloaded asks the Hub for its refs and
  tree, moves its folder's `refs/main` and downloads the newer files by itself when the repository has moved on (read
  in common/download.cpp, common/hf-cache.cpp and tools/server/server.cpp, never seen).
- `--no-agent` reaching the router's model programs through the arguments it hands on, and a config.ini's keys as
  this plugin reads them (an argument without its dashes, or its `LLAMA_ARG_` name, in any section; a value of `off`,
  `disabled`, `false` or `0` leaves a switch off): llama.cpp's own reader of the file was not read.
- A found llama.cpp older than b12359: whether it knows `LLAMA_ARG_OFFLINE` and `--no-agent` (one that does not know
  `--no-agent` refuses to start, its log says so; one that does not know the variable would load online).
- Llama Chat, the server's own page: what it asks of other sites (the Hugging Face Hub for a model's details, say)
  and when; its source at b12359 was not read.
- The Hub's two answers as this plugin reads them (the model card's `gated`, `cardData` and `tags`; the tree's entries
  and its Link header), and how often the Hub answers an address without a token.
- A windowless llama-server on Windows ending at Ctrl+C (the library's console helper), its router's model programs
  ending with it (the job object), and a server started at sign-in (task, user unit).
- Llama Chat with its origin set to the server's own address, and the harnesses reaching `/v1` (no Origin header).
- `llama-server --version` and `--list-devices` run with the platform's environment less Liminal's pairs (the base's
  `short`): a `LLAMA_ARG_*` of yours there could make the install's check fail.
- The disk's needs: 3 GB free asked before an install (the unpacked sizes are not published).

## Install

It is installed with the platform's other plugins; it needs liminal-libs 1.62.0 (`pip install -e .` in this folder
for a checkout). Until that library release has landed it stays in `incoming/.staging/`, which the platform's loader
does not read.

## Tests

```
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
```

Stand-ins only: nothing starts a process, opens a socket, resolves a name or reaches the network (the library's
guard, copied into `tests/conftest.py`, fails a test that does), and nothing is installed or downloaded. The answers
in `tests/data` are written from the shapes llama.cpp's sources and README at b12359 give and the Hub's API document's,
not recorded from a real server.
