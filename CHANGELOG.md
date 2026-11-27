# Changelog

## 0.1.2 (2026-11-27)

Its lookups are asked only from the page: the review of item 14 (plugins-os-ai P3-1), so that the four engines
agree (vLLM 0.1.3 and LMDeploy 0.1.1 did the same with their `find` and `clear_found`). The user, 2026-11-26, of
reads that reach outside or with a key: "Page-only".

- **What only the page asks, besides its changes**: `find` (a lookup at the Hugging Face Hub about a name of the
  asker's choosing, kept an hour for a download's second confirmation), `clear_found` (it empties what the Models tab
  keeps) and `details` (it asks an instance, another machine's with its key). Another plugin's ask of one and an ask
  that names no one are refused (403) before the Hub or an instance is asked. The Models tab asks them as the page,
  as before.
- **Before**, the three were in `READ_HOOKS`: anyone on this PC could have the Hub asked, the lookups forgotten, or
  an instance asked about a model, with no click.
- It has no reads of its own now; the base's (`choose`, `log_of`, `models`, `serving`, `models_of`) and `status` are
  asked by anyone, as before. What it sends the Hub names this release (`User-Agent: plugin-platform-llamacpp/0.1.2`).
- Tests: `tests/test_who_asks.py`, each of the three refused to another plugin and to no one, nothing asked and the
  kept lookup still there, the page's three answered; `tests/test_contract.py`'s declaration.

## 0.1.1 (2026-11-25)

Its own changes are asked only from the page (`PAGE_ONLY_HOOKS`, liminal-libs 1.66.0's word): the independent review's
finding PLT-H1, the sweep of every plugin's hooks that followed it, and the review of that sweep (2026-11-25). The
user, 2026-11-25, chose "Plugin releases": each plugin says which of its hooks only its page asks.

- **What only the page asks**: `pull`, `delete_model`, `load`, `unload` and `build`. Another plugin's ask of one is
  refused (403), and so is an ask that names no asker; liminal-platform 0.34.0 refuses one over HTTP (401) unless its
  login shows that you are at the page. A platform that holds no login asks them as before. The 401 rests on the
  login's word that your browser is signed in, and a local login signs `secret` and `default` in with no password, so
  a program on this PC can sign itself in and ask them still: a proof that a program cannot make is the login's and
  the platform's to give, not this plugin's.
- **Before**, the platform handed an ask of any program on this PC (`POST /api/machines/llamacpp/ask` or `/job`, with
  a header anyone can send and a Host the client chooses, no sign-in) straight to the hook it named. `delete_model`
  with `[instance, model]` deleted a model its router had downloaded, asked once, with the instance's key. `pull`
  downloaded a model after a `find` anyone asks, its licence accepted by an argument the asker sets
  (`accept_licence`). `load` had a router load a model, `unload` took one out from under whoever was using it, and
  `build` changed the build the next Install brings.
- **The base's changes** (an install, an instance added, a start, a stop, a key, a prompt sent) are liminal-libs
  1.99.0's to say, on `ModelServer`: this plugin names its own five alone, and the names add up along the class line.
  On an older library those are asked by anyone still.
- **As before**: its reads are asked by anyone (`find`, `clear_found`, `details`, and the base's), and so are `serving`
  and `models_of`, which the harnesses ask through the platform. The library floor stays 1.62.0: a library before
  1.66.0 does not read the word, and liminal-platform 0.34.0 reads it from the plugin itself. What it sends the Hub
  names this release (`User-Agent: plugin-platform-llamacpp/0.1.1`).
- Tests: `tests/test_who_asks.py`, through the library's own App over the stand-in router and Hub: each of the five
  refused to another plugin and to no one, nothing asked of the router or the Hub and no build kept; the same five
  asked from the page, each sent or kept; the reads and `serving` answered to a plugin. A hook taken out of
  `PAGE_ONLY_HOOKS` fails it. Skipped on a library before 1.66.0.

## 0.1.0 (2026-11-18)

The first release: llama.cpp for Liminal (the user, 2026-11-16: "claude i need 3 plugins for inference serving local
models on the platform, :: lama.cpp, ollama and vLLM"; "they will be classified as AI plugins"; "all three plugins need
to allow installation and configuration of their respective code"; "also remote connection to instances if
possible"; "the plugins manage the services, the app can be started @ boot like a service, ports can be configured via
the plugin, the plugin should report conflicts"; 2026-11-17: "all plugins are middleware"). A machine plugin at plugin
API 2.2 on liminal-libs 1.62.0's model server base, under the AI heading's Inference group (`SECTION = "ai"`,
`GROUP = "inference"`, `order = 510`, `ICON = "llamacpp"`).

- **Install, update, uninstall**: llama.cpp 1.0.0 (build b12359), its build chosen on the page or settled as it
  installs by what nvidia-smi reads (CUDA 13 from driver R620, or from 580 for a card of compute capability 12 on
  Windows; CUDA 12 from 525, NVIDIA's caveat said; else Vulkan, under WSL the processor's), every archive pinned by
  name, size and sha256 and checked, taken from GitHub's one storage host the figures saw, unpacked into the plugin's
  own folder, its program run to say build 12359 and its devices before it is put in use (a Cancel meanwhile
  honoured). An install found on PATH is your own, never replaced. The command line's `install` says the size, the
  hosts and the licence before it asks for `--accept-licence`.
- **Instances**: a server run here (always a router, bound to this machine unless shared, its port its own, no other
  site's page allowed to read it, offline unless its Offline setting is off, `--no-agent`, llama.cpp's own variables
  and the others it reads dropped, its key by environment alone, refused while a config.ini turns on remote access or
  MCP servers), one found here, or another machine's; Start, Stop, At boot, Share, Key and the conflicts as the base
  gives them.
- **Models**: Find a model at the Hugging Face Hub (its files picked by llama.cpp's own rules, their size and licence
  and what fits this GPU said first; the quantization typed kept, nothing offered where the Hub says there is nothing
  to download), Download… as a job through a router started online (asked twice, Cancel pauses it), Delete…, Load…,
  Unload…, Details (the context the harnesses are told among it), Try it.
- **What leaves this machine**, said on the page: no telemetry, no update check; a server started here asks
  huggingface.co nothing unless its Offline setting is off (online, llama.cpp asks the Hub at each load of a model it
  downloaded and fetches a newer revision by itself); downloads on your click.
- Six API cards (llama-server's API, the command, the releases, the Hugging Face Hub API, nvidia-smi, Start at
  sign-in), `api_facts()`, `READ_HOOKS` and `KEEPS` for the Internal API tab, a command line, and tests over stand-ins
  for everything outside.

Needs liminal-libs>=1.62.0.
