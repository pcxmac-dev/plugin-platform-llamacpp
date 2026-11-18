# Changelog

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
  Windows; CUDA 12 from 525; else Vulkan, under WSL the processor's), every archive pinned by name, size and sha256
  and checked, unpacked into the plugin's own folder, its program run to say build 12359 and its devices before it is
  put in use. An install found on PATH is your own, never replaced.
- **Instances**: a server run here (always a router, bound to this machine unless shared, its port its own, no other
  site's page allowed to read it, llama.cpp's own variables dropped, its key by environment alone), one found here, or
  another machine's; Start, Stop, At boot, Share, Key and the conflicts as the base gives them.
- **Models**: Find a model at the Hugging Face Hub (its files picked by llama.cpp's own rules, their size and licence
  and what fits this GPU said first), Download… as a job through the router (asked twice, Cancel pauses it), Delete…,
  Load…, Unload…, Details, Try it.
- **What leaves this machine**, said on the page: no telemetry, no update check; downloads on your click; and the
  Hub asked for a downloaded model's files each time it is loaded (llama.cpp's own way, with no switch that keeps it
  off while its downloads work).
- Six API cards (llama-server's API, the command, the releases, the Hugging Face Hub API, nvidia-smi, Start at
  sign-in), `api_facts()`, `READ_HOOKS` and `KEEPS` for the Internal API tab, a command line, and tests over stand-ins
  for everything outside.

Needs liminal-libs>=1.62.0.
