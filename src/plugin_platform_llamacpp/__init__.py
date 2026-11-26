"""llama.cpp for Liminal (plugin-platform-llamacpp 0.1.0): llama.cpp's server installed, started and stopped on this
machine as a router, found here or reached on another machine, its GGUF models looked up, downloaded, loaded and asked.

The user's words (2026-11-16, in order): "claude i need 3 plugins for inference serving local models on the platform,
:: lama.cpp, ollama and vLLM, i also need 3 harness plugins for the platform :: OpenCode, Aider, and Continue"; "they
will be classified as AI plugins"; "all three plugins need to allow installation and configuration of their
respective code"; "also remote connection to instances if possible"; "the plugins manage the services, the app can be
started @ boot like a service, ports can be configured via the plugin, the plugin should report conflicts". Then
(2026-11-17): "all plugins are middleware", "even for other plugins", "and then the app object code".

A machine plugin of the platform on liminal-libs 1.62.0's model server base (machines/modelserver.py: ModelServer),
drawn under the AI heading's Inference group: it is middleware between llama.cpp (its server's HTTP API, its
command, its release files, the Hugging Face Hub its models come from) and the platform's object code and the other
plugins (the harnesses ask its `serving` through the platform: NATIVE "llamacpp", OpenAI-compatible at /v1).

- Install, update, uninstall (release.py): the pinned llama.cpp 1.0.0 (build b12359), the build chosen (automatic,
  CUDA 13, CUDA 12, Vulkan, the processor's) into its own folder, each archive checked against the size and sha256
  pinned here; an install found on PATH is the user's own, used as it is and never replaced.
- Instances (the base's): a server run here (always a router, server.py), one found on this machine, or another
  machine's by its address and an optional key; each one's port its own setting, its conflicts said before a start
  and in the status; Start and Stop as a windowless child the plugin owns; At boot as a task or a user unit.
- Models (models.py): looked up at the Hugging Face Hub (size and licence before anything is downloaded),
  downloaded through the router as a job that can be cancelled, deleted, loaded, unloaded and described; Try it.

Nothing is asked of a server while its page is not shown, nothing runs by itself, and nothing is sent anywhere
unasked: llama.cpp has no telemetry and no update check; a server started here is offline unless its Offline setting
is off (online, each load of a model it downloaded asks the Hub first and may fetch a newer revision by itself); its
remote access, tools, agent and MCP are never switched on here (--no-agent, and a start refused while a config.ini
turns on what no switch turns off); and what still leaves the machine is said on the page (What leaves this machine).

    python -m plugin_platform_llamacpp status | serving | models NAME | launch NAME | find MODEL
    python -m plugin_platform_llamacpp install [--accept-licence]
"""
from __future__ import annotations

from pathlib import Path

from liminal_libs.instances import Refusal
from liminal_libs.machines.ai import Setting
from liminal_libs.machines.modelserver import ModelServer, ServerRun

from . import apis as _apis
from . import models as _models
from . import release as _release
from . import server as _server

__version__ = "0.1.1"
LIBS_FLOOR = "1.62.0"  # the AI plugins' bases (machines/ai.py, machines/modelserver.py) and llama.cpp's mark


class LlamaRun(ServerRun):
    """The plugin's run state (the base's and more), shared by its copies: its lookups, what Read again read of a
    found install, the download under way."""

    def __init__(self):
        super().__init__()
        self.found: list = []  # models.Found, the newest first
        self.installs: dict = {}  # a found install's {"build", "line", "devices"}, by its path
        self.pull = None  # the panel of the download under way: a detach cancels it


class FolderSetting(Setting):
    """A folder of a setting: a whole path, or left empty."""

    def clean(self, value):
        text = super().clean(value)
        if text is not None and not Path(text).is_absolute():
            raise Refusal(f"{self.label} is a whole path (from its drive or its root), not {text[:60]!r}")
        return text


class LayersSetting(Setting):
    """Layers on the GPU: auto, all or a whole number."""

    def clean(self, value):
        text = super().clean(value)
        if text is None:
            return None
        word = text.strip().lower()
        if not _server.LAYERS.fullmatch(word):
            raise Refusal(f"{self.label} is auto, all or a whole number, not {text[:20]!r}")
        return word


SETTINGS = (
    FolderSetting("models", "Models folder",
                  note="empty: this plugin's own (its folder's models); where its downloads go (LLAMA_CACHE)"),
    FolderSetting("gguf", "A folder of GGUF files of yours",
                  note="listed and loaded from there (--models-dir), never deleted from here"),
    Setting("context", "Context size (tokens)", "number", min=0, max=1_048_576,
            note="empty: the model's own, lowered by llama.cpp until it fits; a coding harness wants 32k or more"),
    LayersSetting("layers", "Layers on the GPU", default="auto", note="auto, all, or a number (-ngl)"),
    Setting("most", "Models loaded at once", "number", default=4, min=1, max=16,
            note="the one used longest ago is unloaded for the next (--models-max)"),
    Setting("idle", "Unload after idle (seconds)", "number", default=300, min=0, max=86400,
            note="0: never (llama.cpp's own default); 300 frees the GPU when nothing asks (--sleep-idle-seconds)"),
    # (the review of 0.1.0: started online, each load of a model it downloaded asks the Hub and may fetch a newer
    # revision by itself; offline, its router downloads nothing either)
    Setting("offline", "Offline", "choice", default="on",
            choices=(("on", "On", "it asks huggingface.co nothing: a model it downloaded loads from its folder as it "
                                  "is, and Download… is off"),
                     ("off", "Off", "Download… works; and each load of a model it downloaded asks huggingface.co "
                                    "first and fetches a newer revision of it by itself")),
            note="LLAMA_ARG_OFFLINE: on (the default), nothing is asked of huggingface.co; read when it starts"),
)


class LlamaCppPlugin(ModelServer):
    """llama.cpp's server on this machine or another (the module's words)."""

    name = "llamacpp"
    title = "llama.cpp"
    about = ("llama.cpp's server on this machine or another: installed, started and stopped here as a router, its GGUF "
             "models found, downloaded, loaded and asked.")
    order = 510
    ICON = "llamacpp"
    apis = _apis.APIS
    LIBS_FLOOR = LIBS_FLOOR
    DEFAULT_PORT = 9931
    NATIVE = "llamacpp"
    KEYED = True
    READY_S = 60.0
    SETTINGS = SETTINGS
    MODEL_COLUMNS = ("Details", "Load", "Delete")
    HOOKS = (*ModelServer.HOOKS, "find", "clear_found", "pull", "delete_model", "load", "unload", "details", "build")
    JOB_HOOKS = (*ModelServer.JOB_HOOKS, "pull")
    CANCEL_HOOKS = (*ModelServer.CANCEL_HOOKS, "pull")
    # What it says of the interface it shares (read by name by the Internal API tab): its own reads beside its
    # bases' (a lookup at the Hub, its kept lookups forgotten, a model's details); what it keeps through the login
    READ_HOOKS = ("find", "clear_found", "details")
    # Who may ask its own changes (0.1.1; liminal-libs 1.66.0's word, which liminal-platform 0.34.0 reads): a model
    # downloaded, deleted, loaded or unloaded, and the build Install brings, are asked only from its page, never by
    # another plugin nor by an ask that names no one. Its own five alone: the names add up along the class line, so
    # the base's changes (an install, a start, a stop, a key: liminal-libs 1.99.0 says them on ModelServer) are the
    # library's to say
    PAGE_ONLY_HOOKS = ("pull", "delete_model", "load", "unload", "build")
    RUN_STATE = LlamaRun
    arch: str | None = None  # this processor's word (platform.machine() when None): a test's stand-in

    def __init__(self, outside=None, *, arch: str | None = None):
        super().__init__(outside)
        self.arch = arch

    @property
    def KEEPS(self) -> tuple:  # noqa: N802 - a declared word of the contract, as READ_HOOKS
        """What it keeps through the login beside its keys."""
        return ((self.variable("instances"), "its instances: each one's name, where it is, its port, whether it is "
                                             "shared or starts at boot, and its settings"),
                (f"{self.variable('instances')[:-len('INSTANCES')]}<INSTANCE>_KEY", "an instance's key, its own pair"),
                (self.variable("build"), "the build Install brings: automatic, CUDA 13, CUDA 12, Vulkan or the "
                                         "processor's"))

    # -- what Install brings and what is installed (release.py) ------------------------------------------------------
    def package(self):
        return _release.package(self)

    def installed(self):
        return _release.installed(self)

    def read_version(self, installed) -> str:
        return _release.read_version(self, installed)

    def install_into(self, log, cancel) -> list[str]:
        return _release.install_into(self, log, cancel)

    def remove_from(self) -> list[str]:
        return _release.remove_from(self)

    def runs_here(self) -> str:
        return _release.runs_here(self)

    def install(self, accept_licence=False, log=None) -> dict:
        """llama.cpp installed (or updated to the build chosen) into the plugin's own folder, as a job: the base's,
        with this system's own refusals said first (Windows' C runtime missing, too little free on the disk)."""
        self._usable()
        pkg, have = self.package(), self.installed()
        if pkg is not None and not self.runs_here() and (have is None or have.how == "plugin"):
            in_use = self.current()
            if not (have is not None and in_use is not None and in_use.name == pkg.pin):
                why = _release.install_refusal(self)
                if why:
                    raise Refusal(why)
        return super().install(accept_licence, log)

    def build(self, which) -> dict:
        """The build Install brings, kept through the login: automatic, cuda13, cuda12, vulkan or cpu."""
        return _release.choose_build(self, which)

    # -- a server run here, and how one is recognised (server.py) ----------------------------------------------------
    def launch(self, instance, key: str = ""):
        return _server.launch(self, instance, key)

    def look(self, at, instance):
        return _server.look(self, at, instance)

    def start_words(self, instance) -> str:
        return _server.start_words(self, instance)

    def share_words(self, instance) -> str:
        return f"{super().share_words(instance)} {_server.share_words(self, instance)}"

    # -- models (models.py) --------------------------------------------------------------------------------------------
    def found_of(self, model):
        """The kept lookup of a model (models.Found), None when it was not looked up within the hour."""
        return _models.found_of(self, model)

    def find(self, model) -> dict:
        """A model looked up at the Hugging Face Hub: its files, size and licence, and what fits this GPU."""
        return _models.find(self, model)

    def clear_found(self) -> dict:
        """The lookups kept forgotten."""
        return _models.clear_found(self)

    def pull(self, name, model, accept_licence=False, log=None) -> dict:
        """A model downloaded by an instance's router, as a job that can be cancelled, asked twice."""
        return _models.pull(self, name, model, accept_licence, log)

    def delete_model(self, name, model) -> dict:
        """A model deleted from an instance's router."""
        return _models.delete_model(self, name, model)

    def load(self, name, model) -> dict:
        """A model loaded by an instance's router now."""
        return _models.load(self, name, model)

    def unload(self, name, model) -> dict:
        """A model an instance's router runs unloaded now."""
        return _models.unload(self, name, model)

    def details(self, name, model) -> dict:
        """What an instance says of one of its models."""
        return _models.details(self, name, model)

    def model_cells(self, instance, model) -> list:
        return _models.cells(self, instance, model)

    def more(self, shown, seen) -> dict:
        return _models.page(self, shown, seen)

    def main_more(self) -> dict:
        tables = [_server.leaves(self)]
        for table in (_release.lacks(self), _release.brings_table(self)):
            if table is not None:
                tables.append(table)
        forms = [] if self.runs_here() else [_release.build_form(self)]
        return {"tables": tables, "forms": forms}

    # -- what the other AI plugins are told (the base's services, without a model not whole yet) -------------------
    def _servable(self, name: str, models: list) -> list:
        seen = self.seen(name)
        if seen is None:
            return models
        unfit = {m["id"] for m in seen.models if m.get("status") in _server.UNSERVED}
        return [m for m in models if m.get("id") not in unfit]

    def serving(self, name: str = "") -> dict:
        """What this plugin offers the other AI plugins (the base's answer), without a model still downloading or
        whose download is not whole: a harness asking for one would be refused by the router."""
        out = super().serving(name)
        for row in out.get("instances") or []:
            row["models"] = self._servable(row["name"], row.get("models") or [])
        return out

    def models_of(self, name) -> dict:
        """An instance asked now for its models (the base's answer), without a model not whole yet."""
        out = super().models_of(name)
        out["models"] = self._servable(self.instance(name).name, out.get("models") or [])
        return out

    # -- what it learned, asking nothing --------------------------------------------------------------------------
    def api_facts(self) -> dict:
        facts: dict = {}
        seen = self.seen_facts()
        if seen:
            facts[_apis.SERVER.name] = seen
        hub = _models.facts(self)
        if hub:
            facts[_apis.HUB_API.name] = hub
        facts[_apis.RELEASES.name] = {"Pinned": f"llama.cpp {_release.VERSION}, build {_release.BUILD}"}
        return facts

    def detach(self) -> None:
        """Detached: a download under way asked to stop (paused by its router, as Cancel does), then the base's (an
        install cancelled, a click's servers stopped)."""
        with self._rt.lock:
            pulling = self._rt.pull
        if pulling is not None:
            try:
                pulling.cancel()
            except Exception:  # noqa: BLE001 - a panel that takes no cancel: the download ends by itself
                pass
        super().detach()


MACHINE = LlamaCppPlugin()
