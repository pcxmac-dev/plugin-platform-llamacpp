"""What the tests share.

Nothing here starts a process, opens a socket or resolves a name: the library's guard (liminal-libs' tests/conftest.py
`no_outside`, copied) stands in for every way to do so and fails the test that tries, and no program is found on
PATH. llama.cpp's, ggml's, Hugging Face's and Liminal's AI variables are taken out of the environment, the platform's
folder is a temporary one, no folder is another system's drive, and the disk has room.

The plugin is driven with stand-ins (`world`): a clock the test moves; a transport answering by method, host and path
(an instance's server, the Hugging Face Hub) and noting every ask; an opener serving release files (github.com
redirecting to its storage); a runner answering nvidia-smi and llama-server with scripted lines, noting each command;
a socket table; a login keeping pairs in memory; a children class; the boot registrations; an app with the plugin in
it. The answers in tests/data are written from the shapes llama.cpp's sources and README at b12359 give, and the
Hub's API document's: nothing was recorded from a real server.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

try:
    import requests  # noqa: F401 - imported before any guard: urllib3 binds a socket of its own when first imported
except ImportError:  # (not every test run has it)
    pass

import liminal_libs
from liminal_libs import mounts
from liminal_libs.apps.authorization import Login, Unauthorized, pair_name, variable_name
from liminal_libs.machines import autostart, children
from liminal_libs.machines.admin import Startup
from liminal_libs.machines.ai import Outside
from liminal_libs.machines.endpoint import Reply
from liminal_libs.resources import Conn

import plugin_platform_llamacpp as plugin
from plugin_platform_llamacpp import release

DATA = Path(__file__).parent / "data"
NOW = 1_763_400_000.0
KEY = "sk-llama-secret-0123456789"  # a key a test keeps: in no answer, line or error
DECLARED = ("MASSIVE_API_KEY", "OPENAI_API_KEY")  # what the registry says the installed plugins declare
GPU_5090 = "0, NVIDIA GeForce RTX 5090 Laptop GPU, 24463, 1200, 23263, 595.71\n"
VERSION_LINE = "version: 1.0.0-dev (build 12359, commit b1b770d9)\nbuilt with Clang 19.1.5 for Windows x86_64\n"
DEVICES = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU (24463 MiB, 23000 MiB free)\n"

# -- the library's guard (tests/conftest.py of liminal-libs 1.62.0), copied --------------------------------------------
STOOD_IN = [
    (subprocess, "subprocess", ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
                                "_fork_exec")),
    (os, "os", ("system", "startfile", "posix_spawn", "posix_spawnp", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                "fork", "forkpty", "execv", "execve")),
    (socket, "socket", ("create_connection", "create_server", "socketpair", "getaddrinfo", "gethostbyname",
                        "gethostbyname_ex", "gethostbyaddr", "getnameinfo")),
    (socket.socket, "socket.socket", ("connect", "connect_ex", "bind", "listen", "sendto", "sendmsg")),
]
if os.name == "nt":
    import _winapi

    STOOD_IN.append((_winapi, "_winapi", ("CreateProcess",)))
else:
    import _posixsubprocess

    STOOD_IN.append((_posixsubprocess, "_posixsubprocess", ("fork_exec",)))
THREADS = ("boot-llamacpp", "child-", "short-read")  # the plugin's and the base's threads: none left after a test


class Reached(BaseException):
    """A test reached for real what it must stand in for."""


@pytest.fixture(autouse=True)
def no_outside(monkeypatch, tmp_path):
    reached: list[str] = []

    def refuse(what):
        def stand_in(*args, **kwargs):
            __tracebackhide__ = True  # its arguments (a child's environment, with its keys) are never printed
            reached.append(what)
            raise Reached(f"the test reached {what} for real: give the code a stand-in")
        return stand_in

    for owner, said, names in STOOD_IN:
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, refuse(f"{said}.{name}"))
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    for name in list(os.environ):
        if name.upper().startswith(("LLAMA_", "GGML_", "HF_", "HUGGINGFACE_", "LIMINAL_AI_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIMINAL_PLATFORM_DIR", str(tmp_path / "platform"))
    monkeypatch.setattr(liminal_libs, "__version__", "1.62.0")
    monkeypatch.setattr(mounts, "foreign", lambda path, *a, **k: False)
    monkeypatch.setattr(release, "exists", lambda path: False)  # WSL's nvidia-smi is never this machine's
    monkeypatch.setattr(release, "disk_free", lambda path: 500 << 30)
    monkeypatch.setattr(release, "glibc", lambda: (2, 39))
    yield reached
    for t in threading.enumerate():
        if t.name.startswith(THREADS):
            t.join(2)
    left = [t.name for t in threading.enumerate() if t.name.startswith(THREADS) and t.is_alive()]
    assert not left, f"threads left running: {left}"
    assert not reached, f"the test reached the outside for real: {reached}"


# -- stand-ins --------------------------------------------------------------------------------------------------------
class Pairs(Login):
    """The platform's login as a test has it: pairs kept in memory."""

    def __init__(self):
        self.kept: dict[str, str] = {}

    def user(self) -> dict:
        return {"name": "someone"}

    def vector(self, name: str) -> str:
        variable = variable_name(name)
        if variable not in self.kept:
            raise Unauthorized(f"no {name}", 404)
        return self.kept[variable]

    def names(self) -> set[str]:
        return {pair_name(v) for v in self.kept}

    def save(self, variable: str, value: str | None) -> str:
        if value is None:
            self.kept.pop(variable, None)
        else:
            self.kept[variable] = value
        return "the test's pairs"


class Clock:
    def __init__(self, now: float = NOW):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Probe:
    """This system's socket table as a test sets it."""

    def __init__(self):
        self.conns: list[Conn] = []
        self.names: dict[int, str] = {}

    def listen(self, pid: int, port: int, name: str = "", address: str = "127.0.0.1") -> None:
        self.conns.append(Conn(pid, "listening", f"{address}:{port}", ""))
        if name:
            self.names[pid] = name

    def connections(self):
        return list(self.conns)

    def program_names(self, pids=None):
        return {p: n for p, n in self.names.items() if pids is None or p in pids}


def reply(status: int, data=None, headers: dict | None = None) -> Reply:
    body = data if isinstance(data, bytes) else json.dumps(data if data is not None else {}).encode()
    return Reply(status, {"content-type": "application/json", **(headers or {})}, [body])


def sse(*events: dict, then=None) -> Reply:
    """A router's event stream: each event a `data:` line, as server-models.cpp writes it; `then` raised after them
    (a stream that goes quiet: TimeoutError), else the stream ends."""
    def chunks():
        for event in events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        if then is not None:
            raise then
    return Reply(200, {"content-type": "text/event-stream"}, chunks())


class Transport:
    """http as a test answers it, by method, host and path; each ask noted."""

    def __init__(self):
        self.asked: list[tuple[str, str, dict, bytes | None]] = []
        self.routes: dict[tuple[str, str, str], object] = {}

    def on(self, method: str, path: str, answer, host: str = "127.0.0.1") -> None:
        self.routes[(method, host, path)] = answer

    def paths(self, host: str = "127.0.0.1") -> list[str]:
        return [f"{m} {urlsplit(u).path}" for m, u, _, _ in self.asked if urlsplit(u).hostname == host]

    def __call__(self, method, url, headers, body, timeout, stream):
        self.asked.append((method, url, dict(headers), body))
        parts = urlsplit(url)
        answer = self.routes.get((method, parts.hostname, parts.path))
        if answer is None:
            return reply(404, {"error": "not here"})
        if callable(answer) and not isinstance(answer, Reply):
            answer = answer(url, headers, body)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, list):  # answers in turn, the last one kept
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
            if isinstance(answer, BaseException):
                raise answer
        if isinstance(answer, Reply):
            return answer
        status, data, *more = answer
        return reply(status, data, more[0] if more else None)


def query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


class Runner:
    """Short commands as a test answers them (nvidia-smi, llama-server --version and --list-devices), each noted."""

    def __init__(self):
        self.ran: list[list[str]] = []
        self.gpu = GPU_5090
        self.caps = "0, 12.0\n"
        self.version = (0, "", VERSION_LINE)
        self.devices = (0, DEVICES, "")
        self.fails: dict[str, BaseException] = {}

    def __call__(self, args, timeout):
        args = [str(a) for a in args]
        self.ran.append(args)
        words = " ".join(args)
        for start, error in self.fails.items():
            if start in words:
                raise error
        if Path(args[0]).stem == "nvidia-smi":
            if any("compute_cap" in a for a in args):
                return (0, self.caps, "") if self.caps is not None else (6, "", 'Field "compute_cap" is not valid')
            return (0, self.gpu, "") if self.gpu else (9, "", "NVIDIA-SMI has failed: no devices were found")
        if "--list-devices" in args:
            return self.devices
        if "--version" in args or args[1:2] == ["version"]:
            return self.version
        return 0, "", ""


def archive(path: Path, files: dict[str, bytes]) -> bytes:
    """A stand-in release archive: a zip, or a .tar.gz, holding `files` by their names."""
    buffer = io.BytesIO()
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(buffer, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
    else:
        with tarfile.open(fileobj=buffer, mode="w:gz") as t:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o755
                t.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class Opener:
    """The release's files as github.com serves them: a redirect to its storage, then the bytes."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.asked: list[str] = []
        self.storage = "release-assets.githubusercontent.com"

    def __call__(self, url, headers, timeout):
        self.asked.append(url)
        parts = urlsplit(url)
        name = parts.path.rsplit("/", 1)[-1]
        if parts.hostname == "github.com":
            return Reply(302, {"location": f"https://{self.storage}/github-production-release-asset/1/{name}"}, [])
        data = self.files.get(name)
        if data is None:
            return Reply(404, {}, [b""])
        return Reply(200, {"content-length": str(len(data))}, [data])


class Kids:
    """A stand-in children.Child class: what it started, nothing real."""

    def __init__(self, clock: Clock):
        made: list = []
        self.made = made

        class Child:
            def __init__(self, name, launch, log, pidfile, *, popen=None, interrupt=None, clock=None, sleep=None,
                         system=None):
                self.name, self.launch, self.log, self.pidfile = name, launch, Path(log), Path(pidfile)
                self.alive, self.pid, self.code, self.since, self.note = False, None, None, None, ""
                made.append(self)

            def start(self):
                self.alive, self.pid, self.since = True, 4300 + len(made), clock()
                return self.pid

            @classmethod
            def adopt(cls, name, log, pidfile, **kw):
                return None

            def owns(self, pid):
                return pid == self.pid

            def stop(self, grace=8.0):
                self.alive, self.code = False, 0
                return "ended when asked (Ctrl+C), after 0.2 s"

            def wait(self, timeout=None):
                return self.code

            def tail(self, lines=40):
                return []

        self.Child = Child


class Boot:
    """autostart as a test has it: each call noted, the registrations kept here."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.kept: dict[str, Startup] = {}

    def register(self, plugin, instance, command, about, *, system, powershell=None, shell=None, home=None, env=None):
        self.calls.append(("register", plugin, instance, system))
        self.kept[instance] = Startup(instance, autostart.name_of(plugin, instance, system), "ready", " ".join(command))
        return self.kept[instance]

    def remove(self, plugin, instance, *, system, powershell=None, shell=None, home=None, env=None):
        self.calls.append(("remove", plugin, instance, system))
        return self.kept.pop(instance, None) is not None

    def act(self, plugin, instance, action, *, system, powershell=None, shell=None, env=None):
        self.calls.append(("act", plugin, instance, action, system))

    def read(self, plugin, *, system, powershell=None, shell=None, home=None, env=None):
        self.calls.append(("read", plugin, system))
        return dict(self.kept)


class App:
    """The platform as a plugin reaches it."""

    def __init__(self, *plugins, once=False):
        self.by = {p.name: p for p in plugins}
        self.once = once
        self.asked: list[tuple[str, str]] = []

    def plugins(self):
        return {n: p.interface() for n, p in self.by.items()}

    def ask(self, name, hook, args):
        self.asked.append((name, hook))
        return self.by[name].ask(hook, *args)


def load(name: str):
    text = (DATA / name).read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else text


class World:
    """Everything outside the plugin, and the plugin made with it."""

    def __init__(self, tmp: Path, boot: Boot):
        self.tmp, self.boot = tmp, boot
        self.clock, self.probe, self.transport = Clock(), Probe(), Transport()
        self.runner, self.opener, self.kids = Runner(), Opener(), Kids(self.clock)
        self.found: dict[str, str] = {"nvidia-smi": "C:/Windows/System32/nvidia-smi.exe"}  # what is on PATH
        self.heard: list = []
        windows = tmp / "Windows" / "System32"
        windows.mkdir(parents=True)
        (windows / "vcruntime140.dll").write_bytes(b"MZ")

    def which(self, name, *a, **k):
        return self.found.get(name)

    def listen(self, host, port):
        self.heard.append((host, port))
        return False

    def env(self, system: str) -> dict:
        t = self.tmp
        return {"PATH": str(t / "bin"), "SystemRoot": str(t / "Windows"), "APPDATA": str(t / "AppData" / "Roaming"),
                "PROGRAMDATA": str(t / "ProgramData"), "HOME": str(t / "home"), "XDG_CONFIG_HOME": "",
                "LIMINAL_PLATFORM_DIR": str(t / "platform"), "LIMINAL_PLUGINS": str(t / "plugins"),
                "LIMINAL_AI_LLAMACPP_INSTANCES": "kept elsewhere", "MASSIVE_API_KEY": "m-key-1",
                "LLAMA_ARG_CONNECT": "1", "LLAMA_API_KEY": "someone-elses-key", "LLAMA_CACHE": str(t / "other"),
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1", "HF_TOKEN": "hf_someone", "HF_ENDPOINT": "https://elsewhere",
                "CUDA_VISIBLE_DEVICES": "0"}

    def outside(self, system: str = "windows", wsl: str = "", **more) -> Outside:
        given = dict(env=self.env(system), system=system, wsl=wsl, home=self.tmp / "platform" / "ai" / "llamacpp",
                     clock=self.clock, which=self.which, probe=self.probe, transport=self.transport,
                     opener=self.opener, runner=self.runner, listening=self.listen, variables=lambda: DECLARED)
        given.update(more)
        return Outside(**given)

    def plugin(self, system: str = "windows", wsl: str = "", arch: str = "AMD64", attach: bool = True, **more):
        p = plugin.LlamaCppPlugin(self.outside(system, wsl, **more), arch=arch)
        p.CHILD = self.kids.Child
        p.login = Pairs()
        if attach:
            p.app = App(p)
        return p

    # -- the plugin's own install, laid out as an install leaves it ------------------------------------------------
    def installed(self, p, which: str = "cuda13", devices=("CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU",)) -> Path:
        folder = p.home / "app" / release.pin_of(which)
        folder.mkdir(parents=True)
        (folder / release.exe_name(p.system)).write_bytes(b"llama-server")
        (folder / release.RECORD).write_text(json.dumps({"which": which, "devices": list(devices)}), encoding="utf-8")
        (p.home / "app" / "CURRENT").write_text(folder.name + "\n", encoding="utf-8")
        return folder

    def stand_in_release(self, monkeypatch, system: str, which: str) -> list[release.Part]:
        """The build's archives made here and pinned in their place (their own sizes and sums), served by the
        opener."""
        server = "llama-server.exe" if system == "windows" else "llama-server"
        top = "" if system == "windows" else "llama-b12359/"
        parts = []
        for real in release.BUILDS[(system, which)]:
            if real.name.startswith("cudart"):
                inner = (f"cudart-{real.name[:-len('.tar.gz')]}/" if system == "linux" else "")
                files = {f"{inner}{'cudart64_13.dll' if system == 'windows' else 'libcudart.so.13'}": b"runtime"}
            else:
                files = {f"{top}{server}": b"server", f"{top}LICENSE": b"MIT"}
            data = archive(Path(real.name), files)
            self.opener.files[real.name] = data
            parts.append(release.Part(real.name, len(data), hashlib.sha256(data).hexdigest(), real.what))
        monkeypatch.setitem(release.BUILDS, (system, which), tuple(parts))
        return parts


@pytest.fixture
def boot(monkeypatch):
    b = Boot()
    for name in ("register", "remove", "act", "read"):
        monkeypatch.setattr(autostart, name, getattr(b, name))
    return b


@pytest.fixture
def world(tmp_path, boot):
    return World(tmp_path, boot)


def local(p, name: str = "default", port=None, **settings) -> None:
    """A local instance added as the page adds one (its port, then its settings by name)."""
    values = [settings.get(s.name) for s in p.SETTINGS]
    p.add_instance(name, "local", None, port, False, *values)


def running(p, world, name: str = "default") -> None:
    """A local instance's server started (a stand-in child), listening on its port."""
    p.start(name)
    child = world.kids.made[-1]
    world.probe.listen(child.pid, p.port_of(p.instance(name)), "llama-server.exe")
    world.clock.now += 6  # the socket table is read again (it is kept 5 s)


def router(world, models: str = "router-models.json", props: str = "router-props.json", host: str = "127.0.0.1",
           prefix: str = ""):
    """A llama.cpp router answering at `host` (under `prefix`): its health, props and models."""
    world.transport.on("GET", prefix + "/health", (200, {"status": "ok"}, {"server": "llama.cpp"}), host)
    world.transport.on("GET", prefix + "/props", (200, load(props), {"server": "llama.cpp"}), host)
    world.transport.on("GET", prefix + "/models", (200, load(models), {"server": "llama.cpp"}), host)


def children_shown(launch, system):
    return children.shown(launch, system, base={})
