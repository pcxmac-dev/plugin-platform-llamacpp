"""What Install brings and what is installed here (the user, 2026-11-16: "all three plugins need to allow installation
and configuration of their respective code").

The pin: llama.cpp v1.0.0, whose binaries are the build b12359 (the v1.0.0 release holds one file, nightly-tag.txt,
naming it; the archives hang on the pre-release b12359). Each archive is pinned here by its name, its size and its
sha256 as the release's own metadata gives them (GitHub's release API: each asset's size and digest; the build's
attestation repeats all 38 sums), read on 2026-11-16 without downloading anything, and fetch checks both before the
file gets its name. Downloaded from github.com, which sends each file on to release-assets.githubusercontent.com (all
twelve were seen so); objects.githubusercontent.com, GitHub's earlier host for release files, is allowed too though
none was sent there: the pinned sum checks a file whichever of GitHub's hosts serves it.

The build is the user's choice (the hook `build`, kept through the Login): automatic, CUDA 13, CUDA 12, Vulkan or the
processor's. Automatic is settled inside the install's job, where nvidia-smi may run (never for a status):

- no NVIDIA GPU read: the Vulkan build, under WSL the processor's (llama.cpp's makers say nothing of Vulkan there);
- a driver of CUDA 13.5's own branch, R620, or newer: the CUDA 13 build (A-revisions M9: the conservative pick below
  that branch is CUDA 12; figures' finding 12: the threshold is the branch, 620);
- below it, on Windows, a card of compute capability 12 (an RTX 50, Blackwell): the CUDA 12.4 build carries no code
  of its own for it (ggml's architectures for a toolkit before 12.8 end at 8.9 and 9.0's PTX), so the CUDA 13.5 build
  from driver 580 on, NVIDIA's minor version compatibility, else the Vulkan build (figures' finding 13);
- else the CUDA 12 build from driver 525 on (NVIDIA's minor version compatibility for any 12.x, on both systems:
  finding 12 asked for one meaning), else the Vulkan build.

The job's first line says which and why; once unpacked, `llama-server --version` must say build 12359 and its
`--list-devices` lines are logged and kept with the install: a CUDA build that lists no CUDA device is kept and said.
The install lies in app/<pin>/ (the library's one layout, A-revisions M11; the pin is b12359-<build>), with
liminal-install.json beside the programs: what was installed, from which archives, and what it saw.

An install found on PATH (llama-server, or the one program `llama` of llama.cpp's own install script, started as
`llama serve`) is the user's own: used as it is, never replaced or removed from here; its build is read by Read again
(`--version`, windowless, 10 s) and never by a status."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from liminal_libs.instances import Refusal
from liminal_libs.machines import download, gpu
from liminal_libs.machines.ai import Installed, Package, version_tuple
from liminal_libs.machines.ai import _which as default_which  # the Outside's own lookup: a stand-in replaces it
from liminal_libs.machines.children import ChildError
from liminal_libs.machines.download import DownloadError, Source
from liminal_libs.machines.terminal import Cancelled
from liminal_libs.machines.world import size_words

VERSION = "1.0.0"  # llama.cpp's release
BUILD = "b12359"  # its binaries (v1.0.0's nightly-tag.txt names them): the build its programs say
BUILD_NUMBER = 12359
TAG_DAY = "2026-11-13"  # when b12359 was published
RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/" + BUILD + "/"
HOSTS = ("github.com", "release-assets.githubusercontent.com")  # what the page names: where a download comes from
# where github.com may send a release file on to (download.Source.hosts): the one seen for all twelve archives, and
# GitHub's earlier host for release files, seen for none (the pinned sum checks a file whichever host serves it)
ASSET_HOSTS = ("release-assets.githubusercontent.com", "objects.githubusercontent.com")
HUB = "https://huggingface.co"  # the Hugging Face Hub: a model's files, size and licence, asked on the user's click
ROUTER_FROM = 7218  # the first build that serves without a model: a router (PR 17470)
MODELS_FROM = 9688  # the first whose router downloads, deletes and tells of models over its API (PR 23976)
PORT_FROM = 11521  # the first build listening on 9931 by default (8080 before it)
CUDA13_DRIVER = 620  # CUDA 13.5's own driver branch, R620: the CUDA 13 build is chosen from it on (A-revisions M9)
MINOR13_DRIVER = 580  # NVIDIA's lowest driver for any CUDA 13.x (minor version compatibility)
CUDA12_DRIVER = 525  # NVIDIA's lowest driver for any CUDA 12.x (minor version compatibility), on both systems
NEW_CARDS = 12  # a compute capability from this major on has no code in the Windows CUDA 12.4 build (finding 13)
NEED_FREE = 3 << 30  # free on the plugin's disk before an install: the largest build is 757 MB, unpacked about twice
LIST_S = 30.0  # what `--list-devices` is given: it loads the GPU's libraries
PULL_S = 21600.0  # a model's download is followed six hours at most, then paused (its part kept)
RECORD ="liminal-install.json"  # what an install of this plugin's says of itself, beside its programs
WHICH = ("automatic", "cuda13", "cuda12", "vulkan", "cpu")
TITLES = {"automatic": "Automatic", "cuda13": "CUDA 13", "cuda12": "CUDA 12", "vulkan": "Vulkan", "cpu": "Processor"}
VC_RUNTIME = "vcruntime140.dll"  # Microsoft's Visual C++ runtime, which llama.cpp's winget package names
VC_WORDS = ("llama.cpp's Windows build needs Microsoft's Visual C++ runtime (x64), which is not on this PC "
            "(System32\\vcruntime140.dll): its installer is Microsoft's, yours to run")
LICENCE = "MIT, llama.cpp's own."
CUDA_LICENCE = " The CUDA runtime in the second archive is NVIDIA's, under NVIDIA's terms."
MODELS_LICENCE = " Each model carries its own licence, said before it is downloaded."
CHECKED = ("Each archive is checked against the size and sha256 pinned in this plugin (as the release's metadata and "
           "its attestation give them).")
UBUNTU = {"cuda13": ("24.04", (2, 39)), "cuda12": ("24.04", (2, 39)), "vulkan": ("22.04", (2, 35)),
          "cpu": ("22.04", (2, 35))}  # what each Linux build was made on, and that Ubuntu's C library
exists = os.path.exists  # a file's presence (WSL's place for nvidia-smi): a test's stand-in replaces it


@dataclass(frozen=True)
class Part:
    """One archive of the pinned release: its name, size in bytes, sha256 and what it is."""

    name: str
    size: int
    sha256: str
    what: str

    def source(self) -> Source:
        return Source(RELEASE + self.name, self.name, self.size, self.sha256, ASSET_HOSTS)


def _part(name: str, size: int, sha256: str, what: str) -> Part:
    return Part(name, size, sha256, what)


# Every archive of b12359 the plugin installs from: (this system, the build) -> the build's archive, then NVIDIA's
# runtime for a CUDA build. Names, sizes and sums read from the release's metadata (figures/llama-cpp.md, Files).
BUILDS: dict[tuple[str, str], tuple[Part, ...]] = {
    ("windows", "cuda13"): (
        _part("llama-b12359-bin-win-cuda-13.5-x64.zip", 179546112,
              "1de9ca1ff408d6e348ea0007c6880ad2393311a02e7799bcde26dc16daef03dd", "llama.cpp's CUDA 13 build"),
        _part("cudart-llama-bin-win-cuda-13.5-x64.zip", 426314199,
              "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba", "NVIDIA's CUDA 13.5 runtime")),
    ("windows", "cuda12"): (
        _part("llama-b12359-bin-win-cuda-12.4-x64.zip", 318976332,
              "e9c0abf442e8339c06728366c1ad2c711332f9939c0c00603fb75d85fd44c914", "llama.cpp's CUDA 12 build"),
        _part("cudart-llama-bin-win-cuda-12.4-x64.zip", 391443627,
              "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", "NVIDIA's CUDA 12.4 runtime")),
    ("windows", "vulkan"): (
        _part("llama-b12359-bin-win-vulkan-x64.zip", 38090643,
              "e8d18c6e4210cca87e7307c2ea78363ac8d158e4e1397358ec96ab70b53a0d5e", "llama.cpp's Vulkan build"),),
    ("windows", "cpu"): (
        _part("llama-b12359-bin-win-cpu-x64.zip", 25097509,
              "837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69", "llama.cpp's processor build"),),
    ("linux", "cuda13"): (
        _part("llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz", 178419847,
              "9440f417615bed0e191acef03bff0249117d2be74335358265a6d28475a9baf0", "llama.cpp's CUDA 13 build"),
        _part("cudart-llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz", 444078822,
              "976a9146bb8a487a6ea701224d7fb14b05e494ab3c8f5c1799ffc4e5805d640c", "NVIDIA's CUDA 13.5 runtime")),
    ("linux", "cuda12"): (
        _part("llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz", 199223370,
              "50c937bec5bbd9f74c99138034f7403283ee21f1943f2429d7b1e4c079eafc11", "llama.cpp's CUDA 12 build"),
        _part("cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz", 594371571,
              "75a12d227f44a391608992329241765343bd143ee9445b90dc62e154ee726e5d", "NVIDIA's CUDA 12.8 runtime")),
    ("linux", "vulkan"): (
        _part("llama-b12359-bin-ubuntu-vulkan-x64.tar.gz", 36153377,
              "19b60f00395a3f4954e46a5037695bc1d843404fc9d3124f82a612ecd8fbe9f0", "llama.cpp's Vulkan build"),),
    ("linux", "cpu"): (
        _part("llama-b12359-bin-ubuntu-x64.tar.gz", 23311084,
              "0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e", "llama.cpp's processor build"),),
}
CUDA = {"windows": {"cuda13": "13.5", "cuda12": "12.4"}, "linux": {"cuda13": "13.5", "cuda12": "12.8"}}


# -- this system ------------------------------------------------------------------------------------------------------
def exe_name(system: str) -> str:
    """llama-server's file name on a system."""
    return "llama-server.exe" if system == "windows" else "llama-server"


def processor(p) -> str:
    """This processor as Python names it (AMD64, x86_64, ARM64, aarch64): the plugin's `arch` where a test sets it."""
    return str(getattr(p, "arch", None) or platform.machine() or "").strip()


def is_x86(p) -> bool:
    return processor(p).lower() in ("amd64", "x86_64", "x64", "em64t")


def runs_here(p) -> str:
    """"" on Windows x64 and Linux x86-64; else why not, in words."""
    if is_x86(p):
        return ""
    return f"this plugin installs and starts llama.cpp for x86-64 only (this processor is {processor(p) or 'unnamed'})"


def environment(p) -> dict:
    return dict(p.out.env if p.out.env is not None else os.environ)


def env_value(env: dict, name: str) -> str:
    """A variable of `env` by its name, its case ignored (as Windows reads it)."""
    for key, value in env.items():
        if str(key).upper() == name.upper():
            return str(value)
    return ""


def vc_runtime_missing(p) -> bool:
    """Whether Microsoft's Visual C++ runtime is missing from this Windows (System32's vcruntime140.dll): looked at
    only."""
    if p.system != "windows":
        return False
    root = env_value(environment(p), "SystemRoot") or env_value(environment(p), "windir") or r"C:\Windows"
    return not (Path(root) / "System32" / VC_RUNTIME).is_file()


def disk_free(path) -> int | None:
    """The bytes free on the disk `path` lies on (its nearest folder that is there); None when it cannot be said."""
    here = Path(path)
    for _ in range(64):
        try:
            return int(shutil.disk_usage(here).free)
        except OSError:
            if here.parent == here:
                return None
            here = here.parent
    return None


def glibc() -> tuple[int, ...] | None:
    """This Linux's GNU C library version, as the system says it (no program run); None for another C library."""
    try:
        said = os.confstr("CS_GNU_LIBC_VERSION") or ""
    except (AttributeError, ValueError, OSError):
        return None
    found = re.search(r"glibc\s+(\d+)\.(\d+)", said)
    return (int(found.group(1)), int(found.group(2))) if found else None


def nvidia_smi(p) -> str:
    """Where nvidia-smi is (gpu.command): looked up on PATH and at WSL's place for it, nothing run."""
    which = None if p.out.which is default_which else p.out.which
    return gpu.command(which=which, exists=lambda path: exists(path))


def install_refusal(p) -> str:
    """"" or why Install cannot begin here, said before the job's first line: Windows' C runtime missing; too little
    free on the plugin's disk."""
    if p.system == "windows" and vc_runtime_missing(p):
        return VC_WORDS
    free = disk_free(p.home)
    if free is not None and free < NEED_FREE:
        return (f"llama.cpp's install needs {size_words(NEED_FREE)} free on the disk of {p.home}, which has "
                f"{size_words(free)} free")
    return ""


# -- the build --------------------------------------------------------------------------------------------------------
def pin_of(which: str) -> str:
    """The folder an install of a build lies in, app/<pin>: b12359-<build>."""
    return f"{BUILD}-{which}"


def builds_here(system: str) -> tuple[str, ...]:
    """The builds this plugin installs on a system, automatic first."""
    return ("automatic", *(w for w in WHICH[1:] if (system, w) in BUILDS))


def kept_build(p) -> str:
    """The build kept through the login (`build`), "automatic" when none is kept or what is kept is no build."""
    try:
        kept = p.pair(p.variable("build")).strip().lower()
    except Exception:  # noqa: BLE001 - a login that cannot say: the build is chosen as it installs
        return "automatic"
    return kept if kept in WHICH and (p.system, kept) in BUILDS else "automatic"


def parts_of(system: str, which: str) -> tuple[Part, ...]:
    return BUILDS.get((system, which), ())


def bytes_of(system: str, which: str) -> int:
    return sum(part.size for part in parts_of(system, which))


def _automatic(system: str) -> tuple[str, ...]:
    """What automatic may bring on a system, in its order of preference."""
    return ("cuda13", "cuda12", "vulkan", "cpu") if system == "linux" else ("cuda13", "cuda12", "vulkan")


def build_words(system: str, which: str) -> str:
    """A build in words, its archives named."""
    parts = parts_of(system, which)
    if not parts:
        return f"no {TITLES.get(which, which)} build of llama.cpp is installed on this system"
    words = f"{parts[0].what} ({parts[0].name})"
    if len(parts) > 1:
        words += f" with {parts[1].what} ({parts[1].name})"
    return words


def automatic_words(p) -> str:
    """What automatic brings, every case named with its size: settled as it installs, by what nvidia-smi reads then."""
    system, wsl = p.system, bool(p.wsl)
    cuda12 = CUDA[system]["cuda12"]
    cases = [f"its CUDA 13 build with NVIDIA's CUDA 13.5 runtime where nvidia-smi reads an NVIDIA GPU with a driver "
             f"of {CUDA13_DRIVER} or newer" + (f", or of {MINOR13_DRIVER} or newer for a card of compute capability "
                                               f"{NEW_CARDS} (an RTX 50)" if system == "windows" else "")
             + f" ({size_words(bytes_of(system, 'cuda13'))})",
             f"its CUDA 12 build with NVIDIA's CUDA {cuda12} runtime where the driver is older, from "
             f"{CUDA12_DRIVER} on ({size_words(bytes_of(system, 'cuda12'))})"]
    if system == "linux" and wsl:
        cases.append(f"its processor's build where none is read ({size_words(bytes_of(system, 'cpu'))}): llama.cpp's "
                     "makers say nothing of its Vulkan build inside WSL")
    else:
        vulkan = size_words(bytes_of(system, "vulkan"))
        cases.append(f"its Vulkan build where none is read, or the driver is older ({vulkan})")
    return "llama.cpp's build chosen as it installs: " + "; ".join(cases)


def licence(system: str, which: str) -> str:
    cuda = which == "automatic" or len(parts_of(system, which)) > 1
    note = (" A CUDA build's second archive is NVIDIA's CUDA runtime, under NVIDIA's terms." if which == "automatic"
            else CUDA_LICENCE if cuda else "")
    return LICENCE + note + MODELS_LICENCE


def in_use_pin(p) -> str:
    """The pin of the install in use when it is one of this release's builds ("" otherwise): automatic takes any."""
    folder = p.current()
    if folder is None:
        return ""
    return folder.name if folder.name in {pin_of(w) for w in WHICH[1:]} else ""


def package(p) -> Package | None:
    """What Install brings here at the pinned version, the build kept (automatic: the largest it may bring, every
    case named); None where nothing is installed from here (another processor)."""
    if runs_here(p):
        return None
    which = kept_build(p)
    keeps = f"Its models ({p.home / 'models'}), its logs and your instances stay."
    if which == "automatic":
        size = max(bytes_of(p.system, w) for w in _automatic(p.system))
        return Package("llama.cpp", f"{VERSION} ({BUILD})", size, licence(p.system, which), HOSTS,
                       automatic_words(p), keeps, pin=in_use_pin(p) or BUILD, checked=CHECKED)
    return Package("llama.cpp", f"{VERSION} ({BUILD}), its {TITLES[which]} build", bytes_of(p.system, which),
                   licence(p.system, which), HOSTS, build_words(p.system, which), keeps, pin=pin_of(which),
                   checked=CHECKED)


# -- what is installed ------------------------------------------------------------------------------------------------
def record_of(folder: Path) -> dict:
    """What an install of this plugin's says of itself (liminal-install.json): {} when it says nothing readable."""
    try:
        got = json.loads((Path(folder) / RECORD).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def _which_of_pin(name: str) -> str:
    return next((w for w in WHICH[1:] if pin_of(w) == name), "")


def installed(p) -> Installed | None:
    """What is installed here, files looked at only: this plugin's (app/CURRENT naming a folder with llama-server),
    else llama-server or llama on PATH (the user's own, its build not read yet). Nothing is run."""
    folder = p.current()
    if folder is not None and (folder / exe_name(p.system)).is_file():
        record = record_of(folder)
        which = str(record.get("which") or _which_of_pin(folder.name))
        version = (f"{VERSION} ({BUILD}), its {TITLES[which]} build" if which in TITLES
                   else f"{BUILD} ({folder.name})")
        devices = [str(d) for d in record.get("devices") or [] if str(d).strip()][:4]
        note = f"it sees {'; '.join(devices)}" if devices else ""
        if record and not devices:
            note = "it saw no GPU when it was installed: it runs on the processor"
        return Installed(version, str(folder), "plugin", note)
    for name in ("llama-server", "llama"):
        try:
            found = p.out.which(name)
        except Exception:  # noqa: BLE001 - a lookup that fails finds nothing
            found = None
        if found:
            return Installed("", str(found), "found", found_note(p, str(found)))
    return None


def is_unified(path: str) -> bool:
    """Whether a found program is llama.cpp's one program `llama` (started as `llama serve`)."""
    return Path(str(path)).stem.lower() == "llama"


def found_note(p, path: str) -> str:
    """What a found install means, and what its last Read again saw (nothing is run for it)."""
    words = (f"{path} is your own install (winget's is the Vulkan build): it is used as it is, never replaced from "
             "here")
    if is_unified(path):
        words += "; it is llama.cpp's one program, started as `llama serve`"
    seen = found_reading(p, path)
    if seen.get("devices"):
        words += f". It sees: {'; '.join(seen['devices'][:4])}"
    elif seen.get("build"):
        words += ". It sees no GPU"
    return words


def found_reading(p, path: str) -> dict:
    """What Read again read of a found install: {"build": n, "line": its version line, "devices": [...]}."""
    with p._rt.lock:
        return dict(p._rt.installs.get(str(path), {}))


def found_build(p, have: Installed | None) -> int | None:
    """A found install's build, as Read again read it; None when not read (or this plugin's own: b12359)."""
    if have is None:
        return None
    if have.how == "plugin":
        return BUILD_NUMBER
    build = found_reading(p, have.path).get("build")
    return build if isinstance(build, int) else None


def program(p, have=None) -> tuple[str, ...]:
    """The server's command head: this plugin's llama-server, a found llama-server, or a found llama with `serve`;
    () when nothing is installed."""
    have = p.installed() if have is None else have
    if have is None:
        return ()
    if have.how == "plugin":
        return (str(Path(have.path) / exe_name(p.system)),)
    return (have.path, "serve") if is_unified(have.path) else (have.path,)


def build_of(text: str) -> int | None:
    """The build a version line says ("version: 1.0.0-dev (build 12359, commit b1b770d9)"); read leniently, since
    llama.cpp said its version otherwise before 2026: "build <n>", "build = <n>", or the number alone before its
    commit ("version: 6900 (1a2b3c4)")."""
    text = text or ""
    found = (re.search(r"\bbuild\s*[:=]?\s*(\d{3,7})\b", text, re.I)
             or re.search(r"\bversion:\s*(\d{3,7})\s*\(", text, re.I))
    return int(found.group(1)) if found else None


def version_line(text: str) -> str:
    for line in (text or "").splitlines():
        if build_of(line) is not None:
            return " ".join(line.split())[:200]
    return ""


def device_lines(text: str) -> list[str]:
    """The devices `--list-devices` names ("CUDA0: NVIDIA GeForce RTX 5090 (24463 MiB, 23000 MiB free)"); none for
    its "(none)"."""
    out, listing = [], False
    for line in (text or "").splitlines():
        if line.strip().lower().startswith("available devices"):
            listing = True
            continue
        if listing and line.startswith(" ") and line.strip() and line.strip() != "(none)":
            out.append(" ".join(line.split())[:200])
    return out[:8]


def read_version(p, have: Installed) -> str:
    """A found install's build, read by running it once, windowless, 10 s (`--version`, or `llama version`), with
    the devices its `--list-devices` names: kept until the next Read again. This plugin's own says its version in
    its files: "" (nothing run)."""
    if have.how == "plugin":
        return ""
    head = program(p, have)
    argv = [have.path, "version"] if is_unified(have.path) else [have.path, "--version"]
    code, out, err = p.short(argv)
    said = out + "\n" + err
    build = build_of(said)
    devices: list[str] = []
    try:
        _, listed, _ = p.short([*head, "--list-devices"], timeout=LIST_S)
        devices = device_lines(listed)
    except ChildError:
        devices = []
    with p._rt.lock:
        p._rt.installs[str(have.path)] = {"build": build, "line": version_line(said), "devices": devices}
    return f"b{build}" if build else ""


# -- the install ------------------------------------------------------------------------------------------------------
def compute_capability(p, index: int) -> tuple[int, int] | None:
    """A card's compute capability, as nvidia-smi says it (its --query-gpu field compute_cap, windowless): inside a
    job only; None when it does not say."""
    program_ = nvidia_smi(p) or gpu.NAME  # where the reading of its memory found it, as gpu.read finds it
    try:
        code, out, _ = p.short([program_, "--query-gpu=index,compute_cap", "--format=csv,noheader"],
                               timeout=gpu.TIMEOUT_S)
    except ChildError:
        return None
    if code != 0:
        return None
    for line in str(out).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(index):
            found = re.fullmatch(r"(\d+)\.(\d+)", parts[1])
            if found:
                return int(found.group(1)), int(found.group(2))
    return None


def settle(p) -> tuple[str, str]:
    """The build automatic brings here now, and why, in words: by the NVIDIA card nvidia-smi reads (inside the job)."""
    which = None if p.out.which is default_which else p.out.which
    reading = gpu.read(runner=p.out.runner, which=which, exists=None if p.out.runner is not None else exists,
                       env=p.child_env())
    if not reading.cards:
        if p.system == "linux" and p.wsl:
            return "cpu", (f"no NVIDIA GPU is read here ({reading.why}), and llama.cpp's makers say nothing of its "
                           "Vulkan build inside WSL: the processor's build")
        return "vulkan", (f"no NVIDIA GPU is read here ({reading.why}): the Vulkan build, for any GPU with a Vulkan "
                          "driver")
    card = max(reading.cards, key=lambda g: (g.total, -g.index))
    driver = version_tuple(card.driver) if card.driver else ()
    named = f"the {card.name} with driver {card.driver or '(not said)'}"
    cuda12 = CUDA[p.system]["cuda12"]
    if driver and driver >= (CUDA13_DRIVER,):
        return "cuda13", f"{named}, of CUDA 13.5's own branch (R{CUDA13_DRIVER}) or newer: the CUDA 13 build"
    if p.system == "windows":
        cap = compute_capability(p, card.index)
        if cap is not None and cap[0] >= NEW_CARDS:
            where = f"{named}, of compute capability {cap[0]}.{cap[1]}: the CUDA 12.4 build carries no code for it"
            if driver and driver >= (MINOR13_DRIVER,):
                return "cuda13", (f"{where}, so the CUDA 13 build, which NVIDIA's minor version compatibility runs on "
                                  f"drivers from {MINOR13_DRIVER} on (what is new in CUDA 13.5 may be refused by one "
                                  f"older than R{CUDA13_DRIVER})")
            return "vulkan", f"{where}, and the CUDA 13 build needs {MINOR13_DRIVER} or newer: the Vulkan build"
    if not driver or driver >= (CUDA12_DRIVER,):
        return "cuda12", (f"{named}, older than CUDA 13.5's own branch (R{CUDA13_DRIVER}): the CUDA 12 build (CUDA "
                          f"{cuda12}), which NVIDIA's minor version compatibility runs on drivers from "
                          f"{CUDA12_DRIVER} on")
    return "vulkan", f"{named}, older than {CUDA12_DRIVER}, the oldest CUDA 12 runs on: the Vulkan build"


def _cause(p, which: str) -> str:
    """Why a program just unpacked may not start, in words."""
    if p.system == "windows":
        return "Microsoft's Visual C++ runtime (x64) may be missing, or the archive is not this build's"
    made, need = UBUNTU.get(which, ("24.04", (2, 39)))
    return (f"this system's libraries may be older than the build needs (OpenSSL 3, and glibc {need[0]}.{need[1]} "
            f"or newer: it was made on Ubuntu {made})")


def check_program(p, exe: Path, which: str) -> tuple[str, list[str], list[str]]:
    """The program just unpacked, run twice (windowless): `--version` must say build 12359; `--list-devices` names
    what it sees. (its version line, its devices, what is said of them). DownloadError when it does not start or is
    not this build: nothing is put in use."""
    try:
        code, out, err = p.short([str(exe), "--version"])
    except ChildError as e:
        raise DownloadError(f"{exe.name} did not start ({e}): {_cause(p, which)}") from None
    said = out + "\n" + err
    build = build_of(said)
    if build is None:
        first = next((x.strip() for x in said.splitlines() if x.strip()), f"exit code {code}")
        raise DownloadError(f"{exe.name} did not say its build ({first[:160]}): {_cause(p, which)}")
    if build != BUILD_NUMBER:
        raise DownloadError(f"{exe.name} says build {build}, not {BUILD_NUMBER}: it is not put in use")
    devices: list[str] = []
    notes: list[str] = []
    try:
        _, listed, _ = p.short([str(exe), "--list-devices"], timeout=LIST_S)
        devices = device_lines(listed)
    except ChildError as e:
        notes.append(f"its devices could not be listed ({e})")
    if which in ("cuda13", "cuda12") and not any(d.upper().startswith("CUDA") for d in devices):
        notes.append("it lists no CUDA device: it will run on the processor. The driver may be older than the build "
                     "needs, or NVIDIA's runtime files are not beside llama-server. Choose another build, then Update.")
    return version_line(said), devices, notes


def install_into(p, log, cancel) -> list[str]:
    """The install: the build settled (automatic: nvidia-smi read now), each archive fetched into downloads/b12359
    and checked against its pinned size and sha256, unpacked into app/<pin> (Linux's tarballs each without their
    top folder, into the same folder), llama-server run to say its build and its devices, liminal-install.json
    written, then put in use (adopt) and the archives removed."""
    say = log
    which = kept_build(p)
    if which == "automatic":
        which, why = settle(p)
    else:
        why = "chosen on this page (The build…)"
    say(f"the build: {TITLES[which]}: {why}")
    parts = parts_of(p.system, which)
    if not parts:
        raise DownloadError(build_words(p.system, which))
    kept = p.home / "downloads" / BUILD
    archives = []
    for part in parts:
        say(f"{part.name} ({size_words(part.size)}): {part.what}, checked against the size and sha256 pinned in this "
            "plugin")
        archives.append(p.fetch(part.source(), kept, log=log, cancel=cancel))
    if cancel():
        raise Cancelled("cancelled")
    pin = pin_of(which)
    folder = p.building(pin)
    strip = 1 if p.system == "linux" else 0  # Linux's tarballs each hold one top folder; Windows' zips are flat
    for archive in archives:
        p.unpack(archive, folder, log=log, cancel=cancel, strip=strip)
    exe = folder / exe_name(p.system)
    if not exe.is_file():
        raise DownloadError(f"{exe.name} is not among what was unpacked into {folder}: nothing is put in use")
    if cancel():
        raise Cancelled("cancelled")
    line, devices, notes = check_program(p, exe, which)
    say(f"{exe.name} says: {line}")
    for device in devices or ["no GPU (it runs on the processor)"]:
        say(f"it sees {device}")
    for note in notes:
        say(note)
    record = {"version": VERSION, "build": BUILD, "which": which, "why": why, "devices": devices, "line": line,
              "files": [{"name": part.name, "size": part.size, "sha256": part.sha256} for part in parts],
              "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (folder / RECORD).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    lines = [f"{TITLES[which]} build: {why}", f"{exe.name} says: {line}",
             *(f"it sees {d}" for d in devices), *notes, "each archive matched the size and sha256 pinned here"]
    lines += p.adopt(pin)
    for archive in archives:  # (after adopt nothing here may fail: what cannot be removed is said)
        try:
            download.remove_tree(archive, p.home / "downloads")
            lines.append(f"removed {archive.name} from downloads")
        except (OSError, ValueError) as e:
            lines.append(f"{archive.name} stays in downloads ({e})")
    return lines


def remove_from(p) -> list[str]:
    """The uninstall: the install's folder and what was downloaded for it removed; its models, logs and instances
    kept and said."""
    lines = [*p.forget_install(), *p.forget_downloads()]
    models = p.home / "models"
    held, files, whole = _held(models)
    if files:
        lines.append(f"kept: {models} ({files:,}{'' if whole else '+'} files, {size_words(held)}), its logs and your "
                     "instances")
    else:
        lines.append("kept: its logs and your instances")
    return lines


def _held(folder: Path, most: int = 4096) -> tuple[int, int, bool]:
    """(the bytes, the files) under `folder`, no link followed, `most` entries looked at; and whether all were."""
    total, files, looked, todo = 0, 0, 0, [Path(folder)]
    while todo:
        path = todo.pop()
        try:
            entries = list(os.scandir(path))
        except OSError:
            continue
        for entry in entries:
            looked += 1
            if looked > most:
                return total, files, False
            try:
                if entry.is_dir(follow_symlinks=False):
                    todo.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return total, files, True


# -- the page's parts -------------------------------------------------------------------------------------------------
def lacks(p) -> dict | None:
    """The table "On this system": what this system lacks or changes for llama.cpp, a row each where it applies;
    None where nothing does. Files and lookups only: nothing is run."""
    rows: list[list[str]] = []
    here = runs_here(p)
    if here:
        rows.append(["Processor", processor(p) or "not said", here])
    if p.system == "windows" and vc_runtime_missing(p):
        rows.append(["Runtime", "Microsoft's Visual C++ runtime (x64) is not found",
                     "llama.cpp's Windows build needs it (its winget package names it): its installer is Microsoft's, "
                     "yours to run. Install refuses until it is there."])
    if not here and not nvidia_smi(p):
        wsl = p.system == "linux" and bool(p.wsl)
        rows.append(["GPU", "nvidia-smi is not on this system",
                     "no NVIDIA GPU can be read, and what fits is not known: Install brings "
                     + ("the processor's build (llama.cpp's makers say nothing of its Vulkan build inside WSL)" if wsl
                        else "the Vulkan build, for any GPU with a Vulkan driver, or the processor")])
    if p.system == "linux" and not here:
        libc = glibc()
        rows.append(["System", f"glibc {libc[0]}.{libc[1]}" if libc else "its C library is not GNU's, or not said",
                     "llama.cpp's Linux builds are made on Ubuntu (the processor's and the Vulkan build on 22.04, "
                     "glibc 2.35; the CUDA builds on 24.04, glibc 2.39) and use this system's OpenSSL 3: on an older "
                     "system the install's check says that llama-server cannot run"])
        if p.wsl:
            from liminal_libs.machines import linux

            generation = linux.wsl_generation()
            rows.append(["This Linux", f"WSL {generation or 2}: {p.wsl}",
                         "Windows' NVIDIA driver is the only one (none is installed inside); a CUDA build brings "
                         "NVIDIA's runtime itself. Windows reaches a server started here at its own 127.0.0.1 through "
                         "WSL's relay, where Windows' llama.cpp plugin lists it as found"])
    if not rows:
        return None
    return {"title": "On this system", "note": "What this system lacks or changes for llama.cpp.",
            "columns": [{"title": "What"}, {"title": "Here"}, {"title": "What it means"}], "rows": rows}


def brings_table(p) -> dict | None:
    """The table "What Install brings": the build kept, what is downloaded and checked, what it needs, the one in
    use; a button to choose the build."""
    if runs_here(p):
        return None
    which = kept_build(p)
    system = p.system
    if which == "automatic":
        build = ["Build", "Automatic", automatic_words(p)]
        size = max(bytes_of(system, w) for w in _automatic(system))
        download_ = ["Download", f"up to {size_words(size)}", f"from {' and '.join(HOSTS)}. {CHECKED}"]
    else:
        build = ["Build", TITLES[which], f"{build_words(system, which)}: chosen on this page"]
        download_ = ["Download", size_words(bytes_of(system, which)), f"from {' and '.join(HOSTS)}. {CHECKED}"]
    needs = (f"a CUDA build: an NVIDIA GPU and its driver ({CUDA13_DRIVER} or newer for CUDA 13"
             + (f", {MINOR13_DRIVER} for a card of compute capability {NEW_CARDS}" if system == "windows" else "")
             + f"; {CUDA12_DRIVER} or newer for CUDA 12); Vulkan: a GPU with a Vulkan driver, else the processor")
    needs += ("; and Microsoft's Visual C++ runtime (x64)" if system == "windows"
              else "; and OpenSSL 3 with glibc 2.35 (2.39 for the CUDA builds)")
    rows = [build, download_, ["Needs", "", needs]]
    have = p.installed()
    if have is not None and have.how == "plugin":
        rows.append(["In use", have.version, have.path])
    return {"title": "What Install brings", "note": f"llama.cpp {VERSION}, build {BUILD} ({TAG_DAY}), pinned here.",
            "columns": [{"title": "What"}, {"title": "Here"}, {"title": "What it means"}], "rows": rows,
            "actions": [{"form": "ai-build", "title": "The build…"}]}


def build_form(p) -> dict:
    """The form that keeps the build Install brings (ai-build, A-revisions M2): one choice, confirmed."""
    kept = kept_build(p)
    choices = []
    for which in builds_here(p.system):
        note = ("settled as it installs, by what nvidia-smi reads then" if which == "automatic"
                else f"{build_words(p.system, which)}, {size_words(bytes_of(p.system, which))}")
        choices.append({"value": which, "title": TITLES[which], "note": note})
    return {"id": "ai-build", "hook": "build", "title": "The build Install brings", "submit": "Review",
            "fields": [{"name": "which", "label": "Build", "type": "choice", "value": kept, "choices": choices}],
            "confirm": {"title": "Install brings the {which} build?",
                        "words": "The build is kept through your login and used by the next Install.", "yes": "Keep"}}


def choose_build(p, which) -> dict:
    """The build Install brings, kept through the login (automatic forgets the pair): changes only what the next
    Install brings."""
    word = str(which or "").strip().lower()
    if word not in builds_here(p.system):
        raise Refusal(f"the build is one of {', '.join(TITLES[w] for w in builds_here(p.system))}, not {word[:20]!r}")
    where = p.keep_pair(p.variable("build"), None if word == "automatic" else word)
    brings = automatic_words(p) if word == "automatic" else build_words(p.system, word)
    lines = [f"kept in {where}"]
    have = p.installed()
    if have is not None and have.how == "plugin" and word != "automatic" and Path(have.path).name != pin_of(word):
        lines.append(f"the install in use is {have.version}: Update… brings the {TITLES[word]} build")
    return p._reply(f"Install brings {brings}", True, lines)
