"""What Install brings and what is installed (release.py): the pins as the release's metadata gives them; package()
on each system and for each build (automatic naming every case); installed() for this plugin's install, a found
llama-server, a found llama and none, files looked at only; Read again reading a found install's build and devices;
the build automatic settles from what nvidia-smi reads (the figures review's findings 13 and 14, numbered from 1),
NVIDIA's caveat said for CUDA 12; the install as a job: fetched through github.com's redirect to its one storage host
(another refused), checked, unpacked (Linux's top folders stripped), its program run to say its build, recorded, put
in use and its archives removed; its refusals before the first line (Windows' C runtime, the disk, the licence) and
its failures (a wrong sum, another build, a program that does not start, a cancel, one while its program is checked);
the uninstall inside the plugin's folder alone; the build kept through the login."""
from __future__ import annotations

import json
import re
import threading

import pytest

from liminal_libs.instances import Refusal
from liminal_libs.machines.ai import ACCEPT_LICENCE, CANCELLED
from liminal_libs.machines.children import ChildError

from plugin_platform_llamacpp import release
from conftest import GPU_5090


class Panel:
    """A job's panel as the platform hands one: its lines, a cancel."""

    def __init__(self, cancel_after: str = ""):
        self.lines: list[str] = []
        self._cancel = threading.Event()
        self.cancel_after = cancel_after

    def __call__(self, line: str) -> None:
        self.lines.append(str(line))
        if self.cancel_after and self.cancel_after in str(line):
            self._cancel.set()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait_cancelled(self, timeout: float) -> bool:
        return self.cancelled


def refusal(call) -> Refusal:
    with pytest.raises(Refusal) as caught:
        call()
    return caught.value


# -- the pins ---------------------------------------------------------------------------------------------------------
def test_every_build_is_pinned_by_name_size_and_sha256_from_github():
    assert release.VERSION == "1.0.0" and release.BUILD == "b12359" and release.BUILD_NUMBER == 12359
    assert set(release.BUILDS) == {(s, w) for s in ("windows", "linux") for w in ("cuda13", "cuda12", "vulkan", "cpu")}
    for (system, which), parts in release.BUILDS.items():
        assert len(parts) == (2 if which.startswith("cuda") else 1)
        for part in parts:
            assert re.fullmatch(r"[0-9a-f]{64}", part.sha256) and part.size > 1_000_000
            source = part.source()
            assert source.url == release.RELEASE + part.name and source.url.startswith("https://github.com/")
            assert (source.size, source.sha256) == (part.size, part.sha256)
            assert source.hosts == ("release-assets.githubusercontent.com",)  # the one the figures saw, for all 12
    # the sizes as bytes (the figures review's finding 32: the plan's "MB" were MiB)
    assert release.bytes_of("windows", "cuda13") == 605_860_311 and release.bytes_of("linux", "cuda12") == 793_594_941
    assert release.BUILDS[("windows", "cuda13")][1].name == "cudart-llama-bin-win-cuda-13.5-x64.zip"  # no tag
    assert release.BUILDS[("linux", "cuda12")][1].name == "cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz"
    assert (release.CUDA13_DRIVER, release.MINOR13_DRIVER, release.CUDA12_DRIVER) == (620, 580, 525)


def test_package_brings_the_build_kept_and_automatic_names_every_case(world):
    p = world.plugin("windows")
    pkg = p.package()
    assert (pkg.title, pkg.version, pkg.pin) == ("llama.cpp", "1.0.0 (b12359)", "b12359")
    assert pkg.size == release.bytes_of("windows", "cuda12")  # the largest it may bring
    assert pkg.hosts == ("github.com", "release-assets.githubusercontent.com")
    for words in ("CUDA 13 build", "620 or newer", "580 or newer for a card of compute capability 12", "CUDA 12 build",
                  "from 525 on", "Vulkan build", "578 MB", "678 MB", "36.3 MB"):
        assert words in pkg.parts, words
    assert "sha256 pinned in this plugin" in pkg.checked and "NVIDIA's terms" in pkg.licence
    assert "Each model carries its own licence" in pkg.licence and "models" in pkg.keeps
    p.build("vulkan")
    pkg = p.package()
    assert (pkg.pin, pkg.size) == ("b12359-vulkan", 38_090_643) and "llama-b12359-bin-win-vulkan-x64.zip" in pkg.parts
    assert "NVIDIA" not in pkg.licence and "Vulkan build" in pkg.version


def test_linux_under_wsl_brings_the_processors_build_where_no_gpu_is_read(world):
    p = world.plugin("linux", wsl="Ubuntu")
    pkg = p.package()
    assert "its processor's build where none is read, or the driver is older than 525 (22.2 MB)" in pkg.parts
    assert "inside WSL" in pkg.parts
    assert "its Vulkan build where none is read" not in pkg.parts  # not a case inside WSL
    assert pkg.size == release.bytes_of("linux", "cuda12") == 793_594_941


def test_nothing_is_installed_from_here_on_another_processor(world):
    p = world.plugin("linux", arch="aarch64")
    assert p.package() is None and "x86-64 only (this processor is aarch64)" in p.runs_here()
    assert "x86-64 only" in refusal(lambda: p.install(True)).args[0]


# -- what is installed ------------------------------------------------------------------------------------------------
def test_installed_reads_files_alone_and_runs_nothing(world):
    p = world.plugin("windows")
    assert p.installed() is None
    world.found["llama-server"] = "C:/tools/llama/llama-server.exe"
    found = p.installed()
    assert (found.how, found.version, found.path) == ("found", "", "C:/tools/llama/llama-server.exe")
    assert "your own install (winget's is the Vulkan build)" in found.note and "never replaced" in found.note
    assert p.status()["installed"].startswith("version not read yet at C:/tools/llama/llama-server.exe")
    folder = world.installed(p, "cuda13")
    mine = p.installed()
    assert (mine.how, mine.path, mine.version) == ("plugin", str(folder), "1.0.0 (b12359), its CUDA 13 build")
    assert mine.note == "it sees CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU"
    assert world.runner.ran == []  # nothing was run for any of it


def test_a_found_llama_is_started_as_llama_serve(world):
    p = world.plugin("linux")
    world.found["llama"] = "/home/me/.local/bin/llama"
    have = p.installed()
    assert have.how == "found" and "started as `llama serve`" in have.note
    assert release.program(p, have) == ("/home/me/.local/bin/llama", "serve")


def test_read_again_reads_a_found_installs_build_and_devices_once(world):
    p = world.plugin("windows")
    world.found["llama-server"] = "C:/tools/llama-server.exe"
    world.runner.version = (0, "", "version: 1.0.0-dev (build 12441, commit 9f8e7d6)\nbuilt with MSVC\n")
    said = p.refresh()
    assert said["ok"] and "b12441 at C:/tools/llama-server.exe" in said["said"]
    assert world.runner.ran == [["C:/tools/llama-server.exe", "--version"],
                                ["C:/tools/llama-server.exe", "--list-devices"]]
    assert "It sees: CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU (24463 MiB, 23000 MiB free)" in p.installed().note
    assert release.found_build(p, p.installed()) == 12441
    world.runner.ran.clear()
    p.status()
    assert world.runner.ran == []  # a status reads what Read again kept


def test_this_plugins_own_install_is_never_run_to_read_its_version(world):
    p = world.plugin("windows")
    world.installed(p)
    p.refresh()
    assert world.runner.ran == []


@pytest.mark.parametrize("text, build", [
    ("version: 1.0.0-dev (build 12359, commit b1b770d9)", 12359),
    ("version: 4000 (b1c2d3e4)\nbuilt with cc", 4000),
    ("version: 1.0.0-dev (commit b1c2d3e4)", None),
    ("build = 7218 (abcdef1)", 7218),
    ("nothing here", None)])
def test_a_build_is_read_leniently_by_its_number(text, build):
    assert release.build_of(text) == build


# -- the build automatic settles ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("system, gpu, caps, which, words", [
    ("windows", GPU_5090, "0, 12.0\n", "cuda13", "compute capability 12.0: the CUDA 12.4 build carries no code"),
    ("windows", "0, NVIDIA GeForce RTX 5090, 32607, 0, 32607, 575.51\n", "0, 12.0\n", "vulkan", "needs 580 or newer"),
    ("windows", "0, NVIDIA GeForce RTX 4090, 24564, 0, 24564, 595.71\n", "0, 8.9\n", "cuda12", "CUDA 12.4"),
    ("windows", "0, NVIDIA GeForce RTX 4090, 24564, 0, 24564, 621.02\n", "0, 8.9\n", "cuda13", "R620"),
    ("windows", "0, NVIDIA GeForce RTX 3060, 12288, 0, 12288, 522.06\n", "0, 8.6\n", "vulkan", "older than 525"),
    ("windows", "", None, "vulkan", "no NVIDIA GPU is read here"),
    ("linux", GPU_5090, "0, 12.0\n", "cuda12", "CUDA 12.8"),
    ("linux", "0, NVIDIA L4, 23034, 0, 23034, 620.10\n", "0, 8.9\n", "cuda13", "R620"),
    ("linux", "0, NVIDIA GeForce GTX 1080, 8192, 0, 8192, 516.94\n", "0, 6.1\n", "vulkan", "older than 525"),
])
def test_automatic_settles_the_build_by_what_nvidia_smi_reads(world, system, gpu, caps, which, words):
    world.runner.gpu, world.runner.caps = gpu, caps
    p = world.plugin(system)
    got, why = release.settle(p)
    assert got == which and words in why, why


def test_under_wsl_with_no_gpu_automatic_brings_the_processors_build(world):
    world.runner.gpu = ""
    p = world.plugin("linux", wsl="Ubuntu")
    assert release.settle(p)[0] == "cpu" and not any("compute_cap" in " ".join(r) for r in world.runner.ran)


def test_under_wsl_a_driver_too_old_for_cuda_12_brings_the_processors_build_as_its_words_say(world):
    """The review: under WSL an NVIDIA card whose driver is older than 525 got the Vulkan build, though the words
    name no Vulkan case inside WSL; it gets the processor's, and the words name that case."""
    world.runner.gpu = "0, NVIDIA GeForce GTX 1080, 8192, 0, 8192, 516.94\n"
    p = world.plugin("linux", wsl="Ubuntu")
    which, why = release.settle(p)
    assert which == "cpu" and "older than 525" in why and "inside WSL" in why, why
    words = release.automatic_words(p)
    assert "its processor's build where none is read, or the driver is older than 525" in words
    assert "Vulkan build where" not in words


@pytest.mark.parametrize("system, own", [("windows", "CUDA 12.4 may be refused by a driver older than 551.78"),
                                         ("linux", "CUDA 12.8 may be refused by a driver older than 570.124.06")])
def test_cuda_12_from_driver_525_says_nvidias_caveat(world, system, own):
    """The review: CUDA 12 from 525 rests on NVIDIA's minor version compatibility, whose caveat was said only for the
    CUDA 13 case: the job's line, Automatic's words and the Needs row say it for CUDA 12 too."""
    world.runner.gpu, world.runner.caps = "0, NVIDIA GeForce RTX 4090, 24564, 0, 24564, 530.30\n", "0, 8.9\n"
    p = world.plugin(system)
    which, why = release.settle(p)
    assert which == "cuda12" and own in why, why
    assert own in release.automatic_words(p)
    needs = {r[0]: r for r in release.brings_table(p)["rows"]}["Needs"][2]
    assert own in needs, needs


def test_a_compute_capability_that_cannot_be_read_falls_back_to_the_drivers_rule(world):
    world.runner.caps = None  # nvidia-smi does not know the field
    p = world.plugin("windows")
    assert release.settle(p)[0] == "cuda12"


# -- the install ------------------------------------------------------------------------------------------------------
def _installs(world, monkeypatch, system="windows", which="cuda13", chosen=True, **plugin):
    """The plugin, the build's archives stood in for; the build chosen on the page unless automatic settles it."""
    parts = world.stand_in_release(monkeypatch, system, which)
    p = world.plugin(system, **plugin)
    if chosen:
        p.build(which)
    return p, parts


def test_the_install_brings_the_settled_build_checks_it_and_puts_it_in_use(world, monkeypatch):
    p, parts = _installs(world, monkeypatch, "windows", "cuda13", chosen=False)
    asked = refusal(lambda: p.install())
    assert asked.accept == ACCEPT_LICENCE and world.opener.asked == []  # its licence is accepted first
    panel = Panel()
    said = p.install(True, log=panel)
    assert said["ok"], said
    folder = p.home / "app" / "b12359-cuda13"
    assert p.current() == folder and (folder / "llama-server.exe").is_file() and (folder / "cudart64_13.dll").is_file()
    record = json.loads((folder / release.RECORD).read_text(encoding="utf-8"))
    assert record["which"] == "cuda13" and [f["name"] for f in record["files"]] == [x.name for x in parts]
    assert record["devices"] == ["CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU (24463 MiB, 23000 MiB free)"]
    assert panel.lines[1].startswith("the build: CUDA 13: the NVIDIA GeForce RTX 5090 Laptop GPU with driver 595.71")
    assert any("compute capability 12.0" in x for x in panel.lines)
    # each archive through github.com, sent on to its storage
    assert [u.split("/")[2] for u in world.opener.asked] == ["github.com", "release-assets.githubusercontent.com"] * 2
    assert not list((p.home / "downloads" / "b12359").glob("*.zip"))  # the archives are removed once in use
    assert p.installed().version == "1.0.0 (b12359), its CUDA 13 build"
    assert p.package().pin == "b12359-cuda13" and refusal(lambda: p.install(True)).args[0].endswith("installed already")
    ran = [r for r in world.runner.ran if "llama-server.exe" in r[0]]
    assert [r[1:] for r in ran] == [["--version"], ["--list-devices"]]


def test_linux_tarballs_are_unpacked_without_their_top_folders_into_one(world, monkeypatch):
    world.runner.gpu = ""  # no card: the processor's build under WSL
    p, _ = _installs(world, monkeypatch, "linux", "cpu", chosen=False, wsl="Ubuntu")
    world.runner.version = (0, "", "version: 1.0.0-dev (build 12359, commit b1b770d9)\n")
    world.runner.devices = (0, "Available devices:\n  (none)\n", "")
    said = p.install(True, log=Panel())
    assert said["ok"], said
    folder = p.home / "app" / "b12359-cpu"
    assert (folder / "llama-server").is_file() and (folder / "LICENSE").is_file()
    assert not (folder / "llama-b12359").exists()
    assert p.installed().note == "it saw no GPU when it was installed: it runs on the processor"


def test_a_linux_cuda_build_brings_its_runtime_beside_llama_server(world, monkeypatch):
    p, _ = _installs(world, monkeypatch, "linux", "cuda12")
    assert p.install(True, log=Panel())["ok"]
    folder = p.home / "app" / "b12359-cuda12"
    assert (folder / "llama-server").is_file() and (folder / "libcudart.so.13").is_file()


def test_a_cuda_build_that_lists_no_cuda_device_is_kept_and_said(world, monkeypatch):
    p, _ = _installs(world, monkeypatch, "windows", "cuda13")
    world.runner.devices = (0, "Available devices:\n  (none)\n", "")
    panel = Panel()
    said = p.install(True, log=panel)
    assert said["ok"] and any("lists no CUDA device: it will run on the processor" in x for x in said["lines"])
    assert any("Choose another build, then Update" in x for x in panel.lines)


def test_an_archive_that_is_not_the_pinned_one_installs_nothing(world, monkeypatch):
    p, parts = _installs(world, monkeypatch, "windows", "vulkan")
    world.opener.files[parts[0].name] = b"PK" + b"x" * (parts[0].size - 2)  # the same size, another file
    said = p.install(True, log=Panel())
    assert not said["ok"] and "not installed" in said["said"] and p.current() is None and p.installed() is None


def test_a_program_of_another_build_is_not_put_in_use(world, monkeypatch):
    p, _ = _installs(world, monkeypatch, "windows", "vulkan")
    world.runner.version = (0, "", "version: 1.0.0-dev (build 12000, commit 1234567)\n")
    said = p.install(True, log=Panel())
    assert not said["ok"] and "says build 12000, not 12359" in said["said"]
    assert p.current() is None and not (p.home / "app" / "b12359-vulkan").exists()  # the half-built folder removed


def test_a_program_that_does_not_start_is_said_with_its_likely_cause(world, monkeypatch):
    p, _ = _installs(world, monkeypatch, "windows", "vulkan")
    world.runner.fails["--version"] = ChildError("llama-server --version could not run: [WinError 126]")
    said = p.install(True, log=Panel())
    assert not said["ok"] and "Visual C++ runtime (x64) may be missing" in said["said"]
    p, _ = _installs(world, monkeypatch, "linux", "cuda12")
    world.runner.fails["--version"] = ChildError("llama-server --version could not run")
    said = p.install(True, log=Panel())
    assert "glibc 2.39 or newer: it was made on Ubuntu 24.04" in said["said"]


def test_a_cancel_keeps_what_came_and_installs_nothing(world, monkeypatch):
    p, parts = _installs(world, monkeypatch, "windows", "cuda13")
    said = p.install(True, log=Panel(cancel_after=f"{parts[1].name} ("))
    assert not said["ok"] and said["said"] == CANCELLED  # "what was downloaded is kept for the next try"
    assert p.current() is None and not (p.home / "app" / "b12359-cuda13").exists()
    kept = p.home / "downloads" / "b12359" / parts[0].name  # the first archive, whole, for the next try
    assert kept.is_file() and kept.stat().st_size == parts[0].size
    assert not (p.home / "downloads" / "b12359" / parts[1].name).exists()


def test_a_cancel_while_the_program_is_checked_installs_nothing(world, monkeypatch):
    """The review: a Cancel pressed while the unpacked llama-server is run (--version, --list-devices) was not
    honoured: the install was put in use and answered as installed. It is now answered as cancelled, nothing in use,
    the half-built folder removed and the archives kept."""
    p, parts = _installs(world, monkeypatch, "windows", "vulkan")
    panel = Panel()
    real = world.runner.__call__

    def runner(args, timeout):
        if "--version" in args:
            panel.cancel()  # the user presses Cancel while llama-server --version runs
        return real(args, timeout)
    p.out.runner = runner
    said = p.install(True, log=panel)
    assert not said["ok"] and said["said"].startswith("cancelled: nothing was installed"), said
    assert p.current() is None and p.installed() is None and not (p.home / "app" / "b12359-vulkan").exists()
    assert (p.home / "downloads" / "b12359" / parts[0].name).is_file()


def test_a_redirect_to_a_host_the_figures_did_not_see_is_refused_in_words(world, monkeypatch):
    """The review: ASSET_HOSTS allowed objects.githubusercontent.com, which the figures saw for none of the twelve
    archives and the install's words do not name. A redirect there is refused, nothing asked of it."""
    p, parts = _installs(world, monkeypatch, "windows", "vulkan")
    world.opener.storage = "objects.githubusercontent.com"
    said = p.install(True, log=Panel())
    assert not said["ok"] and "was sent on to https://objects.githubusercontent.com," in said["said"], said
    assert "release-assets.githubusercontent.com alone: nothing was asked of it" in said["said"]
    assert [u.split("/")[2] for u in world.opener.asked] == ["github.com"] and p.current() is None


def test_windows_without_its_c_runtime_and_a_full_disk_are_refused_before_the_job(world, monkeypatch):
    p = world.plugin("windows")
    (world.tmp / "Windows" / "System32" / "vcruntime140.dll").unlink()
    words = refusal(lambda: p.install(True, log=Panel())).args[0]
    assert "needs Microsoft's Visual C++ runtime (x64)" in words and world.opener.asked == []
    assert any(r[0] == "Runtime" for r in release.lacks(p)["rows"])
    (world.tmp / "Windows" / "System32" / "vcruntime140.dll").write_bytes(b"MZ")
    monkeypatch.setattr(release, "disk_free", lambda path: 1 << 30)
    assert "needs 3.0 GB free" in refusal(lambda: p.install(True, log=Panel())).args[0]


def test_a_found_install_is_never_replaced(world):
    p = world.plugin("windows")
    world.found["llama-server"] = "C:/tools/llama-server.exe"
    assert "your own install: it is used as it is" in refusal(lambda: p.install(True)).args[0]


# -- the uninstall and the build ---------------------------------------------------------------------------------------
def test_the_uninstall_removes_the_install_alone_and_keeps_the_models(world):
    p = world.plugin("windows")
    world.installed(p)
    (p.home / "models").mkdir(parents=True)
    (p.home / "models" / "blob").write_bytes(b"x" * 2048)
    (p.home / "downloads" / "b12359").mkdir(parents=True)
    outside = world.tmp / "keep-me.txt"
    outside.write_text("mine", encoding="utf-8")
    assert refusal(lambda: p.uninstall()).accept == "accept_removal"
    said = p.uninstall(True)
    assert said["ok"] and p.installed() is None and not (p.home / "app").exists()
    assert (p.home / "models" / "blob").is_file() and outside.is_file()
    assert any(x.startswith(f"kept: {p.home / 'models'} (1 files") for x in said["lines"])


def test_the_build_is_kept_through_the_login_and_offers_an_update(world):
    p = world.plugin("windows")
    world.installed(p, "cuda13")
    assert p.package().pin == "b12359-cuda13"  # automatic takes the build in use
    said = p.build("vulkan")
    assert said["ok"] and said["said"].startswith("Install brings llama.cpp's Vulkan build")
    assert p.login.kept["LIMINAL_AI_LLAMACPP_BUILD"] == "vulkan"
    assert "Update… brings the Vulkan build" in said["lines"][1]
    out = p.status()
    assert any(a.get("title") == "Update to 1.0.0 (b12359), its Vulkan build…" for a in out["actions"])
    p.build("automatic")
    assert "LIMINAL_AI_LLAMACPP_BUILD" not in p.login.kept and p.package().pin == "b12359-cuda13"
    words = refusal(lambda: p.build("rocm")).args[0]
    assert "the build is one of Automatic, CUDA 13, CUDA 12, Vulkan, Processor" in words
