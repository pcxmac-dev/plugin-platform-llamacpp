"""llama.cpp's models: looked up at the Hugging Face Hub, downloaded through a router as a job that can be
cancelled, deleted, loaded, unloaded and described, and what the Models tab adds (the user, 2026-11-16: "all three
plugins need to allow installation and configuration of their respective code"; the decisions' A.10: "a download is
a job with progress (size and licence first where the maker gives them); delete is confirmed; load and unload where
the server can").

- Find a model (`find`, a read): `user/repository[:quantization]` is asked of the Hub's API, without a token: the
  model's card (gated or not, its licence) and its files (GET .../tree/main, page by page through its Link header).
  The files llama.cpp would take are picked by its own rules (its common/download.cpp at b12359, ported): the
  quantization named, else Q4_K_M, else Q8_0, else the first model file; every part of a split file; its best
  projector (mmproj). Their sizes summed are what a download brings. A lookup is kept an hour, 12 at most, newest
  first, and nothing is downloaded until a row's Download… is asked.
- Download (`pull`, a job): a router of b9688 or newer only, through its own API (POST /models, then its event
  stream /models/sse, whose download_progress the panel shows); refused before its first line, from what is kept
  alone (the lookup, the instance's state), then asked twice with the model's size, files, folder, free space and
  licence. Cancel pauses it (POST /models/unload: llama.cpp keeps its part, listed as incomplete, and a next
  download goes on from it); it stops by itself after 6 hours. No Hugging Face token is given to the server. A server
  run here that is started offline (its Offline setting, on by default) downloads nothing, so Download… is off for it
  until Offline is turned off. A lookup the Hub answered with nothing to download (no such repository, or none of its
  files for the quantization typed) offers none; one the Hub did not answer can be downloaded unseen, asked twice, the
  quantization as typed and the projector as the router picks it.
- Delete (DELETE /models), Load (POST /models/load), Unload (POST /models/unload), Details (the router's listing and
  the loaded model's /props, asked with autoload off): each confirmed but Details, each refused in words where the
  server cannot (one model's server; a model of your GGUF folder, which llama.cpp never deletes; one loaded)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from liminal_libs.instances import Refusal, has_control
from liminal_libs.machines import gpu
from liminal_libs.machines.ai import ACCEPT_LICENCE, exactly
from liminal_libs.machines.endpoint import Endpoint, EndpointError, Progress
from liminal_libs.machines.terminal import Cancelled
from liminal_libs.machines.world import size_words

from . import release, server

FOUND_S = 3600.0  # a lookup is kept this long
FOUND_MOST = 12  # lookups kept at once, the newest first
TREE_PAGES = 10  # pages of a repository's files read at most (1,000 entries each)
ASK_S = 10.0  # what an ask of the Hub or of a server for a change waits
POST_S = 60.0  # a download's first ask: the router asks the Hub itself before it answers
QUIET_S = 30.0  # the event stream may say nothing this long before the router is asked how the download is
MODEL_CHARS = 200
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
PLACEHOLDER = "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
DEFAULT_TAGS = ("Q4_K_M", "Q8_0")  # llama.cpp's own choice when no quantization is named
SIDECARS = ("mmproj", "mtp-", "dflash-", "eagle3-", "dspark-")  # what a file's name says it is beside a model
SPLIT = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.I)
TAGGED = re.compile(r"[-.]([A-Z0-9_]+)$", re.I)
NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?', re.I)
# what this module asks a server (its router's model calls) and the Hub
CALLS = ("GET /models/sse", "POST /models", "POST /models/load", "POST /models/unload", "DELETE /models")
HUB_CALLS = ("GET /api/models/{repo}", "GET /api/models/{repo}/tree/main")
FIT = ("llama.cpp fits a model by itself: it lowers the context (never under 4,096 tokens) and the layers on the GPU "
       "until 1,024 MiB of the card stay free; what is left runs in system memory, slower.")
GATED = ("gated: its files need a Hugging Face account that was given access, and no token is given to llama.cpp in "
         "this version")


@dataclass(frozen=True)
class Found:
    """A model looked up: what is downloaded (`model`, as llama.cpp names it: user/repository:QUANTIZATION, the
    quantization typed kept whatever the Hub answered), where it was looked up and its page, when (`at`), its size in
    bytes and its files, its licence (and its page), whether it is gated, why its size or licence is not said (""
    when the lookup was whole), whether that is because the Hub said there is nothing to download for it (`absent`:
    no such repository, or none of its files for the quantization: nothing that was asked can be downloaded; else the
    Hub did not answer, and a download would be unseen), what was typed (`asked`), its projector's file and every
    file's name."""

    model: str
    where: str
    page: str
    at: float
    size: int | None = None
    files: int = 0
    licence: str = ""
    gated: bool = False
    why: str = ""
    asked: str = ""
    mmproj: str = ""
    names: tuple = ()
    licence_page: str = ""
    absent: bool = False

    @property
    def ok(self) -> bool:
        return not self.why

    @property
    def unseen(self) -> bool:
        """Whether a download of it would be unseen: the Hub did not answer, so its size and licence are not said."""
        return bool(self.why) and not self.absent


@dataclass(frozen=True)
class File:
    """One file of a repository, as the Hub's tree lists it: its path and its size in bytes."""

    path: str
    size: int


# -- a model's name -------------------------------------------------------------------------------------------------
def clean_model(text) -> tuple[str, str]:
    """(repository, quantization) of a model typed as llama.cpp names one; a Refusal in words otherwise."""
    said = " ".join(str(text or "").split())
    if not said:
        raise Refusal(f"type a model's name first: user/repository, and its quantization after a colon "
                      f"({PLACEHOLDER})")
    repo, _, tag = said.partition(":")
    if not REPO.fullmatch(repo) or ".." in repo or (tag and not TAG.fullmatch(tag)):
        raise Refusal(f"{said[:80]!r} is no model llama.cpp downloads: a repository of huggingface.co (user/name), "
                      f"and its quantization after a colon ({PLACEHOLDER})")
    return repo, tag


def model_id(text) -> str:
    """A model as a server names it (a router's id, a file's name): words on one line, MODEL_CHARS at most."""
    said = " ".join(str(text or "").split())
    if not said or len(said) > MODEL_CHARS or has_control(said):
        raise Refusal(f"name a model as the server lists it, {MODEL_CHARS} characters at most")
    return said


# -- llama.cpp's own way of picking a repository's files (common/download.cpp, b12359) ----------------------------
def split_info(path: str) -> tuple[str, str, int, int] | None:
    """(prefix, quantization, part, parts) of a .gguf's path; None for another file."""
    if not path.endswith(".gguf"):
        return None
    prefix, index, count = path[:-len(".gguf")], 1, 1
    found = SPLIT.match(prefix)
    if found:
        prefix, index, count = found.group(1), int(found.group(2)), int(found.group(3))
    tag = TAGGED.search(prefix)
    return prefix, tag.group(1).upper() if tag else "", index, count


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def is_model(path: str) -> bool:
    """A model's own file: a .gguf whose name says neither an importance matrix nor a file beside a model."""
    name = _name(path)
    return path.endswith(".gguf") and "imatrix" not in name and not any(k in name for k in SIDECARS)


def best_model(files: list[File], tag: str) -> File | None:
    """The file llama.cpp loads for `tag` (its quantization as a whole word of the path), else its defaults', else
    the first model file when no quantization was named; a later part of a split file never."""
    for want in ([tag] if tag else list(DEFAULT_TAGS)):
        pattern = re.compile(r"(^|[-./])" + re.escape(want) + r"[.-]", re.I)
        for f in files:
            info = split_info(f.path)
            if is_model(f.path) and pattern.search(f.path) and info and not (info[3] > 1 and info[2] != 1):
                return f
    if not tag:
        for f in files:
            info = split_info(f.path)
            if is_model(f.path) and info and not (info[3] > 1 and info[2] != 1):
                return f
    return None


def split_files(files: list[File], primary: File) -> list[File]:
    """Every part of a split file (the same prefix, the same number of parts), else the file alone."""
    info = split_info(primary.path)
    if info is None or info[3] <= 1:
        return [primary]
    return [f for f in files if (split_info(f.path) or ("", "", 0, 0))[0::3] == (info[0], info[3])]


def _bits(path: str) -> int:
    info = split_info(path)
    digits = re.search(r"\d+", info[1]) if info else None
    return int(digits.group(0)) if digits else 0


def best_sibling(files: list[File], model: str, keyword: str = "mmproj") -> File | None:
    """The projector llama.cpp takes with a model: a .gguf naming `keyword` in a folder on the model's own path, the
    deepest first, then the nearest quantization."""
    folders = model.split("/")[:-1]
    want = _bits(model)
    best, best_depth, best_diff = None, 0, 0
    for f in files:
        if not f.path.endswith(".gguf") or keyword not in f.path:
            continue
        own = f.path.split("/")[:-1]
        if len(own) > len(folders) or folders[:len(own)] != own:
            continue
        depth, diff = len(own), abs(_bits(f.path) - want)
        if best is None or depth > best_depth or (depth == best_depth and diff < best_diff):
            best, best_depth, best_diff = f, depth, diff
    return best


def pick(files: list[File], tag: str) -> tuple[list[File], File | None, str]:
    """(the model's files, its projector, its quantization) as llama.cpp would download them for `tag`."""
    primary = best_model(files, tag)
    if primary is None:
        return [], None, ""
    info = split_info(primary.path)
    return split_files(files, primary), best_sibling(files, primary.path), (info[1] if info else "") or tag.upper()


# -- the Hub ----------------------------------------------------------------------------------------------------------
def agent() -> str:
    from . import __version__

    return f"plugin-platform-llamacpp/{__version__}"


def hub(p) -> Endpoint:
    """The Hugging Face Hub's API, asked over https, without a token, with this plugin's name and version alone."""
    return Endpoint(release.HUB, timeout=ASK_S, transport=p.out.transport, agent=agent())


def hub_host() -> str:
    return urlsplit(release.HUB).hostname or ""


def _licence(info: dict) -> tuple[str, str]:
    """A model's licence as its card says it (cardData.license; `other` by its license_name), else its tags'
    license:..., and the page its words are at."""
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    named = card.get("license")
    named = ", ".join(str(x) for x in named) if isinstance(named, list) else str(named or "")
    if named.strip().lower() == "other" and card.get("license_name"):
        named = f"{card.get('license_name')} (its own)"
    if not named.strip():
        tags = [str(t) for t in info.get("tags") or [] if str(t).startswith("license:")]
        named = tags[0].split(":", 1)[1] if tags else ""
    link = card.get("license_link")
    return named.strip()[:120], (str(link) if isinstance(link, str) and link.startswith("https://") else "")


def tree(at: Endpoint, repo: str) -> list[File]:
    """A repository's files (GET /api/models/{repo}/tree/main?recursive=true), page by page through its Link header,
    TREE_PAGES at most; each file's real size (a large file's lfs size)."""
    files: list[File] = []
    path = f"/api/models/{repo}/tree/main?recursive=true"
    for _ in range(TREE_PAGES):
        got = at.answer("GET", path)
        rows = got.data if isinstance(got.data, list) else []
        for row in rows:
            if not isinstance(row, dict) or row.get("type") != "file" or not isinstance(row.get("path"), str):
                continue
            lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
            size = lfs.get("size") if isinstance(lfs.get("size"), int) else row.get("size")
            files.append(File(row["path"], size if isinstance(size, int) and size >= 0 else 0))
        following = NEXT.search(str(got.headers.get("link", "")))
        if not following:
            return files
        parts = urlsplit(following.group(1))
        if parts.scheme != "https" or (parts.hostname or "") != hub_host():
            return files  # a next page elsewhere is never asked
        path = parts.path + (f"?{parts.query}" if parts.query else "")
    return files


def keep(p, found: Found) -> None:
    """A lookup kept: the newest first, FOUND_MOST at most, one per model."""
    with p._rt.lock:
        now = p.out.clock()
        same = {found.model.lower(), found.asked.lower()}
        kept = [f for f in p._rt.found if not same & {f.model.lower(), f.asked.lower()} and 0 <= now - f.at < FOUND_S]
        p._rt.found = [found, *kept][:FOUND_MOST]


def lookups(p) -> list[Found]:
    """The lookups kept, younger than FOUND_S, the newest first."""
    with p._rt.lock:
        now = p.out.clock()
        return [f for f in p._rt.found if 0 <= now - f.at < FOUND_S]


def found_of(p, model) -> Found | None:
    """The kept lookup of a model, by what is downloaded or by what was typed, its case ignored."""
    said = " ".join(str(model or "").split()).lower()
    return next((f for f in lookups(p) if said in (f.model.lower(), f.asked.lower())), None)


def find(p, model) -> dict:
    """A model looked up at the Hugging Face Hub (its card, its files), kept an hour: what llama.cpp would download
    for it, its size, its licence, whether it is gated and what fits this machine's GPU. A read."""
    repo, tag = clean_model(model)
    typed = f"{repo}:{tag}" if tag else repo
    at, where = hub(p), hub_host()
    info, why, files, absent = None, "", [], False
    try:
        info = at.get(f"/api/models/{repo}")
    except EndpointError as e:
        if e.status in (401, 404):  # (no token is sent: a private one is as none, for this plugin and llama.cpp)
            why, absent = "it knows no such model, or it is private", True
        else:
            why = f"it did not answer ({' '.join(str(e).split())[:160]})"
    if info is not None and not isinstance(info, dict):
        info, why = None, "its answer is not a model's card"
    if info is not None:
        try:
            files = tree(at, repo)
        except EndpointError as e:
            why = f"its files could not be listed ({' '.join(str(e).split())[:160]})"
    chosen, mmproj, quant = pick(files, tag)
    if info is not None and not why and not chosen:  # (llama.cpp's router refuses it too: "resolves no file")
        why, absent = f"{repo} holds no GGUF file llama.cpp takes" + (f" for {tag}" if tag else ""), True
    quant = quant or tag.upper()  # what was typed is what is asked for, whatever the Hub answered
    named, link = _licence(info or {})
    gated = bool(info) and info.get("gated") not in (None, False, "false", "False")
    taken = [*chosen, *([mmproj] if mmproj else [])]
    size = sum(f.size for f in taken) if taken else None
    found = Found(model=f"{repo}:{quant}" if quant else repo, where=where, page=f"{release.HUB}/{repo}",
                  at=p.out.clock(), size=size or None, files=len(taken), licence=named, gated=gated, why=why,
                  asked=typed, mmproj=mmproj.path if mmproj else "", names=tuple(f.path for f in taken),
                  licence_page=link, absent=absent)
    keep(p, found)
    if absent:
        return p._reply(f"{typed}: {where} says there is nothing to download for it: {why}", False,
                        ["kept an hour: Download… is off for it (nothing that was asked can be downloaded)"])
    if why:
        return p._reply(f"{typed}: {where} did not say its size or licence: {why}", False,
                        ["kept an hour: Download… would ask twice, saying it would be downloaded unseen"])
    lines = [_fit(p, size), f"read at {where}: {', '.join(found.names[:4])}"
             + (f" and {len(found.names) - 4} more" if len(found.names) > 4 else "")]
    if gated:
        lines.insert(0, GATED)
    plural = "" if found.files == 1 else "s"
    return p._reply(f"{found.model}: {size_words(size or 0)} to download in {found.files} file{plural}; licence: "
                    f"{named or 'not said'}", True, lines)


def clear_found(p) -> dict:
    """The lookups forgotten (only what this page shows)."""
    with p._rt.lock:
        n, p._rt.found = len(p._rt.found), []
    return p._reply(f"{n} lookup{'' if n == 1 else 's'} forgotten", True)


def _fit(p, size) -> str:
    if not size:
        return "its size is not said, so what fits is not known"
    try:
        said = gpu.fit(size, p._gpus())
    except Exception:  # noqa: BLE001 - a GPU that cannot be read: what fits is not known
        said = ""
    return f"{said}. {FIT}" if said else FIT


# -- what an instance can do ------------------------------------------------------------------------------------------
def server_kind(p, instance) -> tuple[bool | None, int | None]:
    """(whether it is a router, its build) as it last answered; a server run here is this install's router."""
    seen = p.seen(instance.name)
    if seen is not None and seen.ok and seen.facts.get("router"):
        build = re.match(r"^b(\d+)", str(seen.facts.get("build") or ""))
        return seen.facts.get("router") == "yes", int(build.group(1)) if build else None
    if instance.where == "local":
        return True, release.found_build(p, p.installed())
    return None, None


def unable(p, instance, what: str) -> str:
    """"" or why an instance cannot do `what` (download, delete, load) now, in words: from kept state alone. A server
    run here that is started offline (its Offline setting, read as it starts: Change… is refused while it runs)
    downloads nothing: its router refuses what is not in its folder."""
    name = instance.name
    verbs = {"download": "downloads", "delete": "deletes", "load": "loads and unloads"}
    if what == "download" and instance.where == "local" and server.offline(instance):
        return server.offline_words(instance)
    if instance.where == "local" and name not in p.running():
        return f"start {name} first: llama.cpp {verbs[what]} a model through its running server"
    router, build = server_kind(p, instance)
    if router is None:
        seen = p.seen(name)
        if seen is None:
            return f"{name} has not been asked yet: its Models tab asks it while it is shown"
        return f"{name} did not answer as llama.cpp when it was last asked: {seen.said}"
    if not router:
        return f"{name} serves one model: it loads, downloads and deletes none over its API"
    if what in ("download", "delete") and build is not None and build < release.MODELS_FROM:
        return (f"this llama.cpp (b{build}) is older than b{release.MODELS_FROM}: its router downloads and deletes "
                "nothing over its API")
    return ""


def _where_words(p, instance) -> str:
    if instance.where == "remote":
        return f"on {instance.host}"
    return f"into {server.models_folder(p, instance)}" if instance.where == "local" else "into its own models folder"


def _accept_words(p, instance, found: Found) -> str:
    """The second confirmation of a download, built from the kept lookup alone (nothing asked)."""
    if found.unseen:
        return (f"{found.where} did not say {found.model}'s size or licence ({found.why}): it would be downloaded "
                "unseen, its quantization as typed and its projector as llama.cpp's router picks them.")
    folder = ""
    if instance.where == "local":
        free = release.disk_free(server.models_folder(p, instance))
        folder = f", {_where_words(p, instance)}" + (f" ({size_words(free)} free)" if free is not None else "")
    elif instance.where == "remote":
        folder = f", into {instance.name}'s models folder on {instance.host}"
    page = found.licence_page or found.page
    plural = "" if found.files == 1 else "s"
    return (f"{found.model}: {size_words(found.size or 0)} in {found.files} file{plural} from {found.where}{folder}. "
            f"Licence: {found.licence or 'not said'}; its words are at {page}.")


def pull(p, name, model, accept_licence=False, log=None) -> dict:
    """A model downloaded by a router (a job: its progress in the panel, Cancel pauses it). Refused before its first
    line, from what is kept alone: an instance that cannot download now (a server run here started offline among
    them); a name that is none; no lookup kept; one the Hub said there is nothing to download for; a gated model;
    too little free in a local server's folder; then its licence and size, asked twice."""
    instance = p.instance(name)
    clean_model(model)
    why = unable(p, instance, "download")
    if why:
        raise Refusal(why)
    found = found_of(p, model)
    if found is None:
        raise Refusal(f"look {model} up first (Find a model): its size and licence are read before a download")
    if found.absent:
        raise Refusal(f"{found.model}: {found.where} says {found.why}: nothing that was asked can be downloaded")
    if found.gated:
        raise Refusal(f"{found.model} is {GATED}")
    if instance.where == "local" and found.size:
        folder = server.models_folder(p, instance)
        free = release.disk_free(folder)
        if free is not None and free < found.size:
            raise Refusal(f"{found.model} needs {size_words(found.size)}; {folder} has {size_words(free)} free")
    if not exactly(accept_licence):
        raise Refusal(_accept_words(p, instance, found), accept=ACCEPT_LICENCE)
    return _download(p, instance, found, log)


def _download(p, instance, found: Found, log) -> dict:
    say = log if callable(log) else (lambda line: None)
    cancel = p.cancel_of(log)
    target, name = found.model, instance.name
    say(f"{name} downloads {target} from huggingface.co {_where_words(p, instance)} and lists it once it is whole. No "
        "Hugging Face token is given to it.")
    with p._rt.lock:
        p._rt.pull = log
    try:
        if cancel():
            return p._reply(f"cancelled: nothing of {target} was asked for", False)
        at = p.endpoint(instance, timeout=POST_S)
        # the files the lookup read, its projector among them where it found one; downloaded unseen, the router's
        # own picks (its default sidecars bring a projector where the repository has one)
        asked = {"model": target} if found.unseen else {"model": target,
                                                         "sidecars": ["mmproj"] if found.mmproj else []}
        try:
            at.post("/models", asked)
        except EndpointError as e:
            return p._reply(f"{target} was not downloaded: {name} said {' '.join(str(e).split())[:300]}", False)
        ended = _follow(p, at, target, Progress(log, clock=p.out.clock), cancel, log)
    finally:
        with p._rt.lock:
            if p._rt.pull is log:
                p._rt.pull = None
    p.forget_seen(name)
    kept = "what is here of {m} is kept, and the next download goes on from it".format(m=target)
    if ended in ("cancelled", "timeout"):
        _pause(p, instance, target)
        said = f"cancelled: {kept}" if ended == "cancelled" else f"stopped after 6 hours: {kept}"
        say(said)
        return p._reply(said, False)
    if ended == "stopped":
        return p._reply(f"the download of {target} was paused from elsewhere: what is here of it is kept", False)
    if ended == "lost":
        return p._reply(f"{name} stopped answering while it downloaded {target}: {kept}", False)
    if ended == "finished":
        size = f" ({size_words(found.size)})" if found.size else ""
        say(f"{target} is here{size}")
        return p._reply(f"{target} is here{size}", True, [f"listed on {name}: its Models tab shows it"])
    return p._reply(f"{target} was not downloaded: {name} says its download failed (its log says why)", False)


def _event(line: str) -> dict | None:
    """One event of a router's stream: its `data:` line's JSON {"model", "event", "data"}."""
    if not line.startswith("data:"):
        return None
    try:
        event = json.loads(line[5:].strip())
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _bytes(data) -> tuple[int, int]:
    """(done, total) of a download_progress event, its files summed (under its data, or its data's "progress")."""
    files = data.get("progress") if isinstance(data, dict) and isinstance(data.get("progress"), dict) else data
    done = total = 0
    for value in (files or {}).values() if isinstance(files, dict) else ():
        if isinstance(value, dict):
            done += int(value.get("done") or 0) if isinstance(value.get("done"), int) else 0
            total += int(value.get("total") or 0) if isinstance(value.get("total"), int) else 0
    return done, total


def _state(at: Endpoint, target: str) -> str | None:
    """A model's status as the router lists it now ("downloading", "unloaded", "incomplete", ...); None: not listed."""
    try:
        listed = at.get("/models", timeout=ASK_S)
    except EndpointError:
        return "unknown"
    for row in (listed.get("data") if isinstance(listed, dict) else None) or []:
        if isinstance(row, dict) and str(row.get("id") or "").lower() == target.lower():
            status = row.get("status") if isinstance(row.get("status"), dict) else {}
            return str(status.get("value") or "unloaded")
    return None


def _rest(log, seconds: float) -> None:
    """A short wait between two rounds of a download's following, ended at once by a cancel (the job's Output's
    wait_cancelled); none where the log cannot wait."""
    wait = getattr(log, "wait_cancelled", None)
    if callable(wait):
        try:
            wait(seconds)
        except Exception:  # noqa: BLE001 - a log that cannot wait: the next round comes at once
            pass


def _follow(p, at: Endpoint, target: str, progress: Progress, cancel, log=None) -> str:
    """A download followed on the router's event stream until it ends: "finished", "failed", "stopped" (paused by
    another client), "cancelled", "timeout" (PULL_S) or "lost" (the router stopped answering). A stream quiet for
    QUIET_S, or broken off, is followed again once the router is asked how the download is."""
    start, left, misses = p.out.clock(), release.PULL_S, 0
    while True:
        try:
            for line in at.lines("GET", "/models/sse", timeout=QUIET_S, deadline=max(1.0, left), cancel=cancel):
                event = _event(line)
                if event is None or str(event.get("model") or "").lower() != target.lower():
                    continue
                kind = event.get("event")
                if kind == "download_progress":
                    done, total = _bytes(event.get("data"))
                    progress.say(done, total or None, target)
                elif kind == "download_finished":
                    return "finished"
                elif kind == "download_failed":
                    data = event.get("data") if isinstance(event.get("data"), dict) else {}
                    return "stopped" if data.get("stopped") is True else "failed"
        except Cancelled:
            return "cancelled"
        except EndpointError:
            pass  # quiet, or broken off: the router is asked below
        if cancel():
            return "cancelled"
        state = _state(at, target)
        misses = misses + 1 if state == "unknown" else 0
        if misses >= 3:
            return "lost"
        if state not in ("downloading", "unknown"):
            return "finished" if state not in (None, "incomplete") else "failed"
        left = release.PULL_S - (p.out.clock() - start)
        if left <= 0:
            return "timeout"
        _rest(log, 2.0)


def _pause(p, instance, target: str) -> None:
    """A download paused (POST /models/unload): llama.cpp keeps its part, listed as incomplete."""
    try:
        p.endpoint(instance, timeout=ASK_S).post("/models/unload", {"model": target})
    except EndpointError:
        pass


def _row(p, instance, ident: str) -> dict | None:
    """A model of a router as it lists it now (GET /models), its id's case ignored."""
    listed = p.endpoint(instance, timeout=ASK_S).get("/models")
    for row in (listed.get("data") if isinstance(listed, dict) else None) or []:
        if isinstance(row, dict) and str(row.get("id") or "").lower() == ident.lower():
            return server.router_model(p, instance, row)
    return None


def delete_model(p, name, model) -> dict:
    """A model deleted from a router's cache (DELETE /models): refused where the server cannot (one model's server,
    older than b9688), for one not listed there, one of your GGUF folder or a preset's (llama.cpp deletes only what
    it downloaded), one loaded or downloading."""
    instance = p.instance(name)
    ident = model_id(model)
    why = unable(p, instance, "delete")
    if why:
        raise Refusal(why)
    row = _row(p, instance, ident)
    if row is None:
        raise Refusal(f"{ident} is not listed on {name}")
    if not row["can_remove"]:
        where = "in your GGUF folder" if row["source"] == "models_dir" else "a preset's"
        raise Refusal(f"{ident} is {where}: llama.cpp deletes only what it downloaded")
    if row["loaded"] or row["status"] == "loading":
        raise Refusal(f"{ident} is loaded on {name}: unload it first")
    if row["status"] == "downloading":
        raise Refusal(f"{ident} is downloading on {name}: Cancel its download first")
    p.endpoint(instance, timeout=ASK_S).answer("DELETE", "/models", params={"model": row["id"]})
    p.forget_seen(name)
    return p._reply(f"{row['id']} is deleted from {name}", True,
                    [f"its files go from {name}'s models folder {_where_words(p, instance)}; downloaded again, it "
                     "comes back"])


def load(p, name, model) -> dict:
    """A model loaded by a router now (POST /models/load): a program of its own beside it."""
    instance = p.instance(name)
    ident = model_id(model)
    why = unable(p, instance, "load")
    if why:
        raise Refusal(why)
    row = _row(p, instance, ident)
    if row is None:
        raise Refusal(f"{ident} is not listed on {name}")
    if row["loaded"] or row["status"] == "loading":
        return p._reply(f"{ident} is loaded on {name} already", True)
    if row["status"] in ("downloading", "incomplete"):
        raise Refusal(f"{ident} is not whole on {name} yet: its download "
                      + ("is under way" if row["status"] == "downloading" else "goes on with Download… again"))
    p.endpoint(instance, timeout=ASK_S).post("/models/load", {"model": row["id"]})
    p.forget_seen(name)
    return p._reply(f"{row['id']} is loading: Models shows when it is loaded", True)


def unload(p, name, model) -> dict:
    """A model a router runs unloaded now (POST /models/unload): its memory freed."""
    instance = p.instance(name)
    ident = model_id(model)
    why = unable(p, instance, "load")
    if why:
        raise Refusal(why)
    row = _row(p, instance, ident)
    if row is None:
        raise Refusal(f"{ident} is not listed on {name}")
    if not (row["loaded"] or row["status"] == "loading"):
        return p._reply(f"{ident} is not loaded on {name}", True)
    p.endpoint(instance, timeout=ASK_S).post("/models/unload", {"model": row["id"]})
    p.forget_seen(name)
    return p._reply(f"{row['id']} is unloaded from {name}", True)


def _props_lines(props: dict) -> list[str]:
    lines = []
    if props.get("model_path"):
        lines.append(f"file: {props['model_path']}")
    settings = props.get("default_generation_settings") if isinstance(props.get("default_generation_settings"),
                                                                      dict) else {}
    if isinstance(settings.get("n_ctx"), int) and settings["n_ctx"] > 0:
        lines.append(f"context: {settings['n_ctx']:,} tokens")
    if isinstance(props.get("total_slots"), int):
        lines.append(f"slots: {props['total_slots']} (requests answered at once)")
    modalities = props.get("modalities") if isinstance(props.get("modalities"), dict) else {}
    takes = ["text", *(k for k in ("vision", "audio") if modalities.get(k) is True)]
    lines.append(f"takes in: {', '.join(takes)}")
    if props.get("is_sleeping") is True:
        lines.append("sleeping: its memory is freed until it is asked again")
    return lines


def details(p, name, model) -> dict:
    """What a server says of a model: its listing (where it is from, its state, what it takes in, the context it was
    trained for) and, once loaded, its file, context and slots (/props, asked with autoload off). A read."""
    instance = p.instance(name)
    ident = model_id(model)
    router, _ = server_kind(p, instance)
    at = p.endpoint(instance, timeout=ASK_S)
    if router is False:
        props = at.get("/props")
        return p._reply(f"{ident} on {name}", True, _props_lines(props if isinstance(props, dict) else {}))
    row = _row(p, instance, ident)
    if row is None:
        raise Refusal(f"{ident} is not listed on {name}")
    source = {"cache": "downloaded into its models folder", "models_dir": "your GGUF folder", "preset": "a preset"}
    lines = [f"from {source.get(row['source'], row['source'] or 'its listing')}",
             f"state: {row['status'] or 'unloaded'}" + (f" ({row['detail']})" if row["detail"] else "")]
    if row["train"]:
        lines.append(f"trained for a context of {row['train']:,} tokens")
    if row["takes"]:
        lines.append(f"takes in: {', '.join(row['takes'])}")
    if row["sidecars"]:
        lines.append(f"with: {', '.join(row['sidecars'])}")
    if row["loaded"]:
        props = at.get("/props", params={"model": row["id"], "autoload": "false"})
        lines += _props_lines(props if isinstance(props, dict) else {})
    else:
        lines.append(f"{row['id']} is not loaded: its file, its context and its slots are said once it is")
    lines.append(_told(instance, row))
    return p._reply(f"{row['id']} on {name}", True, lines)


def _told(instance, row: dict) -> str:
    """What context the harnesses are told a router's model runs with (its `serving` row), in words."""
    context = row.get("context")
    if context:
        setting = instance.where == "local" and instance.settings.get("context") == context
        return f"the harnesses are told its context is {context:,} tokens" + (" (its Context size)" if setting else
                                                                                  " (as its router reports it)")
    return ("the harnesses are told no context for it: llama.cpp's router does not report the context a loaded model "
            "runs with (only with its memory accounting on), so one is told only where Context size is set (Change…, "
            "for a server run here)")


# -- the Models tab's own parts ---------------------------------------------------------------------------------------
def _button(form: str, title: str, values: dict, why: str = "", label: str = "") -> dict:
    out = {"form": form, "title": title, "values": values}
    if label:
        out["label"] = label
    if why:
        out.update(disabled=True, note=why)
    return out


def cells(p, instance, model: dict) -> list:
    """A model's own cells under Details, Load and Delete."""
    ident = model["id"]
    values = {"name": instance.name, "model": ident}
    status = str(model.get("status") or "")
    details_ = _button("model-details", "Details", values, label=f"Details of {ident}")
    why = unable(p, instance, "load")
    if model.get("loaded") or status == "loading":
        loaded = _button("model-unload", "Unload…", values, why, f"Unload {ident}")
    else:
        busy = {"downloading": "its download is under way",
                "incomplete": "its download is not whole: Download… again goes on from it"}.get(status, "")
        loaded = _button("model-load", "Load…", values, why or busy, f"Load {ident}")
    remove = unable(p, instance, "delete")
    if not remove and model.get("can_remove") is not True:
        remove = ("in your GGUF folder: llama.cpp deletes only what it downloaded"
                  if model.get("source") == "models_dir" else "llama.cpp deletes only what it downloaded")
    if not remove and (model.get("loaded") or status == "loading"):
        remove = "unload it first"
    if not remove and status == "downloading":
        remove = "its download is under way: Cancel it first"
    return [details_, loaded, _button("model-delete", "Delete…", values, remove, f"Delete {ident}")]


def find_table(p, shown) -> dict:
    """The table "Find a model": its search box, the kept lookups and each one's Download… on the shown instance."""
    found = lookups(p)
    gpus = None
    if found and shown is not None and shown.where != "remote" and any(f.size for f in found):
        try:
            gpus = p._gpus()
        except Exception:  # noqa: BLE001 - a GPU that cannot be read: what fits is not known
            gpus = None
    cannot = "add an instance first" if shown is None else unable(p, shown, "download")
    rows = []
    for f in found:
        if not f.size:
            fits = "–"
        elif shown is not None and shown.where == "remote":
            fits = "the other machine's GPU is not read from here"
        else:
            fits = (gpu.fit(f.size, gpus) if gpus is not None else "") or "–"
        why = (f"nothing to download: {f.where} says {f.why}" if f.absent else GATED if f.gated else "") or cannot
        values = {"name": shown.name if shown is not None else "", "model": f.model}
        rows.append([{"text": f.model, "sub": f.where + (f" · {f.why}" if f.why else "")},
                     size_words(f.size) if f.size else "not said", f.licence or "not said",
                     {"text": fits, "sub": "llama.cpp lowers the context and the layers to fit"} if f.size else fits,
                     _button("model-pull", "Download…", values, why, f"Download {f.model}")])
    where = (f"on {shown.name}" if shown is not None else "on an instance")
    return {"title": "Find a model",
            "search": {"form": "model-find", "field": "model", "value": found[0].asked if found else "",
                       "placeholder": PLACEHOLDER},
            "note": (f"Looked up at {hub_host()} on your click, without a token: a model's files, size and licence are "
                     f"read there before anything is downloaded {where}. {FIT}"),
            "columns": [{"title": "Model"}, {"title": "Download", "num": True}, {"title": "Licence"},
                        {"title": "Fits this GPU"}, {"title": ""}],
            "rows": rows,
            "empty": (f"Type a model's name above, then Look up: its size and licence are read at {hub_host()} before "
                      "anything is downloaded."),
            "actions": [{"hook": "clear_found", "args": [], "title": "Clear", "note": "forgets these lookups"}]}


def _fixed(name: str, label: str, value: str = "") -> dict:
    return {"name": name, "label": label, "type": "fixed", **({"value": value} if value else {})}


def forms(p, shown) -> list[dict]:
    """The model forms: Find a model (a read), Download (a job), Delete, Load, Unload, Details (a read)."""
    name = shown.name if shown is not None else ""
    where = (f"into its models folder on {shown.host}" if shown is not None and shown.where == "remote"
             else "into its models folder")
    return [
        {"id": "model-find", "hook": "find", "title": "Find a model", "confirm": False, "submit": "Look up",
         "note": (f"user/repository at {hub_host()}, its quantization after a colon (Q4_K_M without one): its files, "
                  "size and licence are asked there, without a token"),
         "fields": [{"name": "model", "label": "Model", "type": "text", "required": True, "placeholder": PLACEHOLDER}]},
        {"id": "model-pull", "hook": "pull", "title": "Download a model", "job": True, "submit": "Review",
         "fields": [_fixed("name", "Instance", name), _fixed("model", "Model"),
                    {"name": ACCEPT_LICENCE, "type": "accept", "title": "Download it under its licence?",
                     "yes": "Accept and download",
                     "words": "{model} is downloaded on {name} under the licence named above. Cancel in the panel "
                              "stops it."}],
         "confirm": {"title": "Download {model} on {name}?", "yes": "Go on",
                     "words": (f"{{name}} downloads {{model}} from huggingface.co {where} and lists it once it "
                               "is whole. No Hugging Face token is given to it. Its size and licence are said "
                               "next, before anything is downloaded. Its progress shows in the panel below; "
                               "Cancel stops it (what came is kept for the next download), and it stops by "
                               "itself after 6 hours.")}},
        {"id": "model-delete", "hook": "delete_model", "title": "Delete a model",
         "fields": [_fixed("name", "Instance", name), _fixed("model", "Model")],
         "confirm": {"title": "Delete {model} from {name}?", "yes": "Delete", "danger": True,
                     "words": ("Its files go from {name}'s models folder (what another quantization of the same "
                               "repository still loads stays). Downloaded again, it comes back.")}},
        {"id": "model-load", "hook": "load", "title": "Load a model",
         "fields": [_fixed("name", "Instance", name), _fixed("model", "Model")],
         "confirm": {"title": "Load {model} on {name}?", "yes": "Load",
                     "words": ("It is loaded as a program of its own beside the router, on a port of this machine "
                               "alone that llama.cpp chooses; with the most models loaded already, the one used "
                               "longest ago is unloaded for it.")}},
        {"id": "model-unload", "hook": "unload", "title": "Unload a model",
         "fields": [_fixed("name", "Instance", name), _fixed("model", "Model")],
         "confirm": {"title": "Unload {model} from {name}?", "yes": "Unload",
                     "words": "Its memory is freed; asked again, it is loaded again."}},
        {"id": "model-details", "hook": "details", "title": "Details", "confirm": False,
         "fields": [_fixed("name", "Instance", name), _fixed("model", "Model")]},
    ]


def page(p, shown, seen) -> dict:
    """What the Models tab adds (the base's more()): the Find table and the model forms, drawn whether or not the
    shown instance answers."""
    return {"tables": [find_table(p, shown)], "forms": forms(p, shown)}


def facts(p) -> dict:
    """The Hub's card's facts, asking nothing: the lookups kept and the last one."""
    found = lookups(p)
    if not found:
        return {}
    last = found[0]
    said = last.why or f"{size_words(last.size or 0)}, {last.licence or 'no licence said'}"
    return {"Lookups kept": f"{len(found)} (an hour each)", "Last looked up": f"{last.model}: {said}"}
