"""The plugin from the command line (the platform's page asks the same hooks):

    python -m plugin_platform_llamacpp status
    python -m plugin_platform_llamacpp serving
    python -m plugin_platform_llamacpp models NAME
    python -m plugin_platform_llamacpp launch NAME
    python -m plugin_platform_llamacpp find MODEL
    python -m plugin_platform_llamacpp install [--accept-licence]

status (the default) prints the plugin's status as JSON (files and the socket table only: nothing is asked of a
server); serving what it tells the other AI plugins; models asks an instance for its models now; launch prints the
lines that start an instance's server in a terminal of yours (PowerShell's on Windows, sh's on Linux), its key the
placeholder you replace, never the key; find looks a model up at the Hugging Face Hub (its files, size and licence).
install installs the pinned llama.cpp with the build kept (A-critique M5: this command line has the install the
page has): without --accept-licence it prints what it would download and its licence, and stops. Its instances are
read through this process's environment, the login of a command line."""
from __future__ import annotations

import json
import sys

from liminal_libs.instances import Refusal
from liminal_libs.machines import children

from . import MACHINE

USAGE = ("python -m plugin_platform_llamacpp status | serving | models NAME | launch NAME | find MODEL | "
         "install [--accept-licence]")


def said(answer: dict) -> int:
    print(answer.get("said", ""))
    for line in answer.get("lines") or []:
        print(f"  {line}")
    return 0 if answer.get("ok", True) else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    what, rest = (argv[0], argv[1:]) if argv else ("status", [])
    m = MACHINE
    old = m.too_old()
    if old:  # nothing is read or run on an older library
        print(old, file=sys.stderr)
        return 1
    try:
        if what == "status" and not rest:
            print(json.dumps(m.status(), indent=1, default=str))
            return 0
        if what == "serving" and not rest:
            print(json.dumps(m.serving(), indent=1))
            return 0
        if what == "models" and len(rest) == 1:
            got = m.models_of(rest[0])
            print(got["said"])
            for model in got["models"]:
                context = f", context {model['context']:,}" if model.get("context") else ""
                print(f"  {model['id']}{context}")
            return 0 if got["ok"] else 1
        if what == "launch" and len(rest) == 1:
            instance = m.instance(rest[0])
            for line in children.shown(m.prepared(instance), m.system):
                print(line)
            return 0
        if what == "find" and len(rest) == 1:
            return said(m.find(rest[0]))
        if what == "install" and set(rest) <= {"--accept-licence"}:
            return said(m.install(accept_licence="--accept-licence" in rest, log=print))
    except Refusal as e:
        more = " (--accept-licence accepts it)" if getattr(e, "accept", None) == "accept_licence" else ""
        print(f"refused: {e}{more}", file=sys.stderr)
        return 1
    except (ValueError, LookupError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
