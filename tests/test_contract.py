"""The plugin conforms to the class contract (plugin API 2.2: liminal_libs.sources.conformance, read from the plugin
alone, nothing run): a machine plugin of the platform on the model server base, drawn under the AI heading's
Inference group, its icon llama.cpp's mark as liminal-libs 1.62.0 draws it, its hooks the base's and its own, its
jobs and what may be cancelled, what it says of the interface it shares (READ_HOOKS, KEEPS), its version and floor
as pyproject says them, and one run state across the registry's copies."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import liminal_libs
import plugin_platform_llamacpp as module
from liminal_libs.apps.tabs import Tab
from liminal_libs.machines.modelserver import SERVING, ModelServer
from liminal_libs.sources.conformance import assert_module_conforms
from liminal_libs.ui import STATIC

PYPROJECT = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
OWN = ("find", "clear_found", "pull", "delete_model", "load", "unload", "details", "build")


def test_the_plugin_conforms_to_the_class_contract():
    found = assert_module_conforms(module)
    assert [p.name for p in found] == ["llamacpp"] and all(p.api == "2.2" and not p.is_record for p in found)
    p = found[0]
    assert (p.SYSTEM, p.KIND, p.SECTION, p.GROUP, p.title, p.order) == ("platform", "machine", "ai", "inference",
                                                                         "llama.cpp", 510)
    assert isinstance(p, ModelServer) and p.interface()["apis"] == [a.name for a in p.apis]
    assert p.TABS == (Tab("models", "Models", "models", 15.0),)
    assert (p.DEFAULT_PORT, p.NATIVE, p.KEYED, p.READY_S) == (9931, "llamacpp", True, 60.0)
    assert p.MODEL_COLUMNS == ("Details", "Load", "Delete")


def test_its_hooks_are_the_bases_and_its_own():
    p = module.MACHINE
    hooks = set(p.interface()["hooks"])
    assert hooks == {"present", "status", "refresh", "install", "uninstall", "choose", "add_instance",
                     "change_instance", "remove_instance", "set_key", "forget_key", "share", "start", "stop", "boot",
                     "log_of", "models", "try_it", SERVING, "models_of", *OWN}
    assert "instances" not in hooks and p.instances() == ()  # Docker's hook: a server there would be an engine
    assert p.HOOKS[:len(ModelServer.HOOKS)] == ModelServer.HOOKS and p.HOOKS[len(ModelServer.HOOKS):] == OWN
    assert p.JOB_HOOKS == ("install", "pull") and p.CANCEL_HOOKS == ("install", "pull")


def test_it_says_what_it_shares_its_reads_and_what_it_keeps():
    """The Internal API tab reads these by name (C's internal.declared): its own reads beside the bases', none a job
    nor a hook that takes an accept_*; and what it keeps through the login beside its keys."""
    p = module.MACHINE
    # no reads of its own since 0.1.2: its lookup at the Hub (find), the lookups forgotten (clear_found) and a model's
    # details asked of an instance are its page's alone
    assert "READ_HOOKS" not in type(p).__dict__ and {"find", "clear_found", "details"} <= set(p.PAGE_ONLY_HOOKS)
    assert not {"find", "clear_found", "details"} & set(p.READ_HOOKS) and set(p.READ_HOOKS) <= set(p.HOOKS)
    assert not set(p.READ_HOOKS) & set(p.JOB_HOOKS)
    for hook in p.READ_HOOKS:
        assert not [a for a in inspect.signature(getattr(p, hook)).parameters if a.startswith("accept_")]
    assert [v for v, _ in p.KEEPS] == ["LIMINAL_AI_LLAMACPP_INSTANCES", "LIMINAL_AI_LLAMACPP_<INSTANCE>_KEY",
                                       "LIMINAL_AI_LLAMACPP_BUILD"]
    assert all(isinstance(v, str) and isinstance(a, str) and a for v, a in p.KEEPS)
    assert p.key_variable("default") == "LIMINAL_AI_LLAMACPP_DEFAULT_KEY" == p.KEEPS[1][0].replace("<INSTANCE>",
                                                                                                    "DEFAULT")


def test_each_hook_of_its_own_says_what_it_does():
    """The first sentence of a hook's docstring is its line on the Internal API tab."""
    for hook in (*OWN, "install"):
        doc = inspect.getdoc(getattr(module.LlamaCppPlugin, hook))
        assert doc and doc.split(".")[0].strip(), hook


def test_its_icon_is_its_makers_mark_drawn_by_the_library():
    drawn = set(re.findall(r"^\s*([a-z]+):\s*'", (STATIC / "icons.js").read_text(encoding="utf-8"), re.M))
    assert module.MACHINE.ICON == "llamacpp" and "llamacpp" in drawn


def test_its_version_and_floor_are_pyprojects():
    assert re.search(r'^version = "([^"]+)"', PYPROJECT, re.M).group(1) == module.__version__ == "0.1.1"
    assert re.search(r'"liminal-libs>=([^"]+)"', PYPROJECT).group(1) == module.LIBS_FLOOR == "1.62.0"
    assert module.MACHINE.LIBS_FLOOR == module.LIBS_FLOOR and module.MACHINE.too_old() == ""
    assert 'llamacpp = "plugin_platform_llamacpp"' in PYPROJECT


def test_an_older_library_is_said_and_every_hook_refuses(monkeypatch):
    monkeypatch.setattr(liminal_libs, "__version__", "1.61.2")
    p = module.LlamaCppPlugin()
    assert "needs liminal-libs 1.62.0 or newer" in p.too_old() and not p.present()
    out = p.status()
    assert out["valid"] is False and "1.62.0" in out["why"]
    try:
        p.ask("find", "ggml-org/gemma-3-4b-it-GGUF")
    except ValueError as e:
        assert "1.62.0" in str(e)
    else:
        raise AssertionError("a hook was answered on an older library")


def test_a_copy_the_registry_hands_out_shares_what_the_plugin_keeps():
    p = module.LlamaCppPlugin(arch="AMD64")
    copy = p.with_(origin="test")
    assert copy is not p and copy._rt is p._rt and copy._rt.found is p._rt.found and copy.arch == "AMD64"


def test_each_call_on_its_cards_names_one_of_its_cards_own_parts():
    for api in module.MACHINE.apis:
        areas = {a.name for a in api.areas}
        assert all(c.area in areas for c in api.calls), (api.name, [c.name for c in api.calls if c.area not in areas])
