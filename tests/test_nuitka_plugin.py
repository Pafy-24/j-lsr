import pytest

nuitka = pytest.importorskip("nuitka", reason="build-tool-only policy tests")

from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.ModuleNames import ModuleName

from tools.lsr_nuitka_plugin import (
    BYTECODE_NAMESPACES,
    NuitkaPluginJesterLsrThirdPartyBytecode,
    REQUIRED_COMPLETE_PACKAGES,
    required_include_package_options,
)


def test_plugin_uses_supported_nuitka_api_and_exact_allowlist() -> None:
    assert issubclass(NuitkaPluginJesterLsrThirdPartyBytecode, NuitkaPluginBase)
    assert BYTECODE_NAMESPACES == {
        "transformers", "torch", "sympy", "scipy", "pygments", "sklearn",
        "sentence_transformers", "numpy", "huggingface_hub", "pydantic",
    }


def test_allowlisted_namespaces_are_bytecode_with_prefix_boundaries() -> None:
    plugin = NuitkaPluginJesterLsrThirdPartyBytecode()
    for namespace in BYTECODE_NAMESPACES:
        assert plugin.decideCompilation(ModuleName(namespace)) == "bytecode"
        assert plugin.decideCompilation(ModuleName(f"{namespace}.child.module")) == "bytecode"
        assert plugin.decideCompilation(ModuleName(f"{namespace}_lookalike")) is None


def test_first_party_is_explicitly_compiled_and_unrelated_modules_are_unchanged() -> None:
    plugin = NuitkaPluginJesterLsrThirdPartyBytecode()
    compiled_names = (
        "__main__", "lsr", "lsr.router", "jester", "jester_core",
        "jester_rm.enforcement", "jester_anything.component", "mixed_inference",
        "mixed_inference.runtime", "trainer", "engine", "resource_manager", "rm",
    )
    assert all(plugin.decideCompilation(ModuleName(name)) == "compiled" for name in compiled_names)
    assert all(plugin.decideCompilation(ModuleName(name)) is None for name in ("json", "faiss"))


def test_lazy_packages_are_explicitly_included_but_remain_bytecode() -> None:
    plugin = NuitkaPluginJesterLsrThirdPartyBytecode()
    assert REQUIRED_COMPLETE_PACKAGES == ("huggingface_hub",)
    assert required_include_package_options() == ("--include-package=huggingface_hub",)
    assert all(
        plugin.decideCompilation(ModuleName(name)) == "bytecode"
        for name in REQUIRED_COMPLETE_PACKAGES
    )
    assert plugin.decideCompilation(ModuleName("lsr")) == "compiled"
    assert plugin.decideCompilation(ModuleName("lsr.embedding")) == "compiled"
