from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.ModuleNames import ModuleName

from tools.lsr_nuitka_plugin import (
    BYTECODE_NAMESPACES,
    NuitkaPluginJesterLsrThirdPartyBytecode,
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


def test_first_party_and_unrelated_modules_are_never_demoted() -> None:
    plugin = NuitkaPluginJesterLsrThirdPartyBytecode()
    compiled_names = (
        "__main__", "lsr", "lsr.router", "jester", "jester_core",
        "jester_rm.enforcement", "jester_anything.component", "mixed_inference",
        "mixed_inference.runtime", "trainer", "engine", "resource_manager", "rm",
        "json", "faiss",
    )
    assert all(plugin.decideCompilation(ModuleName(name)) is None for name in compiled_names)
