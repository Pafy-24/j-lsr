import pytest

nuitka = pytest.importorskip("nuitka", reason="build-tool-only policy tests")

from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.ModuleNames import ModuleName

from tools.lsr_nuitka_plugin import (
    BYTECODE_NAMESPACES,
    NuitkaPluginJesterLsrThirdPartyBytecode,
    REQUIRED_COMPLETE_PACKAGES,
    required_include_package_options,
    required_exclude_data_file_options,
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


def test_runtime_distribution_excludes_only_torch_development_headers() -> None:
    from fnmatch import fnmatchcase

    options = required_exclude_data_file_options()
    assert options == ("--noinclude-data-files=torch/include/**",)
    pattern = options[0].split("=", 1)[1]
    assert fnmatchcase("torch/include/torch/csrc/api/include/torch/torch.h", pattern)
    assert not fnmatchcase("torch/lib/torch_cpu.dll", pattern)
    assert not fnmatchcase("lsr/resources/component-manifest.json", pattern)


def test_lsr_command_uses_component_output_and_preserves_runtime_metadata(tmp_path) -> None:
    from tools.lsr_nuitka_build import build_command

    command = build_command(tmp_path / "python.exe", tmp_path, "release-01", tmp_path / "report.xml")
    assert f"--output-dir={tmp_path / 'jester-engine/jester-components/built/j-lsr/windows-x64/release-01'}" in command
    source = tmp_path / "jester-engine/jester-components/code/j-lsr"
    assert command[-1] == str(source / "router.py")
    assert f"--user-plugin={source / 'tools/lsr_nuitka_plugin.py'}" in command
    assert required_exclude_data_file_options()[0] in command
    assert "--include-distribution-metadata=torch" in command
    assert "--include-distribution-metadata=filelock" in command
    assert "--include-distribution-metadata=tokenizers" in command
    assert "--include-package=huggingface_hub" in command
    assert not any("built/engine" in arg.replace("\\", "/") for arg in command)
    with pytest.raises(ValueError):
        build_command(tmp_path / "python.exe", tmp_path, "../escape", tmp_path / "report.xml")


def test_release_runtime_rejects_cpu_only_torch_without_hardware_probe(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from tools import lsr_nuitka_build

    captured = []
    def run(argv, **kwargs):
        captured.append(argv)
        return SimpleNamespace(stdout=json.dumps({"torch": "2.11.0+cpu", "cuda": None}))
    monkeypatch.setattr(lsr_nuitka_build.subprocess, "run", run)
    with pytest.raises(ValueError, match="CPU-only"):
        lsr_nuitka_build.validate_runtime(tmp_path / "python.exe")
    assert "is_available" not in captured[0][-1]
    monkeypatch.setattr(lsr_nuitka_build.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({"torch": "2.11.0+cu128", "cuda": "12.8"})))
    assert lsr_nuitka_build.validate_runtime(tmp_path / "python.exe")["cuda"] == "12.8"
