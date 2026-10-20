"""Nuitka build policy for freezing reviewed third-party modules as bytecode."""

from nuitka.plugins.PluginBase import NuitkaPluginBase


BYTECODE_NAMESPACES = frozenset(
    {
        "huggingface_hub",
        "numpy",
        "pydantic",
        "pygments",
        "scipy",
        "sentence_transformers",
        "sklearn",
        "sympy",
        "torch",
        "transformers",
    }
)

# These packages resolve modules from runtime import maps. Their complete
# namespaces must be requested explicitly from Nuitka's standalone builder.
REQUIRED_COMPLETE_PACKAGES = ("huggingface_hub",)

# LSR runs eager inference; it does not build C++/CUDA extensions at runtime.
# Keep torch's DLLs, but omit the development headers automatically collected
# by Nuitka's torch data-file rule.
EXCLUDED_DATA_FILES = ("torch/include/**",)


def required_include_package_options():
    return tuple(f"--include-package={name}" for name in REQUIRED_COMPLETE_PACKAGES)


def required_exclude_data_file_options():
    return tuple(f"--noinclude-data-files={pattern}" for pattern in EXCLUDED_DATA_FILES)


FIRST_PARTY_NAMESPACES = frozenset(
    {
        "__main__",
        "engine",
        "lsr",
        "mixed_inference",
        "resource_manager",
        "rm",
        "trainer",
    }
)


class NuitkaPluginJesterLsrThirdPartyBytecode(NuitkaPluginBase):
    """Keep Jester code compiled while freezing measured third-party families."""

    plugin_name = __file__

    def decideCompilation(self, module_name):
        top_level = str(module_name.getTopLevelPackageName())
        if (
            top_level in FIRST_PARTY_NAMESPACES
            or top_level == "jester"
            or top_level.startswith("jester_")
        ):
            return "compiled"
        if top_level in BYTECODE_NAMESPACES:
            return "bytecode"
        return None
