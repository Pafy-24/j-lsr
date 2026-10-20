"""Render the Windows LSR build command for an external guarded build launcher."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

RUNTIME_DISTRIBUTIONS = (
    "filelock", "huggingface_hub", "numpy", "packaging", "PyYAML", "regex",
    "safetensors", "sentence-transformers", "tokenizers", "torch", "tqdm",
    "transformers",
)


def validate_runtime(python: Path) -> dict:
    """Reject CPU-only Torch without initializing CUDA or probing hardware."""
    probe = subprocess.run(
        [str(python.resolve()), "-I", "-B", "-c",
         "import json,torch; print(json.dumps({'torch':torch.__version__,"
         "'cuda':torch.version.cuda,'torchOrigin':torch.__file__}))"],
        check=True, capture_output=True, text=True,
    )
    runtime = json.loads(probe.stdout)
    if not runtime.get("cuda"):
        raise ValueError("LSR Windows executable must support CPU and CUDA; CPU-only Torch is not a release runtime")
    return runtime


def build_command(python: Path, workspace: Path, version: str, report: Path) -> list[str]:
    """Render a standalone build in the component catalog for the target platform."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", version):
        raise ValueError("version must be one ordinary directory name")
    workspace = workspace.resolve()
    components = workspace / "jester-engine/jester-components"
    source = components / "code/j-lsr"
    output = components / "built/j-lsr/windows-x64" / version
    return [
        str(python.resolve()), "-I", "-B", "-m", "nuitka",
        "--mode=standalone", "--msvc=latest", "--jobs=1",
        "--output-filename=LSR.exe", f"--output-dir={output}",
        f"--report={report.resolve()}",
        "--include-package=lsr", "--include-package-data=lsr",
        "--include-package=huggingface_hub",
        *(f"--include-distribution-metadata={name}" for name in RUNTIME_DISTRIBUTIONS),
        "--noinclude-data-files=torch/include/**",
        f"--user-plugin={source / 'tools/lsr_nuitka_plugin.py'}",
        str(source / "router.py"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--version", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"executed": False,
                     "runtime": validate_runtime(args.python) if args.check_runtime else None,
                     "argv": build_command(
        args.python, args.workspace, args.version, args.report,
    )}, indent=2))


if __name__ == "__main__":
    main()
