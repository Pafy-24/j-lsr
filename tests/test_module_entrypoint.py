from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


PACKAGE = Path(__file__).parents[1]
SOURCE = PACKAGE / "src"


def _module(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE)
    return subprocess.run(
        [sys.executable, "-B", "-m", "lsr", *arguments],
        cwd=PACKAGE,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_module_help_uses_real_cli_without_starting_runtime() -> None:
    result = _module("--help")
    assert result.returncode == 0
    assert "LSR (Layered Signatures Router) command line interface." in result.stdout
    for command in ("serve", "loadsig", "unloadsig", "sign", "rank", "strategy"):
        assert command in result.stdout


def test_module_invalid_argument_preserves_argparse_exit_two() -> None:
    result = _module("--definitely-invalid")
    assert result.returncode == 2
    assert "unrecognized arguments: --definitely-invalid" in result.stderr


def test_module_delegates_to_the_console_script_main() -> None:
    pyproject = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    assert 'jester-lsr = "lsr.commands:main"' in pyproject

    tree = ast.parse((SOURCE / "lsr" / "__main__.py").read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "commands" and any(alias.name == "main" for alias in node.names) for node in imports)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == "exit"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "main"
        for node in calls
    )
