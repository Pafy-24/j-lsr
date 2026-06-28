from __future__ import annotations

from typing import Any


class LazyConsole:
    def __init__(self, stderr: bool = False):
        self.stderr = stderr
        self._console = None

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self._console is None:
            from rich.console import Console

            self._console = Console(stderr=self.stderr)
        self._console.print(*args, **kwargs)


console = LazyConsole()
error_console = LazyConsole(stderr=True)
