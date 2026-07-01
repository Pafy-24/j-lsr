from __future__ import annotations

from lsr.args import parse_args
from lsr.commands import (
    RouterCLI,
    main,
    print_route_table,
    print_signature_summary,
    route_batch,
    route_prompt,
    sign_dataset,
    sign_raw_text,
)
from lsr.config import *
from lsr.console import LazyConsole, console, error_console
from lsr.dataset import *
from lsr import dependencies as _dependencies
from lsr.dependencies import (
    ensure_faiss,
    ensure_numpy,
    get_kmeans_class,
    get_sentence_transformer_class,
    make_table,
)
from lsr.der import *
from lsr.devices import *
from lsr.embedding import *
from lsr.errors import RouterError
from lsr.io_utils import *
from lsr.models import *
from lsr.routing import *
from lsr.signatures import *


def __getattr__(name: str):
    if name in {"np", "SentenceTransformer", "MiniBatchKMeans", "faiss"}:
        return getattr(_dependencies, name)
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
