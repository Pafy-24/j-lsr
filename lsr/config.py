from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULTS_PATH = Path(__file__).with_name("defaults.json")


def load_defaults(path: Path = DEFAULTS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_config_value(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        value = value[key]
    return value


CONFIG = load_defaults()

DEFAULT_MODEL = str(get_config_value(CONFIG, "model.default"))
DEFAULT_BENCHMARK_MODELS = ",".join(get_config_value(CONFIG, "model.benchmark_candidates"))
DEFAULT_SIGNATURES = str(get_config_value(CONFIG, "paths.signatures"))
DEFAULT_BENCHMARK_INPUT = str(get_config_value(CONFIG, "paths.benchmark_input"))
DEFAULT_BENCHMARK_DATASET = str(get_config_value(CONFIG, "paths.benchmark_dataset"))
DEFAULT_SERVER_HOST = str(get_config_value(CONFIG, "server.host"))
DEFAULT_SERVER_PORT = int(get_config_value(CONFIG, "server.port"))
DEFAULT_SERVER_URL = str(get_config_value(CONFIG, "server.url"))

DEFAULT_MAX_VECTORS = int(get_config_value(CONFIG, "signature.max_vectors"))
SIGNATURE_VERSION = int(get_config_value(CONFIG, "signature.version"))
TOP_LAYER_CENTROIDS = int(get_config_value(CONFIG, "signature.top_layer_centroids"))
LAYER_FANOUT = int(get_config_value(CONFIG, "signature.layer_fanout"))

DEFAULT_TOP_K = int(get_config_value(CONFIG, "routing.top_k"))
SEARCH_L1_TOP = int(get_config_value(CONFIG, "routing.search_l1_top"))
SEARCH_NEXT_TOP = int(get_config_value(CONFIG, "routing.search_next_top"))
SEARCH_EXACT_MAX = int(get_config_value(CONFIG, "routing.search_exact_max"))
LAYER_THRESHOLDS = get_config_value(CONFIG, "routing.layer_thresholds")
LAYER_DESCEND_THRESHOLD = float(LAYER_THRESHOLDS["default"]["pass"])
LAYER_ACCEPT_THRESHOLD = float(LAYER_THRESHOLDS["default"]["accept"])
GLOBAL_L1_TOP = int(get_config_value(CONFIG, "routing.global_l1_top"))
DEFAULT_MEMORY_LIMIT_MB = float(get_config_value(CONFIG, "routing.memory_limit_mb"))
DEFAULT_CACHE_POLICY = str(get_config_value(CONFIG, "routing.cache_policy"))
CONFIDENCE_DECAY = get_config_value(CONFIG, "routing.confidence_decay")
CONFIDENCE_DECAY_ENABLED = bool(CONFIDENCE_DECAY["enabled"])
CONFIDENCE_DECAY_MAX_DELTA = float(CONFIDENCE_DECAY["max_delta"])
CONFIDENCE_DECAY_PENALTY = float(CONFIDENCE_DECAY["penalty"])
SCORING = get_config_value(CONFIG, "routing.scoring")
SCORING_ANCHOR_WEIGHT = float(SCORING["anchor_weight"])
SCORING_TOP_AVERAGE_WEIGHT = float(SCORING["top_average_weight"])
SCORING_LAYER_WEIGHT = float(SCORING["layer_weight"])
SCORING_LEXICAL_WEIGHT = float(SCORING["lexical_weight"])
SCORING_GENERIC_PENALTY = float(SCORING["generic_penalty"])
SCORING_REPETITION_PENALTY = float(SCORING["repetition_penalty"])
SEPARATION = SCORING["separation"]
SEPARATION_ENABLED = bool(SEPARATION["enabled"])
SEPARATION_MIN_TOP_DELTA = float(SEPARATION["min_top_delta"])
SEPARATION_RATIO_THRESHOLD = float(SEPARATION["ratio_threshold"])
SEPARATION_PENALTY = float(SEPARATION["penalty"])
SEPARATION_REFUSE_DELTA = float(SEPARATION["refuse_delta"])
LARGE_INPUT = get_config_value(CONFIG, "routing.large_input")
DEFAULT_LARGE_INPUT_MODE = str(LARGE_INPUT["mode"])
LARGE_INPUT_THRESHOLD_CHARS = int(LARGE_INPUT["threshold_chars"])
LARGE_INPUT_MAX_DIGEST_CHARS = int(LARGE_INPUT["max_digest_chars"])
LARGE_INPUT_CHUNK_CHARS = int(LARGE_INPUT["chunk_chars"])
LARGE_INPUT_FIRST_CHUNKS = int(LARGE_INPUT["first_chunks"])
LARGE_INPUT_LAST_CHUNKS = int(LARGE_INPUT["last_chunks"])
LARGE_INPUT_KEYWORD_CHUNKS = int(LARGE_INPUT["keyword_chunks"])

INDEX_HNSW_THRESHOLD = int(get_config_value(CONFIG, "index.hnsw_threshold"))
INDEX_HNSW_M = int(get_config_value(CONFIG, "index.hnsw_m"))
INDEX_HNSW_EF_CONSTRUCTION = int(get_config_value(CONFIG, "index.hnsw_ef_construction"))
INDEX_HNSW_EF_SEARCH = int(get_config_value(CONFIG, "index.hnsw_ef_search"))

DATASET_SCHEMA = str(get_config_value(CONFIG, "dataset.schema"))
WEB_CONTENT_CHARS = int(get_config_value(CONFIG, "dataset.web_content_chars"))
WEB_TITLE_CHARS = int(get_config_value(CONFIG, "dataset.web_title_chars"))
CRYPTO_QUESTION_CHARS = int(get_config_value(CONFIG, "dataset.crypto_question_chars"))
DEFAULT_ASCII_TABLE = str(get_config_value(CONFIG, "output.ascii_table"))


def layer_thresholds(level: int) -> dict[str, float]:
    raw = LAYER_THRESHOLDS.get(str(level), LAYER_THRESHOLDS["default"])
    return {
        "pass": float(raw["pass"]),
        "accept": float(raw["accept"]),
    }


def all_layer_thresholds() -> dict[str, dict[str, float]]:
    return {
        str(level): {"pass": float(values["pass"]), "accept": float(values["accept"])}
        for level, values in LAYER_THRESHOLDS.items()
    }
