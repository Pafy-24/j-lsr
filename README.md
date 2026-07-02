# LSR

Layered Signatures Router (LSR) is a prototype CLI for deriving routing embeddings from faucet training datasets and routing prompts to the closest faucet.

The project should run with the VS Code interpreter:

```powershell
C:\Users\xxsho\.conda\envs\licenta\python.exe
```

From a terminal:

```powershell
conda activate licenta
cd E:\Licenta\Proiect\LSR
```

## Project Layout

- `router.py` - thin CLI compatibility entry point
- `lsr\args.py` - CLI argument definitions
- `lsr\defaults.json` - central defaults for models, paths, thresholds, index policy, and layer sizing
- `lsr\config.py` - Python accessors for `defaults.json`
- `lsr\commands.py` - command orchestration for signing and routing
- `lsr\dataset.py` - dataset-to-route-text extraction
- `lsr\dataset_schema.py` - `pretrain/core/finetune` JSONL loader/adapters
- `lsr\devices.py` - processing-unit discovery and selection
- `lsr\embedding.py` - model loading and embedding helpers
- `lsr\input_digest.py` - deterministic digest mode for very large inputs
- `lsr\registry.py` - server-side signature registry and stale-file refresh
- `lsr\signatures.py` - signature building, loading, saving, and DER codec
- `lsr\routing.py` - layered routing, lazy loading policy, and hierarchical search
- `requirements.txt` - Python dependencies
- `..\facets\Crypto\datasets\train\crypto_dataset.json` - full raw Crypto training dataset
- `..\facets\Web\datasets\train\WebMainBench_545.jsonl` - WebMainBench calibrated subset for Web routing tests
- `..\facets\signatures\crypto.sig` - ASN.1 DER hierarchical routing signature for Crypto
- `..\facets\signatures\web.sig` - ASN.1 DER hierarchical routing signature for Web

There is no `signatures.json` in the active flow. A faucet signature is stored as:

```text
[faucet].sig
```

The `.sig` file is binary ASN.1 DER. It contains:

```text
FacetSignature ::= SEQUENCE {
  version     INTEGER,
  metadata    Metadata,
  layers      SEQUENCE OF Layer
}

Metadata ::= SEQUENCE {
  modelName      OCTET STRING,
  modelHash   OCTET STRING,
  dimension   INTEGER,
  normalize      INTEGER,
  distanceMetric OCTET STRING
}

Layer ::= SEQUENCE {
  level       INTEGER,
  vectors     OCTET STRING,
  links       SEQUENCE OF Link OPTIONAL
}
```

Vectors are stored as contiguous little-endian `float32` values in `OCTET STRING`. Links are numeric parent-to-child IDs, so no separate `.idx` file is needed.

## Crypto Dataset

The current Crypto dataset is the full raw downloaded dataset from:

```text
natnitaract/exams-basic-and-quantum-cryptography-and-security-latex
```

It contains 104 exam-level records and 347 questions. The dataset keeps training fields such as `solution`; the router ignores solution/output fields when building routing signatures.

## Web Dataset

The Web facet uses the calibrated `WebMainBench_545.jsonl` subset from `opendatalab/WebMainBench` on Hugging Face. WebMainBench is a benchmark for web main-content extraction, and the local copy is stored at:

```text
..\facets\Web\datasets\train\WebMainBench_545.jsonl
```

LSR extracts routing text from `url`, `groundtruth_content`, `convert_main_content`, `main_html`, and `meta` fields such as language, style, difficulty, tables, code, and equations.

Generate the Web signature:

```powershell
python router.py sign `
  -in ..\facets\Web\datasets\train\WebMainBench_545.jsonl `
  --facet web `
  --signatures ..\facets\signatures `
  --max-vectors 2000 `
  --debug
```

With both `crypto.sig` and `web.sig` present, routing can compare at least two facets locally.

## Generate Crypto Signature

LSR reads the full dataset and automatically extracts routing text from useful prompt-side fields:

- `topic`
- `describe`
- `questions[].question`
- `tags`
- `questions[].tags`
- `questions[].prerequisite_knowledge`
- `questions[].difficulty`
- `questions[].problem_type`

It does not embed `solution` as answer text. It may extract short technical keywords from `solution` and `question`, such as `ECDSA`, `nonce`, `POVM`, or `private key`, because those improve routing without training on the full answer.

```powershell
python router.py sign `
  -in ..\facets\Crypto\datasets\train\crypto_dataset.json `
  --signatures ..\facets\signatures `
  --debug
```

Because the path is under `facets\Crypto`, the facet name is inferred as `crypto`. You can also pass it explicitly:

```powershell
python router.py sign `
  -in ..\facets\Crypto\datasets\train\crypto_dataset.json `
  --facet crypto `
  --signatures ..\facets\signatures
```

## Layer Rule

Layer sizing is intentionally not exposed as normal CLI tuning. LSR uses a fixed geometric rule:

```text
L1 = min(128, anchor_count)
L(n+1) = min(anchor_count, L(n) * 10)
stop when L(n) == anchor_count
```

That gives a stable top-level comparison cost while allowing larger facets to grow deeper:

```text
977 anchors    -> L1=128, L2=977
10,000 anchors -> L1=128, L2=1280, L3=10000
100,000 anchors -> L1=128, L2=1280, L3=12800, L4=100000
```

The last layer is always the anchor-vector layer. Intermediate layers are centroids linked numerically to the next layer.

Build order is bottom-up:

```text
1. generate/cap the maximum anchor-vector layer
2. cluster that layer to create the previous/coarser layer
3. repeat until L1 has at most 128 centroids
```

For the current Crypto dataset, the router extracts 977 focused routing texts and stores:

```text
L1 = 128 macro-centroids
L2 = 977 anchor vectors
```

The signature is written to:

```text
..\facets\signatures\crypto.sig
```

## Loading And Descend Policy

LSR loads signature metadata first. For v3 `.sig` files, layers are lazy-loaded from the DER file only when the router needs them.

Default runtime behavior:

```text
1. load metadata for all facets
2. load and keep global L1 warm
3. score every facet on L1
4. descend only facets whose current layer decision is `passed`
5. load deeper layers on demand
6. unload L2+ after the request; keep only L1 warm
```

For memory-constrained runs, pass `--memory-limit-mb N`. In that mode, L1 is processed in chunks: load what fits, compare, unload, then continue with the next chunk. After the route finishes, all loaded layers are released.

Layer decisions use per-layer thresholds from `lsr\defaults.json`:

```text
score >= accept threshold -> accepted
score >= pass threshold   -> passed; continue to next layer
score < pass threshold    -> refused
```

Default config shape:

```json
"layer_thresholds": {
  "default": { "pass": 0.18, "accept": 0.82 },
  "1": { "pass": 0.18, "accept": 0.82 },
  "2": { "pass": 0.20, "accept": 0.84 },
  "3": { "pass": 0.22, "accept": 0.86 }
}
```

Routing depth is no longer decided by top-k or a fixed number of accepted facets. `--top-k` only limits how many sorted results are returned after all threshold decisions have been evaluated.

For diagnostics, `router.py rank --full-depth ...` evaluates all layers even when an earlier layer already returned `accepted` or `refused`. JSON and table output include each layer score, threshold, decision, confidence decay, and entropy.

Inside a selected faucet, the router expands the best linked clusters until the anchor layer, then performs exact search only on the selected anchor pool.

## Route Prompts

Route against all `.sig` files in a signature directory:

```powershell
python router.py rank "How can I recover an ECDSA private key if the nonce k is reused?"
```

`rank` uses the running server by default, so it ranks against the signatures already loaded by `serve` or `loadsig`.

For an ad-hoc local rank against one specific signature, bypass the server explicitly:

```powershell
python router.py rank "How do I verify a DSA signature?" `
  --local `
  --signatures ..\facets\signatures\crypto.sig
```

Input file:

```powershell
python router.py rank -in prompt.txt `
  --top-k 5
```

Batch route many prompts in one process:

```powershell
python router.py rank --batch --json `
  -in ..\router_test_crypto_web_100.json `
  --top-k 3
```

Batch mode sends one HTTP request to the running server, which embeds all prompts together and emits JSON. This is the preferred mode for evaluation and bulk routing.

Very large inputs are routed through a deterministic digest by default when they exceed `routing.large_input.threshold_chars` in `lsr\defaults.json`. The digest uses headings, first chunks, last chunks, and keyword-heavy chunks; it does not call an LLM summarizer.

```powershell
python router.py rank -in large_document.txt --large-input auto --json
python router.py rank -in large_document.txt --large-input force --json
python router.py rank -in large_document.txt --large-input off --json
```

Rank JSON includes `inputMode`, `inputChars`, `routingTextChars`, and `chunksUsed`. Use `--ascii-table` if the terminal renders Unicode table borders incorrectly.

By default, model loading is local-only:

```text
SentenceTransformer(..., local_files_only=True)
```

Use `--online` only when the model is not already cached and the machine is allowed to download it.

## Serve Mode

For interactive use or integrations, LSR can keep the model and signatures loaded in a local HTTP server:

```powershell
python router.py serve `
  --signatures ..\facets\signatures `
  --host 127.0.0.1 `
  --port 8765 `
  --PU 1
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

Route one prompt:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8765/rank `
  -ContentType application/json `
  -Body '{"input":"How can I recover an ECDSA private key if the nonce k is reused?","top_k":1}'
```

The same server can be used through the CLI:

```powershell
python router.py rank `
  "How can I recover an ECDSA private key if the nonce k is reused?" `
  --server http://127.0.0.1:8765 `
  --top-k 1
```

Batch route prompts:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8765/batch `
  -ContentType application/json `
  -Body '{"inputs":["How do I verify an RSA signature?","Configure nginx reverse proxy for FastAPI"],"top_k":2}'
```

Supported endpoints:

- `GET /health` - returns model, loading policy, cache policy, thresholds, signature path, loaded facets, loaded layers, loaded vector bytes, signature files, stale signatures, and default `top_k`
- `POST /rank` - accepts `{"input": "..."}`; `prompt`, `text`, `full_depth`, and `large_input` are also accepted
- `POST /route` - compatibility alias for `/rank`
- `POST /batch` - accepts `{"inputs": ["..."]}`; `prompts`, `texts`, `full_depth`, and `large_input` are also accepted
- `POST /loadsig` - accepts `{"signatures": "path/to/file-or-directory"}`
- `POST /unloadsig` - accepts `{"facet": "crypto"}`, `{"facets": ["crypto"]}`, or `{"all": true}`

Before each rank request, the server refreshes the signature registry. Deleted `.sig` files are unloaded, modified files are reloaded, and new `.sig` files are loaded automatically when the active signature path is a directory.

Signature loading can be controlled through CLI commands while the server is running:

```powershell
python router.py serve --no-load --host 127.0.0.1 --port 8765 --PU 1
python router.py loadsig --signatures ..\facets\signatures --server http://127.0.0.1:8765
python router.py unloadsig --facet crypto --server http://127.0.0.1:8765
python router.py unloadsig --all --server http://127.0.0.1:8765
```

Signing remains a CLI operation. When `serve` watches a signature directory, updated `.sig` files are picked up automatically on the next rank request.

The server uses the same local-only model loading rule as CLI routing. Use `--online` only when the model is not already cached locally.

First run on a fresh environment:

```powershell
python router.py serve --signatures ..\facets\signatures --online
```

After the model is cached, omit `--online`.

## Processing Units

List processing units visible to the router:

```powershell
python router.py --PUs
```

Select a processing unit for signing or routing:

```powershell
python router.py sign `
  -in ..\facets\Crypto\datasets\train\crypto_dataset.json `
  --signatures ..\facets\signatures `
  --PU 1
```

```powershell
python router.py serve `
  --signatures ..\facets\signatures `
  --PU 1

python router.py rank "How do I verify an ECDSA signature?"
```

`--PUs` always includes CPU index `0`. It also reports GPUs/NPUs detected by Windows even when the current Python backend cannot use them. The `Usable` column tells whether `--PU <index>` can actually be used for embeddings.

On this machine, the `licenta` environment uses Python 3.11 and `torch 2.11.0+cu128`. The NVIDIA RTX GPU is usable through `torch.cuda` and appears as `--PU 1`. Windows also detects the AMD NPU, but it remains unusable for the current SentenceTransformer/PyTorch backend.

## Example Result

```text
Rank | Facet  | Score  | Best Vector | Cluster | Exact | Mode | Best Match
1    | crypto | 0.6042 | 0.6276      | 0.6029  | 23    | deep | vector #671
```

With only one faucet signature, every prompt returns `crypto`; the score is semantic similarity, not a calibrated probability. Add more faucet `.sig` files to compare facets directly. The `Exact` column shows how many layer-3 vectors were actually scanned after hierarchical pruning.

## Dataset Format

The canonical facet dataset layout is:

```text
facets
`-- [Facet]
    `-- datasets
        |-- test
        |   |-- [facet]_pretrain.jsonl
        |   |-- [facet]_core.jsonl
        |   `-- [facet]_finetune.jsonl
        `-- train
            |-- [facet]_pretrain.jsonl
            |-- [facet]_core.jsonl
            `-- [facet]_finetune.jsonl
```

`*_pretrain.jsonl` uses one plain text per line:

```json
{"id":"crypto_pretrain_0001","text":"AES-GCM is an authenticated encryption mode..."}
```

`*_core.jsonl` and `*_finetune.jsonl` use chat messages:

```json
{"id":"crypto_core_0001","messages":[{"role":"user","content":"Explain AES-GCM."},{"role":"assistant","content":"AES-GCM is..."}]}
```

Current normalized files:

```text
..\facets\Crypto\datasets\train\crypto_pretrain.jsonl
..\facets\Crypto\datasets\train\crypto_core.jsonl
..\facets\Crypto\datasets\train\crypto_finetune.jsonl
..\facets\Crypto\datasets\test\crypto_pretrain.jsonl
..\facets\Crypto\datasets\test\crypto_core.jsonl
..\facets\Crypto\datasets\test\crypto_finetune.jsonl
..\facets\Web\datasets\train\web_pretrain.jsonl
..\facets\Web\datasets\train\web_core.jsonl
..\facets\Web\datasets\train\web_finetune.jsonl
..\facets\Web\datasets\test\web_pretrain.jsonl
..\facets\Web\datasets\test\web_core.jsonl
..\facets\Web\datasets\test\web_finetune.jsonl
```

Evaluate signatures:

```powershell
python router.py evaluate `
  --signatures ..\facets\signatures `
  -in ..\facets\Crypto\datasets\test\crypto_pretrain.jsonl,..\facets\Web\datasets\test\web_pretrain.jsonl `
  --model auto `
  --output report.json
```

Sweep threshold candidates without editing `defaults.json`:

```powershell
python router.py evaluate `
  --signatures ..\facets\signatures `
  -in ..\facets\Crypto\datasets\test\crypto_pretrain.jsonl,..\facets\Web\datasets\test\web_pretrain.jsonl `
  --model auto `
  --sweep-thresholds `
  --output sweep-report.json
```

The report includes recall@1, recall@3, MRR, false-positive rate, refused rate, per-layer accepted/passed/refused counts, confidence-decay stats, latency p50/p95/p99, traced memory, and loaded vector bytes.

## Other Supported Dataset Shapes

LSR still supports simpler JSON lists:

```json
[
  {
    "facet": "crypto",
    "text": "Verify RSA signatures using OpenSSL EVP."
  }
]
```

It also supports prompt-side fields:

```json
[
  {
    "facet": "crypto",
    "instruction": "Explain signature verification.",
    "input": "Use OpenSSL EVP with RSA.",
    "output": "Answer text is ignored unless no other text exists.",
    "tags": ["rsa", "openssl"]
  }
]
```

For those shapes, routing text uses `text` first, otherwise `instruction + input + tags`, and only uses `output` as a last resort.

## CLI Options

```text
python router.py serve [--options]
python router.py loadsig [--options]
python router.py unloadsig [--options]
python router.py sign [--options] [-in dataset.json | "raw text"]
python router.py rank [--options] [-in prompt.txt | "prompt text"]
python router.py benchmark-models [--options]
python router.py evaluate [--options]

Global:
--PUs                 List available processing units and exit

Common runtime options:
--model NAME|auto     SentenceTransformer model name, default sentence-transformers/all-MiniLM-L6-v2,
                      or infer from .sig metadata
--online              Allow model downloads/checks; default is local-only
--PU N|auto           Use processing unit index N, or auto-select a usable PU
--index auto|flat|hnsw
                       Vector index backend; auto uses global L1 FlatIP and HNSW for large layers
--memory-limit-mb N   0 keeps L1 warm; >0 enables chunked lazy loading under this memory budget
--descend-threshold N Override all per-layer pass thresholds from defaults.json
--accept-threshold N  Override all per-layer accept thresholds from defaults.json
--large-input auto|off|force
                      Large input routing mode

serve:
--host HOST           Server host, default 127.0.0.1
--port N              Server port, default 8765
--signatures PATH     Initial .sig file or directory of .sig files
--no-load             Start with no signatures loaded; use loadsig later
--top-k N             Default returned result limit, default 3
--index auto|flat|hnsw
                      Index policy, default auto

loadsig:
--signatures PATH     .sig file or directory to load into server RAM/VRAM
--server URL          Server URL, default http://127.0.0.1:8765

unloadsig:
--facet NAME          Facet to unload; repeatable
--all                 Unload all signatures
--server URL          Server URL, default http://127.0.0.1:8765

sign:
-in, --input PATH     Dataset JSON input
--facet NAME          Optional when facet can be inferred from the faucet path
--signatures PATH     Output .sig file or directory
--max-vectors N       Maximum anchor vectors stored per faucet, default 10000
--debug               Print extra details

rank:
-in, --input PATH     Prompt file, or batch input with --batch
--signatures PATH     .sig file or directory for --local ranking
--model NAME|auto     With --local, auto infers modelName from loaded signatures
--top-k N             Returned result limit; routing depth is threshold-based
--full-depth          Evaluate all layers even after accepted/refused layer decisions
--batch               Rank many prompts from JSON/JSONL/stdin in one process
--large-input auto|off|force
                      Use raw text or deterministic digest mode for large inputs
--index auto|flat|hnsw
                      Index policy for --local; default auto
--server URL          Server URL, default http://127.0.0.1:8765
--local               Load model/signatures locally instead of using serve
--remote              Compatibility option; server mode is already the default
--json                Print machine-readable JSON output
--ascii-table         Use ASCII-safe table borders
--debug               Print extra details

benchmark-models:
-in, --input PATH     Labeled benchmark path, default test pretrain files under ../facets
--dataset PATH        Training dataset path, directory, or comma-separated paths, default train pretrain files
--models A,B          Candidate models, default MiniLM-L6 and gte-small
--output PATH         Optional JSON report output
--index auto|flat|hnsw
                      Index policy to benchmark

evaluate:
-in, --input PATH     Test dataset file, directory, or comma-separated paths, default test pretrain files
--signatures PATH     .sig file or directory to evaluate
--model NAME|auto     Auto infers modelName from loaded signatures
--sweep-thresholds    Evaluate a small pass/accept threshold grid
--output PATH         Optional JSON report output
```
