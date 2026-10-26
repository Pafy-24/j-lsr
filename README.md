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
python -m pip install -r requirements-windows-ml-dev.txt
```

## Resource Manager boundary

LSR nu mai deține inventarul sau alocarea hardware. Orice comandă care încarcă modelul local (`serve`, `sign`, `benchmark-models`, `evaluate`, `rank --local`) trebuie pornită prin Jester Resource Manager și primește CPU/CUDA/DirectML/VitisAI din lease. `--PUs` a fost înlocuit de:

```powershell
cd E:\Licenta\Proiect\RM
jester-rm resources list --json
jester-rm instance create examples\instance-lsr-npu.json
jester-rm instance plan lsr-npu-01
jester-rm instance start lsr-npu-01
```

`--PU` și `--memory-limit-mb` sunt respinse în workloadurile managed; limitele se declară în profilul RM. Fragmentele locale de comandă din secțiunile algoritmice de mai jos descriu argumentele adaptorului `lsr.task`/`lsr.serve`, nu o cale operațională care ocolește RM. `loadsig`, `unloadsig` și rank către un server existent nu alocă hardware local și pot fi apelate direct.

## Project Layout

- `router.py` - thin CLI compatibility entry point
- `lsr\args.py` - CLI argument definitions
- `lsr\defaults.json` - central defaults for models, paths, thresholds, index policy, and layer sizing
- `lsr\resources\component-manifest.json` - installed portable `lsr.route` manifest; its four bound JSON resources declare the narrow configuration schema, partial defaults, capability contract, and closed `lsr.serve` mapping. The defaults supply only `topK`; encoder and signature assets remain required. The mapping is declarative and is not executed by LSR. Schema validation fixes the two asset roles and their reference shapes, but cannot prove that resolved bytes contain a compatible encoder or signature set; that later check belongs to the Runtime asset resolver.
- `lsr\config.py` - Python accessors for `defaults.json`
- `lsr\commands.py` - command orchestration for signing and routing
- `lsr\dataset.py` - dataset-to-route-text extraction
- `lsr\dataset_schema.py` - `pretrain/core/finetune` JSONL loader/adapters
- `lsr\devices.py` - processing-unit discovery and selection
- `lsr\embedding.py` - model loading and embedding helpers
- `lsr\windows_ml.py` - Windows ML device registration and ONNX embeddings
- `lsr\input_digest.py` - deterministic digest mode for very large inputs
- `lsr\memory.py` - functional allocator cleanup and release hooks; no process-footprint sampling
- `lsr\registry.py` - server-side signature registry and stale-file refresh
- `lsr\scoring.py` - lexical/generic/repetition diagnostics and score helpers
- `lsr\signatures.py` - signature building, loading, saving, and DER codec
- `lsr\routing.py` - layered routing, lazy loading policy, and hierarchical search
- `requirements.txt` - Python dependencies
- `requirements-windows-ml-dev.txt` - AMD NPU model-preparation dependencies
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
2. load one needed layer from one .sig file
3. score that layer
4. copy only scalar diagnostics/candidate ids
5. release that layer from RAM/VRAM immediately
6. descend only facets whose current layer decision is `passed`
```

There is no persistent global L1 index in runtime routing. `loadingPolicy=streaming-release-layer` means each vector layer is read with `seek/read`, compared, and removed before the next layer is loaded. Immutable parent-child topology is retained separately in a compact CSR-style `uint32` cache, so repeated requests do not decode thousands of DER integers again. The cache is shared safely by request-local routing copies and is reported independently from transient vector memory.

The router has a library-level budget for loaded signature-vector bytes before loading each layer. This is not process RAM/VRAM measurement or a process limit. Operational limits remain RM `ramBytes` and accelerator `memoryBudgetBytes`, applied by RMA. The managed CLI continues to reject `--memory-limit-mb`; local/library use interprets it as the vector budget. Allocator cleanup remains functional component behavior, while RMA observes process consumption externally.

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

Routing depth is no longer decided by top-k or a fixed number of accepted
facets. `--top-k` only limits the returned diagnostic results and selected
facets after all threshold decisions have been evaluated.

For diagnostics, `router.py rank --full-depth ...` evaluates all layers even when an earlier layer already returned `accepted` or `refused`. JSON output includes each layer score, threshold, decision, confidence decay, entropy, plus facet-level `semanticScore`, `confidence`, `weight`, `separation`, `anchorSimilarity`, `lexicalScore`, `genericPenalty`, and `repetitionPenalty`.

Routing is multi-label. The machine-readable response exposes `selectedFacets`,
an independently weighted list of every non-refused facet up to `top_k`:

```json
{
  "schemaVersion": "lsr-route-v2",
  "weightSemantics": "independent-pass-to-max-v1",
  "selectedFacets": [
    {"facet": "web", "weight": 0.61, "decision": "passed"},
    {"facet": "crypto", "weight": 0.43, "decision": "passed"}
  ]
}
```

`weight` is always in `[0, 1]`, is not normalized across facets, and is not a
probability. Passing the `accepted` threshold does not force a weight of `1`;
that value is reserved for the maximum semantic score. Downstream inference
can therefore apply a facet gradually instead of allowing a weakly relevant
facet to dominate the base model. If every facet is refused, `selectedFacets`
is an empty list. No synthetic `unknown` route is emitted. The full `results`
list remains available for diagnostics and includes refused facets with
`weight: 0`.

`schemaVersion` and `weightSemantics` make this contract explicit for Mixed
Inference and allow the weighting formula to evolve later without silently
changing downstream behavior.

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

For interactive use or integrations, LSR can keep the model and signatures loaded in an RM-managed HTTP server. Configurează `component.arguments` cu `--signatures ...`; hostul și portul sunt injectate de adaptor:

```powershell
jester-rm instance start lsr-serve-01
jester-rm operations inspect <OPERATION_ID>
jester-rm instance status lsr-serve-01
```

Health check:

Folosește portul din lease/status (sau declară `network.port: 8765`):

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

- `GET /health` - also returns `embeddingBackend`, `selectedProcessingUnit`, and live CPU/GPU/iGPU/NPU counters under `processingUnits`
- `POST /rank` - accepts `{"input": "..."}`; `prompt`, `text`, `full_depth`, and `large_input` are also accepted
- `POST /route` - compatibility alias for `/rank`
- `POST /batch` - accepts `{"inputs": ["..."]}`; `prompts`, `texts`, `full_depth`, and `large_input` are also accepted
- `POST /loadsig` - accepts `{"signatures": "path/to/file-or-directory"}`
- `POST /unloadsig` - accepts `{"facet": "crypto"}`, `{"facets": ["crypto"]}`, or `{"all": true}`

Explicitly unloaded facets remain disabled during automatic signature refresh.
Calling `loadsig` creates a fresh registry and enables the signatures found at
the requested path again.

The server applies resource backpressure: JSON requests are limited to 4 MiB,
batches to 256 prompts, and the worker pool accepts at most eight queued
requests beyond the active workers. Signature directory scans are throttled to
once per second; explicit `loadsig` and `unloadsig` operations remain immediate.

Concurrent rank requests are not serialized by a global request lock. Each
request routes against an atomic metadata snapshot and owns its lazy layer
cache, so one request cannot release another request's vectors. Embedding
requests are sent through one micro-batching worker. By default it waits up to
5 ms and combines up to 256 semantic units into one model invocation. Existing
`/batch` requests and simultaneous `/rank` calls share the same batching path.
`GET /health` exposes cumulative batching counters under `embeddingBatcher`.
Signature load/unload/refresh operations use a short state lock while active
routes continue safely on their existing snapshots.

Before each rank request, the server refreshes the signature registry. Deleted `.sig` files are unloaded, modified files are reloaded, and new `.sig` files are loaded automatically when the active signature path is a directory.

Signature loading can be controlled through CLI commands while the server is running:

```powershell
# pornește o instanță lsr.serve al cărei component.arguments conține --no-load
jester-rm instance start lsr-serve-empty-01
python router.py loadsig --signatures ..\facets\signatures --server http://127.0.0.1:8765
python router.py unloadsig --facet crypto --server http://127.0.0.1:8765
python router.py unloadsig --all --server http://127.0.0.1:8765
```

Signing remains a CLI operation. When `serve` watches a signature directory, updated `.sig` files are picked up automatically on the next rank request.

The server uses the same local-only model loading rule as CLI routing. Use `--online` only when the model is not already cached locally.

La primul run într-un mediu fără model cached, adaugă `--online` în `component.arguments`, apoi recreează generația/lease-ul. După cache, elimină flag-ul; hash-ul effective config va reflecta alegerea.

```powershell
jester-rm instance inspect lsr-serve-01
jester-rm instance restart lsr-serve-01
```

## RM bindings și pregătirea Windows ML

Inventarul canonic și selecția sunt făcute cu ID-uri stabile:

```powershell
jester-rm resources list --json
jester-rm profile validate lsr-npu --adapter lsr.serve
jester-rm instance plan lsr-npu-01
```

CPU folosește `processorIds`, NVIDIA folosește UUID, iar iGPU/NPU folosesc provider + LUID. Lease-ul traduce binding-ul fizic în vizibilitatea runtime; LSR verifică providerul și LUID-ul înainte de embedding și refuză fallback-ul tăcut.

On the Ryzen AI 9 365 development machine, the current mapping is:

```text
0  CPU   torch                 Ryzen AI 9 CPU
1  GPU   torch.cuda            NVIDIA RTX 5070 Laptop GPU
2  iGPU  windowsml.dml         AMD Radeon 880M
3  NPU   windowsml.vitisai     AMD NPU Compute Accelerator
```

Numerele din stânga sunt doar vechea ordine diagnostică și nu se persistă. Copiază `resourceId` din RM. DirectML este legat prin LUID de Radeon-ul integrat și nu poate selecta silențios placa NVIDIA discretă.

The iGPU uses the upstream `onnx/model.onnx` export. Cache it once if needed:

```powershell
python scripts\probe_windows_ml.py --PU 2 --iterations 1
```

The NPU needs a static A16W8 graph prepared once with AMD Quark. Run this from a
Visual Studio Developer PowerShell so `cl.exe` is available:

```powershell
conda activate licenta
cd E:\Licenta\Proiect\LSR
$env:CUDA_VISIBLE_DEVICES = "-1"
python scripts\prepare_windows_ml_model.py `
  --calibration ..\facets\Crypto\datasets\train `
  --calibration ..\facets\Web\datasets\train
```

The generated `onnx/model_npu_a16w8.onnx` stays in the Hugging Face model cache;
it is not committed to the repository. Vitis currently requires batch 1 for
this MiniLM graph. `WindowsMLEmbeddingModel` handles that constraint internally,
including when the server micro-batcher submits several semantic units.
The compiled Vitis artifact is cached under `.lsr_vitis_cache`, keyed by model
hash and EP version. The first NPU session still performs a long compilation;
later processes reuse the compiled `xmodel` and start faster.

Verified on this machine on 2026-07-29:

- iGPU output cosine versus CPU FP32: at least `0.99999994`; physical Radeon
  utilization reached `2.01%` during the sustained probe.
- NPU output cosine versus CPU FP32: `0.9546` to `0.9791` on the smoke prompts;
  physical NPU utilization reached `2.59%`.
- The 100-item Crypto/Web evaluation on the 64-sample calibrated NPU model
  matched the CPU routing metrics: recall@1 `0.81`, recall@3 `0.88`, MRR
  `0.845`, false-positive rate `0.08`. Its p50 routing latency was `88.08 ms`;
  NPU execution is therefore functional but is not a latency win for this
  small, partially offloaded MiniLM graph.

Run a sustained hardware probe with:

```powershell
python scripts\probe_windows_ml.py --PU 2 --iterations 100
python scripts\probe_windows_ml.py --PU 3 --iterations 100
```

## Example Result

```text
Rank | Facet  | Score  | Best Vector | Cluster | Exact | Mode | Best Match
1    | crypto | 0.6042 | 0.6276      | 0.6029  | 23    | deep | vector #671
```

With only one faucet signature, the diagnostic `results` list compares against
`crypto`, but `selectedFacets` remains empty when that signature is refused.
The score and weight are not calibrated probabilities. Add more faucet `.sig`
files to compare facets directly. The `Exact` column shows how many final-layer
vectors were scanned after hierarchical pruning.

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

The server terminal displays a live dashboard by default. It shows the current
routing/embedding operation, CPU, CUDA GPU, Radeon iGPU, and AMD NPU utilization,
process RAM, CUDA VRAM, and physical dedicated/shared accelerator memory,
loaded signature layers, vector bytes, request count, and last-request latency.
RAM and VRAM inventories are separate and attribute memory to the embedding
model, signature metadata, active vector layers, CUDA allocator cache, and
unattributed runtime/native allocations. The same tracked allocations are
available as `memoryComponents` from `GET /health`.
The Windows accelerator counters are read by physical LUID through a persistent
PDH query, so monitoring does not launch PowerShell once per refresh.
Model parameter and buffer bytes are counted from their actual tensor devices.
The RAM delta caused by importing PyTorch/Transformers, loading the tokenizer,
CUDA driver mappings, and native libraries is reported separately as
`embedding-framework/tokenizer/driver`; it is not presented as a CPU copy of
GPU model weights.

Every `/rank` response includes `performance.timing` (`rankingMs`,
`embeddingMs`, `routingMs`, and process `cpuTimeMs`) plus per-request RAM and
CUDA allocator peaks under `performance.resources`. The terminal rank output
prints both transient peak and retained-after-rank deltas. The server does not
run inference warm-up by default: PyTorch CUDA warm-up was measured to retain
roughly 500 MB of additional native working memory on this Windows setup. The
first rank therefore includes lazy backend initialization, while subsequent
ranks show steady-state cost. The dashboard retains the last rank's
peaks, also exposed as `lastRankPerformance` by `GET /health`.
On Windows, per-rank resources distinguish volatile physical working set from
process Private Bytes; the latter is the better retained-allocation/leak signal.

Each rank also reports `signatureMemory`: the theoretical vector RAM required
to load every layer eagerly, the size of all facets' L1 layers, the measured
peak simultaneously resident vector bytes under the streaming policy, total
layer bytes read, loads and facet counts per level, and the facets that
descended beyond L1. This makes memory scaling measurable without actually
loading hundreds or thousands of complete signatures at once.
It also projects the observed average signature sizes to 100, 1,000, and
10,000 facets for both all-L1-resident and eager-all-layers policies.
`persistentTopologyCacheBytes`, `persistentTopologyCacheLevels` and
`peakResidentSignatureBytes` make the latency/memory trade-off explicit.

The route-only hot path can be reproduced without embedding-model noise:

```powershell
python scripts\benchmark_routing_hotpath.py `
  --signatures ..\facets\signatures-data2k `
  --anchor-facet coder `
  --iterations 20 `
  --warmup 1
```

The script reloads signatures independently for normal and `fullDepth` modes,
reports cold and steady-state latency, and does not include embedding time.

Short and large inputs share the same bounded semantic-unit pipeline. At most
six units (global input plus domain-neutral clauses/chunks) are embedded in one
batch. Every signature layer is loaded once and scored with one matrix
multiplication against the unit matrix; `similarityEvaluationsUpperBound`
reports the bounded vector-by-unit work. Scores combine 35% global similarity
and 65% best-unit similarity, preserving overall intent while allowing mixed
facets to activate independently.
Use `--no-monitor` for redirected logs or non-interactive service deployments.
The HTTP server uses a bounded worker pool (`--workers 4` by default), avoiding
the native thread-local memory growth caused by creating one PyTorch request
thread per connection.

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

The report includes recall@1, recall@3, MRR, false-positive rate, refused rate,
per-layer accepted/passed/refused counts, selected-facet count, weight,
confidence/separation/decay stats, lexical/generic/repetition diagnostics,
latency p50/p95/p99, process RAM/VRAM footprint, and loaded vector bytes.
Recall and false-positive metrics use `selectedFacets`, so refused diagnostic
results cannot be counted as successful routing decisions.

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
python router.py strategy build|load [--options]
python router.py benchmark-models [--options]
python router.py evaluate [--options]

Global:
--PUs                 Deprecated; exits with RM_LEASE_REQUIRED. Use jester-rm resources list.

Common runtime options:
--model NAME|auto     SentenceTransformer model name, default sentence-transformers/all-MiniLM-L6-v2,
                      or infer from .sig metadata
--online              Allow model downloads/checks; default is local-only
--PU N|auto           Legacy parser field; rejected for managed workloads. Device comes from lease.
--index auto|flat|hnsw
                       Vector index backend; auto uses global L1 FlatIP and HNSW for large layers
--memory-limit-mb N   Local/library signature-vector budget in MiB, not process RAM/VRAM.
                      Rejected for managed workloads; process limits come from the RM lease.
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
--strategy-asset PATH Versioned strategy asset validated before encoder load

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

strategy:
build                 Requires --output PATH; writes an asset from current signatures/router settings
load                  Requires --asset PATH; validates the asset against current signatures/router settings
--strategy-id ID      semantic-baseline-v1, fingerprint-only-v1, or fingerprint-gate-semantic-v1
--fingerprint-bits N, --fingerprint-seed TEXT, --minimum-jaccard N
                      Experimental fingerprint configuration for build

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
--strategy-asset PATH Select an asset for --local ranking; remote services select it at startup
--strategy-operation route|explain|profile
                      Explicit local/HTTP asset operation; managed channels authorize route only

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

## Shared SDK runtime

Managed-channel validation is shared through `jester_sdk_runtime`; the existing
component CLI/API and resource checks remain in place. Native builds include
this support as compiled code. Recipients of the standard Engine Native Blocks
do not install the authoring SDK separately.

Source development and build hosts require the matching SDK runtime sources
(`JesterWorkspace/jester-sdk/src` on the development import path), plus Core
and its RM client. See the [SDK documentation](../../../../jester-sdk/README.md)
and [workspace build recipes](../../../../compile/README.md). This is a build
dependency, not a new runtime package installation step.
