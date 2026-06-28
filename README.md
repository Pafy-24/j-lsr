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
- `lsr\commands.py` - command orchestration for signing and routing
- `lsr\dataset.py` - dataset-to-route-text extraction
- `lsr\devices.py` - processing-unit discovery and selection
- `lsr\embedding.py` - model loading and embedding helpers
- `lsr\signatures.py` - signature building, loading, saving, and DER codec
- `lsr\routing.py` - query expansion and hierarchical search
- `requirements.txt` - Python dependencies
- `..\faucets\Crypto\datasets\train\crypto_dataset.json` - full raw Crypto training dataset
- `..\faucets\signatures\crypto.sig` - ASN.1 DER hierarchical routing signature for Crypto

There is no `signatures.json` in the active flow. A faucet signature is stored as:

```text
[faucet].sig
```

The `.sig` file is binary ASN.1 DER. It contains:

```text
FacetSignature ::= SEQUENCE {
  version     INTEGER,
  modelHash   OCTET STRING,
  dimension   INTEGER,
  layers      SEQUENCE OF Layer
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
python router.py --sign `
  -in ..\faucets\Crypto\datasets\train\crypto_dataset.json `
  --signatures ..\faucets\signatures `
  --debug
```

Because the path is under `faucets\Crypto`, the facet name is inferred as `crypto`. You can also pass it explicitly:

```powershell
python router.py --sign `
  -in ..\faucets\Crypto\datasets\train\crypto_dataset.json `
  --facet crypto `
  --signatures ..\faucets\signatures
```

## Layer Rule

Layer sizing is intentionally not exposed as normal CLI tuning. LSR uses a fixed geometric rule:

```text
L1 = min(128, anchor_count)
L(n+1) = min(anchor_count, L(n) * 10)
stop when L(n) == anchor_count
```

That gives a stable top-level comparison cost while allowing larger faucets to grow deeper:

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
..\faucets\signatures\crypto.sig
```

## Descend Policy

Routing does not send every faucet to the deepest layer.

```text
1. score every faucet on L1 only
2. always descend at least the top-k L1 faucets
3. also descend near-tie faucets if:
   score >= 0.18 and score >= best_l1_score - 0.08
4. never descend more than 8 faucets per query
```

Inside a selected faucet, the router expands the best linked clusters until the anchor layer, then performs exact search only on the selected anchor pool.

## Route Prompts

Route against all `.sig` files in a signature directory:

```powershell
python router.py "How can I recover an ECDSA private key if the nonce k is reused?" `
  --signatures ..\faucets\signatures
```

Route against one specific signature:

```powershell
python router.py "How do I verify a DSA signature?" `
  --signatures ..\faucets\signatures\crypto.sig
```

Input file:

```powershell
python router.py -in prompt.txt `
  --signatures ..\faucets\signatures `
  --top-k 5
```

Batch route many prompts in one process:

```powershell
python router.py --batch --json `
  -in ..\router_test_crypto_web_100.json `
  --signatures ..\faucets\signatures `
  --PU 1
```

Batch mode loads the model and signatures once, embeds all prompts together, and emits JSON. This is the preferred mode for evaluation and bulk routing.

By default, model loading is local-only:

```text
SentenceTransformer(..., local_files_only=True)
```

Use `--online` only when the model is not already cached and the machine is allowed to download it.

## Serve Mode

For interactive use or integrations, LSR can keep the model and signatures loaded in a local HTTP server:

```powershell
python router.py --serve `
  --signatures ..\faucets\signatures `
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
  -Uri http://127.0.0.1:8765/route `
  -ContentType application/json `
  -Body '{"input":"How can I recover an ECDSA private key if the nonce k is reused?","top_k":1}'
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

- `GET /health` - returns model, signature path, loaded facets, and default `top_k`
- `POST /route` - accepts `{"input": "..."}`; `prompt` and `text` are also accepted
- `POST /batch` - accepts `{"inputs": ["..."]}`; `prompts` and `texts` are also accepted

The server uses the same local-only model loading rule as CLI routing. Use `--online` only when the model is not already cached locally.

## Processing Units

List processing units visible to the router:

```powershell
python router.py --PUs
```

Select a processing unit for signing or routing:

```powershell
python router.py --sign `
  -in ..\faucets\Crypto\datasets\train\crypto_dataset.json `
  --signatures ..\faucets\signatures `
  --PU 1

python router.py "How do I verify an ECDSA signature?" `
  --signatures ..\faucets\signatures `
  --PU 1
```

`--PUs` always includes CPU index `0`. It also reports GPUs/NPUs detected by Windows even when the current Python backend cannot use them. The `Usable` column tells whether `--PU <index>` can actually be used for embeddings.

On this machine, the `licenta` environment uses Python 3.11 and `torch 2.11.0+cu128`. The NVIDIA RTX GPU is usable through `torch.cuda` and appears as `--PU 1`. Windows also detects the AMD NPU, but it remains unusable for the current SentenceTransformer/PyTorch backend.

## Example Result

```text
Rank | Facet  | Score  | Best Vector | Cluster | Exact | Mode | Best Match
1    | crypto | 0.6042 | 0.6276      | 0.6029  | 23    | deep | vector #671
```

With only one faucet signature, every prompt returns `crypto`; the score is semantic similarity, not a calibrated probability. Add more faucet `.sig` files to compare facets directly. The `Exact` column shows how many layer-3 vectors were actually scanned after hierarchical pruning.

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
--sign                Generate or update vector signatures
-in, --input PATH     Input file path
--facet NAME          Optional when facet can be inferred from the faucet path
--signatures PATH     .sig file or directory of .sig files, default signatures
--top-k N             Number of routed facets to show, default 3
--batch               Route many prompts from JSON/JSONL/stdin in one process
--json                Print machine-readable JSON output
--model NAME          SentenceTransformer model name
--online              Allow model downloads/checks; default is local-only
--max-vectors N       Maximum layer-3 vectors stored per faucet, default 10000
--PUs                 List available processing units and exit
--PU N                Use processing unit index N for embeddings
--debug               Print extra details
```
