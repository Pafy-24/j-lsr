# J5-J7 strategy foundation checkpoint

This tranche adds a library-level strategy asset around the existing semantic
`LSRRouter`. The current semantic route remains the default and its route
payload, independent facet weights, threshold decisions, and abstention
behavior are preserved.

## Contract

`StrategyAsset` is versioned, canonical-JSON serialized, and identified by a
digest over its strategy ID, version, immutable configuration, registry
snapshot ID, and per-facet fingerprint metadata. A registry snapshot records
the loaded signature facet names, the source file digest captured at load time,
loaded vector/topology digest, model metadata, layer sizes, and a snapshot
identity. Mutating loaded vectors therefore changes the identity; snapshot
capture does not reread the backing file on every request. A loaded asset
refuses to route against a different snapshot.

The strategy surface is:

- `build(registry, router)` creates an asset from the actual loaded registry;
- `load(path, registry, router)` verifies the serialized asset and snapshot;
- `route(prompt, prompt_embedding, candidate_facets=...)` routes through the
  real router and applies only request-local filtering;
- `explain(...)` returns the asset, snapshot, candidates, and decisions;
- `profile(...)` reports search time separately from caller-supplied embedding
  time.

An empty candidate label set is valid and returns an empty multilabel route.
The registry and its signatures are never mutated by filtering. Lazy signatures
are copied with the existing `routing_copy()` adapter before subset routing.

## Variants

`semantic-baseline-v1` delegates to the current semantic router. Its default
route is the compatibility baseline. `fingerprint-only-v1` computes a
deterministic CPU signed-projection bit fingerprint from the query embedding and
each facet's first-layer prototype, then emits independent facet weights from
Jaccard similarity. `fingerprint-gate-semantic-v1` uses the same score only as a
candidate gate and delegates the survivors to the semantic router. Neither
variant is selected implicitly, and neither uses softmax competition. Opposite
vectors are separated by the signed projections; non-finite vectors are
rejected.

Fingerprints record algorithm, bit count, seed, and feature configuration in
the asset. They are an experimental deterministic prefilter, not a semantic
quality claim.

## Evaluation boundary

Offline evaluation reports multilabel precision, recall, false activation,
abstention, and binned expected calibration error (confidence versus empirical
accuracy). Case/route length mismatches fail closed. A calibration label
universe is explicit; missing scores must be declared unavailable or
no-candidate-zero. Timing reports embedding and search as separate fields; the
strategy measures search and accepts embedding time from the caller. Empty
expected labels are valid and contribute to abstention metrics. No pilot-v3
performance or independent-heldout claim is made.

Server, CLI, RM lease, and channel integration are intentionally outside this
checkpoint and remain for the follow-up handshake.
