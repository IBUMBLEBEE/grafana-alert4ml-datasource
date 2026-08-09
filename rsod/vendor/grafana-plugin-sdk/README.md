# vendored grafana-plugin-sdk 0.5.0

A local copy of the crates.io package `grafana-plugin-sdk = "0.5.0"` with a single deserialization fix.

## Why vendored

Grafana serializes the frame JSON (`FrameJSON`) `schema.name` / `schema.refId` with
`omitempty`: when the frame name is empty (upstream frames from prometheus, testdata,
etc. always have an empty frame name), the `name` key is omitted entirely. The `Schema`
struct in SDK 0.5.0 declares both fields as required, so deserialization fails when
proxying queries through `/api/ds/query`:

```
query A failed: datasource query: failed to decode /api/ds/query response:
error decoding response body
```

(The Go backend is unaffected: the Go SDK's `json:"name,omitempty"` silently sets the
value to empty when deserializing missing fields.)

## Local change

The `Schema` struct in `src/data/frame/de.rs`:

```rust
#[serde(default)]
name: String,
#[serde(default)]
ref_id: String,
```

That is the only change (2 lines). Everything else matches crates.io 0.5.0.

## When to remove

Once the upstream fix is released (a new crates.io version), delete this directory and
the `[patch.crates-io]` section in `rsod/Cargo.toml`, and restore
`grafana-plugin-sdk = "<new version>"`.
