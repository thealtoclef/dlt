---
title: JSON column flattening (Arrow path)
description: Flatten JSON struct or string columns into typed dlt columns on the Arrow normalize path
keywords: [arrow, parquet, json, flatten, schema, normalize, x-json-flatten]
---

# JSON column flattening on the Arrow path

When a resource emits `pyarrow.Table` / `pyarrow.RecordBatch` data (the high-throughput "Arrow path"), dlt can flatten nested JSON inside individual columns into typed sibling columns during normalize. This mirrors the row-by-row JSON-path behavior controlled by `x-json-flatten` hints, but runs vectorized over Arrow buffers instead of materializing rows as Python dicts.

Use it when:

- Your source produces wide JSON blobs (a `meta` struct, a stringified API payload) and you want them landed as flat typed columns at the destination — without writing a custom transform.
- You want destination schema parity between an Arrow-emitting resource and a Python-dict-emitting resource that share the same downstream consumers.

## Quick start

```py
import dlt
import pyarrow as pa

data = pa.table({
    "id": [1, 2],
    "meta": [
        {"user": {"name": "alice", "email": "a@x"}, "extra": 1},
        {"user": {"name": "bob",   "email": "b@x"}, "extra": 2},
    ],
})

@dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}))
def my_resource():
    yield data

pipeline = dlt.pipeline("demo", destination="duckdb")
pipeline.run(my_resource())
```

The destination table `my_resource` now has columns `id`, `meta__user__name`, `meta__user__email`, `meta__extra` — the `meta` struct is gone, its leaves are typed columns named with the active naming convention's separator (`__` for snake_case).

## The hint API

`dlt.mark.with_json_flatten` attaches `x-json-*` hints to one or more columns. All keyword arguments are optional and default to the row-by-row JSON path's defaults.

```py
dlt.mark.with_json_flatten(
    {
        "meta":      True,                 # full recursive flatten
        "payload":   ["user.name", "id"],  # extract specific paths only
        "raw_blob":  True,
    },
    keep_original=False,         # also keep the source column alongside flattened siblings
    force_string=False,          # cast every emitted leaf to text
    max_depth=None,              # stop recursion at depth N; keep deeper as struct (or JSON string with engine="duckdb")
    schema_inference="incremental",  # how to discover the union schema of string-JSON columns
)
```

| Hint argument | Equivalent schema hint | Effect |
|---|---|---|
| dict value `True` | `x-json-flatten: true` | recursive full flatten |
| dict value `[paths]` | `x-json-flatten: [paths]` | only the listed dotted paths are emitted |
| `keep_original=True` | `x-json-keep-original: true` | source column kept alongside flattened columns |
| `force_string=True` | `x-json-flatten-force-string: true` | every leaf cast to text |
| `max_depth=N` | `x-json-flatten-max-depth: N` | flatten up to depth N; deeper kept as struct (pyarrow) or JSON string (duckdb) |
| `schema_inference="incremental" \| "full-scan" \| int` | `x-json-flatten-schema-inference` | controls union-schema discovery for string-JSON columns |

## Input shapes

The flatten step inspects the actual Arrow type of each hinted column and dispatches accordingly. Two shapes are supported:

### Struct column

```text
meta: struct<user: struct<name: string, email: string>, extra: int64>
```

Flattening is **zero-copy** — emitted leaves are `pa.compute.struct_field` projections of the source buffer. Naming follows `naming.shorten_fragments(parent, *child_path)`, identical to the JSON path's `_flatten`.

### String / large_string column carrying JSON text

```text
raw: string  # values like '{"a": 1, "b": "x"}'
```

Each row is parsed and a union struct schema is inferred. By default this uses pyarrow's native dict-to-struct builder; `engine="duckdb"` switches to a vectorized DuckDB SQL pipeline.

Rows whose JSON parses to anything other than an object (scalar, array, invalid) are preserved as-is in the source column — they are never silently dropped.

:::tip
If your Arrow data comes from SQL JSON/JSONB columns via connectorx, those columns usually arrive as `pa.string()` JSON text. Set `json_engine = "duckdb"` for that workload: DuckDB parses string JSON and extracts `flatten=[paths]` selections in vectorized SQL, while the default pyarrow engine must materialize each batch to Python strings and call `json.loads` row by row.
:::

## Schema inference modes (string-JSON columns)

Use `schema_inference=...` to control how the union schema is discovered when a column carries JSON text.

| Mode | When the schema is learned | Memory cost | Late-appearing keys |
|---|---|---|---|
| `"incremental"` (default) | Per batch, additively | bounded — one batch at a time | Captured if they appear before the column is locked at the destination; **may be missed** if a destination's schema is fixed earlier |
| `"full-scan"` | One pre-scan over the entire parquet file, before any batch is flattened | **proportional to the full column** (every row's parsed dict held in memory during the scan) | Always captured |
| `int N` | Pre-scan limited to the first `N` rows | bounded — N rows | Missed if they appear after row N |

:::warning
`schema_inference="full-scan"` materializes every row of the column as a Python dict during the pre-scan. For a 10M-row column with ~1KB JSON per value, expect on the order of 10GB of Python dicts to be resident before flatten starts. Prefer `int N` sampling on wide columns; reserve `"full-scan"` for small but type-diverse columns.
:::

## Engines

A single config knob selects the flatten engine. The default is pure pyarrow and adds no new dependency.

```toml
[normalize.arrow_normalizer]
json_engine = "pyarrow"          # default
duckdb_memory_limit = "2GB"      # only honored when json_engine = "duckdb"
duckdb_threads = 1
```

Or via environment variable: `NORMALIZE__ARROW_NORMALIZER__JSON_ENGINE=duckdb`.

| Engine | Requires | What it accelerates |
|---|---|---|
| `"pyarrow"` | nothing | All struct work (zero-copy). String-JSON parsing via `pa.array` of dicts. |
| `"duckdb"` | `pip install dlt[duckdb]` (already a dlt extra) | Vectorized SQL parsing of string-JSON columns, including selective `flatten=[paths]` extraction. Also enables `to_json` serialization at `max_depth` boundaries and for `keep_original` on struct inputs. |

`engine="duckdb"` opens one in-memory DuckDB connection per normalizer worker, lazily on first use; the connection is closed when the worker shuts down.

For string-JSON columns, DuckDB avoids the Python `to_pylist()` + `json.loads` loop used by the pyarrow engine during normal batch flattening. Full-scan schema inference still performs a Python pre-scan by design; use it only when you need its stronger late-key guarantees.

## Documented divergences from the JSON path

Two hints behave differently on the Arrow path than on the row-by-row JSON path. These are intentional — the Arrow buffers expose information the dict-based path doesn't have, and the cheapest correct option differs.

### `keep_original=True` on a struct column

| Path | Source column after flatten |
|---|---|
| JSON (dict-based) | JSON string |
| Arrow (default `engine="pyarrow"`) | **The original struct column, unchanged** |
| Arrow (`engine="duckdb"`) | JSON string (via DuckDB `to_json`) |

Rationale: keeping the source as a struct is free on the Arrow path; serializing to JSON text is not. Users who specifically want a JSON-string copy can opt into `engine="duckdb"`.

### `max_depth=N` on a struct input

| Path | Sub-tree at depth > N |
|---|---|
| JSON (dict-based) | JSON string |
| Arrow (default `engine="pyarrow"`) | **Kept as a struct column** |
| Arrow (`engine="duckdb"`) | JSON string (via DuckDB `to_json`) |

Same rationale as above.

For string-JSON inputs both behaviors above are produced natively under either engine — the divergences only apply when the input was already a struct.

### Boolean stringification under `force_string`

| Path | `True` becomes | `False` becomes |
|---|---|---|
| JSON (dict-based) | `"True"` (Python `str()`) | `"False"` |
| Arrow (either engine) | `"true"` (Arrow native cast, JSON-literal) | `"false"` |

Both are accepted dlt outputs. Downstream consumers that depend on exact case should normalize at query time.

## Mixed-type coercion

When the same flattened path has incompatible types across rows (e.g. `{"a": 1}` and `{"a": "x"}`) dlt widens the column to `text` and stringifies values. This applies in two cases:

- **Same batch**: rows in one row-group disagree → the offending field is built field-by-field with per-key coercion to text.
- **Across batches** (incremental mode only): a later batch carries a type that conflicts with what an earlier batch already emitted → the canonical column type advances to `text` from that batch onward.

If you need a single canonical text type across **all** batches (including the first one), use `schema_inference="full-scan"`: the pre-scan observes the conflict before any batch is flattened, locks the union schema with the path widened to text, and every batch — including the first — is emitted as text.

## Type-conflict and naming-collision policies

- **Type conflict** at a flattened path: widen to `text`, log one warning per `(table, path)`. Same policy as dlt's variant resolution elsewhere.
- **Name collision** — a flattened name (e.g. `meta__a`) already exists on the batch as a regular column: raises `NameNormalizationCollision` (the existing dlt exception used by the Arrow normalizer). Surfaces immediately rather than silently overwriting.

## Pros

- **Vectorized**: struct input is zero-copy via `pa.compute.struct_field`; no row-by-row Python materialization.
- **No new mandatory dependency**: default engine is pure pyarrow.
- **Schema parity with the JSON path** for all hints, including `force_string`, `max_depth`, `keep_original`.
- **Lossless**: rows whose JSON can't be parsed as an object stay in the source column rather than being dropped to null.
- **Late-key recovery**: `schema_inference="full-scan"` guarantees no late-appearing key is missed.
- **Honors mixed types**: same-batch and cross-batch type conflicts are coerced to text rather than crashing the normalize step.
- **Multi-row-group safe**: a single parquet file with many row groups gets the flatten applied uniformly to every row group; no schema drift between groups.
- **Opt-in DuckDB acceleration** for the cases where Python serialization is the bottleneck or where struct→JSON-string semantics are required.
- **Interoperates with `_dlt_id` / `_dlt_load_id`**: both are added by their normal mechanisms (extract-time for `_dlt_load_id`, normalize-time for `_dlt_id`); the flatten step never touches them.

## Cons / limitations

- **Default pyarrow string-JSON parsing materializes one batch as Python dicts** (`to_pylist` + `json.loads`). Peak transient memory per batch is roughly 3× the batch's arrow size. Set `json_engine="duckdb"` for vectorized string-JSON parsing and path extraction. For struct-typed sources the cost is zero.
- **`schema_inference="full-scan"` holds the entire column in Python memory during the pre-scan.** Avoid for very wide columns; prefer `int N` sampling.
- **Cross-batch type conflicts can't rewrite already-written batches.** The first batch's data lands with its original type; only batches from the conflicting one onward become text. Use `"full-scan"` for a globally consistent type.
- **`keep_original=True` on struct + `engine="pyarrow"`** preserves the struct column (documented divergence); set `engine="duckdb"` to get a JSON-string copy instead.
- **`max_depth=N` on struct + `engine="pyarrow"`** keeps depth-`N+1` sub-trees as struct columns (documented divergence); set `engine="duckdb"` for JSON-string at the boundary.
- **Top-level lists inside hinted columns are not extracted to child tables.** If a hinted column contains an array root, that row's value is preserved in the source column as-is. Child-table extraction is a separate feature outside the flatten scope.
- **DuckDB engine adds one round-trip per batch** for the structure-detection query. At very small batch sizes the overhead may be visible; the break-even vs the pyarrow path is at a few thousand rows.
- **Adds a normalize-stage rewrite of the parquet file.** Files that would otherwise have been directly imported (`must_rewrite=False`) are rewritten when any hinted column is present on the table.

## Configuration summary

```toml
# dlt configuration (config.toml / env)
[normalize.arrow_normalizer]
json_engine = "pyarrow"             # "pyarrow" | "duckdb"
duckdb_memory_limit = "2GB"
duckdb_threads = 1
```

Per-column hints — three equivalent forms:

```py
# (1) in-code via dlt.mark
@dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}, force_string=True))
def res(): ...

# (2) as resource column hints
@dlt.resource(columns={"meta": {"x-json-flatten": True, "x-json-flatten-force-string": True}})
def res(): ...

# (3) in a schema file (yaml/json)
# tables:
#   res:
#     columns:
#       meta:
#         x-json-flatten: true
#         x-json-flatten-force-string: true
```

## Related

- [Schema](../schema.md) — overall schema model
- [Schema evolution](../schema-evolution.md) — how dlt evolves columns over time
- [Naming convention](../naming-convention.md) — how flattened column names are built
