"""AC parity tests: each AC from tests/common/normalizers/test_json_expansion.py
is re-asserted on the Arrow normalize path.

For every AC we run TWO variants:
  - struct input: `pa.struct<...>` column
  - string-JSON input: `pa.string()` column carrying the same JSON text

Each test goes through `pipeline.extract(loader_file_format='parquet')` then
`pipeline.normalize()` and reads the normalized parquet job back. This exercises
the real `ArrowItemsNormalizer._write_with_dlt_columns` rewrite path, the
schema-update plumbing, and the prescan integration — i.e. the same end-to-end
surface the JSON-path tests cover, but on the Arrow path.
"""

import json as _json
from typing import Any

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import dlt
from dlt.common.utils import uniq_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_pipeline(name: str) -> Any:
    return dlt.pipeline(
        "arrow_ac_" + uniq_id() + "_" + name,
        destination="duckdb",
        dev_mode=True,
    )


def _normalized_table(pipeline: Any, table_substr: str) -> pa.Table:
    """Run normalize, return the first parquet job table whose name contains substr."""
    pipeline.normalize()
    load_id = pipeline.list_normalized_load_packages()[0]
    storage = pipeline._get_load_storage()
    jobs = storage.normalized_packages.list_new_jobs(load_id)
    job = [j for j in jobs if table_substr in j][0]
    with storage.normalized_packages.storage.open_file(job, "rb") as f:
        return pq.read_table(f)


def _row(tbl: pa.Table, idx: int = 0) -> dict:
    """Materialize a single row as a plain dict for parity-style assertions."""
    return {name: tbl.column(name)[idx].as_py() for name in tbl.schema.names}


# ---------------------------------------------------------------------------
# AC parity — each case exercised as struct AND as string-JSON
# ---------------------------------------------------------------------------


def test_ac1_basic_json_flatten_struct() -> None:
    """AC1 (struct input): nested struct expanded into __ sub-columns."""
    pipeline = _make_pipeline("ac1_struct")
    item = pa.table(
        {
            "id": [1],
            "metadata": [{"name": "John", "email": "john@example.com"}],
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"metadata": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["id"] == 1
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"
    assert "metadata" not in row


def test_ac1_basic_json_flatten_string() -> None:
    """AC1 (string input): JSON string parsed and expanded into __ sub-columns."""
    pipeline = _make_pipeline("ac1_string")
    item = pa.table(
        {
            "id": [1],
            "metadata": [_json.dumps({"name": "John", "email": "john@example.com"})],
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"metadata": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["id"] == 1
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"
    assert "metadata" not in row


def test_ac2_keep_original_string() -> None:
    """AC2: original JSON string preserved alongside flattened sub-columns."""
    pipeline = _make_pipeline("ac2")
    raw = _json.dumps({"name": "John", "email": "john@example.com"})
    item = pa.table({"id": [1], "metadata": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"metadata": True}, keep_original=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["metadata"] == raw
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"


def test_ac3_path_based_with_keep_original_string() -> None:
    """AC3: only specified paths flattened; original JSON preserved."""
    pipeline = _make_pipeline("ac3")
    raw = _json.dumps({"user": {"name": "John", "age": 30}, "timestamp": "2024-01-01"})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": ["user.name"]}, keep_original=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["data"] == raw
    assert row["data__user__name"] == "John"
    assert "data__user__age" not in row
    assert "data__timestamp" not in row


def test_ac4_keep_original_without_flatten_string() -> None:
    """AC4: keep_original without flatten — value preserved as-is, no parsing."""
    pipeline = _make_pipeline("ac4")
    raw = _json.dumps({"nested": {"field": "value"}})
    item = pa.table({"id": [1], "raw_json": [raw]})

    # flatten=None with keep_original=True → no expansion, source preserved
    @dlt.resource(columns=dlt.mark.with_json_flatten({"raw_json": None}, keep_original=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["raw_json"] == raw
    assert "raw_json__nested" not in row


def test_ac5_keep_original_with_dict_native_struct() -> None:
    """AC5 (arrow struct equivalent): struct keep_original keeps the source column
    AS struct (documented Arrow-path divergence vs JSON path which serializes to text)."""
    pipeline = _make_pipeline("ac5")
    item = pa.table(
        {
            "id": [1],
            "user_profile": [{"name": "John", "email": "j@x", "settings": {"theme": "dark"}}],
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"user_profile": True}, keep_original=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    # struct keep_original — flattened columns present
    assert tbl.column("user_profile__name").to_pylist() == ["John"]
    assert tbl.column("user_profile__email").to_pylist() == ["j@x"]
    assert tbl.column("user_profile__settings__theme").to_pylist() == ["dark"]
    # documented divergence: source column kept as struct (NOT as JSON string)
    assert "user_profile" in tbl.schema.names
    assert pa.types.is_struct(tbl.schema.field("user_profile").type)


def test_ac6_path_based_only_string() -> None:
    """AC6: only specified dot-paths flattened; non-listed keys dropped."""
    pipeline = _make_pipeline("ac6")
    raw = _json.dumps({"user": {"name": "John", "email": "j@x", "age": 30}})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": ["user.name", "user.email"]}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["data__user__name"] == "John"
    assert row["data__user__email"] == "j@x"
    assert "data__user__age" not in row
    assert "data" not in row


def test_ac8_invalid_json_preserved() -> None:
    """AC8: invalid JSON kept as-is, no expansion (lossless preservation)."""
    pipeline = _make_pipeline("ac8")
    item = pa.table({"id": [1, 2], "metadata": ["not-valid-json", _json.dumps({"a": 1})]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"metadata": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    raw_vals = tbl.column("metadata").to_pylist()
    assert raw_vals[0] == "not-valid-json"
    a_vals = tbl.column("metadata__a").to_pylist()
    assert a_vals[1] == 1


def test_ac9_missing_path_silently_skipped_string() -> None:
    """AC9: paths absent from the data are silently skipped (no null column emitted)."""
    pipeline = _make_pipeline("ac9")
    raw = _json.dumps({"user": {"name": "John"}})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": ["user.name", "user.email"]}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    assert "data__user__name" in tbl.schema.names
    # missing path must NOT produce an all-null column (matches JSON path behavior)
    assert "data__user__email" not in tbl.schema.names


def test_ac10_arrow_struct_with_path_filter() -> None:
    """AC10: native arrow struct + path-based flatten — only specified paths emitted."""
    pipeline = _make_pipeline("ac10")
    item = pa.table({"id": [1], "struct_col": [{"name": "John", "age": 30}]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"struct_col": ["name"]}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["struct_col__name"] == "John"
    assert "struct_col__age" not in row


def test_multiple_json_columns_independent() -> None:
    """Multiple hinted columns with different configurations are handled independently."""
    pipeline = _make_pipeline("multi_cols")
    item = pa.table(
        {
            "id": [1],
            "metadata": [_json.dumps({"name": "John", "email": "j@x"})],
            "config": [_json.dumps({"settings": {"theme": "dark", "lang": "en"}})],
        }
    )

    @dlt.resource(
        columns=dlt.mark.with_json_flatten(
            {"metadata": True, "config": ["settings.theme"]},
            keep_original=False,
        )
    )
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "j@x"
    assert row["config__settings__theme"] == "dark"
    assert "config__settings__lang" not in row


def test_null_value_handling_string() -> None:
    """Null source value with flatten hint stays null; no sub-columns emitted."""
    pipeline = _make_pipeline("null_val")
    item = pa.table({"id": [1, 2], "metadata": [None, _json.dumps({"name": "John"})]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"metadata": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    name_vals = tbl.column("metadata__name").to_pylist()
    assert name_vals[0] is None
    assert name_vals[1] == "John"


def test_empty_object_drops_source_no_subcols_string() -> None:
    """Empty JSON object — no sub-columns and (because all values are None) the
    pyarrow `parsed_rows` of {} produces no fields; nothing to emit."""
    pipeline = _make_pipeline("empty_obj")
    item = pa.table({"id": [1, 2], "metadata": ["{}", _json.dumps({"k": "v"})]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"metadata": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    # only "k" appears; nothing exotic from the empty row
    assert "metadata__k" in tbl.schema.names
    assert tbl.column("metadata__k").to_pylist() == [None, "v"]


def test_deeply_nested_paths_string() -> None:
    """Deeply nested dot-paths are extracted; non-listed branches dropped."""
    pipeline = _make_pipeline("deep_paths")
    raw = _json.dumps({"a": {"b": {"c": {"d": "value1"}}}, "x": {"y": "value2"}, "z": "ignored"})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": ["a.b.c.d", "x.y"]}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["data__a__b__c__d"] == "value1"
    assert row["data__x__y"] == "value2"
    assert "data__z" not in row


# ---------------------------------------------------------------------------
# force_string parity
# ---------------------------------------------------------------------------


def test_force_string_scalars_string() -> None:
    """force_string coerces int/float/bool leaves to text.

    Documented divergence vs JSON path: booleans serialize as `"true"`/`"false"`
    (arrow's native cast, JSON-literal form) on the Arrow path, vs `"True"`/`"False"`
    (Python `str()`) on the row-by-row JSON path. Both are accepted dlt outputs;
    downstream consumers should normalize case if needed.
    """
    pipeline = _make_pipeline("force_str_scalar")
    raw = _json.dumps({"count": 42, "score": 3.14, "active": True, "label": "ok"})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": True}, force_string=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["data__count"] == "42"
    assert row["data__score"] == "3.14"
    # divergence — Arrow path uses JSON-literal casing for booleans
    assert row["data__active"] == "true"
    assert row["data__label"] == "ok"


def test_force_string_nested_string() -> None:
    """force_string applies at every nesting level."""
    pipeline = _make_pipeline("force_str_nested")
    raw = _json.dumps({"user": {"age": 30, "score": 9.5}})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": True}, force_string=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    row = _row(tbl)
    assert row["data__user__age"] == "30"
    assert row["data__user__score"] == "9.5"


def test_force_string_struct_input() -> None:
    """Struct-input parity for force_string: all leaf primitives become text."""
    pipeline = _make_pipeline("force_str_struct")
    item = pa.table({"id": [1], "data": [{"count": 42, "active": True}]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": True}, force_string=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    assert pa.types.is_string(tbl.schema.field("data__count").type)
    assert pa.types.is_string(tbl.schema.field("data__active").type)


# ---------------------------------------------------------------------------
# max_depth parity (divergence-aware)
# ---------------------------------------------------------------------------


def test_max_depth_1_struct_keeps_substruct_arrow_divergence() -> None:
    """max_depth=1 on struct (pyarrow engine) keeps deeper levels as struct columns —
    this is the documented Arrow-path divergence vs the JSON path's JSON-string
    serialization. Use engine='duckdb' to opt into JSON-string behavior."""
    pipeline = _make_pipeline("max_depth_struct")
    item = pa.table({"id": [1], "data": [{"a": {"b": {"c": 1}}}]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"data": True}, max_depth=1))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    assert "data__a" in tbl.schema.names
    # documented divergence: kept as struct, NOT as JSON string
    assert pa.types.is_struct(tbl.schema.field("data__a").type)


def test_max_depth_with_force_string_string_input() -> None:
    """max_depth + force_string combined: top-level keys are text leaves."""
    pipeline = _make_pipeline("max_depth_force_str")
    raw = _json.dumps({"a": 1, "b": "x", "c": True})
    item = pa.table({"id": [1], "data": [raw]})

    @dlt.resource(
        columns=dlt.mark.with_json_flatten({"data": True}, force_string=True, max_depth=1)
    )
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalized_table(pipeline, "res")
    assert pa.types.is_string(tbl.schema.field("data__a").type)
    assert pa.types.is_string(tbl.schema.field("data__b").type)
    assert pa.types.is_string(tbl.schema.field("data__c").type)
    row = _row(tbl)
    assert row["data__a"] == "1"
    assert row["data__b"] == "x"
    # arrow-path divergence: bool stringified as JSON-literal "true"/"false"
    assert row["data__c"] == "true"


# ---------------------------------------------------------------------------
# Cross-cutting: arrow path passes both arrow inputs (struct and string)
# through the SAME normalize pipeline as the JSON path produces equivalent
# canonical column sets.
# ---------------------------------------------------------------------------


def test_struct_and_string_inputs_produce_equivalent_columns() -> None:
    """Sanity: the same logical JSON content produces the same flattened
    column set whether the source column type is struct or string."""
    raw_dict = {"user": {"name": "alice", "age": 30}, "extra": "x"}

    # struct variant
    p1 = _make_pipeline("equiv_struct")
    item1 = pa.table({"id": [1], "m": [raw_dict]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"m": True}))
    def res1():
        yield item1

    p1.extract(res1(), loader_file_format="parquet")
    tbl1 = _normalized_table(p1, "res1")
    cols1 = set(tbl1.schema.names) - {"_dlt_id", "_dlt_load_id"}

    # string variant
    p2 = _make_pipeline("equiv_string")
    item2 = pa.table({"id": [1], "m": [_json.dumps(raw_dict)]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"m": True}))
    def res2():
        yield item2

    p2.extract(res2(), loader_file_format="parquet")
    tbl2 = _normalized_table(p2, "res2")
    cols2 = set(tbl2.schema.names) - {"_dlt_id", "_dlt_load_id"}

    assert cols1 == cols2, (cols1, cols2)
