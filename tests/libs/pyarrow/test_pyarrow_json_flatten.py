"""Unit tests for the Arrow-native JSON column flatten module."""
import json as _json
from typing import Any, Dict, List

import pytest
import pyarrow as pa

from dlt.common.libs.pyarrow import NameNormalizationCollision
from dlt.common.libs.pyarrow_json_flatten import flatten_arrow_batch
from dlt.common.normalizers.json.helpers import TJsonColumnExpansionSpec
from dlt.common.normalizers.naming.snake_case import NamingConvention as SnakeCase


def _naming() -> SnakeCase:
    return SnakeCase()


def _spec(
    flatten: Any = True,
    keep_original: bool = False,
    force_string: bool = False,
    max_depth: int = None,
    schema_inference: Any = None,
) -> TJsonColumnExpansionSpec:
    return TJsonColumnExpansionSpec(
        flatten, keep_original, force_string, max_depth, schema_inference
    )


def test_struct_full_flatten_zero_copy() -> None:
    batch = pa.RecordBatch.from_pylist(
        [
            {"id": 1, "meta": {"a": 1, "b": {"c": "x"}}},
            {"id": 2, "meta": {"a": 2, "b": {"c": "y"}}},
        ]
    )
    out, partial = flatten_arrow_batch(
        batch, table_name="t", expansion_specs={"meta": _spec()}, naming=_naming()
    )
    names = set(out.schema.names)
    assert names == {"id", "meta__a", "meta__b__c"}
    assert out.column("meta__a").to_pylist() == [1, 2]
    assert out.column("meta__b__c").to_pylist() == ["x", "y"]
    assert "meta__a" in partial["columns"]


def test_struct_flatten_normalizes_camelcase_child_fields() -> None:
    """Struct sub-fields with camelCase names must be normalized before shorten_fragments
    to avoid producing unnormalized column names like `request__cardType` instead of
    `request__card_type`. Regression test for collision with pre-existing normalized columns.
    """
    batch = pa.RecordBatch.from_pylist(
        [
            {"request": {"cardType": "visa", "refId": "abc", "userId": 42}},
            {"request": {"cardType": "mc", "refId": "def", "userId": 99}},
        ]
    )
    out, partial = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"request": _spec()},
        naming=_naming(),
    )
    names = set(out.schema.names)
    assert names == {"request__card_type", "request__ref_id", "request__user_id"}
    assert out.column("request__card_type").to_pylist() == ["visa", "mc"]
    assert out.column("request__ref_id").to_pylist() == ["abc", "def"]
    assert out.column("request__user_id").to_pylist() == [42, 99]
    assert "request__card_type" in partial["columns"]
    assert "request__ref_id" in partial["columns"]
    assert "request__user_id" in partial["columns"]


def test_struct_path_list_projection() -> None:
    batch = pa.RecordBatch.from_pylist([{"meta": {"user": {"name": "a", "email": "a@x"}, "id": 1}}])
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec(flatten=["user.name", "user.email"])},
        naming=_naming(),
    )
    assert set(out.schema.names) == {"meta__user__name", "meta__user__email"}


def test_struct_missing_path_silently_skipped() -> None:
    batch = pa.RecordBatch.from_pylist([{"meta": {"user": {"name": "a"}}}])
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec(flatten=["user.name", "user.missing"])},
        naming=_naming(),
    )
    assert "meta__user__name" in out.schema.names
    assert "meta__user__missing" not in out.schema.names


def test_keep_original_on_struct_keeps_source_column() -> None:
    batch = pa.RecordBatch.from_pylist([{"meta": {"a": 1}}, {"meta": {"a": 2}}])
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec(keep_original=True)},
        naming=_naming(),
    )
    # source struct column preserved AS struct (documented divergence vs JSON path)
    assert "meta" in out.schema.names
    assert pa.types.is_struct(out.schema.field("meta").type)
    assert "meta__a" in out.schema.names


def test_force_string_casts_leaves() -> None:
    batch = pa.RecordBatch.from_pylist([{"meta": {"a": 1, "b": True}}])
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec(force_string=True)},
        naming=_naming(),
    )
    assert pa.types.is_string(out.schema.field("meta__a").type)
    assert pa.types.is_string(out.schema.field("meta__b").type)


def test_max_depth_string_json_with_duckdb_serializes_subtree_to_json_string() -> None:
    """P1: with engine='duckdb', a string-JSON column hinted `max_depth=N` produces
    a JSON-string column at depth N+1 (the documented duckdb-only divergence)."""
    pytest.importorskip("duckdb")
    from dlt.common.libs.duckdb import make_connection

    conn = make_connection(memory_limit="256MB", threads=1)
    try:
        batch = pa.RecordBatch.from_pylist(
            [
                {"raw": _json.dumps({"user": {"name": "alice", "age": 30}})},
                {"raw": _json.dumps({"user": {"name": "bob", "age": 25}})},
            ]
        )
        out, partial = flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"raw": _spec(max_depth=1)},
            naming=_naming(),
            engine="duckdb",
            duckdb_connection=conn,
        )
        # depth 1 means only top-level keys typed; deeper levels serialized to JSON text
        assert pa.types.is_string(out.schema.field("raw__user").type)
        vals = out.column("raw__user").to_pylist()
        # the values must round-trip to dicts and contain the expected keys
        parsed = [_json.loads(v) for v in vals]
        assert parsed[0]["name"] == "alice" and parsed[0]["age"] == 30
        assert parsed[1]["name"] == "bob" and parsed[1]["age"] == 25
        assert partial["columns"]["raw__user"]["data_type"] == "text"
    finally:
        conn.close()


def test_duckdb_string_json_preserves_non_object_rows_without_python_prefilter() -> None:
    """DuckDB engine preserves malformed, array, and scalar JSON roots while flattening objects."""
    pytest.importorskip("duckdb")
    from dlt.common.libs.duckdb import make_connection

    conn = make_connection(memory_limit="256MB", threads=1)
    try:
        batch = pa.RecordBatch.from_pylist(
            [
                {"raw": _json.dumps({"a": 1})},
                {"raw": "not-valid-json"},
                {"raw": "[1, 2, 3]"},
                {"raw": "42"},
                {"raw": None},
                {"raw": _json.dumps({"a": 2})},
            ]
        )
        out, _ = flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"raw": _spec()},
            naming=_naming(),
            engine="duckdb",
            duckdb_connection=conn,
        )
        assert out.column("raw__a").to_pylist() == [1, None, None, None, None, 2]
        assert out.column("raw").to_pylist() == [
            None,
            "not-valid-json",
            "[1, 2, 3]",
            "42",
            None,
            None,
        ]
    finally:
        conn.close()


def test_duckdb_string_json_preserves_all_non_object_rows() -> None:
    """If a batch has no object JSON rows, DuckDB emits no subcolumns but keeps source values."""
    pytest.importorskip("duckdb")
    from dlt.common.libs.duckdb import make_connection

    conn = make_connection(memory_limit="256MB", threads=1)
    try:
        batch = pa.RecordBatch.from_pylist(
            [{"raw": "not-valid-json"}, {"raw": "[1, 2, 3]"}, {"raw": None}]
        )
        out, partial = flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"raw": _spec()},
            naming=_naming(),
            engine="duckdb",
            duckdb_connection=conn,
        )
        assert out.schema.names == ["raw"]
        assert out.column("raw").to_pylist() == ["not-valid-json", "[1, 2, 3]", None]
        assert partial["columns"]["raw"]["data_type"] == "text"
    finally:
        conn.close()


def test_max_depth_default_keeps_substruct_as_struct() -> None:
    batch = pa.RecordBatch.from_pylist([{"meta": {"a": {"b": {"c": 1}}}}])
    out, partial = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec(max_depth=1)},
        naming=_naming(),
    )
    # depth 1 means only top-level keys; deeper kept as struct
    assert "meta__a" in out.schema.names
    assert pa.types.is_struct(out.schema.field("meta__a").type)


def test_null_struct_produces_no_subcolumns() -> None:
    batch = pa.RecordBatch.from_pylist([{"meta": None}, {"meta": None}])
    out, partial = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec()},
        naming=_naming(),
    )
    # null-typed columns are dropped without emission
    assert "meta" not in out.schema.names
    assert partial["columns"] == {} or all(c.startswith("meta__") for c in partial["columns"])


def test_collision_raises() -> None:
    # pre-existing "meta__a" column collides with what flatten would emit
    batch = pa.RecordBatch.from_pylist([{"meta__a": 99, "meta": {"a": 1}}])
    with pytest.raises(NameNormalizationCollision):
        flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"meta": _spec()},
            naming=_naming(),
        )


def test_string_column_path_list_extraction() -> None:
    rows: List[Dict[str, Any]] = [
        {"raw": _json.dumps({"user": {"name": "a", "email": "a@x"}})},
        {"raw": _json.dumps({"user": {"name": "b", "email": "b@x"}})},
    ]
    batch = pa.RecordBatch.from_pylist(rows)
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec(flatten=["user.name", "user.email"])},
        naming=_naming(),
    )
    assert out.column("raw__user__name").to_pylist() == ["a", "b"]
    assert out.column("raw__user__email").to_pylist() == ["a@x", "b@x"]


def test_string_column_path_list_extraction_duckdb_engine() -> None:
    """DuckDB engine extracts listed paths without the Python string path parser."""
    pytest.importorskip("duckdb")
    from dlt.common.libs.duckdb import make_connection

    rows: List[Dict[str, Any]] = [
        {"raw": _json.dumps({"user": {"name": "a", "age": 30}, "active": True})},
        {"raw": _json.dumps({"user": {"name": "b", "age": 31}, "active": False})},
        {"raw": "not-json"},
        {"raw": "[1, 2, 3]"},
    ]
    conn = make_connection(memory_limit="256MB", threads=1)
    try:
        batch = pa.RecordBatch.from_pylist(rows)
        out, _ = flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"raw": _spec(flatten=["user.name", "user.age", "active"])},
            naming=_naming(),
            engine="duckdb",
            duckdb_connection=conn,
        )
        assert out.column("raw__user__name").to_pylist() == ["a", "b", None, None]
        assert out.column("raw__user__age").to_pylist() == [30, 31, None, None]
        assert out.column("raw__active").to_pylist() == [True, False, None, None]
        assert out.column("raw").to_pylist() == [None, None, "not-json", "[1, 2, 3]"]
    finally:
        conn.close()


def test_string_column_path_list_duckdb_missing_path_skipped() -> None:
    pytest.importorskip("duckdb")
    from dlt.common.libs.duckdb import make_connection

    conn = make_connection(memory_limit="256MB", threads=1)
    try:
        batch = pa.RecordBatch.from_pylist(
            [{"raw": _json.dumps({"a": 1})}, {"raw": _json.dumps({"a": 2})}]
        )
        out, _ = flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"raw": _spec(flatten=["a", "missing.path"])},
            naming=_naming(),
            engine="duckdb",
            duckdb_connection=conn,
        )
        assert "raw__a" in out.schema.names
        assert "raw__missing__path" not in out.schema.names
    finally:
        conn.close()


def test_string_column_path_list_duckdb_struct_target_recurses() -> None:
    pytest.importorskip("duckdb")
    from dlt.common.libs.duckdb import make_connection

    conn = make_connection(memory_limit="256MB", threads=1)
    try:
        batch = pa.RecordBatch.from_pylist(
            [
                {"raw": _json.dumps({"user": {"name": "a", "email": "a@x"}})},
                {"raw": _json.dumps({"user": {"name": "b", "email": "b@x"}})},
            ]
        )
        out, _ = flatten_arrow_batch(
            batch,
            table_name="t",
            expansion_specs={"raw": _spec(flatten=["user"])},
            naming=_naming(),
            engine="duckdb",
            duckdb_connection=conn,
        )
        assert out.column("raw__user__name").to_pylist() == ["a", "b"]
        assert out.column("raw__user__email").to_pylist() == ["a@x", "b@x"]
    finally:
        conn.close()


def test_string_column_keep_original_preserves_text() -> None:
    rows = [{"raw": '{"a": 1}'}, {"raw": '{"a": 2}'}]
    batch = pa.RecordBatch.from_pylist(rows)
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={
            "raw": _spec(flatten=["a"], keep_original=True),
        },
        naming=_naming(),
    )
    assert out.column("raw").to_pylist() == ['{"a": 1}', '{"a": 2}']
    assert out.column("raw__a").to_pylist() == [1, 2]


def test_string_column_full_flatten_pyarrow_engine_incremental_schema() -> None:
    # batch 1: only "a"; batch 2: "a" + "b" — locking after first batch is per call,
    # so each call returns the structures present
    rows1 = [{"raw": '{"a": 1}'}, {"raw": '{"a": 2}'}]
    rows2 = [{"raw": '{"a": 3, "b": 9}'}]
    batch1 = pa.RecordBatch.from_pylist(rows1)
    batch2 = pa.RecordBatch.from_pylist(rows2)
    locks: Dict[str, Any] = {}
    out1, _ = flatten_arrow_batch(
        batch1,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
        string_schema_locks=locks,
    )
    out2, _ = flatten_arrow_batch(
        batch2,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
        string_schema_locks=locks,
    )
    assert "raw__a" in out1.schema.names
    # second batch reveals the new key — incremental schema picks it up
    assert "raw__a" in out2.schema.names
    assert "raw__b" in out2.schema.names


def test_non_record_batch_table_input_returns_table() -> None:
    table = pa.table({"meta": [{"a": 1}, {"a": 2}]})
    out, _ = flatten_arrow_batch(
        table,
        table_name="t",
        expansion_specs={"meta": _spec()},
        naming=_naming(),
    )
    assert isinstance(out, pa.Table)
    assert out.column("meta__a").to_pylist() == [1, 2]


def test_no_hinted_columns_returns_input_unchanged() -> None:
    batch = pa.RecordBatch.from_pylist([{"x": 1}])
    out, partial = flatten_arrow_batch(batch, table_name="t", expansion_specs={}, naming=_naming())
    assert out is batch
    assert partial["columns"] == {}


def test_invalid_json_string_preserves_source() -> None:
    """P1: invalid JSON strings keep their original value (JSON path parity)."""
    rows = [{"raw": '{"a": 1}'}, {"raw": "not-valid-json"}, {"raw": '{"a": 3}'}]
    batch = pa.RecordBatch.from_pylist(rows)
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
    )
    assert "raw" in out.schema.names
    raw_vals = out.column("raw").to_pylist()
    # null where parse succeeded, original where it failed
    assert raw_vals[0] is None
    assert raw_vals[1] == "not-valid-json"
    assert raw_vals[2] is None
    assert out.column("raw__a").to_pylist() == [1, None, 3]


def test_root_array_json_preserves_source() -> None:
    """P1: JSON whose root is an array (not object) keeps the original value."""
    rows = [{"raw": "[1, 2, 3]"}, {"raw": '{"a": 1}'}]
    batch = pa.RecordBatch.from_pylist(rows)
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
    )
    assert out.column("raw").to_pylist() == ["[1, 2, 3]", None]


def test_path_list_skips_columns_with_no_matches() -> None:
    """P2: requested path that never resolves to a value is dropped from output."""
    rows = [{"raw": _json.dumps({"a": 1})}, {"raw": _json.dumps({"a": 2})}]
    batch = pa.RecordBatch.from_pylist(rows)
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec(flatten=["a", "missing.path"])},
        naming=_naming(),
    )
    assert "raw__a" in out.schema.names
    assert "raw__missing__path" not in out.schema.names


def test_same_batch_mixed_types_coerce_to_text() -> None:
    """P1: mixed types at the same path within ONE batch are coerced to text rather
    than raising ArrowInvalid. This is the case the pyarrow JSON parser cannot
    handle natively."""
    rows = [
        {"raw": _json.dumps({"a": 1})},
        {"raw": _json.dumps({"a": "hello"})},
        {"raw": _json.dumps({"a": True})},
    ]
    batch = pa.RecordBatch.from_pylist(rows)
    out, partial = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
    )
    assert pa.types.is_string(out.schema.field("raw__a").type)
    assert out.column("raw__a").to_pylist() == ["1", "hello", "True"]
    assert partial["columns"]["raw__a"]["data_type"] == "text"


def test_same_batch_nested_mixed_types_coerce_to_text() -> None:
    """Nested keys with mixed types are also coerced; recursion is exercised."""
    rows = [
        {"raw": _json.dumps({"u": {"k": 1}})},
        {"raw": _json.dumps({"u": {"k": "x"}})},
    ]
    batch = pa.RecordBatch.from_pylist(rows)
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
    )
    assert pa.types.is_string(out.schema.field("raw__u__k").type)
    assert out.column("raw__u__k").to_pylist() == ["1", "x"]


def test_cross_batch_type_conflict_emits_text_in_second_batch() -> None:
    """Cross-batch (incremental mode): the first batch is already written before the
    conflict is observed, so only the second batch can be coerced. The recorded
    canonical type for the path advances to text; subsequent batches stay text."""
    locks: Dict[str, Any] = {}
    batch1 = pa.RecordBatch.from_pylist([{"raw": _json.dumps({"a": 1})}])
    batch2 = pa.RecordBatch.from_pylist([{"raw": _json.dumps({"a": "hello"})}])
    batch3 = pa.RecordBatch.from_pylist([{"raw": _json.dumps({"a": 99})}])
    out1, _ = flatten_arrow_batch(
        batch1,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
        string_schema_locks=locks,
    )
    out2, partial2 = flatten_arrow_batch(
        batch2,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
        string_schema_locks=locks,
    )
    out3, partial3 = flatten_arrow_batch(
        batch3,
        table_name="t",
        expansion_specs={"raw": _spec()},
        naming=_naming(),
        string_schema_locks=locks,
    )
    # incremental semantics: batch1 is already-emitted integer (cannot rewrite the past)
    assert pa.types.is_integer(out1.schema.field("raw__a").type)
    # from the conflicting batch onward the path is text — stable canonical schema
    assert pa.types.is_string(out2.schema.field("raw__a").type)
    assert pa.types.is_string(out3.schema.field("raw__a").type)
    assert partial2["columns"]["raw__a"]["data_type"] == "text"
    assert partial3["columns"]["raw__a"]["data_type"] == "text"


def test_full_scan_locks_text_for_mixed_types_from_start(tmp_path: Any) -> None:
    """P1: with `schema_inference='full-scan'` mixed-type paths are detected during
    pre-scan and locked to text BEFORE any batch is flattened, so even the first
    batch's data is emitted as text."""
    from dlt.common.libs.pyarrow_json_flatten import prescan_string_json_schemas

    table = pa.table(
        {
            "raw": [
                _json.dumps({"a": 1}),
                _json.dumps({"a": 2}),
                _json.dumps({"a": "later"}),
            ]
        }
    )
    fpath = tmp_path / "in.parquet"
    pa.parquet.write_table(table, fpath, row_group_size=1)
    with open(fpath, "rb") as f:
        locked = prescan_string_json_schemas(f, {"raw": _spec(schema_inference="full-scan")})
    # union struct must have raw.a as string after the full scan
    assert "raw" in locked
    field_a = locked["raw"].field(locked["raw"].get_field_index("a"))
    assert pa.types.is_string(field_a.type)

    # exercise a batch with the lock in place — first batch's integer should
    # be widened to text via the locked struct type
    batch = pa.RecordBatch.from_pylist([{"raw": _json.dumps({"a": 1})}])
    out, partial = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"raw": _spec(schema_inference="full-scan")},
        naming=_naming(),
        string_schema_locks={"raw": locked["raw"]},
    )
    assert pa.types.is_string(out.schema.field("raw__a").type)
    assert partial["columns"]["raw__a"]["data_type"] == "text"


def test_full_scan_picks_up_late_key_via_prescan(tmp_path: Any) -> None:
    """P1: full-scan mode pre-scans the entire column so late-appearing keys are
    captured before any batch is flattened."""
    from dlt.common.libs.pyarrow_json_flatten import prescan_string_json_schemas

    # write a parquet file where the rare key shows up only in the last row
    table = pa.table(
        {
            "raw": [_json.dumps({"a": 1})] * 5 + [_json.dumps({"a": 6, "rare": "x"})],
        }
    )
    fpath = tmp_path / "in.parquet"
    pa.parquet.write_table(table, fpath, row_group_size=2)
    with open(fpath, "rb") as f:
        locked = prescan_string_json_schemas(f, {"raw": _spec(schema_inference="full-scan")})
    assert "raw" in locked
    field_names = [locked["raw"].field(i).name for i in range(locked["raw"].num_fields)]
    assert "rare" in field_names


def test_int_sample_size_limits_prescan(tmp_path: Any) -> None:
    """P1: int sample size only scans the first N rows (documented as may-miss)."""
    from dlt.common.libs.pyarrow_json_flatten import prescan_string_json_schemas

    table = pa.table(
        {
            "raw": [_json.dumps({"a": 1})] * 5 + [_json.dumps({"a": 6, "rare": "x"})],
        }
    )
    fpath = tmp_path / "in.parquet"
    pa.parquet.write_table(table, fpath, row_group_size=2)
    with open(fpath, "rb") as f:
        locked = prescan_string_json_schemas(f, {"raw": _spec(schema_inference=3)})
    field_names = [locked["raw"].field(i).name for i in range(locked["raw"].num_fields)]
    assert "rare" not in field_names


def test_prescan_tolerates_columns_absent_from_file(tmp_path: Any) -> None:
    """P2: a parquet file may legitimately omit a hinted column (sparse batches);
    prescan must skip it the same way flatten does, instead of raising KeyError."""
    from dlt.common.libs.pyarrow_json_flatten import prescan_string_json_schemas

    fpath = tmp_path / "in.parquet"
    pa.parquet.write_table(pa.table({"id": [1, 2]}), fpath)
    with open(fpath, "rb") as f:
        # hinted column "raw" is absent from the file — must return {} not raise
        locked = prescan_string_json_schemas(
            f,
            {
                "raw": _spec(schema_inference="full-scan"),
                "missing2": _spec(schema_inference=10),
            },
        )
    assert locked == {}


def test_prescan_partial_overlap_with_file_columns(tmp_path: Any) -> None:
    """When some hinted columns are present and others absent, only the present
    ones contribute to the lock."""
    from dlt.common.libs.pyarrow_json_flatten import prescan_string_json_schemas

    fpath = tmp_path / "in.parquet"
    pa.parquet.write_table(
        pa.table({"id": [1], "raw": [_json.dumps({"a": 1, "b": "x"})]}),
        fpath,
    )
    with open(fpath, "rb") as f:
        locked = prescan_string_json_schemas(
            f,
            {
                "raw": _spec(schema_inference="full-scan"),
                "absent": _spec(schema_inference="full-scan"),
            },
        )
    assert "raw" in locked
    assert "absent" not in locked


def test_unhinted_column_passes_through() -> None:
    batch = pa.RecordBatch.from_pylist([{"id": 1, "meta": {"a": 1}, "other": "x"}])
    out, _ = flatten_arrow_batch(
        batch,
        table_name="t",
        expansion_specs={"meta": _spec()},
        naming=_naming(),
    )
    assert out.column("id").to_pylist() == [1]
    assert out.column("other").to_pylist() == ["x"]
