"""End-to-end Arrow-path JSON flatten tests.

Parity coverage with `tests/common/normalizers/test_json_expansion.py` for the cases
that translate to Arrow shapes. Each case is exercised through `extract → normalize`
on a duckdb pipeline so the schema-update plumbing and the rewrite path are covered.
"""
import json as _json
from typing import Any, List

import pytest
import pyarrow as pa

import dlt
from dlt.common.utils import uniq_id


def _make_pipeline(name: str) -> Any:
    return dlt.pipeline("arrow_flatten_" + uniq_id() + "_" + name, destination="duckdb")


def _normalize_and_read_first_job(pipeline: Any, table_substr: str) -> pa.Table:
    pipeline.normalize()
    load_id = pipeline.list_normalized_load_packages()[0]
    storage = pipeline._get_load_storage()
    jobs = storage.normalized_packages.list_new_jobs(load_id)
    job = [j for j in jobs if table_substr in j][0]
    with storage.normalized_packages.storage.open_file(job, "rb") as f:
        return pa.parquet.read_table(f)


def test_struct_full_flatten_arrow() -> None:
    pipeline = _make_pipeline("struct_full")
    item = pa.table(
        {
            "id": [1, 2],
            "meta": [
                {"name": "a", "email": "a@x"},
                {"name": "b", "email": "b@x"},
            ],
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    cols = set(tbl.schema.names)
    assert {"id", "meta__name", "meta__email"} <= cols
    assert "meta" not in cols
    schema_cols = pipeline.default_schema.tables["res"]["columns"]
    assert "meta__name" in schema_cols
    assert "meta__email" in schema_cols


def test_struct_path_list_flatten_arrow() -> None:
    pipeline = _make_pipeline("struct_paths")
    item = pa.table(
        {
            "raw": [{"user": {"name": "a", "email": "a@x"}, "extra": 1}],
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"raw": ["user.name"]}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert "raw__user__name" in tbl.schema.names
    assert "raw__user__email" not in tbl.schema.names


def test_struct_keep_original_keeps_struct_column() -> None:
    pipeline = _make_pipeline("struct_keep")
    item = pa.table({"meta": [{"a": 1}, {"a": 2}]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}, keep_original=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    # documented divergence: struct input + keep_original keeps the struct column
    assert "meta" in tbl.schema.names
    assert "meta__a" in tbl.schema.names


def test_string_path_list_flatten_arrow() -> None:
    pipeline = _make_pipeline("string_paths")
    item = pa.table(
        {
            "raw": [
                _json.dumps({"user": {"name": "a", "email": "a@x"}}),
                _json.dumps({"user": {"name": "b", "email": "b@x"}}),
            ]
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"raw": ["user.name", "user.email"]}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert tbl.column("raw__user__name").to_pylist() == ["a", "b"]
    assert tbl.column("raw__user__email").to_pylist() == ["a@x", "b@x"]


def test_string_keep_original_preserves_text_column() -> None:
    pipeline = _make_pipeline("string_keep")
    raw_json = '{"a": 1}'
    item = pa.table({"raw": [raw_json, raw_json]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"raw": ["a"]}, keep_original=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert "raw" in tbl.schema.names
    assert "raw__a" in tbl.schema.names
    assert tbl.column("raw").to_pylist() == [raw_json, raw_json]


def test_force_string_arrow() -> None:
    pipeline = _make_pipeline("force_string")
    item = pa.table({"meta": [{"a": 1, "b": True}]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}, force_string=True))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert pa.types.is_string(tbl.schema.field("meta__a").type)
    assert pa.types.is_string(tbl.schema.field("meta__b").type)
    assert tbl.column("meta__a").to_pylist() == ["1"]


def test_must_rewrite_when_hinted_blocks_direct_import() -> None:
    """A parquet file with no other rewrite reason still goes through the rewrite path
    when an expansion hint is present on the table."""
    import os

    os.environ["NORMALIZE__PARQUET_NORMALIZER__ADD_DLT_ID"] = "False"
    os.environ["NORMALIZE__PARQUET_NORMALIZER__ADD_DLT_LOAD_ID"] = "False"
    try:
        pipeline = _make_pipeline("must_rewrite")
        item = pa.table({"meta": [{"a": 1}]})

        @dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}))
        def res():
            yield item

        pipeline.extract(res(), loader_file_format="parquet")
        tbl = _normalize_and_read_first_job(pipeline, "res")
        # flatten happened — proof that the rewrite branch was taken
        assert "meta__a" in tbl.schema.names
    finally:
        os.environ.pop("NORMALIZE__PARQUET_NORMALIZER__ADD_DLT_ID", None)
        os.environ.pop("NORMALIZE__PARQUET_NORMALIZER__ADD_DLT_LOAD_ID", None)


def test_invalid_string_json_preserved_in_pipeline() -> None:
    """Pipeline-level proof that invalid JSON rows survive end to end."""
    pipeline = _make_pipeline("invalid_json")
    item = pa.table({"raw": [_json.dumps({"a": 1}), "not-valid-json", _json.dumps({"a": 3})]})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"raw": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert "raw" in tbl.schema.names
    assert "raw__a" in tbl.schema.names
    raw_vals = tbl.column("raw").to_pylist()
    assert raw_vals[1] == "not-valid-json"


def test_full_scan_inference_catches_late_key() -> None:
    """Late-appearing key is picked up under schema_inference='full-scan'."""
    pipeline = _make_pipeline("full_scan")
    rows = [_json.dumps({"a": i}) for i in range(5)]
    rows.append(_json.dumps({"a": 99, "rare": "x"}))
    item = pa.table({"raw": rows})

    @dlt.resource(columns=dlt.mark.with_json_flatten({"raw": True}, schema_inference="full-scan"))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert "raw__a" in tbl.schema.names
    assert "raw__rare" in tbl.schema.names


def test_flatten_applies_to_all_row_groups_in_file() -> None:
    """CRITICAL regression: a parquet file with multiple row groups must have the
    flatten hint applied to EVERY row group. The bug was: `_maybe_flatten_batch`
    deleted the hinted source column from the schema after batch 1, then
    `get_json_expansion_columns` re-read from the mutated schema on batch 2 and
    returned no hints — leaving the original struct column in subsequent row
    groups and producing a schema mismatch on parquet read."""
    import pyarrow.parquet as pq

    pipeline = _make_pipeline("multi_rg")
    item = pa.table(
        {
            "id": list(range(10)),
            "meta": [{"a": i, "b": str(i)} for i in range(10)],
        }
    )

    @dlt.resource(columns=dlt.mark.with_json_flatten({"meta": True}))
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")

    # force multiple row groups in the extracted parquet file so the normalize
    # loop iterates more than once.
    extracted_load_id = pipeline.list_extracted_load_packages()[0]
    extracted = pipeline._get_normalize_storage().extracted_packages
    jobs = extracted.list_new_jobs(extracted_load_id)
    job = [j for j in jobs if "res" in j][0]
    full = extracted.storage.make_full_path(job)
    rewritten = pq.read_table(full)
    pq.write_table(rewritten, full, row_group_size=2)
    assert pq.ParquetFile(full).num_row_groups >= 3

    tbl = _normalize_and_read_first_job(pipeline, "res")
    assert "meta" not in tbl.schema.names
    assert tbl.column("meta__a").to_pylist() == list(range(10))
    assert tbl.column("meta__b").to_pylist() == [str(i) for i in range(10)]


def test_no_hint_leaves_arrow_path_unchanged() -> None:
    """Regression guard: a parquet table without flatten hints should still work."""
    pipeline = _make_pipeline("no_hint")
    item = pa.table({"id": [1, 2], "meta": [{"a": 1}, {"a": 2}]})

    @dlt.resource
    def res():
        yield item

    pipeline.extract(res(), loader_file_format="parquet")
    tbl = _normalize_and_read_first_job(pipeline, "res")
    # meta stays as a struct since no hint was applied
    assert "meta" in tbl.schema.names
    assert pa.types.is_struct(tbl.schema.field("meta").type)
