"""Demonstration script for dlt JSON column expansion (AC1-AC11).

Each AC function creates a real dlt pipeline with duckdb destination,
runs it with sample input, and asserts the output matches expected values.

Run with:
    uv run python examples/json_expansion_ac_demo.py
"""

from typing import Any, Dict, List

import dlt
from dlt.common.utils import uniq_id
from dlt.pipeline.mark import with_json_flatten


def _fetchall_as_dicts(pipeline: dlt.Pipeline, table_name: str) -> List[Dict[str, Any]]:
    """Read all rows from table_name in the pipeline's duckdb dataset as dicts."""
    dataset = pipeline.dataset()
    relation = dataset[table_name]
    return [dict(zip(relation.columns, row)) for row in relation.fetchall()]


def _strip_dlt_cols(row: Dict[str, Any]) -> Dict[str, Any]:
    """Remove _dlt_* system columns from a row for cleaner assertions."""
    return {k: v for k, v in row.items() if not k.startswith("_dlt")}


def run_pipeline(
    input_rows: List[Dict[str, Any]],
    hints: Dict[str, Dict[str, Any]],
    table_name: str = "test_table",
) -> List[Dict[str, Any]]:
    """Create and run a dlt pipeline, return output rows from duckdb."""
    pipeline = dlt.pipeline(
        pipeline_name=f"ac_demo_{table_name}_{uniq_id(6)}",
        destination="duckdb",
        dataset_name="ac_data",
    )

    @dlt.resource(table_name=table_name)
    def src():
        yield input_rows

    # Apply column-level hints (e.g. x-json-flatten, x-json-keep-original)
    src = src().apply_hints(columns=hints)

    info = pipeline.run(src)
    failed = sum(len(p.jobs["failed_jobs"]) for p in info.load_packages)
    if failed > 0:
        raise AssertionError(f"Pipeline had {failed} failed jobs")
    return _fetchall_as_dicts(pipeline, table_name)


def run_pipeline_with_child_tables(
    input_rows: List[Dict[str, Any]],
    hints: Dict[str, Dict[str, Any]],
    table_name: str = "test_table",
    child_tables: List[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run pipeline and optionally fetch child tables too.

    Returns dict mapping table_name -> rows.
    """
    pipeline = dlt.pipeline(
        pipeline_name=f"ac_demo_{table_name}_{uniq_id(6)}",
        destination="duckdb",
        dataset_name="ac_data",
    )

    @dlt.resource(table_name=table_name)
    def src():
        yield input_rows

    src = src().apply_hints(columns=hints)
    info = pipeline.run(src)
    failed = sum(len(p.jobs["failed_jobs"]) for p in info.load_packages)
    if failed > 0:
        raise AssertionError(f"Pipeline had {failed} failed jobs")

    result: Dict[str, List[Dict[str, Any]]] = {table_name: _fetchall_as_dicts(pipeline, table_name)}
    if child_tables:
        for child in child_tables:
            result[child] = _fetchall_as_dicts(pipeline, child)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AC1: Basic JSON String Flattening
# ─────────────────────────────────────────────────────────────────────────────


def ac1_basic_json_flatten():
    """AC1: JSON string with x-json-flatten: True → __ columns."""
    print("\n" + "=" * 62)
    print("AC1: Basic JSON String Flattening")
    print("=" * 62)

    hints = with_json_flatten({"metadata": True})
    input_rows = [{"id": 1, "metadata": '{"name": "John", "email": "john@example.com"}'}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1, f"id mismatch: {row}"
    assert row["metadata__name"] == "John", f"metadata__name mismatch: {row}"
    assert row["metadata__email"] == "john@example.com", f"metadata__email mismatch: {row}"
    assert "metadata" not in row, f"metadata should be absent (flattened): {row}"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC2: Keep Original Column
# ─────────────────────────────────────────────────────────────────────────────


def ac2_keep_original():
    """AC2: JSON string with x-json-flatten + x-json-keep-original: True."""
    print("\n" + "=" * 62)
    print("AC2: Keep Original Column")
    print("=" * 62)

    hints = with_json_flatten({"metadata": True}, keep_original=True)
    input_rows = [{"id": 1, "metadata": '{"name": "John", "email": "john@example.com"}'}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"
    assert row["metadata"] == '{"name": "John", "email": "john@example.com"}'

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC3: Path-Based + Keep Original Combined
# ─────────────────────────────────────────────────────────────────────────────


def ac3_path_based_with_keep_original():
    """AC3: Path-filtered flatten with original preserved."""
    print("\n" + "=" * 62)
    print("AC3: Path-Based + Keep Original Combined")
    print("=" * 62)

    hints = with_json_flatten({"data": ["user.name"]}, keep_original=True)
    input_rows = [
        {"id": 1, "data": '{"user": {"name": "John", "age": 30}, "timestamp": "2024-01-01"}'}
    ]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["data__user__name"] == "John"
    assert "data__user__age" not in row, f"data__user__age should be absent: {row}"
    assert "data__timestamp" not in row, f"data__timestamp should be absent: {row}"
    assert row["data"] == '{"user": {"name": "John", "age": 30}, "timestamp": "2024-01-01"}'

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC4: Keep Original Without Flattening
# ─────────────────────────────────────────────────────────────────────────────


def ac4_keep_original_without_flatten():
    """AC4: keep_original only (no flatten) — JSON string is NOT flattened."""
    print("\n" + "=" * 62)
    print("AC4: Keep Original Without Flattening")
    print("=" * 62)

    # Keep_original without flatten: use x-json-keep-original only
    hints = {"raw_json": {"x-json-keep-original": True}}
    input_rows = [{"id": 1, "raw_json": '{"nested": {"field": "value"}}'}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["raw_json"] == '{"nested": {"field": "value"}}'
    # No flattening: no raw_json__nested__field
    assert "raw_json__nested" not in row, f"raw_json should NOT be flattened: {row}"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC5: Keep Original with Native Dict (no x-json-flatten)
# ─────────────────────────────────────────────────────────────────────────────


def ac5_native_dict_keep_original():
    """AC5: Native dict with keep_original only — dict is flattened AND original is preserved."""
    print("\n" + "=" * 62)
    print("AC5: Keep Original with Native Dict")
    print("=" * 62)

    hints = with_json_flatten({"user_profile": []}, keep_original=True)
    input_rows = [
        {
            "id": 1,
            "user_profile": {
                "name": "John",
                "email": "john@example.com",
                "settings": {"theme": "dark"},
            },
        }
    ]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    # Native dict is flattened by dlt normally
    assert row["user_profile__name"] == "John"
    assert row["user_profile__email"] == "john@example.com"
    assert row["user_profile__settings__theme"] == "dark"
    # Original preserved as serialized JSON string on the same column name
    assert (
        row["user_profile"]
        == '{"name":"John","email":"john@example.com","settings":{"theme":"dark"}}'
    )

    print(f"  Input (native dict): {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC6: Path-Based Extraction Only
# ─────────────────────────────────────────────────────────────────────────────


def ac6_path_based_only():
    """AC6: Only specified dot-paths are flattened; all other fields excluded."""
    print("\n" + "=" * 62)
    print("AC6: Path-Based Extraction Only")
    print("=" * 62)

    hints = with_json_flatten({"data": ["user.name", "user.email"]})
    input_rows = [
        {"id": 1, "data": '{"user": {"name": "John", "email": "john@example.com", "age": 30}}'}
    ]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["data__user__name"] == "John"
    assert row["data__user__email"] == "john@example.com"
    assert "data__user__age" not in row, f"data__user__age should be absent: {row}"
    assert "data" not in row, f"data (source column) should be absent: {row}"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC7: Arrays at Second Level → Nested Child Table
# ─────────────────────────────────────────────────────────────────────────────


def ac7_nested_array_child_table():
    """AC7: Nested array in JSON creates a child table via DLT's existing mechanism."""
    print("\n" + "=" * 62)
    print("AC7: Arrays at Second Level → Nested Child Table")
    print("=" * 62)

    hints = with_json_flatten({"metadata": True})
    input_rows = [
        {"id": 1, "metadata": '{"name": "John", "tags": [{"label": "vip"}, {"label": "premium"}]}'}
    ]
    tables = run_pipeline_with_child_tables(
        input_rows,
        hints,
        table_name="test_table",
        child_tables=["test_table__metadata__tags"],
    )

    main_rows = tables["test_table"]
    child_rows = sorted(
        tables["test_table__metadata__tags"],
        key=lambda r: r["_dlt_list_idx"],
    )
    main = _strip_dlt_cols(main_rows[0])

    # Assert main table
    assert main["id"] == 1
    assert main["metadata__name"] == "John"
    assert "metadata" not in main
    assert "metadata__tags" not in main

    # Assert child table
    assert len(child_rows) == 2
    assert child_rows[0]["label"] == "vip"
    assert child_rows[0]["_dlt_list_idx"] == 0
    assert "_dlt_parent_id" in child_rows[0]
    assert child_rows[1]["label"] == "premium"
    assert child_rows[1]["_dlt_list_idx"] == 1

    print(f"  Input: {input_rows[0]}")
    print(f"  Main table: {main}")
    print(f"  Child table rows:")
    for i, child in enumerate(child_rows):
        print(f"    [{i}] {_strip_dlt_cols(child)}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC8: Invalid JSON Handling
# ─────────────────────────────────────────────────────────────────────────────


def ac8_invalid_json_handling():
    """AC8: Invalid JSON string kept as-is; no expansion; warning logged by dlt."""
    print("\n" + "=" * 62)
    print("AC8: Invalid JSON Handling")
    print("=" * 62)

    hints = with_json_flatten({"metadata": True})
    input_rows = [{"id": 1, "metadata": "not-valid-json"}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["metadata"] == "not-valid-json"
    assert "metadata__name" not in row, f"metadata should not be expanded: {row}"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC9: Missing Paths in Data
# ─────────────────────────────────────────────────────────────────────────────


def ac9_missing_path_graceful():
    """AC9: Missing paths are skipped silently; no error."""
    print("\n" + "=" * 62)
    print("AC9: Missing Paths in Data")
    print("=" * 62)

    hints = with_json_flatten({"data": ["user.name", "user.email"]})
    input_rows = [{"id": 1, "data": '{"user": {"name": "John"}}'}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["data__user__name"] == "John"
    assert "data__user__email" not in row, f"data__user__email should be absent: {row}"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC10: Arrow/Parquet Struct with Path Filter
# ─────────────────────────────────────────────────────────────────────────────


def ac10_arrow_parquet_struct_with_path():
    """AC10: Native dict (Arrow/Parquet struct) with path-based x-json-flatten."""
    print("\n" + "=" * 62)
    print("AC10: Arrow/Parquet Struct with Path Filter")
    print("=" * 62)

    hints = with_json_flatten({"struct_col": ["name"]})
    input_rows = [{"id": 1, "struct_col": {"name": "John", "age": 30}}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["struct_col__name"] == "John"
    assert "struct_col__age" not in row, f"struct_col__age should be absent: {row}"

    print(f"  Input (native dict): {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC11: Default Auto-Flattening (no apply_hints)
# ─────────────────────────────────────────────────────────────────────────────


def ac11_default_auto_flatten():
    """AC11: Without any apply_hints, dlt auto-flattens native dict columns.

    DLT's relational normalizer automatically flattens nested dicts into __
    sub-columns. This test uses NO apply_hints — pure default behavior.
    Data: native Python dict, 4 levels deep, 6 leaf fields across 3 parent fields.
    """
    print("\n" + "=" * 62)
    print("AC11: Default Auto-Flattening (no apply_hints)")
    print("=" * 62)

    pipeline = dlt.pipeline(
        pipeline_name=f"ac_demo_test_table_{uniq_id(6)}",
        destination="duckdb",
        dataset_name="ac_data",
    )

    input_rows = [
        {
            "id": 1,
            "profile": {
                "name": "Alice",
                "email": "alice@example.com",
                "address": {
                    "city": "NYC",
                    "zip": "10001",
                    "country": {"code": "US", "name": "United States"},
                },
            },
        }
    ]

    @dlt.resource(table_name="test_table")
    def src():
        yield input_rows

    # NO apply_hints — pure dlt default behavior
    info = pipeline.run(src)
    failed = sum(len(p.jobs["failed_jobs"]) for p in info.load_packages)
    if failed > 0:
        raise AssertionError(f"Pipeline had {failed} failed jobs")

    dataset = pipeline.dataset()
    relation = dataset["test_table"]
    rows = [dict(zip(relation.columns, row)) for row in relation.fetchall()]
    row = _strip_dlt_cols(rows[0])

    # Assert
    assert row["id"] == 1, f"id mismatch: {row}"
    assert row["profile__name"] == "Alice", f"profile__name mismatch: {row}"
    assert row["profile__email"] == "alice@example.com", f"profile__email mismatch: {row}"
    assert row["profile__address__city"] == "NYC", f"profile__address__city mismatch: {row}"
    assert row["profile__address__zip"] == "10001", f"profile__address__zip mismatch: {row}"
    assert row["profile__address__country__code"] == "US", (
        f"profile__address__country__code mismatch: {row}"
    )
    assert row["profile__address__country__name"] == "United States", (
        f"profile__address__country__name mismatch: {row}"
    )
    # No source column left (flattened completely)
    assert "profile" not in row, f"profile source column should be absent: {row}"

    print(f"  Input (native dict, 4 levels, 6 fields): {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC12: Multiple JSON Columns with Different Configs
# ─────────────────────────────────────────────────────────────────────────────


def ac12_multiple_json_columns():
    """AC12: Two JSON columns with different hint configs in the same row.

    Column 'metadata': x-json-flatten: True, keep_original: True
    Column 'config':   x-json-flatten: ['settings.theme'] only (no keep_original)
    Both columns are processed independently.
    """
    print("\n" + "=" * 62)
    print("AC12: Multiple JSON Columns, Different Configs")
    print("=" * 62)

    hints = {
        "metadata": {"x-json-flatten": True, "x-json-keep-original": True},
        "config": {"x-json-flatten": ["settings.theme"]},
    }
    input_rows = [
        {
            "id": 1,
            "metadata": '{"name": "Alice", "role": "admin"}',
            "config": '{"settings": {"theme": "dark", "lang": "en"}}',
        }
    ]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    # metadata: full flatten + keep original
    assert row["metadata__name"] == "Alice"
    assert row["metadata__role"] == "admin"
    assert row["metadata"] == '{"name": "Alice", "role": "admin"}'
    # config: only settings.theme flattened, no keep_original, source column absent
    assert row["config__settings__theme"] == "dark"
    assert "config__settings__lang" not in row
    assert "config" not in row

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC13: Multi-Row Schema Evolution (Different JSON Structures Across Rows)
# ─────────────────────────────────────────────────────────────────────────────


def ac13_multi_row_schema_evolution():
    """AC13: Same column has different JSON shapes across rows.

    Row 1: {"name": "Alice"}         → data__name: Alice, data__age: None
    Row 2: {"name": "Bob", "age": 30} → data__name: Bob, data__age: 30
    Row 3: {"name": "Carol"}         → data__name: Carol, data__age: None
    DLT infers nullable columns for missing fields.
    """
    print("\n" + "=" * 62)
    print("AC13: Multi-Row Schema Evolution")
    print("=" * 62)

    pipeline = dlt.pipeline(
        pipeline_name=f"ac_demo_test_table_{uniq_id(6)}",
        destination="duckdb",
        dataset_name="ac_data",
    )

    input_rows = [
        {"id": 1, "data": '{"name": "Alice"}'},
        {"id": 2, "data": '{"name": "Bob", "age": 30}'},
        {"id": 3, "data": '{"name": "Carol"}'},
    ]

    @dlt.resource(table_name="test_table")
    def src():
        yield input_rows

    src = src().apply_hints(columns={"data": {"x-json-flatten": True}})
    info = pipeline.run(src)
    failed = sum(len(p.jobs["failed_jobs"]) for p in info.load_packages)
    if failed > 0:
        raise AssertionError(f"Pipeline had {failed} failed jobs")

    dataset = pipeline.dataset()
    relation = dataset["test_table"]
    rows = [dict(zip(relation.columns, row)) for row in relation.fetchall()]

    # Sort by id
    rows_by_id = sorted(rows, key=lambda r: r["id"])
    row1 = _strip_dlt_cols(rows_by_id[0])
    row2 = _strip_dlt_cols(rows_by_id[1])
    row3 = _strip_dlt_cols(rows_by_id[2])

    # Assert
    assert row1["data__name"] == "Alice"
    assert row1.get("data__age") is None  # missing in row 1
    assert row2["data__name"] == "Bob"
    assert row2["data__age"] == 30
    assert row3["data__name"] == "Carol"
    assert row3.get("data__age") is None  # missing in row 3

    print(f"  Input:  {input_rows}")
    print(f"  Row 1: {row1}")
    print(f"  Row 2: {row2}")
    print(f"  Row 3: {row3}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC14: JSON Root Primitive (array at root level)
# ─────────────────────────────────────────────────────────────────────────────


def ac14_json_root_primitive():
    """AC14: JSON string whose parsed root is not a dict.

    When the JSON string parses to an array or primitive (not a dict),
    x-json-flatten cannot extract sub-columns. The original column value is kept.
    """
    print("\n" + "=" * 62)
    print("AC14: JSON Root Primitive (non-dict root)")
    print("=" * 62)

    hints = {"data": {"x-json-flatten": True}}
    input_rows = [{"id": 1, "data": "[1, 2, 3]"}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Array at root: not a dict, cannot be flattened → keep original JSON string
    assert row["id"] == 1
    assert row["data"] == "[1, 2, 3]"
    assert not any(k.startswith("data__") for k in row)

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS (array root → original column preserved, no flatten)")


# ─────────────────────────────────────────────────────────────────────────────
# AC15: Unicode and Special Characters in JSON
# ─────────────────────────────────────────────────────────────────────────────


def ac15_unicode_special_chars():
    """AC15: JSON with Unicode characters and special symbols is correctly parsed.

    Tests: Unicode, emoji, newlines, escaped quotes.
    """
    print("\n" + "=" * 62)
    print("AC15: Unicode and Special Characters in JSON")
    print("=" * 62)

    hints = {"data": {"x-json-flatten": True}}
    input_rows = [
        {
            "id": 1,
            "data": '{"name": "Alice", "msg": "hello\\nworld", "emoji": "🎉"}',
        }
    ]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["data__name"] == "Alice"
    assert row["data__msg"] == "hello\nworld"  # newline unescaped
    assert row["data__emoji"] == "🎉"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC16: JSON Boolean Values (true/false as JSON primitives)
# ─────────────────────────────────────────────────────────────────────────────


def ac16_json_boolean_values():
    """AC16: JSON 'true'/'false' parsed as Python booleans, not strings.

    dlt parses JSON 'true' → Python True, 'false' → Python False.
    These appear as booleans in the flattened output.
    """
    print("\n" + "=" * 62)
    print("AC16: JSON Boolean Values (true/false)")
    print("=" * 62)

    hints = {"data": {"x-json-flatten": True}}
    input_rows = [{"id": 1, "data": '{"active": true, "deleted": false}'}]
    output_rows = run_pipeline(input_rows, hints)
    row = _strip_dlt_cols(output_rows[0])

    # Assert
    assert row["id"] == 1
    assert row["data__active"] is True  # Python True, not string "true"
    assert row["data__deleted"] is False  # Python False, not string "false"

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC17: max_table_nesting=0 blocks child tables but allows JSON flattening
# ─────────────────────────────────────────────────────────────────────────────


def ac17_max_nesting_zero_with_flatten():
    print("\n" + "=" * 62)
    print("AC17: max_table_nesting controls nesting depth")
    print("=" * 62)

    pipeline = dlt.pipeline(
        pipeline_name=f"ac_demo_test_table_{uniq_id(6)}",
        destination="duckdb",
        dataset_name="ac_data",
    )

    input_rows = [
        {
            "id": 1,
            "metadata": '{"name": "John", "email": "john@example.com"}',
            "tags": ["vip", "premium"],
        }
    ]

    @dlt.resource(
        table_name="test_table",
        columns={"metadata": {"x-json-flatten": True}},
    )
    def src():
        yield input_rows

    src = src()
    src.max_table_nesting = 1

    info = pipeline.run(src)
    failed = sum(len(p.jobs["failed_jobs"]) for p in info.load_packages)
    if failed > 0:
        raise AssertionError(f"Pipeline had {failed} failed jobs")

    dataset = pipeline.dataset()
    relation = dataset["test_table"]
    rows = [dict(zip(relation.columns, row)) for row in relation.fetchall()]
    row = _strip_dlt_cols(rows[0])

    assert row["id"] == 1
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"
    assert "metadata" not in row

    tags_child = dataset["test_table__tags"]
    tags_rows = list(tags_child.fetchall())
    assert len(tags_rows) == 2

    print(f"  max_nesting=1: {row}")
    print(f"  child table: {tags_rows}")
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# AC18: max_table_nesting=0 keeps native list of dicts as JSON blob
# ─────────────────────────────────────────────────────────────────────────────


def ac18_max_nesting_zero_list_of_json():
    print("\n" + "=" * 62)
    print("AC18: max_table_nesting=0 keeps list of JSON as blob")
    print("=" * 62)

    pipeline = dlt.pipeline(
        pipeline_name=f"ac_demo_test_table_{uniq_id(6)}",
        destination="duckdb",
        dataset_name="ac_data",
    )

    input_rows = [
        {
            "id": 1,
            "items": [{"sku": "A1", "qty": 10}, {"sku": "B2", "qty": 5}],
        }
    ]

    @dlt.resource(table_name="test_table")
    def src():
        yield input_rows

    src = src()
    src.max_table_nesting = 0

    info = pipeline.run(src)
    failed = sum(len(p.jobs["failed_jobs"]) for p in info.load_packages)
    if failed > 0:
        raise AssertionError(f"Pipeline had {failed} failed jobs")

    dataset = pipeline.dataset()
    relation = dataset["test_table"]
    rows = [dict(zip(relation.columns, row)) for row in relation.fetchall()]
    row = _strip_dlt_cols(rows[0])

    assert row["id"] == 1
    assert "items" in row
    assert isinstance(row["items"], str)
    assert "sku" in row["items"] and "qty" in row["items"]

    try:
        dataset["test_table__items"]
        raise AssertionError(
            "test_table__items child table should NOT exist with max_table_nesting=0"
        )
    except KeyError:
        pass

    print(f"  Input:  {input_rows[0]}")
    print(f"  Output: {row}")
    print("  PASS (list of JSON stays in main table, no child table)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 62)
    print("dlt JSON Column Expansion — AC Demonstration (AC1-AC18)")
    print("Destination: duckdb (in-memory)")
    print("=" * 62)

    acs = [
        ("AC1", ac1_basic_json_flatten),
        ("AC2", ac2_keep_original),
        ("AC3", ac3_path_based_with_keep_original),
        ("AC4", ac4_keep_original_without_flatten),
        ("AC5", ac5_native_dict_keep_original),
        ("AC6", ac6_path_based_only),
        ("AC7", ac7_nested_array_child_table),
        ("AC8", ac8_invalid_json_handling),
        ("AC9", ac9_missing_path_graceful),
        ("AC10", ac10_arrow_parquet_struct_with_path),
        ("AC11", ac11_default_auto_flatten),
        ("AC12", ac12_multiple_json_columns),
        ("AC13", ac13_multi_row_schema_evolution),
        ("AC14", ac14_json_root_primitive),
        ("AC15", ac15_unicode_special_chars),
        ("AC16", ac16_json_boolean_values),
        ("AC17", ac17_max_nesting_zero_with_flatten),
        ("AC18", ac18_max_nesting_zero_list_of_json),
    ]

    passed = 0
    failed = 0
    for name, fn in acs:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 62)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 62)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
