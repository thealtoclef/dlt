"""Tests for JSON column expansion feature (AC1-AC10)."""

import pytest

from dlt.common.schema import Schema
from dlt.common.schema.utils import new_table
from dlt.common.normalizers.json.relational import DataItemNormalizer as RelationalNormalizer

from tests.utils import create_schema_with_name


@pytest.fixture
def norm() -> RelationalNormalizer:
    return Schema("default").data_item_normalizer  # type: ignore[return-value]


def test_ac1_basic_json_flatten(norm: RelationalNormalizer) -> None:
    """AC1: JSON string is parsed and expanded into __ sub-columns."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "metadata", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "metadata": '{"name": "John", "email": "john@example.com"}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"
    assert "metadata" not in row


def test_ac2_keep_original(norm: RelationalNormalizer) -> None:
    """AC2: Original JSON string is preserved alongside flattened sub-columns."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "metadata",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-keep-original": True,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "metadata": '{"name": "John", "email": "john@example.com"}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["metadata"] == '{"name": "John", "email": "john@example.com"}'
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"


def test_ac3_path_based_with_keep_original(norm: RelationalNormalizer) -> None:
    """AC3: Only specified paths are flattened; original JSON string is preserved."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": ["user.name"],
                    "x-json-keep-original": True,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"user": {"name": "John", "age": 30}, "timestamp": "2024-01-01"}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["data"] == '{"user": {"name": "John", "age": 30}, "timestamp": "2024-01-01"}'
    assert row["data__user__name"] == "John"
    assert "data__user__age" not in row
    assert "data__timestamp" not in row


def test_ac4_keep_original_without_flatten(norm: RelationalNormalizer) -> None:
    """AC4: keep_original without x-json-flatten on a JSON string — no parsing, value preserved as-is."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "raw_json", "data_type": "text", "x-json-keep-original": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "raw_json": '{"nested": {"field": "value"}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["raw_json"] == '{"nested": {"field": "value"}}'
    assert "raw_json__nested" not in row


def test_ac5_keep_original_with_dict_native(norm: RelationalNormalizer) -> None:
    """AC5: Native dict with keep_original only — dict is flattened normally AND original is preserved as JSON string."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "user_profile", "data_type": "json", "x-json-keep-original": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {
            "id": 1,
            "user_profile": {
                "name": "John",
                "email": "john@example.com",
                "settings": {"theme": "dark"},
            },
        },
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    # native dict is flattened normally
    assert row["user_profile__name"] == "John"
    assert row["user_profile__email"] == "john@example.com"
    assert row["user_profile__settings__theme"] == "dark"
    # original is preserved as a JSON string on the source column
    assert row["user_profile"] == '{"name":"John","email":"john@example.com","settings":{"theme":"dark"}}'


def test_ac6_path_based_only(norm: RelationalNormalizer) -> None:
    """AC6: Only specified dot-paths are flattened; all other fields are dropped."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": ["user.name", "user.email"],
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"user": {"name": "John", "email": "john@example.com", "age": 30}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["data__user__name"] == "John"
    assert row["data__user__email"] == "john@example.com"
    assert "data__user__age" not in row
    assert "data" not in row


def test_ac7_array_at_level_2(norm: RelationalNormalizer) -> None:
    """AC7: Arrays inside an expanded JSON string become a nested child table."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "metadata", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "metadata": '{"name": "John", "tags": [{"label": "vip"}, {"label": "premium"}]}'},
        "load_id",
        "test_table",
    ))
    main_row = next(r[1] for r in rows if r[0][0] == "test_table")
    tag_rows = [r[1] for r in rows if r[0][0] == "test_table__metadata__tags"]

    assert main_row["id"] == 1
    assert main_row["metadata__name"] == "John"
    assert len(tag_rows) == 2
    assert tag_rows[0]["label"] == "vip"
    assert tag_rows[1]["label"] == "premium"
    assert "_dlt_parent_id" in tag_rows[0]
    assert tag_rows[0]["_dlt_list_idx"] == 0
    assert tag_rows[1]["_dlt_list_idx"] == 1


def test_ac8_invalid_json(norm: RelationalNormalizer) -> None:
    """AC8: Invalid JSON string is kept as-is; no expansion occurs."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "metadata", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "metadata": "not-valid-json"},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["metadata"] == "not-valid-json"
    assert "metadata__name" not in row


def test_ac9_missing_path(norm: RelationalNormalizer) -> None:
    """AC9: Paths absent from the data are silently skipped."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": ["user.name", "user.email"],
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"user": {"name": "John"}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["data__user__name"] == "John"
    assert "data__user__email" not in row


def test_ac10_arrow_parquet_struct_with_path_filter(norm: RelationalNormalizer) -> None:
    """AC10: Native dict (Arrow/Parquet struct) with path-based x-json-flatten — only specified paths produced."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "struct_col", "data_type": "json", "x-json-flatten": ["name"]},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "struct_col": {"name": "John", "age": 30}},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["struct_col__name"] == "John"
    assert "struct_col__age" not in row


def test_multiple_json_columns(norm: RelationalNormalizer) -> None:
    """Multiple JSON columns with different configurations are each handled independently."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "metadata",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-keep-original": True,
                },
                {
                    "name": "config",
                    "data_type": "text",
                    "x-json-flatten": ["settings.theme"],
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {
            "id": 1,
            "metadata": '{"name": "John", "email": "john@example.com"}',
            "config": '{"settings": {"theme": "dark", "lang": "en"}}',
        },
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    # metadata: full flatten + keep original
    assert row["metadata"] == '{"name": "John", "email": "john@example.com"}'
    assert row["metadata__name"] == "John"
    assert row["metadata__email"] == "john@example.com"
    # config: path-based only, original dropped
    assert row["config__settings__theme"] == "dark"
    assert "config__settings__lang" not in row
    assert "config" not in row


def test_null_value_handling(norm: RelationalNormalizer) -> None:
    """Null column value with x-json-flatten is kept as null; no expansion."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "metadata", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "metadata": None},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["metadata"] is None
    assert "metadata__name" not in row


def test_empty_json_object(norm: RelationalNormalizer) -> None:
    """Empty JSON object with x-json-flatten produces no sub-columns and drops the source column."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "metadata", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "metadata": "{}"},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert "metadata__name" not in row
    assert "metadata" not in row


def test_nested_paths_deep(norm: RelationalNormalizer) -> None:
    """Deeply nested dot-paths are correctly extracted and non-listed keys are excluded."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": ["a.b.c.d", "x.y"],
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {
            "id": 1,
            "data": '{"a": {"b": {"c": {"d": "value1"}}}, "x": {"y": "value2"}, "z": "ignored"}',
        },
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["id"] == 1
    assert row["data__a__b__c__d"] == "value1"
    assert row["data__x__y"] == "value2"
    assert "data__z" not in row


# ---------------------------------------------------------------------------
# force_string tests
# ---------------------------------------------------------------------------


def test_force_string_scalars(norm: RelationalNormalizer) -> None:
    """force_string coerces int/float/bool leaf values to str."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-force-string": True,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"count": 42, "score": 3.14, "active": true, "label": "ok"}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["data__count"] == "42"
    assert row["data__score"] == "3.14"
    assert row["data__active"] == "True"
    assert row["data__label"] == "ok"  # already a string — unchanged


def test_force_string_nested(norm: RelationalNormalizer) -> None:
    """force_string coerces leaf scalars at every nesting level."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-force-string": True,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"user": {"age": 30, "score": 9.5}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["data__user__age"] == "30"
    assert row["data__user__score"] == "9.5"


def test_force_string_type_compatibility(norm: RelationalNormalizer) -> None:
    """Two rows where the same key has different value types produce no type conflict with force_string."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-force-string": True,
                },
            ],
        )
    )
    norm._reset()

    # first row: count is int
    rows1 = list(norm.normalize_data_item(
        {"id": 1, "data": '{"count": 42}'},
        "load_id",
        "test_table",
    ))
    # second row: count is string
    rows2 = list(norm.normalize_data_item(
        {"id": 2, "data": '{"count": "hello"}'},
        "load_id",
        "test_table",
    ))

    assert rows1[0][1]["data__count"] == "42"
    assert rows2[0][1]["data__count"] == "hello"
    # both values are str — schema would infer data_type: "text" for both, no conflict


def test_force_string_preserves_null(norm: RelationalNormalizer) -> None:
    """force_string leaves None values as None."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-force-string": True,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"value": null, "count": 5}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["data__value"] is None
    assert row["data__count"] == "5"


def test_force_string_false_is_default(norm: RelationalNormalizer) -> None:
    """Without force_string, scalar types are preserved as-is (existing behaviour)."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "data", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"count": 42, "score": 3.14}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["data__count"] == 42       # int, not string
    assert row["data__score"] == 3.14     # float, not string


# ---------------------------------------------------------------------------
# max_depth tests
# ---------------------------------------------------------------------------


def test_max_depth_1(norm: RelationalNormalizer) -> None:
    """max_depth=1 serializes all dict values at the top level to JSON strings."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-max-depth": 1,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"user": {"name": "John", "age": 30}, "count": 5}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    # scalar at depth 1 — expanded normally
    assert row["data__count"] == 5
    # dict at depth 1 — serialised to JSON string, NOT further flattened
    assert isinstance(row["data__user"], str)
    assert "name" in row["data__user"]
    assert "data__user__name" not in row


def test_max_depth_2(norm: RelationalNormalizer) -> None:
    """max_depth=2 expands one level and serializes dicts at the second level."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-max-depth": 2,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"a": {"b": {"c": "deep"}, "val": 1}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    # scalar at depth 2 — expanded normally
    assert row["data__a__val"] == 1
    # dict at depth 2 — serialised to JSON string
    assert isinstance(row["data__a__b"], str)
    assert "c" in row["data__a__b"]
    assert "data__a__b__c" not in row


def test_max_depth_none_is_unlimited(norm: RelationalNormalizer) -> None:
    """max_depth=None (default) leaves deep flattening unchanged."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {"name": "data", "data_type": "text", "x-json-flatten": True},
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"a": {"b": {"c": {"d": "value"}}}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    assert row["data__a__b__c__d"] == "value"


def test_max_depth_with_path_filter(norm: RelationalNormalizer) -> None:
    """max_depth and path-based x-json-flatten compose correctly.

    Path filter extracts {"a": {"x": 1, "y": {"z": 2}}}. With max_depth=2
    the top-level "a" dict is traversed (depth 1), its scalar "x" is expanded,
    and its nested dict "y" sits at depth 2 and is serialized to a JSON string.
    """
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": ["a"],
                    "x-json-flatten-max-depth": 2,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"a": {"x": 1, "y": {"z": 2}}, "b": "ignored"}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    # scalar at depth 2 inside "a" — expanded normally
    assert row["data__a__x"] == 1
    # dict at depth 2 inside "a" — serialised to JSON string
    assert isinstance(row["data__a__y"], str)
    assert "data__a__y__z" not in row
    # "b" excluded by path filter
    assert "data__b" not in row


def test_max_depth_with_force_string(norm: RelationalNormalizer) -> None:
    """max_depth is applied first, then force_string coerces remaining scalars."""
    norm.schema.update_table(
        new_table(
            "test_table",
            columns=[
                {"name": "id", "data_type": "bigint"},
                {
                    "name": "data",
                    "data_type": "text",
                    "x-json-flatten": True,
                    "x-json-flatten-max-depth": 1,
                    "x-json-flatten-force-string": True,
                },
            ],
        )
    )
    norm._reset()

    rows = list(norm.normalize_data_item(
        {"id": 1, "data": '{"count": 42, "user": {"name": "John"}}'},
        "load_id",
        "test_table",
    ))
    row = rows[0][1]

    # scalar at depth 1 — force_string coerces to str
    assert row["data__count"] == "42"
    # dict at depth 1 — limit_depth serialises it; force_string sees a string, no-op
    assert isinstance(row["data__user"], str)
    assert "data__user__name" not in row
