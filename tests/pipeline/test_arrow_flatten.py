"""Integration tests for Arrow struct flattening with x-json-flatten hints.

Verifies that Arrow tables containing struct columns are correctly flattened
when run through a real dlt pipeline with a duckdb destination.
"""

import json as _json

import dlt
import pytest
from dlt.common.libs.pyarrow import pyarrow as pa


def _make_arrow_table():
    """Create an Arrow table with a struct column."""
    return pa.table(
        {
            "id": [1, 2],
            "data": pa.array(
                [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                type=pa.struct([pa.field("name", pa.string()), pa.field("age", pa.int64())]),
            ),
        }
    )


def _make_json_string_table():
    """Create an Arrow table with JSON string columns."""
    return pa.table(
        {
            "id": [1],
            "meta": pa.array([_json.dumps({"key": "value", "num": 42})]),
        }
    )


def test_struct_flatten_in_pipeline() -> None:
    """Arrow table with struct + x-json-flatten hint produces flattened columns in duckdb."""
    pipeline = dlt.pipeline(
        pipeline_name="test_struct_flatten",
        destination="duckdb",
        dataset_name="test_struct_flatten_dataset",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"data": True}),  # type: ignore[arg-type]
    )
    def source():
        yield _make_arrow_table()

    info = pipeline.run(source())
    assert info.loads_ids  # data was loaded

    # query duckdb to verify flattened columns
    with (
        pipeline.sql_client() as client,
        client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur,
    ):
        columns = [row[0] for row in cur.fetchall()]
        assert "data__name" in columns, f"data__name missing from {columns}"
        assert "data__age" in columns, f"data__age missing from {columns}"
        assert "data" not in columns, f"data unexpectedly in {columns}"

    with (
        pipeline.sql_client() as client,
        client.execute_query("SELECT data__name, data__age FROM test_table ORDER BY id") as cur,
    ):
        rows = cur.fetchall()
        assert rows[0][0] == "Alice"
        assert rows[0][1] == 30
        assert rows[1][0] == "Bob"
        assert rows[1][1] == 25


def test_struct_flatten_keep_original() -> None:
    """Arrow table with struct + keep_original -> original JSON string preserved."""
    pipeline = dlt.pipeline(
        pipeline_name="test_struct_keep_original",
        destination="duckdb",
        dataset_name="test_struct_keep_orig",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"data": True}, keep_original=True),  # type: ignore[arg-type]
    )
    def source():
        yield _make_arrow_table()

    info = pipeline.run(source())
    assert info.loads_ids

    with (
        pipeline.sql_client() as client,
        client.execute_query("SELECT data, data__name FROM test_table ORDER BY id") as cur,
    ):
        rows = cur.fetchall()
        assert rows[0][0] is not None  # original JSON string
        assert rows[0][1] == "Alice"


def test_string_json_flatten() -> None:
    """Arrow table with JSON string column + x-json-flatten hint produces flattened columns."""
    pipeline = dlt.pipeline(
        pipeline_name="test_string_json_flatten",
        destination="duckdb",
        dataset_name="test_string_json_flatten_ds",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"meta": True}),  # type: ignore[arg-type]
    )
    def source():
        yield _make_json_string_table()

    info = pipeline.run(source())
    assert info.loads_ids

    with (
        pipeline.sql_client() as client,
        client.execute_query("SELECT meta__key, meta__num FROM test_table") as cur,
    ):
        rows = cur.fetchall()
        assert rows[0][0] == "value"
        assert rows[0][1] == 42

    with (
        pipeline.sql_client() as client,
        client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur,
    ):
        columns = [row[0] for row in cur.fetchall()]
        assert "meta" not in columns, f"meta unexpectedly in {columns}"


def test_no_hint_no_flatten() -> None:
    """Struct column without hint -> preserved as-is (no flattening)."""
    pipeline = dlt.pipeline(
        pipeline_name="test_no_hint_arrow",
        destination="duckdb",
        dataset_name="test_no_hint_arrow_ds",
        dev_mode=True,
    )

    @dlt.resource(name="test_table", table_name="test_table")
    def source():
        yield _make_arrow_table()

    info = pipeline.run(source())
    assert info.loads_ids

    with (
        pipeline.sql_client() as client,
        client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur,
    ):
        columns = [row[0] for row in cur.fetchall()]
        assert "data__name" not in columns, f"data__name unexpectedly present in {columns}"
        assert "data__age" not in columns, f"data__age unexpectedly present in {columns}"


def test_schema_evolution() -> None:
    """Schema evolution: first run adds flattened columns, second run -> no changes."""
    pipeline = dlt.pipeline(
        pipeline_name="test_schema_evolution",
        destination="duckdb",
        dataset_name="test_schema_evolve",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"data": True}),  # type: ignore[arg-type]
    )
    def source():
        yield _make_arrow_table()

    info1 = pipeline.run(source())
    assert info1.loads_ids

    # Second run with same data
    info2 = pipeline.run(source())
    assert info2.loads_ids

    # Verify ALL rows have non-null flattened values after both runs
    with (
        pipeline.sql_client() as client,
        client.execute_query("SELECT data__name, data__age FROM test_table ORDER BY id") as cur,
    ):
        rows = cur.fetchall()
        assert len(rows) == 4, f"Expected 4 rows (2 per run), got {len(rows)}"
        for data_name, data_age in rows:
            assert data_name is not None, f"Expected non-null data__name, got {data_name}"
            assert data_age is not None, f"Expected non-null data__age, got {data_age}"

    with (
        pipeline.sql_client() as client,
        client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur,
    ):
        columns = [row[0] for row in cur.fetchall()]
        assert "data" not in columns, f"data unexpectedly in {columns}"


@pytest.mark.parametrize(
    "column_hints, test_id",
    [
        # keep_original=True without x-json-flatten: struct serialized to JSON
        # string, no children created (matches JSON normalizer keep_original-only behavior)
        ({"data": {"data_type": "json", "x-json-keep-original": True}}, "keep-original-only"),
        # max_depth=0: root struct collapsed to JSON string column
        (dlt.mark.with_json_flatten({"data": True}, max_depth=0), "max-depth-0"),
    ],
    ids=["keep-original-only", "max-depth-0"],
)
def test_struct_collapse_to_json(column_hints: dict, test_id: str) -> None:
    """Struct column collapses to JSON string without producing child columns."""
    pipeline = dlt.pipeline(
        pipeline_name=f"test_collapse_{test_id}",
        destination="duckdb",
        dataset_name=f"test_collapse_{test_id}",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=column_hints,  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"name": "Alice", "age": 30}],
                    type=pa.struct([pa.field("name", pa.string()), pa.field("age", pa.int64())]),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with (
        pipeline.sql_client() as client,
        client.execute_query(
            "SELECT column_name, data_type FROM information_schema.columns"
            " WHERE table_name='test_table' ORDER BY ordinal_position"
        ) as cur,
    ):
        rows = cur.fetchall()
        col_names = [r[0] for r in rows]
        assert "data" in col_names, f"data missing from {col_names}"
        assert "data__name" not in col_names, f"data__name unexpectedly in {col_names}"
        assert "data__age" not in col_names, f"data__age unexpectedly in {col_names}"

    with (
        pipeline.sql_client() as client,
        client.execute_query("SELECT data FROM test_table") as cur,
    ):
        rows = cur.fetchall()
        assert rows[0][0] is not None  # JSON string present


# ---------------------------------------------------------------------------
# Arrow struct flattening equivalence tests — mirrors JSON normalizer behavior
# ---------------------------------------------------------------------------


def test_path_filter_struct_keep_original() -> None:
    """Only specified struct paths are flattened; original JSON string preserved."""
    pipeline = dlt.pipeline(
        pipeline_name="test_path_filter_ko_arrow",
        destination="duckdb",
        dataset_name="test_path_filter_ko_arrow",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten(
            {"data": ["user.name"]}, keep_original=True
        ),  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"user": {"name": "John", "age": 30}, "timestamp": "2024-01-01"}],
                    type=pa.struct(
                        [
                            pa.field(
                                "user",
                                pa.struct(
                                    [pa.field("name", pa.string()), pa.field("age", pa.int64())]
                                ),
                            ),
                            pa.field("timestamp", pa.string()),
                        ]
                    ),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur:
            columns = [r[0] for r in cur.fetchall()]
            assert "data" in columns, "data missing (keep_original)"
            assert "data__user__name" in columns, "data__user__name missing"
            assert "data__user__age" not in columns, "data__user__age unexpectedly present"
            assert "data__timestamp" not in columns, "data__timestamp unexpectedly present"

        with client.execute_query("SELECT data, data__user__name FROM test_table") as cur:
            rows = cur.fetchall()
            assert rows[0][0] is not None  # original JSON
            assert rows[0][1] == "John"


def test_path_filter_struct_only() -> None:
    """Only specified struct paths are flattened; source column dropped."""
    pipeline = dlt.pipeline(
        pipeline_name="test_path_filter_arrow",
        destination="duckdb",
        dataset_name="test_path_filter_arrow",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"data": ["user.name", "user.email"]}),  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"user": {"name": "John", "email": "j@ex.com", "age": 30}}],
                    type=pa.struct(
                        [
                            pa.field(
                                "user",
                                pa.struct(
                                    [
                                        pa.field("name", pa.string()),
                                        pa.field("email", pa.string()),
                                        pa.field("age", pa.int64()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur:
            columns = [r[0] for r in cur.fetchall()]
            assert "data" not in columns, f"data unexpectedly in {columns}"
            assert "data__user__name" in columns
            assert "data__user__email" in columns
            assert "data__user__age" not in columns, f"data__user__age unexpectedly in {columns}"

        with client.execute_query(
            "SELECT data__user__name, data__user__email FROM test_table"
        ) as cur:
            rows = cur.fetchall()
            assert rows[0][0] == "John"
            assert rows[0][1] == "j@ex.com"


def test_force_string_struct() -> None:
    """force_string on struct: leaf scalars coerced to string; null preserved."""
    pipeline = dlt.pipeline(
        pipeline_name="test_force_str_arrow",
        destination="duckdb",
        dataset_name="test_force_str_arrow",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"data": True}, force_string=True),  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"count": 42, "value": None, "name": "test"}],
                    type=pa.struct(
                        [
                            pa.field("count", pa.int64()),
                            pa.field("value", pa.string()),
                            pa.field("name", pa.string()),
                        ]
                    ),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT data__count, data__value, data__name FROM test_table"
        ) as cur:
            rows = cur.fetchall()
            assert rows[0][0] == "42"  # int → str
            assert rows[0][1] is None  # null preserved
            assert rows[0][2] == "test"  # string unchanged


def test_max_depth_1_struct() -> None:
    """max_depth=1 on nested struct: top-level scalars expanded, nested struct serialized to JSON."""
    pipeline = dlt.pipeline(
        pipeline_name="test_md1_arrow",
        destination="duckdb",
        dataset_name="test_md1_arrow",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten({"data": True}, max_depth=1),  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"user": {"name": "John", "age": 30}, "count": 5}],
                    type=pa.struct(
                        [
                            pa.field(
                                "user",
                                pa.struct(
                                    [
                                        pa.field("name", pa.string()),
                                        pa.field("age", pa.int64()),
                                    ]
                                ),
                            ),
                            pa.field("count", pa.int64()),
                        ]
                    ),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur:
            columns = [r[0] for r in cur.fetchall()]
            assert "data__count" in columns
            assert "data__user" in columns
            assert (
                "data__user__name" not in columns
            ), "nested struct should be serialized, not further flattened"

        with client.execute_query("SELECT data__count, data__user FROM test_table") as cur:
            rows = cur.fetchall()
            assert rows[0][0] == 5
            assert "name" in rows[0][1]  # JSON string


def test_max_depth_1_struct_keep_original() -> None:
    """max_depth=1 + keep_original on nested struct: children expanded to boundary, original kept."""
    pipeline = dlt.pipeline(
        pipeline_name="test_md1_ko_arrow",
        destination="duckdb",
        dataset_name="test_md1_ko_arrow",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten(
            {"data": True}, keep_original=True, max_depth=1
        ),  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"user": {"name": "John", "age": 30}, "count": 5}],
                    type=pa.struct(
                        [
                            pa.field(
                                "user",
                                pa.struct(
                                    [
                                        pa.field("name", pa.string()),
                                        pa.field("age", pa.int64()),
                                    ]
                                ),
                            ),
                            pa.field("count", pa.int64()),
                        ]
                    ),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur:
            columns = [r[0] for r in cur.fetchall()]
            assert "data" in columns, "data missing (keep_original)"
            assert "data__count" in columns
            assert "data__user" in columns
            assert "data__user__name" not in columns

        with client.execute_query("SELECT data, data__count, data__user FROM test_table") as cur:
            rows = cur.fetchall()
            assert rows[0][0] is not None  # original JSON
            assert rows[0][1] == 5
            assert "name" in rows[0][2]


def test_force_string_max_depth_keep_original_combo() -> None:
    """Combo: force_string + max_depth=1 + keep_original on nested struct."""
    pipeline = dlt.pipeline(
        pipeline_name="test_combo_arrow",
        destination="duckdb",
        dataset_name="test_combo_arrow",
        dev_mode=True,
    )

    @dlt.resource(
        name="test_table",
        table_name="test_table",
        columns=dlt.mark.with_json_flatten(
            {"data": True}, keep_original=True, force_string=True, max_depth=1
        ),  # type: ignore[arg-type]
    )
    def source():
        yield pa.table(
            {
                "id": [1],
                "data": pa.array(
                    [{"user": {"name": "John", "age": 30}, "count": 5}],
                    type=pa.struct(
                        [
                            pa.field(
                                "user",
                                pa.struct(
                                    [
                                        pa.field("name", pa.string()),
                                        pa.field("age", pa.int64()),
                                    ]
                                ),
                            ),
                            pa.field("count", pa.int64()),
                        ]
                    ),
                ),
            }
        )

    info = pipeline.run(source())
    assert info.loads_ids

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT column_name FROM information_schema.columns WHERE table_name='test_table'"
        ) as cur:
            columns = [r[0] for r in cur.fetchall()]
            assert "data" in columns, "data missing (keep_original)"
            assert "data__count" in columns
            assert "data__user" in columns
            assert "data__user__name" not in columns

        with client.execute_query("SELECT data, data__count, data__user FROM test_table") as cur:
            rows = cur.fetchall()
            assert rows[0][0] is not None
            assert rows[0][1] == "5"  # force_string: int → str
            assert "name" in rows[0][2]
