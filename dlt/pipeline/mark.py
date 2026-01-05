"""Module with mark functions that make data to be specially processed"""
from typing import Any, Dict, List, Set, Union

from dlt.extract import (
    make_hints,
    make_nested_hints,
    with_file_import,
    with_hints,
    with_table_name,
    materialize_schema_item as materialize_table_schema,
)


def with_json_flatten(
    columns: Union[Set[str], Dict[str, Union[bool, List[str]]]],
    keep_original: bool = False,
) -> Dict[str, Any]:
    """Mark columns to be parsed from JSON strings to Python dicts.

    This enables DLT's built-in flattening and schema inference for JSON columns.
    When a JSON string column is marked with x-json-flatten, DLT will:
    1. Parse the JSON string to a Python dict
    2. Flatten nested fields using DLT's native flattening (creates `__` columns)
    3. Infer schema from the data (persisted across runs)

    Args:
        columns: Can be:
            - Set of column names: Flatten all fields in those columns
            - Dict mapping column names to:
                - True: Flatten all fields
                - List of paths: Only flatten specific paths (e.g., ["user.name", "user.email"])
        keep_original: If True, preserve the original JSON string in `{column}__original`

    Returns:
        Resource hints dict that can be passed to `apply_hints()` or used in `@dlt.resource`

    Examples:
        Basic - Flatten all fields:
        ```python
        import dlt

        @dlt.resource(apply_hints=dlt.mark.with_json_flatten({"metadata", "settings"}))
        def users():
            yield {
                "id": 1,
                "metadata": '{"name": "John", "email": "john@example.com"}',
                "settings": '{"theme": "dark"}'
            }

        # Result in destination:
        # - id: 1
        # - metadata__name: "John"
        # - metadata__email: "john@example.com"
        # - settings__theme: "dark"
        ```

        Path-based - Only flatten specific fields:
        ```python
        @dlt.resource(apply_hints=dlt.mark.with_json_flatten({
            "data": ["user.name", "user.email"]  # Only these paths
        }))
        def users():
            yield {
                "id": 1,
                "data": '{"user": {"name": "John", "email": "john@example.com", "age": 30}}'
            }

        # Result in destination:
        # - id: 1
        # - data__user__name: "John"
        # - data__user__email: "john@example.com"
        # Note: data__user__age is NOT flattened (excluded)
        ```

        Keep original - Preserve JSON string alongside flattened fields:
        ```python
        @dlt.resource(apply_hints=dlt.mark.with_json_flatten(
            {"metadata"}, keep_original=True
        ))
        def users():
            yield {"id": 1, "metadata": '{"name": "John"}'}

        # Result in destination:
        # - id: 1
        # - metadata__name: "John"
        # - metadata__original: '{"name": "John"}'  <-- Original preserved
        ```

        Using with column definitions:
        ```python
        @dlt.resource(
            columns={
                "id": {"data_type": "bigint"},
                "metadata": {
                    "data_type": "json",
                    "x-json-flatten": True,
                    "x-json-keep-original": True
                }
            }
        )
        def users():
            yield {"id": 1, "metadata": '{"name": "John"}'}
        ```

        Arrow/Parquet struct columns also work:
        ```python
        @dlt.resource(apply_hints=dlt.mark.with_json_flatten({"struct_col"}))
        def data():
            yield {
                "struct_col": pa.array([{"field1": "value1"}])
            }
        ```
    """
    column_hints = {}

    if isinstance(columns, set):
        # Simple set - flatten all fields
        for col_name in columns:
            column_hints[col_name] = {
                "data_type": "json",
                "x-json-flatten": True,
                "x-json-keep-original": keep_original,
            }
    else:
        # Dict with per-column config
        for col_name, paths in columns.items():
            column_hints[col_name] = {
                "data_type": "json",
                "x-json-flatten": paths if isinstance(paths, list) else True,
                "x-json-keep-original": keep_original,
            }

    return make_hints(columns=column_hints)


__all__ = [
    "with_table_name",
    "with_hints",
    "with_file_import",
    "make_hints",
    "make_nested_hints",
    "materialize_table_schema",
    "with_json_flatten",
]
