"""Module with mark functions that make data to be specially processed"""
from typing import Any, Dict, List, Optional, Union

from dlt.extract import (
    make_hints,
    make_nested_hints,
    with_file_import,
    with_hints,
    with_table_name,
    materialize_schema_item as materialize_table_schema,
)


def with_json_flatten(
    columns: Dict[str, Union[bool, List[str]]],
    keep_original: bool = False,
    force_string: bool = False,
    max_depth: Optional[int] = None,
    schema_inference: Optional[Union[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Creates column hints for JSON string expansion.

    Produces a hints dict compatible with `apply_hints()` or `make_hints(columns=...)`.

    Args:
        columns (Dict[str, Union[bool, List[str]]]): Mapping of column names to their
            flatten specification. Pass `True` to flatten the entire JSON object, or a
            list of dot-paths (e.g. `["user.name", "user.email"]`) to flatten selectively.
        keep_original (bool): When `True`, the raw JSON string is preserved on the
            source column alongside the flattened sub-columns.
        force_string (bool): When `True`, all scalar values produced by the expansion
            are coerced to `str`. Prevents type-conflict failures when the same JSON
            key carries different value types across rows (e.g. `42` then `"42"`).
        max_depth (Optional[int]): Maximum nesting levels to expand. Dicts at or beyond
            this depth are serialised to a JSON string instead of being further flattened.
            `None` means unlimited (default dlt behaviour).
        schema_inference (Optional[Union[str, int]]): Schema inference mode for string
            JSON columns on the Arrow normalize path. `"incremental"` (default) discovers
            new keys batch by batch and matches the row-by-row JSON path semantics.
            `"full-scan"` pre-scans the column to lock a union schema before flattening.
            An `int` is interpreted as a fixed sample size (may miss late-appearing keys).
            Has no effect on the row-by-row JSON path.

    Returns:
        Dict[str, Dict[str, Any]]: Column hints dict with `x-json-flatten` and
            related entries per column.

    Example:
        >>> import dlt
        >>> @dlt.resource(
        ...     apply_hints=dlt.mark.with_json_flatten(
        ...         {"metadata": True, "data": ["user.name", "user.email"]},
        ...         keep_original=True,
        ...         force_string=True,
        ...         max_depth=2,
        ...     )
        ... )
        ... def my_resource():
        ...     yield {"metadata": '{"name": "John"}'}
    """
    hints: Dict[str, Dict[str, Any]] = {}
    for col, spec in columns.items():
        col_hints: Dict[str, Any] = {"x-json-flatten": spec}
        if keep_original:
            col_hints["x-json-keep-original"] = keep_original
        if force_string:
            col_hints["x-json-flatten-force-string"] = force_string
        if max_depth is not None:
            col_hints["x-json-flatten-max-depth"] = max_depth
        if schema_inference is not None:
            col_hints["x-json-flatten-schema-inference"] = schema_inference
        hints[col] = col_hints
    return hints


__all__ = [
    "with_table_name",
    "with_hints",
    "with_file_import",
    "make_hints",
    "make_nested_hints",
    "materialize_table_schema",
    "with_json_flatten",
]
