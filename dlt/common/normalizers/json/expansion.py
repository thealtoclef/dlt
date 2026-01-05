"""JSON column flattening for DLT normalization.

This module provides utilities for converting JSON string columns to Python dicts
so DLT's built-in flattening and schema inference can handle them.

Features:
- Parse JSON strings to Python dicts
- Selective path-based flattening (only extract specific paths)
- Keep original JSON string alongside flattened columns
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from dlt.common import logger
from dlt.common.typing import DictStrAny

TJsonFlattenHint = Union[bool, None, List[str]]
"""Type for x-json-flatten hint:
- True/None: Flatten all paths (default)
- List[str]: Only flatten specific paths (e.g., ["user.name", "user.email"])
"""


def extract_paths_from_dict(
    data: Dict[str, Any],
    paths: List[str],
) -> Dict[str, Any]:
    """Extract specific paths from a nested dict.

    Args:
        data: Source dictionary to extract from
        paths: List of dot-notation paths (e.g., ["user.name", "user.email"])

    Returns:
        New dict containing only the specified paths

    Example:
        >>> data = {"user": {"name": "John", "email": "john@example.com"}, "id": 1}
        >>> extract_paths_from_dict(data, ["user.name", "user.email"])
        {'user': {'name': 'John', 'email': 'john@example.com'}}
    """
    result: Dict[str, Any] = {}

    for path in paths:
        parts = path.split(".")
        current = data

        # Navigate to the nested value
        try:
            for part in parts:
                current = current[part]
        except (KeyError, TypeError):
            # Path not found or null value encountered - skip
            continue

        # Build the nested structure in result
        temp = result
        for i, part in enumerate(parts[:-1]):
            if part not in temp:
                temp[part] = {}
            temp = temp[part]

        temp[parts[-1]] = current

    return result


def parse_json_columns(
    row: DictStrAny,
    json_columns: Dict[str, Dict[str, Any]],
) -> DictStrAny:
    """Parse JSON string columns to Python dicts with optional path filtering.

    Features:
    - Parses JSON strings to dicts for DLT's built-in flattening
    - Supports selective path extraction (only specific paths flattened)
    - Supports keeping original JSON string alongside parsed data

    Args:
        row: Input data row (will be modified in place)
        json_columns: Dict mapping column names to config with:
            - paths: Optional list of paths to extract (None = all)
            - keep_original: Whether to preserve original JSON string

    Returns:
        The modified row with JSON columns processed

    Example:
        >>> row = {"id": 1, "metadata": '{"name": "John", "email": "john@example.com"}'}
        >>> parse_json_columns(row, {"metadata": {"paths": None, "keep_original": False}})
        {'id': 1, 'metadata': {'name': 'John', 'email': 'john@example.com'}}

        >>> # Path-based extraction
        >>> row = {"id": 1, "data": '{"user": {"name": "John", "age": 30}, "id": 1}'}
        >>> parse_json_columns(row, {"data": {"paths": ["user.name"], "keep_original": False}})
        {'id': 1, 'data': {'user': {'name': 'John'}}}

        >>> # Keep original
        >>> row = {"id": 1, "metadata": '{"name": "John"}'}
        >>> parse_json_columns(row, {"metadata": {"paths": None, "keep_original": True}})
        {'id': 1, 'metadata': {'name': 'John'}, 'metadata__original': '{"name": "John"}'}
    """
    for col_name, config in json_columns.items():
        if col_name not in row:
            continue

        value = row[col_name]
        if value is None:
            continue

        # Store original if requested
        keep_original = config.get("keep_original", False)
        if keep_original and isinstance(value, (str, dict)):
            # Store copy of original value
            original_value = value
            # For dicts, we need to serialize them
            if isinstance(original_value, dict):
                original_value = json.dumps(original_value)
            row[f"{col_name}__original"] = original_value

        # If already a dict and we want all paths, nothing to do
        if isinstance(value, dict):
            if config.get("paths") is not None:
                # Selective extraction from dict
                extracted = extract_paths_from_dict(value, config["paths"])
                row[col_name] = extracted
            continue

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                # Apply path filtering if specified
                if config.get("paths") is not None:
                    parsed = extract_paths_from_dict(parsed, config["paths"])
                row[col_name] = parsed
            except json.JSONDecodeError:
                logger.debug(
                    f"Failed to parse JSON for column '{col_name}': {value[:100]}..."
                )
                # Keep original value if parsing fails
                pass

    return row


def get_json_columns_to_parse(
    table_columns: DictStrAny,
) -> Dict[str, Dict[str, Any]]:
    """Get all columns that should be parsed as JSON with their configuration.

    Checks for the x-json-flatten and x-json-keep-original hints on columns.

    Args:
        table_columns: Dict of column_name -> column_schema

    Returns:
        Dict of column_name -> config dict with:
            - paths: List of paths to extract, or None for all
            - keep_original: Whether to keep the original value

    Example:
        >>> table_columns = {
        ...     "metadata": {"x-json-flatten": True},
        ...     "user": {"x-json-flatten": ["user.name", "user.email"], "x-json-keep-original": True}
        ... }
        >>> get_json_columns_to_parse(table_columns)
        {
            'metadata': {'paths': None, 'keep_original': False},
            'user': {'paths': ['user.name', 'user.email'], 'keep_original': True}
        }
    """
    json_columns = {}
    for col_name, col_schema in table_columns.items():
        flatten_hint = col_schema.get("x-json-flatten")
        if flatten_hint is None and not col_schema.get("x-json-keep-original"):
            continue

        # Determine paths to extract
        paths = None
        if isinstance(flatten_hint, list):
            paths = flatten_hint
        elif flatten_hint is None and col_schema.get("x-json-keep-original"):
            # keep-original without flatten means don't flatten, just store original
            paths = []  # Empty list = no flattening

        json_columns[col_name] = {
            "paths": paths,
            "keep_original": col_schema.get("x-json-keep-original", False),
        }

    return json_columns
