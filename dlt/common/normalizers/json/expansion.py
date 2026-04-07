"""
JSON Column Expansion Module

Pure transformation layer for expanding JSON string columns into dictionaries
before they enter DLT's flattening pipeline.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from dlt.common import logger
from dlt.common.typing import DictStrAny


def parse_json_value(value: Any) -> Optional[DictStrAny]:
    """Parse JSON string or dict-like value into a dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        logger.error(
            "Invalid JSON value for JSON expansion, returning original value without flattening"
        )
        return None


def filter_by_paths(data: Dict[str, Any], paths: List[str]) -> Dict[str, Any]:
    """Extract only specified paths from dict, preserving nested structure."""
    result: Dict[str, Any] = {}

    for path in paths:
        current = data
        parts = path.split(".")
        valid = True

        for p in parts:
            if not isinstance(current, dict) or p not in current:
                valid = False
                break
            current = current[p]

        if not valid:
            continue

        # rebuild nested structure
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = current

    return result


def expand_json_column(
    raw_value: Any,
    flatten_spec: Union[bool, List[str], None],
    keep_original: bool,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Expand JSON column according to flatten specification.

    Returns:
        Tuple of (original_value_to_store, dict_to_flatten)
    """
    if not flatten_spec:
        return raw_value, None

    parsed_value = parse_json_value(raw_value)

    if parsed_value is None:
        return raw_value, None

    if flatten_spec is True:
        return raw_value if keep_original else None, parsed_value

    if isinstance(flatten_spec, list):
        filtered = filter_by_paths(parsed_value, flatten_spec)
        return raw_value if keep_original else None, filtered

    return raw_value, None


def prune_and_cast_json(data: Any, max_depth: int, force_str: bool, current_depth: int = 0) -> Any:
    """
    Preprocess JSON data before DLT's _flatten gets to it.
    - Limits recursion depth to `max_depth` by serializing subtrees into JSON strings.
    - Optionally casts primitive leaves to strings if `force_str=True`.
    """
    if current_depth >= max_depth and isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
        
    if isinstance(data, dict):
        return {
            str(k): prune_and_cast_json(v, max_depth, force_str, current_depth + 1)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [prune_and_cast_json(item, max_depth, force_str, current_depth + 1) for item in data]
    else:
        # Primitive leaf node
        if force_str and data is not None:
            return str(data)
        return data
