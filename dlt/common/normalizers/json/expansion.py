"""
JSON Column Expansion Module

Pure transformation layer for expanding JSON string columns into dictionaries
before they enter DLT's flattening pipeline.
"""

import json
from typing import Any, List, Optional, Tuple, Union

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
        logger.debug("Invalid JSON string, skip flatten: %r", value)
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

    parsed = parse_json_value(raw_value)

    if parsed is None:
        return raw_value, None

    if flatten_spec is True:
        return raw_value if keep_original else None, parsed

    if isinstance(flatten_spec, list):
        filtered = filter_by_paths(parsed, flatten_spec)
        return raw_value if keep_original else None, filtered

    return raw_value, None
