"""
JSON Column Expansion Module

Pure transformation layer for expanding JSON string columns into dictionaries
before they enter DLT's flattening pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from dlt.common.json import json
from dlt.common.data_types.type_helpers import json_to_str

from dlt.common import logger
from dlt.common.typing import DictStrAny


def parse_json_value(value: Any) -> Optional[DictStrAny]:
    """Parse JSON string or dict-like value into a dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return cast(DictStrAny, json.loads(value))
    except Exception:
        logger.warning(
            "invalid JSON value for column expansion, keeping original value"
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


def limit_depth(data: Dict[str, Any], max_depth: int) -> Dict[str, Any]:
    """Serialize dict values that exceed `max_depth` to JSON strings.

    At depth 1, any dict value is serialized in place. At deeper levels the
    function recurses until the limit is reached.
    """
    if max_depth <= 1:
        return {k: json_to_str(v) if isinstance(v, dict) else v for k, v in data.items()}
    return {
        k: limit_depth(v, max_depth - 1) if isinstance(v, dict) else v
        for k, v in data.items()
    }


def apply_force_string(data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce all scalar leaf values to `str` at every nesting level.

    Recurses into nested dicts so that all scalar leaves are converted before
    `_flatten` processes them. Lists are left intact (they become nested tables).
    `None` values are preserved as `None`.
    """
    result: Dict[str, Any] = {}
    for k, v in data.items():
        if v is None:
            result[k] = v
        elif isinstance(v, dict):
            result[k] = apply_force_string(v)
        elif isinstance(v, list):
            result[k] = v
        else:
            result[k] = str(v)
    return result


def expand_json_column(
    raw_value: Any,
    flatten_spec: Union[bool, List[str], None],
    keep_original: bool,
    force_string: bool = False,
    max_depth: Optional[int] = None,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Expand JSON column according to flatten specification.

    Returns:
        Tuple of (original_value_to_store, dict_to_flatten)
    """
    if not flatten_spec:
        return raw_value, None

    parsed_value = parse_json_value(raw_value)

    if parsed_value is None:
        return raw_value, None

    if flatten_spec is True:
        expanded_dict: Optional[Dict[str, Any]] = parsed_value
    elif isinstance(flatten_spec, list):
        expanded_dict = filter_by_paths(parsed_value, flatten_spec)
    else:
        return raw_value, None

    if expanded_dict is not None:
        if max_depth is not None:
            expanded_dict = limit_depth(expanded_dict, max_depth)
        if force_string:
            expanded_dict = apply_force_string(expanded_dict)

    return raw_value if keep_original else None, expanded_dict
