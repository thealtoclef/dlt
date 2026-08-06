"""Arrow-native JSON column flattening.

Mirrors the row-by-row `x-json-*` hints implemented in
`dlt/common/normalizers/json/expansion.py` for the high-throughput Arrow path.
The module exposes pure functions that take a `pyarrow.RecordBatch` plus a set
of expansion hints and return a flattened batch together with a partial table
schema describing the newly emitted columns.

The default engine is pure pyarrow. The optional `duckdb` engine adds
vectorized string-JSON parsing and a struct-to-JSON serializer used by the
opt-in `keep_original` / `max_depth` divergences for struct inputs.
"""

import hashlib
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union, cast

from dlt.common import logger
from dlt.common.libs.pyarrow import (
    NameNormalizationCollision,
    get_column_type_from_py_arrow,
    pyarrow,
)
from dlt.common.libs.pyarrow import pyarrow as pa
from dlt.common.normalizers.json.helpers import TJsonColumnExpansionSpec
from dlt.common.normalizers.naming import NamingConvention
from dlt.common.schema.typing import TColumnSchema, TPartialTableSchema


TJsonFlattenEngine = Literal["pyarrow", "duckdb"]


_FLATTENED_TYPE_FALLBACK: TColumnSchema = {"data_type": "text"}

# emitted columns are returned as tuples of (name, array, schema). a separate
# "preserved source" array is returned alongside when string-JSON rows could not
# be parsed as dicts — those rows keep the original value to match the row-by-row
# JSON path's data-preservation semantics.
TFlattenEmission = Tuple[str, Any, TColumnSchema]
TFlattenResult = Tuple[List[TFlattenEmission], Optional[Any]]


def is_json_string_arrow_type(dtype: Any) -> bool:
    """Returns True for arrow string types that may carry JSON text payloads."""
    return bool(pa.types.is_string(dtype) or pa.types.is_large_string(dtype))


def is_struct_arrow_type(dtype: Any) -> bool:
    """Returns True for arrow struct types."""
    return bool(pa.types.is_struct(dtype))


def flatten_arrow_batch(
    batch: Any,
    table_name: str,
    expansion_specs: Dict[str, TJsonColumnExpansionSpec],
    naming: NamingConvention,
    *,
    engine: TJsonFlattenEngine = "pyarrow",
    duckdb_connection: Optional[Any] = None,
    string_schema_locks: Optional[Dict[str, Any]] = None,
    destination_casefold_identifier: Optional[Callable[[str], str]] = None,
) -> Tuple[Any, TPartialTableSchema]:
    """Applies `x-json-*` hints to an arrow batch and returns the rewritten batch.

    Args:
        batch (Any): A `pyarrow.RecordBatch` or `pyarrow.Table` to flatten.
        table_name (str): dlt table name the batch belongs to.
        expansion_specs (Dict[str, TJsonColumnExpansionSpec]): Per-column hints, as
            returned by `get_json_expansion_columns`.
        naming (NamingConvention): Active naming convention used for column naming.
        engine (TJsonFlattenEngine): Flatten engine. `"pyarrow"` is the default
            zero-copy path; `"duckdb"` enables vectorized struct/string serialization
            for the opt-in divergences.
        duckdb_connection (Optional[Any]): Pre-opened DuckDB connection. Required when
            `engine='duckdb'` and any work needs vectorized JSON ops.
        string_schema_locks (Optional[Dict[str, Any]]): Per-source-column locked
            struct types for streaming string-JSON inference. Updated in place.
        destination_casefold_identifier (Optional[Callable[[str], str]]): Destination
            column-name casefold function. When set, flattened columns that are not
            stable under casefolding get deterministic `__c_<hash>` suffixes.

    Returns:
        Tuple[Any, TPartialTableSchema]: The flattened batch (same type as input —
            `RecordBatch` or `Table`) and a partial table schema describing the
            newly emitted columns.
    """
    if not expansion_specs:
        return batch, _empty_partial(table_name)

    columns_present = set(batch.schema.names)
    hinted = {name: spec for name, spec in expansion_specs.items() if name in columns_present}
    if not hinted:
        return batch, _empty_partial(table_name)

    is_record_batch = isinstance(batch, pa.RecordBatch)
    new_field_names: List[str] = []
    new_arrays: List[Any] = []
    new_columns_schema: Dict[str, TColumnSchema] = {}
    seen_names: Set[str] = set()
    schema_locks = string_schema_locks if string_schema_locks is not None else {}
    path_types: Dict[str, Any] = schema_locks.setdefault("__path_types__", {})

    for field in batch.schema:
        name = field.name
        if name not in hinted:
            _push_column(name, batch.column(field.name), new_field_names, new_arrays, seen_names)
            continue

        spec = hinted[name]
        source_array = batch.column(field.name)

        keep_source_now = spec.keep_original
        if (
            keep_source_now
            and engine == "duckdb"
            and is_struct_arrow_type(field.type)
            and duckdb_connection is not None
        ):
            # duckdb engine opt-in: keep_original on struct serializes to JSON string
            source_array_to_keep = _struct_to_json_string(source_array, duckdb_connection)
        else:
            source_array_to_keep = source_array

        if not spec.flatten_spec:
            if keep_source_now:
                _push_column(name, source_array_to_keep, new_field_names, new_arrays, seen_names)
            continue

        preserved_source: Optional[Any] = None
        if is_struct_arrow_type(field.type):
            emitted = _flatten_struct_column(
                parent_name=name,
                struct_array=source_array,
                spec=spec,
                naming=naming,
                engine=engine,
                duckdb_connection=duckdb_connection,
            )
        elif is_json_string_arrow_type(field.type):
            emitted, preserved_source = _flatten_string_column(
                parent_name=name,
                string_array=source_array,
                spec=spec,
                naming=naming,
                engine=engine,
                duckdb_connection=duckdb_connection,
                schema_locks=schema_locks,
            )
        else:
            logger.warning(
                f"Column {name!r} has JSON flatten hint but arrow type {field.type} is"
                " neither struct nor string. Hint ignored on this batch."
            )
            if keep_source_now:
                _push_column(name, source_array_to_keep, new_field_names, new_arrays, seen_names)
            continue

        # decide what to do with the source column. when keep_original is set, always
        # push it. otherwise, if some rows failed to parse as dicts, preserve those
        # rows' original values to match the row-by-row JSON path semantics.
        if keep_source_now:
            _push_column(name, source_array_to_keep, new_field_names, new_arrays, seen_names)
        elif preserved_source is not None:
            _push_column(name, preserved_source, new_field_names, new_arrays, seen_names)
            new_columns_schema[name] = cast(TColumnSchema, {"name": name, "data_type": "text"})

        emitted = _resolve_casefold_collisions(
            emitted, naming, destination_casefold_identifier, seen_names
        )
        for emitted_name, emitted_array, emitted_col_schema in emitted:
            if emitted_name in seen_names:
                raise NameNormalizationCollision(
                    f"Flattened column {emitted_name!r} collides with an existing column in"
                    f" table {table_name!r}."
                )
            emitted_array, emitted_col_schema = _enforce_path_type(
                emitted_name, emitted_array, emitted_col_schema, path_types
            )
            new_field_names.append(emitted_name)
            new_arrays.append(emitted_array)
            seen_names.add(emitted_name)
            new_columns_schema[emitted_name] = {**emitted_col_schema, "name": emitted_name}

    new_schema = pa.schema(
        [pa.field(n, a.type, nullable=True) for n, a in zip(new_field_names, new_arrays)],
        metadata=batch.schema.metadata,
    )
    if is_record_batch:
        result = pa.RecordBatch.from_arrays(new_arrays, schema=new_schema)
    else:
        result = pa.Table.from_arrays(new_arrays, schema=new_schema)

    partial = cast(TPartialTableSchema, {"name": table_name, "columns": new_columns_schema})
    return result, partial


def _empty_partial(table_name: str) -> TPartialTableSchema:
    return cast(TPartialTableSchema, {"name": table_name, "columns": {}})


def _push_column(
    name: str,
    array: Any,
    names: List[str],
    arrays: List[Any],
    seen: Set[str],
) -> None:
    if name in seen:
        raise NameNormalizationCollision(f"Column {name!r} appears twice in flatten output.")
    names.append(name)
    arrays.append(array)
    seen.add(name)


def _flatten_struct_column(
    parent_name: str,
    struct_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
    engine: TJsonFlattenEngine,
    duckdb_connection: Optional[Any],
) -> List[Tuple[str, Any, TColumnSchema]]:
    """Flattens a struct-typed column into one column per leaf path.

    Honors `flatten_spec` (`True` for full recursion or a list of dotted paths),
    `force_string`, `max_depth`. When `engine='duckdb'` the `max_depth` boundary
    is serialised via `to_json`; otherwise the sub-struct is kept as a struct
    column (documented divergence from the JSON path).
    """
    if pa.types.is_null(struct_array.type):
        return []

    if spec.flatten_spec is True:
        return _recurse_struct(
            parent_name,
            struct_array,
            spec=spec,
            naming=naming,
            depth=1,
            engine=engine,
            duckdb_connection=duckdb_connection,
        )

    if isinstance(spec.flatten_spec, list):
        emitted: List[Tuple[str, Any, TColumnSchema]] = []
        for dotted_path in spec.flatten_spec:
            segments = [s for s in dotted_path.split(".") if s]
            if not segments:
                continue
            extracted = _extract_struct_path(struct_array, segments)
            if extracted is None:
                continue
            leaf_name = naming.shorten_fragments(parent_name, *segments)
            if pa.types.is_struct(extracted.type):
                emitted.extend(
                    _recurse_struct(
                        leaf_name,
                        extracted,
                        spec=spec,
                        naming=naming,
                        depth=len(segments) + 1,
                        engine=engine,
                        duckdb_connection=duckdb_connection,
                        original_path=(parent_name, *segments),
                    )
                )
            else:
                emitted.append(_finalize_leaf(leaf_name, extracted, spec, (parent_name, *segments)))
        return emitted

    return []


def _recurse_struct(
    parent_name: str,
    struct_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
    depth: int,
    engine: TJsonFlattenEngine,
    duckdb_connection: Optional[Any],
    original_path: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, Any, TColumnSchema]]:
    if pa.types.is_null(struct_array.type):
        return []

    if spec.max_depth is not None and depth > spec.max_depth:
        if engine == "duckdb":
            string_array = _struct_to_json_string(struct_array, duckdb_connection)
            return [_finalize_leaf(parent_name, string_array, spec, original_path)]
        return [
            (
                parent_name,
                struct_array,
                _with_flatten_description({"data_type": "json"}, original_path),
            )
        ]

    emitted: List[Tuple[str, Any, TColumnSchema]] = []
    struct_type = struct_array.type
    for child_idx in range(struct_type.num_fields):
        child_field = struct_type.field(child_idx)
        child_array = pa.compute.struct_field(struct_array, [child_idx])
        child_name = naming.shorten_fragments(
            parent_name, naming.normalize_identifier(child_field.name)
        )
        child_path = (*(original_path or (parent_name,)), child_field.name)
        if pa.types.is_struct(child_array.type):
            emitted.extend(
                _recurse_struct(
                    child_name,
                    child_array,
                    spec=spec,
                    naming=naming,
                    depth=depth + 1,
                    engine=engine,
                    duckdb_connection=duckdb_connection,
                    original_path=child_path,
                )
            )
        else:
            emitted.append(_finalize_leaf(child_name, child_array, spec, child_path))
    return emitted


def _extract_struct_path(struct_array: Any, segments: List[str]) -> Optional[Any]:
    current = struct_array
    for seg in segments:
        if not pa.types.is_struct(current.type):
            return None
        try:
            current = pa.compute.struct_field(current, [seg])
        except (KeyError, pyarrow.ArrowInvalid):
            return None
    return current


def _finalize_leaf(
    name: str,
    array: Any,
    spec: TJsonColumnExpansionSpec,
    original_path: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, Any, TColumnSchema]:
    if spec.force_string:
        array = _cast_array_to_string(array)
    col_schema: TColumnSchema
    try:
        col_schema = cast(TColumnSchema, dict(get_column_type_from_py_arrow(array.type)))
    except Exception:  # noqa: BLE001
        col_schema = cast(TColumnSchema, dict(_FLATTENED_TYPE_FALLBACK))
    if not col_schema:
        col_schema = cast(TColumnSchema, dict(_FLATTENED_TYPE_FALLBACK))
    col_schema = _with_flatten_description(col_schema, original_path)
    return name, array, col_schema


def _with_flatten_description(
    col_schema: TColumnSchema, original_path: Optional[Tuple[str, ...]]
) -> TColumnSchema:
    if original_path and "description" not in col_schema:
        col_schema = cast(TColumnSchema, dict(col_schema))
        col_schema["description"] = f"Flattened from original path: {'.'.join(original_path)}"
    return col_schema


def _resolve_casefold_collisions(
    emitted: List[TFlattenEmission],
    naming: NamingConvention,
    casefold_identifier: Optional[Callable[[str], str]],
    seen_names: Set[str],
) -> List[TFlattenEmission]:
    """Suffix flattened names that are unsafe under destination casefolding rules.

    For BigQuery, column names cannot differ only by case. We suffix all flattened
    names whose casefolded representation differs from the emitted name, not only
    names currently observed in a collision group. That keeps the physical name for
    `request.refId` stable even if `request.RefId` appears in a later batch.
    """
    if casefold_identifier is None or not emitted:
        return emitted

    existing_folded = {casefold_identifier(n) for n in seen_names}
    groups: Dict[str, List[int]] = {}
    for idx, (name, _, _) in enumerate(emitted):
        groups.setdefault(casefold_identifier(name), []).append(idx)

    rename_indexes: Set[int] = set()
    for idx, (name, _, _) in enumerate(emitted):
        if casefold_identifier(name) != name:
            rename_indexes.add(idx)
    for folded_name, indexes in groups.items():
        if len(indexes) > 1 or folded_name in existing_folded:
            rename_indexes.update(indexes)

    if not rename_indexes:
        return emitted

    resolved: List[TFlattenEmission] = []
    for idx, (name, array, col_schema) in enumerate(emitted):
        if idx in rename_indexes:
            original_path = _original_path_from_description(col_schema) or name
            name = _case_collision_name(name, original_path, naming)
        resolved.append((name, array, col_schema))
    _verify_casefold_unique(resolved, seen_names, casefold_identifier)
    return resolved


def _case_collision_name(name: str, original_path: str, naming: NamingConvention) -> str:
    suffix = f"__c_{_case_collision_tag(original_path)}"
    return naming.shorten_identifier(name + suffix, original_path, naming.max_length)


def _case_collision_tag(original_path: str) -> str:
    return hashlib.shake_128(original_path.encode("utf-8")).hexdigest(8)


def _verify_casefold_unique(
    emitted: List[TFlattenEmission],
    seen_names: Set[str],
    casefold_identifier: Callable[[str], str],
) -> None:
    folded_names = {casefold_identifier(n) for n in seen_names}
    for name, _, _ in emitted:
        folded_name = casefold_identifier(name)
        if folded_name in folded_names:
            raise NameNormalizationCollision(
                f"Flattened column {name!r} collides with an existing destination column after"
                " applying destination casefolding rules."
            )
        folded_names.add(folded_name)


def _original_path_from_description(col_schema: TColumnSchema) -> Optional[str]:
    description = col_schema.get("description")
    prefix = "Flattened from original path: "
    if description and description.startswith(prefix):
        return description[len(prefix) :]
    return None


def _cast_array_to_string(array: Any) -> Any:
    """Vectorized leaf-cast to string.

    Lists and structs are serialized to JSON via `to_json` semantics. For the
    pure-pyarrow path (no duckdb), nested types are best-effort: we fall back to
    `pa.compute.cast` and let arrow raise if a destination type rejects the cast.
    """
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        return array
    if pa.types.is_null(array.type):
        return pa.nulls(len(array), type=pa.string())
    try:
        return pa.compute.cast(array, pa.string())
    except (pyarrow.ArrowInvalid, pyarrow.ArrowNotImplementedError):
        return pa.array(
            [None if v is None else str(v) for v in array.to_pylist()], type=pa.string()
        )


def _flatten_string_column(
    parent_name: str,
    string_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
    engine: TJsonFlattenEngine,
    duckdb_connection: Optional[Any],
    schema_locks: Dict[str, Any],
) -> TFlattenResult:
    """Flattens a string-typed column that carries JSON text.

    Returns the emitted columns plus an optional `preserved_source` array — populated
    when one or more rows could not be parsed as a JSON object (invalid / scalar /
    root-array). The caller is responsible for re-emitting the source column with
    that array so those rows' original values are not lost.
    """
    if isinstance(spec.flatten_spec, list):
        if engine == "duckdb" and duckdb_connection is not None:
            return _flatten_string_paths_duckdb(
                parent_name,
                string_array,
                spec,
                naming,
                duckdb_connection,
            )
        return _flatten_string_paths_pyarrow(parent_name, string_array, spec, naming)

    if spec.flatten_spec is True:
        if engine == "duckdb" and duckdb_connection is not None:
            return _flatten_string_full_duckdb(
                parent_name,
                string_array,
                spec,
                naming,
                duckdb_connection,
                schema_locks,
            )
        return _flatten_string_full_pyarrow(parent_name, string_array, spec, naming, schema_locks)
    return [], None


def _flatten_string_paths_pyarrow(
    parent_name: str,
    string_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
) -> TFlattenResult:
    from dlt.common.json import json

    paths: List[Tuple[str, List[str]]] = []
    for path in spec.flatten_spec:  # type: ignore[union-attr]
        segments = [seg for seg in path.split(".") if seg]
        if segments:
            paths.append((naming.shorten_fragments(parent_name, *segments), segments))
    columns: Dict[str, List[Any]] = {key: [] for key, _ in paths}
    raw_values = string_array.to_pylist()
    preserved: Optional[List[Any]] = None
    for idx, raw in enumerate(raw_values):
        parsed = _safe_json_load(raw, json)
        if raw is not None and not isinstance(parsed, dict):
            if preserved is None:
                preserved = [None] * len(raw_values)
            preserved[idx] = raw
        for key, segs in paths:
            columns[key].append(_descend_dict(parsed, segs))
    out: List[TFlattenEmission] = []
    for key, values in columns.items():
        # skip columns where no row actually had this path — matches JSON path semantics
        if all(v is None for v in values):
            continue
        array = pa.array([_coerce_scalar(v) for v in values])
        out.append(_finalize_leaf(key, array, spec))
    preserved_array = pa.array(preserved, type=pa.string()) if preserved is not None else None
    return out, preserved_array


def _flatten_string_paths_duckdb(
    parent_name: str,
    string_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
    duckdb_connection: Any,
) -> TFlattenResult:
    """Extracts listed paths from a string-JSON column using DuckDB's JSON parser."""
    try:
        parsed_struct, preserved_array = _parse_string_json_objects_duckdb(
            string_array,
            duckdb_connection,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"DuckDB JSON path extraction for column {parent_name!r} failed: {exc!s}."
            " Falling back to pyarrow parser for this batch."
        )
        return _flatten_string_paths_pyarrow(parent_name, string_array, spec, naming)
    if parsed_struct is None or not pa.types.is_struct(parsed_struct.type):
        return [], preserved_array
    emitted = _flatten_struct_column(
        parent_name,
        parsed_struct,
        spec=spec,
        naming=naming,
        engine="duckdb",
        duckdb_connection=duckdb_connection,
    )
    return emitted, preserved_array


def _flatten_string_full_pyarrow(
    parent_name: str,
    string_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
    schema_locks: Dict[str, Any],
) -> TFlattenResult:
    from dlt.common.json import json

    raw_values = string_array.to_pylist()
    sanitized: List[Optional[Dict[str, Any]]] = []
    preserved: Optional[List[Any]] = None
    for idx, raw in enumerate(raw_values):
        parsed = _safe_json_load(raw, json)
        if isinstance(parsed, dict):
            sanitized.append(parsed)
        else:
            sanitized.append(None)
            if raw is not None:
                if preserved is None:
                    preserved = [None] * len(raw_values)
                preserved[idx] = raw
    preserved_array = pa.array(preserved, type=pa.string()) if preserved is not None else None

    inference_mode = spec.schema_inference if spec.schema_inference is not None else "incremental"
    locked = schema_locks.get(parent_name) if inference_mode != "incremental" else None
    if locked is not None:
        try:
            struct_array = pa.array(sanitized, type=locked)
        except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
            # coerce row values to match the locked target type (paths the
            # lock typed as string get stringified) before retrying.
            coerced = [_coerce_dict_to_type(r, locked) for r in sanitized]
            try:
                struct_array = pa.array(coerced, type=locked)
            except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
                struct_array = _build_struct_or_coerce(coerced, parent_name)
                schema_locks[parent_name] = struct_array.type
    else:
        struct_array = _build_struct_or_coerce(sanitized, parent_name)
        if inference_mode != "incremental":
            schema_locks[parent_name] = struct_array.type

    if not pa.types.is_struct(struct_array.type):
        return [], preserved_array
    emitted = _recurse_struct(
        parent_name,
        struct_array,
        spec=spec,
        naming=naming,
        depth=1,
        engine="pyarrow",
        duckdb_connection=None,
    )
    return emitted, preserved_array


def _flatten_string_full_duckdb(
    parent_name: str,
    string_array: Any,
    spec: TJsonColumnExpansionSpec,
    naming: NamingConvention,
    duckdb_connection: Any,
    schema_locks: Dict[str, Any],
) -> TFlattenResult:
    try:
        array, preserved_array = _parse_string_json_objects_duckdb(
            string_array,
            duckdb_connection,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"DuckDB JSON parse for column {parent_name!r} failed: {exc!s}. Falling back to"
            " pyarrow parser for this batch."
        )
        return _flatten_string_full_pyarrow(parent_name, string_array, spec, naming, schema_locks)
    if array is None or not pa.types.is_struct(array.type):
        return [], preserved_array

    schema_locks[parent_name] = array.type
    emitted = _recurse_struct(
        parent_name,
        array,
        spec=spec,
        naming=naming,
        depth=1,
        engine="duckdb",
        duckdb_connection=duckdb_connection,
    )

    return emitted, preserved_array


def _parse_string_json_objects_duckdb(
    string_array: Any,
    duckdb_connection: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Parses object-root string JSON values to a struct and preserves non-objects."""
    arrow_input = pa.table({"v": string_array})
    duckdb_connection.register("dlt_json_in", arrow_input)
    try:
        # detect a union JSON structure across all object rows. mixed-type keys
        # are reported as `"JSON"` by duckdb and surface as varchar columns in
        # the parsed result — that gives us automatic same-batch coercion.
        structure_row = duckdb_connection.execute("""
            WITH parsed AS (
                SELECT try_cast(v AS JSON) AS j
                FROM dlt_json_in
            )
            SELECT json_group_structure(j)
            FROM parsed
            WHERE json_type(j) = 'OBJECT'
            """).fetchone()
        structure = structure_row[0] if structure_row else None
        if not structure:
            return None, _preserve_non_null_string_values(string_array)

        # json_transform requires the structure to be a SQL constant — interpolate
        # the literal here. the structure value is produced by duckdb so it is a
        # well-formed JSON literal; escape any embedded single quotes defensively.
        struct_literal = structure.replace("'", "''")
        result_arrow = duckdb_connection.execute(f"""
            WITH parsed AS (
                SELECT
                    v,
                    try_cast(v AS JSON) AS j
                FROM dlt_json_in
            ),
            typed AS (
                SELECT
                    v,
                    j,
                    coalesce(json_type(j) = 'OBJECT', false) AS is_object
                FROM parsed
            )
            SELECT
                json_transform(CASE WHEN is_object THEN j ELSE NULL END, '{struct_literal}') AS j,
                (NOT is_object AND v IS NOT NULL) AS preserve_source
            FROM typed
        """).fetch_arrow_table()
    finally:
        duckdb_connection.unregister("dlt_json_in")

    array = result_arrow.column(0)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    preserve_mask = result_arrow.column(1)
    if isinstance(preserve_mask, pa.ChunkedArray):
        preserve_mask = preserve_mask.combine_chunks()
    return array, _build_preserved_source_array(string_array, preserve_mask)


def _enforce_path_type(
    name: str, array: Any, col_schema: TColumnSchema, path_types: Dict[str, Any]
) -> Tuple[Any, TColumnSchema]:
    """Tracks per-path arrow types across batches and coerces conflicts to text.

    Mirrors dlt's variant policy: when the same flattened path resolves to different
    types in different batches, widen to `text` so the destination schema stays
    consistent. A warning is emitted once per `(name, prior_type)` pair.
    """
    recorded = path_types.get(name)
    if recorded is None:
        path_types[name] = array.type
        return array, col_schema
    if recorded == array.type:
        return array, col_schema
    if recorded == pa.string() or recorded == pa.large_string():
        # already widened; coerce this batch too
        return _cast_array_to_string(array), cast(TColumnSchema, {"data_type": "text"})
    logger.warning(
        f"Flattened path {name!r} type changed from {recorded} to {array.type};"
        " coercing to text for schema consistency."
    )
    path_types[name] = pa.string()
    return _cast_array_to_string(array), cast(TColumnSchema, {"data_type": "text"})


def prescan_string_json_schemas(
    parquet_file: Any,
    column_specs: Dict[str, TJsonColumnExpansionSpec],
) -> Dict[str, Any]:
    """Scans a parquet file once to lock the union arrow struct type per hinted column.

    Used by `schema_inference="full-scan"` (entire column scanned) and `int` sample
    modes (first N rows scanned). Returns a mapping `column_name -> pa.StructType`
    suitable for seeding `string_schema_locks`.
    """
    from dlt.common.json import json

    if not column_specs:
        return {}
    reader = pyarrow.parquet.ParquetFile(parquet_file)
    # only request columns actually present in the file. iter_batches raises
    # KeyError on missing names, but a hinted column may legitimately be absent
    # from a sparse Arrow batch — that case is silently ignored by flatten,
    # so prescan must mirror the same tolerance.
    present_in_file = set(reader.schema_arrow.names)
    columns_to_read = [c for c in column_specs.keys() if c in present_in_file]
    if not columns_to_read:
        return {}
    aggregated: Dict[str, List[Any]] = {c: [] for c in columns_to_read}
    sample_limits: Dict[str, Optional[int]] = {}
    for c in columns_to_read:
        si = column_specs[c].schema_inference
        sample_limits[c] = si if isinstance(si, int) else None
    done: Set[str] = set()
    for batch in reader.iter_batches(columns=columns_to_read):
        if len(done) == len(columns_to_read):
            break
        for col_name in columns_to_read:
            if col_name in done:
                continue
            arr = batch.column(col_name).to_pylist()
            limit = sample_limits[col_name]
            remaining = (limit - len(aggregated[col_name])) if limit is not None else len(arr)
            if remaining <= 0:
                done.add(col_name)
                continue
            slice_ = arr[: max(remaining, 0)]
            for v in slice_:
                aggregated[col_name].append(_safe_json_load(v, json) if v is not None else None)
            if limit is not None and len(aggregated[col_name]) >= limit:
                done.add(col_name)

    locked: Dict[str, Any] = {}
    for col_name, rows in aggregated.items():
        dict_rows = [r if isinstance(r, dict) else None for r in rows]
        if not any(isinstance(r, dict) for r in dict_rows):
            continue
        struct = _build_struct_or_coerce(dict_rows, col_name)
        if pa.types.is_struct(struct.type):
            locked[col_name] = struct.type
    return locked


def _build_preserved_source_array(string_array: Any, preserve_mask: Any) -> Optional[Any]:
    """Builds the source-preservation column from a DuckDB-produced boolean mask."""
    if not pa.compute.any(preserve_mask).as_py():
        return None
    source = (
        string_array.combine_chunks() if isinstance(string_array, pa.ChunkedArray) else string_array
    )
    nulls = pa.nulls(len(source), type=source.type)
    return pa.compute.if_else(preserve_mask, source, nulls)


def _preserve_non_null_string_values(string_array: Any) -> Optional[Any]:
    """Returns the original string array when DuckDB found no object rows to flatten."""
    source = (
        string_array.combine_chunks() if isinstance(string_array, pa.ChunkedArray) else string_array
    )
    if not pa.compute.any(pa.compute.is_valid(source)).as_py():
        return None
    return source


def _struct_to_json_string(struct_array: Any, duckdb_connection: Optional[Any]) -> Any:
    if duckdb_connection is None:
        return pa.array(
            [None if v is None else _json_dumps_compact(v) for v in struct_array.to_pylist()],
            type=pa.string(),
        )
    arrow_input = pa.table({"v": struct_array})
    duckdb_connection.register("dlt_struct_in", arrow_input)
    try:
        result_arrow = duckdb_connection.execute(
            "SELECT to_json(v) AS j FROM dlt_struct_in"
        ).fetch_arrow_table()
    finally:
        duckdb_connection.unregister("dlt_struct_in")
    column = result_arrow.column(0)
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return column


def _build_struct_or_coerce(values: List[Optional[Dict[str, Any]]], parent_name: str) -> Any:
    """Builds a `pa.StructArray` from parsed dict rows, coercing mixed-type keys to text.

    Used when `pa.array(values)` raises because the same key has incompatible types
    across rows within a single batch. Recurses into nested dicts so deep mixed-type
    paths are also coerced. Rows that were `None` produce all-null struct entries.

    Returns:
        Any: A `pa.StructArray` (or a string `pa.Array` when no recoverable structure
            exists). Callers must check `pa.types.is_struct` before recursing.
    """
    try:
        return pa.array(values)
    except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
        pass
    all_keys: Dict[str, None] = {}
    for d in values:
        if isinstance(d, dict):
            for k in d:
                all_keys[k] = None
    if not all_keys:
        # nothing structurally recoverable — coerce the whole column to text
        return pa.array(
            [None if v is None else _json_dumps_compact(v) for v in values],
            type=pa.string(),
        )
    fields: List[Any] = []
    arrays: List[Any] = []
    for key in all_keys:
        sub: List[Any] = [d.get(key) if isinstance(d, dict) else None for d in values]
        if any(isinstance(s, dict) for s in sub):
            sub_dicts: List[Optional[Dict[str, Any]]] = [
                s if isinstance(s, dict) or s is None else None for s in sub
            ]
            # any non-dict, non-None values at this key force coercion to text
            if any(s is not None and not isinstance(s, dict) for s in sub):
                arr = _stringify_array(sub, parent_name, key)
            else:
                arr = _build_struct_or_coerce(sub_dicts, f"{parent_name}.{key}")
        else:
            try:
                arr = pa.array(sub)
            except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
                arr = _stringify_array(sub, parent_name, key)
        fields.append(pa.field(key, arr.type))
        arrays.append(arr)
    mask = pa.array([v is None for v in values], type=pa.bool_())
    return pa.StructArray.from_arrays(arrays, fields=fields, mask=mask)


def _coerce_dict_to_type(value: Any, target: Any) -> Any:
    """Coerces a parsed dict's values so it can be loaded into a locked arrow type.

    Used by full-scan / sample-size schema inference: when the pre-scan widened a
    path to text but a later batch carries native types (e.g. integers), this
    walks the dict and stringifies values at paths the target type marks string.
    """
    if value is None:
        return None
    if pa.types.is_struct(target):
        if not isinstance(value, dict):
            return None
        out: Dict[str, Any] = {}
        for i in range(target.num_fields):
            field = target.field(i)
            out[field.name] = _coerce_dict_to_type(value.get(field.name), field.type)
        return out
    if pa.types.is_string(target) or pa.types.is_large_string(target):
        if isinstance(value, (dict, list)):
            return _json_dumps_compact(value)
        return None if value is None else str(value)
    return value


def _stringify_array(values: List[Any], parent_name: str, key: str) -> Any:
    logger.warning(
        f"Mixed types within batch at {parent_name}.{key!r}; coercing to text for"
        " schema consistency."
    )
    return pa.array(
        [
            (
                None
                if v is None
                else (_json_dumps_compact(v) if isinstance(v, (dict, list)) else str(v))
            )
            for v in values
        ],
        type=pa.string(),
    )


def _safe_json_load(value: Any, json_module: Any) -> Optional[Union[Dict[str, Any], List[Any]]]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        result: Optional[Union[Dict[str, Any], List[Any]]] = json_module.loads(value)
        return result
    except Exception:  # noqa: BLE001
        logger.warning("invalid JSON value for arrow column expansion, treating as null")
        return None


def _descend_dict(data: Any, segments: List[str]) -> Any:
    current = data
    for seg in segments:
        if not isinstance(current, dict) or seg not in current:
            return None
        current = current[seg]
    return current


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return _json_dumps_compact(value)
    return value


def _json_dumps_compact(value: Any) -> str:
    from dlt.common.json import json

    return json.dumps(value)
