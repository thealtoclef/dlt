from typing import Callable, List, Dict, Optional, Set, Any

from dlt.common import logger
from dlt.common.data_writers.writers import ArrowToObjectAdapter
from dlt.common.json import json
from dlt.common.metrics import DataWriterMetrics
from dlt.common.normalizers.json.relational import DataItemNormalizer as RelationalNormalizer
from dlt.common.normalizers.json.helpers import get_json_expansion_columns
from dlt.common.normalizers.utils import generate_dlt_ids
from dlt.common.schema.typing import C_DLT_ID
from dlt.common.schema.utils import dlt_id_column, normalize_table_identifiers
from dlt.common.schema import TSchemaUpdate, Schema
from dlt.common.storages.load_storage import LoadStorage
from dlt.common.storages import NormalizeStorage
from dlt.common.storages.data_item_storage import DataItemStorage
from dlt.common.storages.load_package import ParsedLoadJobFileName
from dlt.common.exceptions import MissingDependencyException

from dlt.common.runtime.collector import Collector, NULL_COLLECTOR
from dlt.normalize.configuration import NormalizeConfiguration
from dlt.normalize.items_normalizers.base import ItemsNormalizer

try:
    from dlt.common.libs import pyarrow
    from dlt.common.libs.pyarrow import pyarrow as pa
except MissingDependencyException:
    pyarrow = None
    pa = None

_warned_struct_cast: Set[str] = set()
"""Columns for which the struct→string native cast fallback warning has been logged."""


def _struct_to_json_string(struct_arr: "pa.Array") -> "pa.Array":
    """Serialize a struct array to a JSON string array.

    Tries native `pa.compute.cast` first (PyArrow >= 14), falling back to
    `to_pylist` + `json.dumps` with a one-time warning.
    """
    try:
        return pa.compute.cast(struct_arr, pa.string())
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid, pa.ArrowTypeError):
        if "struct_cast" not in _warned_struct_cast:
            _warned_struct_cast.add("struct_cast")
            logger.warning(
                "pyarrow native struct→string cast unavailable; falling back to json.dumps per row"
            )
        pylist = struct_arr.to_pylist()
        return pa.array(
            [json.dumps(row) if row is not None else None for row in pylist],
            type=pa.string(),
        )


class ArrowItemsNormalizer(ItemsNormalizer):
    REWRITE_ROW_GROUPS = 1

    def __init__(
        self,
        item_storage: DataItemStorage,
        load_storage: LoadStorage,
        normalize_storage: NormalizeStorage,
        schema: Schema,
        load_id: str,
        config: NormalizeConfiguration,
        report_progress: bool = False,
        collector: Collector = NULL_COLLECTOR,
    ) -> None:
        super().__init__(
            item_storage,
            load_storage,
            normalize_storage,
            schema,
            load_id,
            config,
            report_progress=report_progress,
            collector=collector,
        )
        self._null_only_columns: Dict[str, Set[str]] = {}

    @property
    def null_only_columns(self) -> Dict[str, Set[str]]:
        return self._null_only_columns

    def _collect_null_columns_from_arrow_metadata(
        self, arrow_schema: Any, root_table_name: str
    ) -> None:
        """Read dlt.null_columns from arrow schema metadata, normalize names, add to tracker."""
        metadata = arrow_schema.metadata or {}
        null_cols_json = metadata.get(b"dlt.null_columns")
        if not null_cols_json:
            return
        null_col_names = json.loadb(null_cols_json)
        if not null_col_names:
            return
        normalized_names = set()
        for name in null_col_names:
            normalized_names.add(self.schema.naming.normalize_path(name))
        self._null_only_columns.setdefault(root_table_name, set()).update(normalized_names)

    def _write_with_dlt_columns(
        self,
        extracted_items_file: str,
        root_table_name: str,
        add_dlt_id: bool,
        expansion_cols: Optional[Dict[str, Any]] = None,
    ) -> List[TSchemaUpdate]:
        new_columns: List[Any] = []
        schema = self.schema
        load_id = self.load_id
        schema_update: TSchemaUpdate = {}
        data_normalizer = schema.data_item_normalizer

        from dlt.common.libs.pyarrow import (
            is_flattenable_column,
            flatten_struct_column,
            expand_arrow_json_column,
            apply_arrow_path_filter,
            apply_arrow_depth_limit,
            apply_arrow_force_string,
            get_column_type_from_py_arrow,
            build_flatten_schema_update,
        )

        if add_dlt_id and isinstance(data_normalizer, RelationalNormalizer):
            partial_table = normalize_table_identifiers(
                {
                    "name": root_table_name,
                    "columns": {C_DLT_ID: dlt_id_column()},
                },
                schema.naming,
            )
            schema.update_table(partial_table, normalize_identifiers=False)
            table_updates = schema_update.setdefault(root_table_name, [])
            table_updates.append(partial_table)
            # TODO: use get_root_row_id_type to get row id type and generate deterministic
            #  row ids as well (using pandas helper function prepared for scd2)
            #  we could also generate random columns with pandas or duckdb if present
            new_columns.append(
                (
                    -1,
                    pa.field(data_normalizer.c_dlt_id, pyarrow.pyarrow.string(), nullable=False),
                    lambda batch: pa.array(generate_dlt_ids(batch.num_rows)),
                )
            )

        items_count = 0
        columns_schema = schema.get_table_columns(root_table_name)
        flattened_column_types: Dict[str, Any] = {}
        dropped_source_columns: Set[str] = set()
        # if we use adapter to convert arrow to dicts, then normalization is not necessary
        is_native_arrow_writer = not issubclass(self.item_storage.writer_cls, ArrowToObjectAdapter)
        if is_native_arrow_writer and expansion_cols:
            norm_config = schema._normalizers_config["json"].get("config") or {}
            max_nesting = norm_config.get("max_nesting", 1000)
            has_flatten_hints = True
        else:
            has_flatten_hints = False
        # detect case-insensitive destinations (e.g. BigQuery) for casefold collision resolution
        dest_caps = self.config.destination_capabilities
        casefold_id: Optional[Callable[[str], str]] = (
            str.casefold
            if dest_caps and getattr(dest_caps, "sqlglot_dialect", None) == "bigquery"
            else None
        )
        should_normalize: bool = None
        self._maybe_cancel()
        with self.normalize_storage.extracted_packages.storage.open_file(
            extracted_items_file, "rb"
        ) as f:
            batch_new_types: Dict[str, Any] = {}
            for batch in pyarrow.pq_stream_with_new_columns(
                f, new_columns, row_groups_per_read=self.REWRITE_ROW_GROUPS
            ):
                self._maybe_cancel()
                items_count += batch.num_rows
                self._report_progress(root_table_name, batch.num_rows)
                # we may need to normalize
                if is_native_arrow_writer and should_normalize is None:
                    should_normalize = pyarrow.should_normalize_arrow_schema(
                        batch.schema, columns_schema, schema.naming
                    )[0]
                    if should_normalize:
                        logger.info(
                            f"When writing arrow table to {root_table_name} the schema requires"
                            " normalization because its shape does not match the actual schema of"
                            " destination table. Arrow table columns will be reordered and missing"
                            " columns will be added if needed."
                        )
                # flatten BEFORE normalization so that schema-evolution placeholder
                # columns are not added until after flattening produces real data
                if has_flatten_hints:
                    # capture original column names before any transformations
                    # (used for collision detection during flattening)
                    original_names = set(batch.schema.names)
                    for col_name, spec in expansion_cols.items():
                        if col_name not in batch.schema.names:
                            continue
                        field = batch.schema.field(col_name)
                        # keep_original-only without flatten_spec: don't flatten, just
                        # serialize struct to JSON string (matches JSON normalizer which
                        # skips expansion and keeps a json.dumps copy)
                        if spec.keep_original and not spec.flatten_spec:
                            if pa.types.is_struct(field.type):
                                struct_arr = batch.column(col_name).combine_chunks()
                                json_strs = _struct_to_json_string(struct_arr)
                                batch = batch.set_column(
                                    batch.schema.get_field_index(col_name),
                                    pa.field(col_name, pa.string(), nullable=True),
                                    json_strs,
                                )
                                columns_schema[col_name] = {
                                    "name": col_name,
                                    "data_type": "json",
                                    "nullable": True,
                                }
                            # string cols: already kept as-is, nothing to do
                            continue
                        if not is_flattenable_column(
                            field.type, col_name, {"x-json-flatten": spec.flatten_spec}
                        ):
                            continue
                        if pa.types.is_struct(field.type):
                            struct_arr = batch.column(col_name).combine_chunks()
                            # save original struct reference for keep_original (serialization deferred)
                            original_struct_arr = struct_arr if spec.keep_original else None
                            # apply hints before flattening
                            if spec.flatten_spec and isinstance(spec.flatten_spec, list):
                                struct_arr = apply_arrow_path_filter(struct_arr, spec.flatten_spec)
                            if spec.max_depth is not None:
                                struct_arr = apply_arrow_depth_limit(struct_arr, spec.max_depth)
                                # max_depth=0 collapses root struct to a string array; replace
                                # the column in-place and skip further struct processing
                                if not pa.types.is_struct(struct_arr.type):
                                    batch = batch.set_column(
                                        batch.schema.get_field_index(col_name),
                                        pa.field(col_name, struct_arr.type, nullable=True),
                                        struct_arr,
                                    )
                                    columns_schema[col_name] = {
                                        "name": col_name,
                                        "data_type": "json",
                                        "nullable": True,
                                    }
                                    continue
                            if spec.force_string:
                                struct_arr = apply_arrow_force_string(struct_arr)
                            # flatten the struct — pass transformed array directly, skip set_column
                            batch, new_types = flatten_struct_column(
                                batch,
                                col_name,
                                schema.naming,
                                flattened_column_types,
                                max_nesting,
                                existing_names=original_names,
                                struct_arr=struct_arr,
                                casefold_identifier=casefold_id,
                            )
                            batch_new_types.update(new_types)
                            flattened_column_types.update(new_types)
                            # remove dropped source column from destination schema
                            if not spec.keep_original:
                                columns_schema.pop(col_name, None)
                                dropped_source_columns.add(col_name)
                            # re-add original struct column as JSON string (match JSON normalizer)
                            if spec.keep_original and original_struct_arr is not None:
                                json_strs = _struct_to_json_string(original_struct_arr)
                                batch = batch.append_column(
                                    pa.field(col_name, pa.string(), nullable=True),
                                    json_strs,
                                )
                                columns_schema[col_name] = {
                                    "name": col_name,
                                    "data_type": "json",
                                    "nullable": True,
                                }
                        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                            _original, expanded = expand_arrow_json_column(
                                batch.column(col_name), spec
                            )
                            if expanded is not None:
                                # collect all arrays and fields for batched table construction
                                new_arrays: List[Any] = []
                                new_fields_list: List[Any] = []
                                for e_col_name in expanded.column_names:
                                    norm_col = schema.naming.normalize_identifier(e_col_name)
                                    flat_name = schema.naming.shorten_fragments(col_name, norm_col)
                                    e_arr = expanded.column(e_col_name).combine_chunks()
                                    e_type = expanded.schema.field(e_col_name).type
                                    flattened_column_types[flat_name] = e_type
                                    batch_new_types[flat_name] = e_type
                                    new_arrays.append(e_arr)
                                    new_fields_list.append(
                                        pa.field(flat_name, e_type, nullable=True)
                                    )
                                # build table: skip string col if !keep_original, keep rest, add expanded
                                keep_arrays = []
                                keep_fields = []
                                for f in batch.schema:
                                    if f.name == col_name and not spec.keep_original:
                                        continue
                                    keep_arrays.append(batch.column(f.name))
                                    keep_fields.append(f)
                                keep_arrays.extend(new_arrays)
                                keep_fields.extend(new_fields_list)
                                batch = pa.Table.from_arrays(
                                    keep_arrays,
                                    schema=pa.schema(keep_fields, metadata=batch.schema.metadata),
                                )
                                # recursively flatten struct columns produced from nested JSON objects
                                flat_names = [
                                    schema.naming.shorten_fragments(
                                        col_name, schema.naming.normalize_identifier(ec)
                                    )
                                    for ec in expanded.column_names
                                ]
                                for flat_name in flat_names:
                                    if flat_name not in batch.schema.names:
                                        continue
                                    flat_field = batch.schema.field(flat_name)
                                    if pa.types.is_struct(flat_field.type):
                                        batch, new_types = flatten_struct_column(
                                            batch,
                                            flat_name,
                                            schema.naming,
                                            flattened_column_types,
                                            max_nesting,
                                            existing_names=original_names,
                                            casefold_identifier=casefold_id,
                                        )
                                        batch_new_types.update(new_types)
                                        flattened_column_types.update(new_types)
                                        # remove intermediate struct column — replaced by children
                                        flattened_column_types.pop(flat_name, None)
                                # remove dropped source column from destination schema
                                if not spec.keep_original:
                                    columns_schema.pop(col_name, None)
                                    dropped_source_columns.add(col_name)
                            elif col_name in dropped_source_columns and not spec.keep_original:
                                # column was dropped in a previous batch — drop from this batch too
                                keep_arrays = []
                                keep_fields = []
                                for f in batch.schema:
                                    if f.name == col_name:
                                        continue
                                    keep_arrays.append(batch.column(f.name))
                                    keep_fields.append(f)
                                batch = pa.Table.from_arrays(
                                    keep_arrays,
                                    schema=pa.schema(keep_fields, metadata=batch.schema.metadata),
                                )
                                columns_schema.pop(col_name, None)
                # normalize AFTER flattening — flattened columns already present,
                # so normalize_py_arrow_item won't add placeholder nulls for them
                if should_normalize:
                    batch = pyarrow.normalize_py_arrow_item(
                        batch, columns_schema, schema.naming, self.config.destination_capabilities
                    )
                    # normalize may remove null columns and set dlt.null_columns metadata
                    self._collect_null_columns_from_arrow_metadata(batch.schema, root_table_name)
                # update columns_schema with batch-newly flattened column types
                if batch_new_types:
                    for col_name, arrow_type in batch_new_types.items():
                        if col_name not in columns_schema:
                            col_type = get_column_type_from_py_arrow(arrow_type)
                            columns_schema[col_name] = {
                                "name": col_name,
                                "nullable": True,
                                **col_type,
                            }
                    batch_new_types.clear()
                self.item_storage.write_data_item(
                    load_id,
                    schema.name,
                    root_table_name,
                    batch,
                    columns_schema,
                )
        # TODO: better to check if anything is in the buffer and skip writing file
        if items_count == 0 and not is_native_arrow_writer:
            self.item_storage.write_empty_items_file(
                load_id,
                schema.name,
                root_table_name,
                columns_schema,
            )

        if flattened_column_types:
            flatten_update = build_flatten_schema_update(
                root_table_name, flattened_column_types, schema.naming
            )
            if root_table_name in flatten_update:
                existing_table_updates = schema_update.setdefault(root_table_name, [])
                for update_entry in flatten_update[root_table_name]:
                    existing_table_updates.append(update_entry)

        # remove dropped source columns from the actual schema — matches JSON normalizer
        # behavior where the source column is absent from destination
        if dropped_source_columns:
            table_schema = schema.get_table(root_table_name)
            for col_name in dropped_source_columns:
                table_schema["columns"].pop(col_name, None)

        return [schema_update]

    def __call__(self, extracted_items_file: str, root_table_name: str) -> List[TSchemaUpdate]:
        self._maybe_cancel()
        from dlt.common.libs.pyarrow import get_parquet_metadata

        # read schema and counts from file metadata

        with self.normalize_storage.extracted_packages.storage.open_file(
            extracted_items_file, "rb"
        ) as f:
            num_rows, arrow_schema = get_parquet_metadata(f)
            file_metrics = DataWriterMetrics(extracted_items_file, num_rows, f.tell(), 0, 0)

        # collect null column metadata from arrow schema
        self._collect_null_columns_from_arrow_metadata(arrow_schema, root_table_name)

        add_dlt_id = self.config.parquet_normalizer.add_dlt_id
        # TODO: add dlt id only if not present in table
        # if we need to add any columns or the file format is not parquet, we can't just import files
        must_rewrite = add_dlt_id or self.item_storage.writer_spec.file_format != "parquet"
        # always load expansion hints — needed regardless of rewrite reason
        expansion_cols = get_json_expansion_columns(self.schema, root_table_name)
        if not must_rewrite and expansion_cols:
            must_rewrite = True
            logger.info(
                f"Table {root_table_name} has x-json-flatten hints; forcing rewrite for flattening"
            )
        if not must_rewrite:
            # in rare cases normalization may be needed
            must_rewrite = pyarrow.should_normalize_arrow_schema(
                arrow_schema, self.schema.get_table_columns(root_table_name), self.schema.naming
            )[0]
        if must_rewrite:
            logger.info(
                f"Table {root_table_name} parquet file {extracted_items_file} must be rewritten:"
                f" add_dlt_id: {add_dlt_id} destination file"
                f" format: {self.item_storage.writer_spec.file_format} or due to required"
                " normalization "
            )
            schema_update = self._write_with_dlt_columns(
                extracted_items_file, root_table_name, add_dlt_id, expansion_cols
            )
            return schema_update

        logger.info(
            f"Table {root_table_name} parquet file {extracted_items_file} will be directly imported"
            " without normalization"
        )
        parts = ParsedLoadJobFileName.parse(extracted_items_file)
        self.item_storage.import_items_file(
            self.load_id,
            self.schema.name,
            parts.table_name,
            self.normalize_storage.extracted_packages.storage.make_full_path(extracted_items_file),
            file_metrics,
        )
        self._report_progress(root_table_name, num_rows)

        return []
