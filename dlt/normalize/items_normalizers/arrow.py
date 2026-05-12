from typing import List, Dict, Set, Any, Optional, Tuple

from dlt.common import logger
from dlt.common.data_writers.writers import ArrowToObjectAdapter
from dlt.common.json import json
from dlt.common.metrics import DataWriterMetrics
from dlt.common.normalizers.json.expansion import expand_json_column, flatten_expanded_dict
from dlt.common.normalizers.json.helpers import get_json_expansion_columns, TJsonColumnExpansionSpec
from dlt.common.normalizers.json.relational import DataItemNormalizer as RelationalNormalizer
from dlt.common.normalizers.utils import generate_dlt_ids
from dlt.common.schema.typing import C_DLT_ID, TColumnSchema, TTableSchemaColumns
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

    def _expand_json_columns_in_batch(
        self,
        batch: "pa.Table",
        json_expansion_cols: Dict[str, TJsonColumnExpansionSpec],
        columns_schema: TTableSchemaColumns,
    ) -> Tuple["pa.Table", List[TColumnSchema]]:
        """Expand JSON string columns in an Arrow batch according to x-json-* hints.

        For each column with expansion hints, parses JSON string values, flattens
        nested dicts into `col__key` flat columns, and adds them to the batch.
        Returns the modified batch and new column schemas for schema updates.
        """
        new_column_schemas: List[TColumnSchema] = []
        cols_to_remove: List[str] = []
        new_cols_arrays: Dict[str, List[Any]] = {}

        for col_name, spec in json_expansion_cols.items():
            try:
                col_idx = batch.schema.get_field_index(col_name)
            except KeyError:
                continue

            col_array = batch.column(col_idx)
            values = col_array.to_pylist()
            num_rows = len(values)

            if not spec.flatten_spec and spec.keep_original:
                # serialize native dict columns (struct type) to JSON string
                serialized = []
                for v in values:
                    if isinstance(v, dict):
                        serialized.append(json.dumps(v))
                    else:
                        serialized.append(v)
                batch = batch.set_column(
                    col_idx,
                    pa.field(col_name, pa.string()),
                    pa.array(serialized, type=pa.string()),
                )
                continue

            # track expanded keys seen in this batch (may vary across rows)
            batch_new_keys: Dict[str, List[Any]] = {}
            for i, raw_value in enumerate(values):
                original, expanded_dict = expand_json_column(
                    raw_value,
                    spec.flatten_spec,
                    spec.keep_original,
                    spec.force_string,
                    spec.max_depth,
                )
                if expanded_dict is not None:
                    flat = flatten_expanded_dict(expanded_dict, naming=self.schema.naming)
                    for flat_key, flat_val in flat.items():
                        batch_new_keys.setdefault(flat_key, [None] * num_rows)[i] = flat_val

                if spec.keep_original:
                    serialized_original = original
                    if isinstance(original, (dict, list)):
                        serialized_original = json.dumps(original)
                    values[i] = serialized_original

            if spec.keep_original:
                batch = batch.set_column(
                    col_idx,
                    pa.field(col_name, pa.string()),
                    pa.array(
                        [json.dumps(v) if isinstance(v, (dict, list)) else v for v in values],
                        type=pa.string(),
                    ),
                )
            else:
                cols_to_remove.append(col_name)

            # gather new column arrays for this batch
            for new_key, col_vals in batch_new_keys.items():
                # prefix new column name with the source column name to avoid collisions
                prefixed_key = f"{col_name}__{new_key}"
                new_cols_arrays.setdefault(prefixed_key, [None] * num_rows)
                # merge: fill in values from this column (may overlap with other columns)
                for i, val in enumerate(col_vals):
                    if val is not None:
                        new_cols_arrays[prefixed_key][i] = val

        if not new_cols_arrays and not cols_to_remove:
            return batch, []

        # drop columns that had keep_original=False
        if cols_to_remove:
            keep_indices = [
                i for i, field in enumerate(batch.schema) if field.name not in cols_to_remove
            ]
            batch = batch.select(keep_indices)

        # add new expanded columns to the batch
        for new_col_name, new_col_vals in new_cols_arrays.items():
            try:
                new_array = pa.array(new_col_vals)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                logger.warning(
                    f"Failed to infer Arrow type for expanded column {new_col_name}: {e}. "
                    "Falling back to string type."
                )
                new_array = pa.array(
                    [str(v) if v is not None else None for v in new_col_vals],
                    type=pa.string(),
                )

            batch = batch.append_column(
                pa.field(new_col_name, new_array.type),
                new_array,
            )

            # register new column in schema and schema update
            try:
                col_type_info = pyarrow.get_column_type_from_py_arrow(new_array.type)
            except Exception:
                col_type_info = {"data_type": "text"}
            new_col_schema: TColumnSchema = {
                "name": new_col_name,
                "nullable": True,
                **col_type_info,
            }
            new_column_schemas.append(new_col_schema)
            columns_schema[new_col_name] = new_col_schema

        return batch, new_column_schemas

    def _write_with_dlt_columns(
        self,
        extracted_items_file: str,
        root_table_name: str,
        add_dlt_id: bool,
    ) -> List[TSchemaUpdate]:
        new_columns: List[Any] = []
        schema = self.schema
        load_id = self.load_id
        schema_update: TSchemaUpdate = {}
        data_normalizer = schema.data_item_normalizer

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
        # check for JSON expansion hints on this table
        json_expansion_cols = get_json_expansion_columns(schema, root_table_name)
        if json_expansion_cols:
            logger.info(
                f"Table {root_table_name} has JSON expansion hints for columns:"
                f" {list(json_expansion_cols)}. JSON column expansion will be applied."
            )
        # if we use adapter to convert arrow to dicts, then normalization is not necessary
        is_native_arrow_writer = not issubclass(self.item_storage.writer_cls, ArrowToObjectAdapter)
        should_normalize: bool = None
        self._maybe_cancel()
        with self.normalize_storage.extracted_packages.storage.open_file(
            extracted_items_file, "rb"
        ) as f:
            for batch in pyarrow.pq_stream_with_new_columns(
                f, new_columns, row_groups_per_read=self.REWRITE_ROW_GROUPS
            ):
                self._maybe_cancel()
                items_count += batch.num_rows
                self._report_progress(root_table_name, batch.num_rows)

                if json_expansion_cols:
                    expanded_batch, new_cols = self._expand_json_columns_in_batch(
                        batch, json_expansion_cols, columns_schema
                    )
                    if new_cols:
                        table_updates = schema_update.setdefault(root_table_name, [])
                        table_updates.append(
                            {"name": root_table_name, "columns": {c["name"]: c for c in new_cols}}
                        )
                    batch = expanded_batch
                    should_normalize = True

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
                if should_normalize:
                    batch = pyarrow.normalize_py_arrow_item(
                        batch, columns_schema, schema.naming, self.config.destination_capabilities
                    )
                    # normalize may remove null columns and set dlt.null_columns metadata
                    self._collect_null_columns_from_arrow_metadata(batch.schema, root_table_name)
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

        return [schema_update]

    def __call__(self, extracted_items_file: str, root_table_name: str) -> List[TSchemaUpdate]:
        self._maybe_cancel()
        # read schema and counts from file metadata
        from dlt.common.libs.pyarrow import get_parquet_metadata

        with self.normalize_storage.extracted_packages.storage.open_file(
            extracted_items_file, "rb"
        ) as f:
            num_rows, arrow_schema = get_parquet_metadata(f)
            file_metrics = DataWriterMetrics(extracted_items_file, num_rows, f.tell(), 0, 0)

        # collect null column metadata from arrow schema
        self._collect_null_columns_from_arrow_metadata(arrow_schema, root_table_name)

        add_dlt_id = self.config.parquet_normalizer.add_dlt_id
        # check for JSON expansion hints — if present, we must rewrite to expand columns
        has_json_expansion = bool(get_json_expansion_columns(self.schema, root_table_name))
        # TODO: add dlt id only if not present in table
        # if we need to add any columns or the file format is not parquet, we can't just import files
        must_rewrite = add_dlt_id or self.item_storage.writer_spec.file_format != "parquet"
        if not must_rewrite:
            # in rare cases normalization may be needed
            must_rewrite = pyarrow.should_normalize_arrow_schema(
                arrow_schema, self.schema.get_table_columns(root_table_name), self.schema.naming
            )[0]
            if has_json_expansion:
                must_rewrite = True
        if must_rewrite:
            logger.info(
                f"Table {root_table_name} parquet file {extracted_items_file} must be rewritten:"
                f" add_dlt_id: {add_dlt_id} destination file"
                f" format: {self.item_storage.writer_spec.file_format} or due to required"
                " normalization "
            )
            schema_update = self._write_with_dlt_columns(
                extracted_items_file, root_table_name, add_dlt_id
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
