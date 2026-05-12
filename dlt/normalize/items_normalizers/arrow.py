from typing import Any, Optional

from dlt.common import logger
from dlt.common.data_writers.writers import ArrowToObjectAdapter
from dlt.common.exceptions import MissingDependencyException
from dlt.common.json import json
from dlt.common.metrics import DataWriterMetrics
from dlt.common.normalizers.json import helpers as normalize_helpers
from dlt.common.normalizers.json.expansion import expand_json_column, flatten_expanded_dict
from dlt.common.normalizers.json.helpers import (
    TJsonColumnExpansionSpec,
    get_json_expansion_columns,
)
from dlt.common.normalizers.json.relational import DataItemNormalizer as RelationalNormalizer
from dlt.common.normalizers.utils import generate_dlt_ids
from dlt.common.runtime.collector import NULL_COLLECTOR, Collector
from dlt.common.schema import Schema, TSchemaUpdate
from dlt.common.schema.typing import C_DLT_ID, TTableSchemaColumns
from dlt.common.schema.utils import dlt_id_column, normalize_table_identifiers
from dlt.common.storages import NormalizeStorage
from dlt.common.storages.data_item_storage import DataItemStorage
from dlt.common.storages.load_package import ParsedLoadJobFileName
from dlt.common.storages.load_storage import LoadStorage
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
        self._null_only_columns: dict[str, set[str]] = {}

    @property
    def null_only_columns(self) -> dict[str, set[str]]:
        return self._null_only_columns

    def _flatten_py_arrow_item(
        self,
        batch: Any,
        root_table_name: str,
        max_nesting: int = 1000,
        json_expansion_cols: Optional[dict[str, TJsonColumnExpansionSpec]] = None,
        columns_schema: Optional[TTableSchemaColumns] = None,
        protected_columns: Optional[set[str]] = None,
    ) -> tuple[Any, bool]:
        """Flatten Arrow batch by expanding JSON string columns and native struct columns.

        Uses a hybrid approach:
          - JSON strings are expanded via ``expand_json_column`` / ``flatten_expanded_dict``
            (per-column ``to_pylist`` conversion for hinted columns only).
          - Native struct columns are flattened zero-copy by extracting child arrays.

        Recurses until *max_nesting* is exhausted or no further changes are made.
        On recursion only struct flattening is applied; JSON expansion runs once.
        """
        if max_nesting <= 0:
            return batch, False

        naming = self.schema.naming
        fields, columns, changed = [], [], False
        json_cols = json_expansion_cols or {}
        protected = protected_columns or set()

        for field, column in zip(batch.schema, batch.columns):
            spec = json_cols.get(field.name)

            if (
                spec
                and spec.flatten_spec
                and (pa.types.is_string(field.type) or pa.types.is_large_string(field.type))
            ):
                changed = True
                num_rows = batch.num_rows

                if spec.keep_original:
                    fields.append(field)
                    columns.append(column)

                batch_new_keys: dict[str, list[Any]] = {}
                for i, raw_value in enumerate(column.to_pylist()):
                    _, expanded_dict = expand_json_column(
                        raw_value,
                        spec.flatten_spec,
                        spec.keep_original,
                        spec.force_string,
                        spec.max_depth,
                    )
                    if expanded_dict is not None:
                        flat = flatten_expanded_dict(expanded_dict, naming=naming)
                        for flat_key, flat_val in flat.items():
                            batch_new_keys.setdefault(flat_key, [None] * num_rows)[i] = flat_val

                norm_field = naming.normalize_path(field.name) if naming else field.name
                for sub_key, sub_vals in batch_new_keys.items():
                    try:
                        sub_array = pa.array(sub_vals)
                    except (pa.ArrowInvalid, pa.ArrowTypeError):
                        sub_array = pa.array(
                            [str(v) if v is not None else None for v in sub_vals],
                            type=pa.string(),
                        )
                    prefixed_name = f"{norm_field}__{sub_key}"
                    fields.append(pa.field(prefixed_name, sub_array.type))
                    columns.append(sub_array)
                continue

            if pa.types.is_struct(field.type):
                norm_field = naming.normalize_path(field.name) if naming else field.name
                is_nested = normalize_helpers.is_nested_type(
                    self.schema, root_table_name, norm_field, max_nesting
                )
                # For Arrow structs, we only skip flattening if explicitly marked as x-nested-type.
                # is_nested_type may return True for destinations supporting JSON, but we want to flatten by default.
                if is_nested and not (
                    columns_schema and columns_schema.get(norm_field, {}).get("x-nested-type")
                ):
                    is_nested = False

                if spec and spec.keep_original:
                    fields.append(field)
                    columns.append(column)
                    if not spec.flatten_spec:
                        continue
                    # keep_original + flatten_spec: protect the kept original from
                    # re-flattening on recursion by adding it to the protected set.
                    protected.add(field.name)

                # On recursion (no spec), skip struct flattening for columns
                # protected by a previous keep_original pass.
                if not spec and field.name in protected:
                    fields.append(field)
                    columns.append(column)
                    continue

                if not is_nested:
                    changed = True
                    for i, child in enumerate(field.type):
                        norm_child = naming.normalize_path(child.name) if naming else child.name
                        new_name = f"{norm_field}__{norm_child}"
                        fields.append(child.with_name(new_name))
                        if isinstance(column, pa.ChunkedArray):
                            columns.append(pa.chunked_array([c.field(i) for c in column.chunks]))
                        else:
                            columns.append(column.field(i))
                    continue

            fields.append(field)
            columns.append(column)

        if not changed:
            return batch, False

        # Recurse for nested structs; JSON expansion already completed on the first pass.
        flattened_batch, _ = self._flatten_py_arrow_item(
            batch.__class__.from_arrays(
                columns, schema=pa.schema(fields, metadata=batch.schema.metadata)
            ),
            root_table_name,
            max_nesting - 1,
            None,
            columns_schema,
            protected_columns=protected,
        )
        return flattened_batch, True

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
        normalized = {self.schema.naming.normalize_path(n) for n in null_col_names}
        self._null_only_columns.setdefault(root_table_name, set()).update(normalized)

    def _write_with_dlt_columns(
        self,
        extracted_items_file: str,
        root_table_name: str,
        add_dlt_id: bool,
        has_nested: bool = False,
    ) -> list[TSchemaUpdate]:
        new_columns: list[Any] = []
        schema = self.schema
        load_id = self.load_id
        schema_update: TSchemaUpdate = {}
        data_normalizer = schema.data_item_normalizer

        if add_dlt_id and isinstance(data_normalizer, RelationalNormalizer):
            partial_table = normalize_table_identifiers(
                {"name": root_table_name, "columns": {C_DLT_ID: dlt_id_column()}},
                schema.naming,
            )
            schema.update_table(partial_table, normalize_identifiers=False)
            schema_update.setdefault(root_table_name, []).append(partial_table)
            # TODO: use get_root_row_id_type to get row id type and generate deterministic
            #  row ids as well (using pandas helper function prepared for scd2)
            #  we could also generate random columns with pandas or duckdb if present
            new_columns.append(
                (
                    -1,
                    pa.field(data_normalizer.c_dlt_id, pa.string(), nullable=False),
                    lambda batch: pa.array(generate_dlt_ids(batch.num_rows)),
                )
            )

        items_count = 0
        columns_schema = schema.get_table_columns(root_table_name)
        # if we use adapter to convert arrow to dicts, then normalization is not necessary
        is_native_arrow_writer = not issubclass(self.item_storage.writer_cls, ArrowToObjectAdapter)
        should_normalize: bool = None
        json_expansion_cols = get_json_expansion_columns(schema, root_table_name)

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

                changed = False

                if json_expansion_cols or has_nested:
                    if json_expansion_cols:
                        logger.info(
                            "Table %s has JSON expansion hints for columns: %s",
                            root_table_name,
                            list(json_expansion_cols.keys()),
                        )
                    batch, changed = self._flatten_py_arrow_item(
                        batch,
                        root_table_name,
                        normalize_helpers.get_table_nesting_level(schema, root_table_name),
                        json_expansion_cols=json_expansion_cols,
                        columns_schema=columns_schema,
                    )

                if changed:
                    new_cols = pyarrow.py_arrow_to_table_schema_columns(batch.schema)
                    diff_cols = {c: v for c, v in new_cols.items() if c not in columns_schema}
                    if diff_cols:
                        partial_table = normalize_table_identifiers(
                            {"name": root_table_name, "columns": diff_cols},
                            schema.naming,
                        )
                        schema.update_table(partial_table, normalize_identifiers=False)
                        schema_update.setdefault(root_table_name, []).append(partial_table)
                        columns_schema = schema.get_table_columns(root_table_name)
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
                        batch,
                        columns_schema,
                        schema.naming,
                        self.config.destination_capabilities,
                    )
                    # normalize may remove null columns and set dlt.null_columns metadata
                    self._collect_null_columns_from_arrow_metadata(batch.schema, root_table_name)

                self.item_storage.write_data_item(
                    load_id, schema.name, root_table_name, batch, columns_schema
                )

        # TODO: better to check if anything is in the buffer and skip writing file
        if items_count == 0 and not is_native_arrow_writer:
            self.item_storage.write_empty_items_file(
                load_id, schema.name, root_table_name, columns_schema
            )

        return [schema_update]

    def __call__(self, extracted_items_file: str, root_table_name: str) -> list[TSchemaUpdate]:
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

        has_nested_structs = any(pa.types.is_struct(f.type) for f in arrow_schema)

        add_dlt_id = self.config.parquet_normalizer.add_dlt_id
        # TODO: add dlt id only if not present in table

        has_json_expansion = bool(get_json_expansion_columns(self.schema, root_table_name))

        # if we need to add any columns or the file format is not parquet, we can't just import files
        must_rewrite = add_dlt_id or self.item_storage.writer_spec.file_format != "parquet"
        if not must_rewrite:
            # in rare cases normalization may be needed
            must_rewrite = pyarrow.should_normalize_arrow_schema(
                arrow_schema,
                self.schema.get_table_columns(root_table_name),
                self.schema.naming,
            )[0]
        if not must_rewrite and (has_json_expansion or has_nested_structs):
            must_rewrite = True

        if must_rewrite:
            logger.info(
                "Table %s parquet file %s must be rewritten (add_dlt_id=%s, format=%s,"
                " json_expansion=%s, has_nested=%s)",
                root_table_name,
                extracted_items_file,
                add_dlt_id,
                self.item_storage.writer_spec.file_format,
                has_json_expansion,
                has_nested_structs,
            )
            return self._write_with_dlt_columns(
                extracted_items_file, root_table_name, add_dlt_id, has_nested=has_nested_structs
            )

        logger.info(
            "Table %s parquet file %s will be directly imported without normalization",
            root_table_name,
            extracted_items_file,
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
