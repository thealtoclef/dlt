import base64
import gzip
import hashlib
from datetime import datetime, date, time  # noqa: I251
from pendulum.tz import UTC
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    List,
)

from dlt import version
from dlt.common.exceptions import MissingDependencyException, DltException
from dlt.common.schema.typing import (
    C_DLT_ID,
    C_DLT_LOAD_ID,
    TColumnSchema,
    TTableSchemaColumns,
    TPartialTableSchema,
)
from dlt.common import logger
from dlt.common.json import json, custom_encode, map_nested_values_in_place
from dlt.common.destination.capabilities import DestinationCapabilitiesContext
from dlt.common.schema.typing import TColumnType
from dlt.common.schema.utils import is_nullable_column, dlt_load_id_column
from dlt.common.time import get_precision_from_datetime_unit
from dlt.common.typing import AnyType, StrStr, TFileOrPath, TDataItems
from dlt.common.normalizers.naming import NamingConvention
from dlt.common.normalizers.json.expansion import (
    apply_force_string,
    filter_by_paths,
    limit_depth,
    parse_json_value,
)

if TYPE_CHECKING:
    from dlt.common.normalizers.json.helpers import TJsonColumnExpansionSpec

try:
    import pyarrow
    import pyarrow.parquet
    import pyarrow.compute
    import pyarrow.dataset
    from pyarrow.parquet import ParquetFile
    from pyarrow import Table
except ModuleNotFoundError:
    raise MissingDependencyException(
        "dlt pyarrow helpers",
        [f"{version.DLT_PKG_NAME}[parquet]"],
        "Install pyarrow to be allowed to load arrow tables, pandas DataFrames and to use parquet"
        " files.",
    )

import ctypes

TAnyArrowItem = Union[pyarrow.Table, pyarrow.RecordBatch]

ARROW_DECIMAL_MAX_PRECISION = 76

MAX_RECURSION_DEPTH = 100
"""Maximum recursion depth for flattening nested struct columns.

Used by `flatten_struct_column` to prevent infinite recursion when encountering
recursive struct types.
"""

_warned_json_expansion: Set[str] = set()
"""Columns for which the JSON-expansion performance warning has already been logged."""


def is_flattenable_column(
    col_type: "pyarrow.DataType",
    col_name: str,
    column_hints: Optional[Dict[str, Any]],
) -> bool:
    """Checks whether a column can be flattened based on its type and hints.

    A column is flattenable if it has a truthy `x-json-flatten` hint and its type
    is `pyarrow.struct` or `pyarrow.string`. Other types with the hint are skipped
    with a logged warning.

    Args:
        col_type: PyArrow data type of the column.
        col_name: Name of the column.
        column_hints: Column-level hints dict, typically from the schema.

    Returns:
        `True` if the column should be flattened, `False` otherwise.
    """
    if column_hints is None or not column_hints:
        return False
    if not column_hints.get("x-json-flatten"):
        return False
    if pyarrow.types.is_struct(col_type):
        return True
    if pyarrow.types.is_string(col_type) or pyarrow.types.is_large_string(col_type):
        return True
    logger.warning(
        f"Column '{col_name}' has x-json-flatten hint but type {col_type} is not struct or string"
        " — skipping"
    )
    return False


def is_empty_struct(col_type: "pyarrow.DataType") -> bool:
    """Checks whether a PyArrow struct type has zero fields.

    Args:
        col_type: PyArrow data type to check.

    Returns:
        `True` if `col_type` is a struct with no fields, `False` otherwise.
    """
    return bool(pyarrow.types.is_struct(col_type) and col_type.num_fields == 0)


class UnsupportedArrowTypeException(DltException):
    """Exception raised when Arrow type conversion failed.

    The setters are used to update the exception with more context
    such as the relevant field and tablea it is caught downstream.
    """

    def __init__(
        self,
        arrow_type: pyarrow.DataType,
        field_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        self.arrow_type = arrow_type
        self._field_name = field_name if field_name else ""
        self._table_name = table_name if table_name else ""

        msg = self.generate_message(self.arrow_type, self._field_name, self._table_name)
        super().__init__(msg)

    @staticmethod
    def generate_message(arrow_type: pyarrow.DataType, field_name: str, table_name: str) -> str:
        msg = f"Arrow type `{arrow_type}`"
        if field_name:
            msg += f" for field `{field_name}`"
        if table_name:
            msg += f" in table `{table_name}`"

        msg += (
            " is unsupported by dlt. See documentation:"
            " https://dlthub.com/docs/dlt-ecosystem/verified-sources/arrow-pandas#supported-arrow-data-types"
        )
        return msg

    def _update_message(self) -> None:
        """Modify the `Exception.args` tuple to update message."""
        msg = self.generate_message(self.arrow_type, self.field_name, self.table_name)
        self.args = (msg,)  # must be a tuple

    @property
    def field_name(self) -> str:
        return self._field_name

    @field_name.setter
    def field_name(self, value: str) -> None:
        self._field_name = value
        self._update_message()

    @property
    def table_name(self) -> str:
        return self._table_name

    @table_name.setter
    def table_name(self, value: str) -> None:
        self._table_name = value
        self._update_message()


class PyToArrowConversionException(DltException):
    """Exception raised when converting data to Arrow based on a TableSchema"""

    def __init__(
        self,
        data_type: Optional[str],
        inferred_arrow_type: Optional[pyarrow.DataType] = None,
        field_name: Optional[str] = None,
        table_name: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        self.data_type = data_type
        self.inferred_arrow_type = inferred_arrow_type
        self._field_name = field_name if field_name else ""
        self._table_name = table_name if table_name else ""
        self._details = details if details else ""

        super().__init__()
        self._update_message()

    @staticmethod
    def generate_message(
        data_type: Optional[str],
        inferred_arrow_type: Optional[pyarrow.DataType],
        field_name: str,
        table_name: str,
        details: str,
    ) -> str:
        msg = "Conversion to arrow failed"
        if field_name:
            msg += f" for field `{field_name}`"
        if table_name:
            msg += f" in table `{table_name}`"

        msg += f" with dlt hint `{data_type=:}` and `{inferred_arrow_type=:}`"
        msg += " " + details
        return msg

    def _update_message(self) -> None:
        """Modify the `Exception.args` tuple to update message."""
        msg = self.generate_message(
            self.data_type,
            self.inferred_arrow_type,
            self.field_name,
            self.table_name,
            self._details,
        )
        self.args = (msg,)  # must be a tuple

    @property
    def field_name(self) -> str:
        return self._field_name

    @field_name.setter
    def field_name(self, value: str) -> None:
        self._field_name = value
        self._update_message()

    @property
    def table_name(self) -> str:
        return self._table_name

    @table_name.setter
    def table_name(self, value: str) -> None:
        self._table_name = value
        self._update_message()

    @property
    def details(self) -> str:
        return self._details

    @details.setter
    def details(self, value: str) -> None:
        self._details = value
        self._update_message()


class ArrowSchemaNormalizationResult(NamedTuple):
    """Named result of should_normalize_arrow_schema

    Fields:
    - should_normalize: whether any normalization is required
    - rename_mapping: mapping from original field names to normalized names
    - rev_mapping: reverse mapping from normalized names to original
    - nullable_updates: fields that require nullable flag updates (by normalized name)
    - columns: potentially filtered/adjusted TTableSchemaColumns
    """

    should_normalize: bool
    rename_mapping: StrStr
    rev_mapping: Dict[str, str]
    nullable_updates: Dict[str, bool]
    columns: TTableSchemaColumns


def get_py_arrow_datatype(
    column: TColumnType,
    caps: DestinationCapabilitiesContext,
    tz: str,
) -> Any:
    column_type = column["data_type"]
    if column_type == "text":
        return pyarrow.string()
    elif column_type == "double":
        return pyarrow.float64()
    elif column_type == "bool":
        return pyarrow.bool_()
    elif column_type == "timestamp":
        # sets timezone to None when timezone hint is false
        timezone = tz if column.get("timezone", True) else None
        precision = column.get("precision")
        if precision is None:
            precision = caps.timestamp_precision
        return get_py_arrow_timestamp(precision, timezone)
    elif column_type == "bigint":
        return get_pyarrow_int(column.get("precision"))
    elif column_type == "binary":
        return pyarrow.binary(column.get("precision") or -1)
    elif column_type == "json":
        if (nested_type := column.get("x-nested-type")) and caps.supports_nested_types:
            return deserialize_type(nested_type)  # type: ignore[arg-type]
        else:
            return pyarrow.string()
    elif column_type == "decimal":
        precision, scale = column.get("precision"), column.get("scale")
        if (precision is None) and (scale is None):
            precision_tuple = caps.decimal_precision
        elif precision is None:
            precision_tuple = caps.decimal_precision
            logger.warning(
                f"Received decimal column hint `scale={scale}`, but `precision` not set. Will"
                " assume default destination capability `(precision, scale) ="
                f" {caps.decimal_precision}`"
            )
        elif scale is None:
            # setting scale to 0 when unspecified is a common practice across databases
            precision_tuple = (precision, 0)
            logger.warning(
                f"Received decimal column hint `precision={precision}`, but `scale` not set. "
                "Will assume default destination capability `scale=0`"
            )
        else:
            precision_tuple = (precision, scale)

        return get_py_arrow_numeric(precision_tuple)
    elif column_type == "wei":
        return get_py_arrow_numeric(caps.wei_precision)
    elif column_type == "date":
        return pyarrow.date32()
    elif column_type == "time":
        precision = column.get("precision")
        if precision is None:
            precision = caps.timestamp_precision
        return get_py_arrow_time(precision)
    else:
        raise ValueError(column_type)


def get_py_arrow_timestamp(precision: int, tz: str) -> Any:
    tz = tz if tz else None
    if precision == 0:
        return pyarrow.timestamp("s", tz=tz)
    if precision <= 3:
        return pyarrow.timestamp("ms", tz=tz)
    if precision <= 6:
        return pyarrow.timestamp("us", tz=tz)
    return pyarrow.timestamp("ns", tz=tz)


def get_py_arrow_time(precision: int) -> Any:
    if precision == 0:
        return pyarrow.time32("s")
    elif precision <= 3:
        return pyarrow.time32("ms")
    elif precision <= 6:
        return pyarrow.time64("us")
    return pyarrow.time64("ns")


def get_py_arrow_numeric(precision: Tuple[int, int]) -> Any:
    if precision[0] <= 38:
        return pyarrow.decimal128(*precision)
    if precision[0] <= 76:
        return pyarrow.decimal256(*precision)
    # for higher precision use max precision and trim scale to leave the most significant part
    return pyarrow.decimal256(76, max(0, 76 - (precision[0] - precision[1])))


def get_pyarrow_int(precision: Optional[int]) -> Any:
    if precision is None:
        return pyarrow.int64()
    if precision <= 8:
        return pyarrow.int8()
    elif precision <= 16:
        return pyarrow.int16()
    elif precision <= 32:
        return pyarrow.int32()
    return pyarrow.int64()


def get_column_type_from_py_arrow(dtype: pyarrow.DataType) -> TColumnType:
    """Returns (data_type, precision, scale) tuple from pyarrow.DataType"""
    if pyarrow.types.is_string(dtype) or pyarrow.types.is_large_string(dtype):
        return dict(data_type="text")
    elif pyarrow.types.is_floating(dtype):
        return dict(data_type="double")
    elif pyarrow.types.is_boolean(dtype):
        return dict(data_type="bool")
    elif pyarrow.types.is_timestamp(dtype):
        precision = get_precision_from_datetime_unit(dtype.unit)
        timestamp_d: TColumnType = dict(data_type="timestamp", precision=precision)
        if dtype.tz is None:
            timestamp_d["timezone"] = False
        return timestamp_d
    elif pyarrow.types.is_date(dtype):
        return dict(data_type="date")
    elif pyarrow.types.is_time(dtype):
        # Time fields in schema are `DataType` instead of `Time64Type` or `Time32Type`
        precision = get_precision_from_datetime_unit(dtype.unit)
        return dict(data_type="time", precision=precision)
    elif pyarrow.types.is_integer(dtype):
        result: TColumnType = dict(data_type="bigint")
        if dtype.bit_width != 64:  # 64bit is a default bigint
            result["precision"] = dtype.bit_width
        return result
    elif pyarrow.types.is_fixed_size_binary(dtype):
        return dict(data_type="binary", precision=dtype.byte_width)
    elif pyarrow.types.is_binary(dtype) or pyarrow.types.is_large_binary(dtype):
        return dict(data_type="binary")
    elif pyarrow.types.is_decimal(dtype):
        return dict(data_type="decimal", precision=dtype.precision, scale=dtype.scale)
    elif pyarrow.types.is_nested(dtype):
        return get_nested_column_type_from_py_arrow(dtype)
    elif pyarrow.types.is_dictionary(dtype):
        # Dictionary types are essentially categorical encodings. The underlying value_type
        # dictates the "logical" type. We simply delegate to the underlying value_type.
        return get_column_type_from_py_arrow(dtype.value_type)
    elif pyarrow.types.is_null(dtype):
        return {}  # incomplete column, no data_type
    else:
        raise UnsupportedArrowTypeException(arrow_type=dtype)


def py_arrow_to_table_schema_columns(schema: pyarrow.Schema) -> TTableSchemaColumns:
    """Convert a PyArrow schema to a table schema columns dict.

    Args:
        schema (pyarrow.Schema): pyarrow schema

    Returns:
        TTableSchemaColumns: table schema columns
    """
    result: TTableSchemaColumns = {}
    for field in schema:
        try:
            converted_type = get_column_type_from_py_arrow(field.type)
        except UnsupportedArrowTypeException as e:
            # modify attributes inplace to add context instead of re-raising with `raise e`
            e.field_name = field.name
            raise

        result[field.name] = {
            "name": field.name,
            "nullable": field.nullable,
            **converted_type,
        }
    return result


def get_nested_column_type_from_py_arrow(dtype: pyarrow.DataType) -> TColumnType:
    """Creates `json` dlt data type with nested type structure in `x-nested-type` hint.
    Currently the only recognized nested type format is arrow-ipc
    """
    return {"data_type": "json", "x-nested-type": serialize_type(dtype)}  # type: ignore[typeddict-unknown-key]


def serialize_type(dtype: pyarrow.DataType) -> str:
    """Serializes arrow type via arrow ipc as base64 str"""
    schema = pyarrow.schema([pyarrow.field("c", dtype)])
    return "arrow-ipc:" + base64.b64encode(gzip.compress(schema.serialize().to_pybytes())).decode(
        "ascii"
    )


def deserialize_type(type_str: str) -> pyarrow.DataType:
    if type_str.startswith("arrow-ipc:"):
        decompressed = gzip.decompress(base64.b64decode(type_str[10:]))
        schema = pyarrow.ipc.read_schema(pyarrow.BufferReader(decompressed))
        return schema.field(0).type
    else:
        raise TypeError("Cannot deserialize pyarrow type, only arrow-ipc is supported")


def build_flatten_schema_update(
    table_name: str,
    flattened_columns: Dict[str, pyarrow.DataType],
    naming: NamingConvention,
) -> Dict[str, List[TPartialTableSchema]]:
    """Builds a `TSchemaUpdate` from flattened columns with their Arrow data types.

    For each column, infers dlt data type from the Arrow type and constructs a column
    schema dict. If `flattened_columns` is empty, returns an empty dict.

    Args:
        table_name (str): Name of the root table to update.
        flattened_columns (Dict[str, pyarrow.DataType]): Mapping of column names to their
            Arrow data types.
        naming (NamingConvention): Naming convention used to normalize column paths.

    Returns:
        Dict[str, List[Dict[str, Any]]]: `TSchemaUpdate` mapping the table name to a list
            of partial table schemas, each containing `name` and `columns`.
    """
    if not flattened_columns:
        return {}

    columns_dict: Dict[str, TColumnSchema] = {}
    for col_name, arrow_type in flattened_columns.items():
        if pyarrow.types.is_struct(arrow_type):
            type_result = get_nested_column_type_from_py_arrow(arrow_type)
        else:
            type_result = get_column_type_from_py_arrow(arrow_type)

        normalized_name = naming.normalize_path(col_name)
        columns_dict[normalized_name] = {
            "name": normalized_name,
            "nullable": True,
            **type_result,
        }

    return {
        table_name: [
            {
                "name": table_name,
                "columns": columns_dict,
            }
        ]
    }


def remove_null_columns(item: TAnyArrowItem) -> TAnyArrowItem:
    """Remove all columns of datatype pyarrow.null() from the table or record batch.
    Stores removed column names in arrow schema metadata under 'dlt.null_columns' key.
    """
    null_col_names = [field.name for field in item.schema if pyarrow.types.is_null(field.type)]
    if not null_col_names:
        return item
    item = remove_columns(item, null_col_names)
    return add_arrow_metadata(item, {"dlt.null_columns": json.dumps(null_col_names)})


def remove_null_columns_from_schema(schema: pyarrow.Schema) -> Tuple[pyarrow.Schema, bool]:
    """Remove all columns of datatype pyarrow.null() from the schema"""
    fields: List[pyarrow.field] = []
    contains_null: bool = False
    for field in schema:
        if pyarrow.types.is_null(field.type):
            contains_null = True
        else:
            fields.append(field)
    return pyarrow.schema(fields), contains_null


def remove_columns(item: TAnyArrowItem, columns: Sequence[str]) -> TAnyArrowItem:
    """Remove `columns` from Arrow `item`"""
    if not columns:
        return item

    if isinstance(item, pyarrow.Table):
        return item.drop(columns)
    elif isinstance(item, pyarrow.RecordBatch):
        # NOTE: select is available in pyarrow 12 an up
        return item.select([n for n in item.schema.names if n not in columns])  # reverse selection
    else:
        raise ValueError(item)


def append_column(item: TAnyArrowItem, name: str, data: Any) -> TAnyArrowItem:
    """Appends new column to Table or RecordBatch"""
    if isinstance(item, pyarrow.Table):
        return item.append_column(name, data)
    elif isinstance(item, pyarrow.RecordBatch):
        new_field = pyarrow.field(name, data.type)
        return pyarrow.RecordBatch.from_arrays(
            item.columns + [data], schema=item.schema.append(new_field)
        )
    else:
        raise ValueError(item)


def rename_columns(item: TAnyArrowItem, new_column_names: Sequence[str]) -> TAnyArrowItem:
    """Rename arrow columns on Table or RecordBatch, returns same data but with renamed schema"""

    if list(item.schema.names) == list(new_column_names):
        # No need to rename
        return item

    if isinstance(item, pyarrow.Table):
        return item.rename_columns(new_column_names)
    elif isinstance(item, pyarrow.RecordBatch):
        new_fields = [
            field.with_name(new_name) for new_name, field in zip(new_column_names, item.schema)
        ]
        return pyarrow.RecordBatch.from_arrays(item.columns, schema=pyarrow.schema(new_fields))
    else:
        raise TypeError(f"Unsupported data item type: `{type(item)}`")


def fill_empty_source_column_values_with_placeholder(
    table: pyarrow.Table, source_columns: List[str], placeholder: str
) -> pyarrow.Table:
    """
    Replaces empty strings and null values in the specified source columns of an Arrow table with a placeholder string.

    Args:
        table (pa.Table): The input Arrow table.
        source_columns (List[str]): A list of column names to replace empty strings and null values in.
        placeholder (str): The placeholder string to use for replacement.

    Returns:
        pyarrow.Table: The modified Arrow table with empty strings and null values replaced in the specified columns.
    """
    for col_name in source_columns:
        column = table[col_name]
        filled_column = pyarrow.compute.fill_null(column, fill_value=placeholder)
        new_column = pyarrow.compute.replace_substring_regex(
            filled_column, pattern=r"^$", replacement=placeholder
        )
        table = table.set_column(table.column_names.index(col_name), col_name, new_column)
    return table


def should_normalize_arrow_schema(
    schema: pyarrow.Schema,
    columns: TTableSchemaColumns,
    naming: NamingConvention,
) -> ArrowSchemaNormalizationResult:
    """Figure out if any of the normalization steps must be executed. This prevents
    from rewriting arrow tables when no changes are needed. Refer to `normalize_py_arrow_item`
    for a list of normalizations. Note that `column` must be already normalized.
    """
    schema, contains_null_cols = remove_null_columns_from_schema(schema)

    rename_mapping = get_normalized_arrow_fields_mapping(schema, naming)
    # no clashes in rename ensured above
    rev_mapping = {v: k for k, v in rename_mapping.items()}
    nullable_mapping = {k: is_nullable_column(v) for k, v in columns.items()}

    # Key is the renamed column name
    nullable_updates: Dict[str, bool] = {}
    column_cast: Dict[str, bool] = {}
    for field in schema:
        norm_name = rename_mapping[field.name]
        # All fields from arrow schema that have nullable set to different value than in columns
        if norm_name in nullable_mapping and field.nullable != nullable_mapping[norm_name]:
            nullable_updates[norm_name] = nullable_mapping[norm_name]
        # Detect arrow columns that require to be normalized
        if norm_name in columns and should_normalize_py_arrow_item_column(
            columns[norm_name], field.type
        ):
            column_cast[norm_name] = True

    dlt_load_id_col = naming.normalize_identifier(C_DLT_LOAD_ID)
    dlt_id_col = naming.normalize_identifier(C_DLT_ID)
    dlt_columns = {dlt_load_id_col, dlt_id_col}

    # remove all columns that are dlt columns but are not present in arrow schema. we do not want to add such columns
    # that should happen in the normalizer
    columns = {
        name: column
        for name, column in columns.items()
        if name not in dlt_columns or name in rev_mapping
    }

    # check if nothing to rename
    skip_normalize = (
        (list(rename_mapping.keys()) == list(rename_mapping.values()) == list(columns.keys()))
        and not nullable_updates
        and not contains_null_cols
        and not column_cast
    )
    return ArrowSchemaNormalizationResult(
        not skip_normalize,
        rename_mapping,
        rev_mapping,
        nullable_updates,
        columns,
    )


def normalize_py_arrow_item(
    item: TAnyArrowItem,
    columns: TTableSchemaColumns,
    naming: NamingConvention,
    caps: DestinationCapabilitiesContext,
) -> TAnyArrowItem:
    """Normalize arrow `item` schema according to the `columns`. Note that
    columns must be already normalized.

    0. columns with no data type will be dropped
    1. arrow schema field names will be normalized according to `naming`
    2. arrows columns will be reordered according to `columns`
    3. empty columns will be inserted if they are missing, types will be generated using `caps`
    4. arrow columns with different nullability than corresponding schema columns will be updated
    5. timestamps will be normalized according to timezone flag in dlt column schema

    NOTE: nullability is not enforced. it is up to destination to do that.
    """
    item = remove_null_columns(item)
    schema = item.schema
    should_normalize, rename_mapping, rev_mapping, nullable_updates, columns = (
        should_normalize_arrow_schema(schema, columns, naming)
    )
    if not should_normalize:
        return item

    new_fields = []
    new_columns = []

    for column_name, column in columns.items():
        # get original field name
        field_name = rev_mapping.pop(column_name, column_name)
        if field_name in rename_mapping:
            idx = schema.get_field_index(field_name)
            new_field = schema.field(idx).with_name(column_name)
            if column_name in nullable_updates:
                # Set field nullable to match column
                new_field = new_field.with_nullable(nullable_updates[column_name])

            # coerce type
            new_type, new_arrow_column = normalize_py_arrow_item_column(
                column, new_field.type, item.column(idx)
            )

            # use renamed field
            new_fields.append(new_field.with_type(new_type))
            new_columns.append(new_arrow_column)
        else:
            # column does not exist in pyarrow. create empty field and column
            new_field = pyarrow.field(
                column_name,
                get_py_arrow_datatype(column, caps, "UTC"),
                nullable=is_nullable_column(column),
            )
            new_fields.append(new_field)
            new_columns.append(pyarrow.nulls(item.num_rows, type=new_field.type))

    # add the remaining columns
    for column_name, field_name in rev_mapping.items():
        idx = schema.get_field_index(field_name)
        # use renamed field
        new_fields.append(schema.field(idx).with_name(column_name))
        new_columns.append(item.column(idx))

    # preserve schema metadata (e.g. dlt.null_columns) through normalization rebuild
    return item.__class__.from_arrays(
        new_columns, schema=pyarrow.schema(new_fields, metadata=item.schema.metadata)
    )


def should_normalize_py_arrow_item_column(
    column: TColumnSchema, arrow_type: pyarrow.DataType
) -> bool:
    # Only handle timestamp columns
    if not pyarrow.types.is_timestamp(arrow_type):
        return False

    current_tz = arrow_type.tz
    target_tz = "UTC" if column.get("timezone", True) else None

    # normalize if tz different
    return current_tz != target_tz  # type: ignore[no-any-return]


def normalize_py_arrow_item_column(
    column: TColumnSchema, arrow_type: pyarrow.Field, arrow_column: pyarrow.Array
) -> Tuple[pyarrow.DataType, pyarrow.Array]:
    """Normalize arrow timestamp column timezone according to dlt schema column convention.

    Args:
        column: dlt column schema with timezone hint
        arrow_type: actual PyArrow data type
        arrow_column: actual PyArrow column data

    Returns:
        Tuple of (modified_type, modified_column) or (arrow_field, arrow_column) if no changes needed
    """
    if not should_normalize_py_arrow_item_column(column, arrow_type):
        return arrow_type, arrow_column

    unit = arrow_type.unit
    current_tz = arrow_type.tz
    target_tz = "UTC" if column.get("timezone", True) else None

    if target_tz == "UTC":
        # Need tz-aware UTC
        if current_tz is None:
            # Attach UTC without shifting (values already represent UTC)
            col = pyarrow.compute.assume_timezone(
                arrow_column, "UTC", ambiguous="latest", nonexistent="latest"
            )
        else:
            # Metadata-only cast to UTC tz
            col = pyarrow.compute.cast(arrow_column, pyarrow.timestamp(unit, "UTC"))
    else:
        # Need naive
        if current_tz is None:
            col = arrow_column  # already naive
        else:
            # Make naive wall-clock; if you want naive-in-UTC, ensure tz metadata is UTC first
            if current_tz != "UTC":
                arrow_column = pyarrow.compute.cast(arrow_column, pyarrow.timestamp(unit, "UTC"))
            col = pyarrow.compute.local_timestamp(arrow_column)

    return pyarrow.timestamp(unit, target_tz), col


def add_dlt_load_id_column(
    item: TAnyArrowItem,
    columns: TTableSchemaColumns,
    caps: DestinationCapabilitiesContext,
    naming: NamingConvention,
    load_id: str,
) -> TAnyArrowItem:
    """
    Adds or replaces the `_dlt_load_id` column.
    """
    dlt_load_id_col_name = naming.normalize_identifier(C_DLT_LOAD_ID)

    idx = item.schema.get_field_index(dlt_load_id_col_name)
    # if the column already exists, get rid of it
    if idx != -1:
        item = remove_columns(item, dlt_load_id_col_name)

    # get pyarrow.string() type
    pyarrow_string = get_py_arrow_datatype(
        # use already existing column definition or use the default
        # NOTE: the existence of the load id column is ensured by this time
        # since it is added in _compute_tables before files are written
        (
            columns[dlt_load_id_col_name]
            if dlt_load_id_col_name in columns
            else dlt_load_id_column()
        ),
        caps,
        "UTC",  # ts is irrelevant to get pyarrow string, but it's required...
    )

    # Check if destination supports dictionary encoding (default True if not specified)
    use_dictionary = True
    if caps.parquet_format is not None:
        use_dictionary = caps.parquet_format.supports_dictionary_encoding

    # add the column with the new value at previous index or append
    item = add_constant_column(
        item=item,
        name=dlt_load_id_col_name,
        data_type=pyarrow_string,
        value=load_id,
        nullable=(
            columns[dlt_load_id_col_name]["nullable"]
            if dlt_load_id_col_name in columns
            else dlt_load_id_column()["nullable"]
        ),
        index=idx,
        use_dictionary=use_dictionary,
    )

    return item


def get_normalized_arrow_fields_mapping(schema: pyarrow.Schema, naming: NamingConvention) -> StrStr:
    """Normalizes schema field names and returns mapping from original to normalized name. Raises on name collisions"""
    # use normalize_path to be compatible with how regular columns are normalized in dlt.Schema
    norm_f = naming.normalize_path
    name_mapping = {n.name: norm_f(n.name) for n in schema}
    # verify if names uniquely normalize
    normalized_names = set(name_mapping.values())
    if len(name_mapping) != len(normalized_names):
        raise NameNormalizationCollision(
            f"Arrow schema fields normalized from:\n{list(name_mapping.keys())}:\nto:\n"
            f" {list(normalized_names)}"
        )
    return name_mapping


def columns_to_arrow(
    columns: TTableSchemaColumns,
    caps: DestinationCapabilitiesContext,
    timestamp_timezone: str = "UTC",
) -> pyarrow.Schema:
    """Convert a table schema columns dict to a pyarrow schema.

    Args:
        columns (TTableSchemaColumns): table schema columns

    Returns:
        pyarrow.Schema: pyarrow schema

    """
    caps = caps or DestinationCapabilitiesContext.generic_capabilities()
    return pyarrow.schema(
        [
            pyarrow.field(
                name,
                get_py_arrow_datatype(
                    schema_item,
                    caps,
                    timestamp_timezone,
                ),
                nullable=schema_item.get("nullable", True),
            )
            for name, schema_item in columns.items()
            if schema_item.get("data_type") is not None
        ]
    )


def get_parquet_metadata(parquet_file: TFileOrPath) -> Tuple[int, pyarrow.Schema]:
    """Gets parquet file metadata (including row count and schema)

    Args:
        parquet_file (str): path to parquet file

    Returns:
        FileMetaData: file metadata
    """
    with pyarrow.parquet.ParquetFile(parquet_file) as reader:
        return reader.metadata.num_rows, reader.schema_arrow


def is_arrow_item(item: Any) -> bool:
    return isinstance(item, (pyarrow.Table, pyarrow.RecordBatch))


def to_arrow_scalar(value: Any, arrow_type: pyarrow.DataType) -> Any:
    """Converts python value to an arrow compute friendly version"""
    return pyarrow.scalar(value, type=arrow_type)


def from_arrow_scalar(arrow_value: pyarrow.Scalar) -> Any:
    """Converts arrow scalar into Python type."""
    return arrow_value.as_py()


TNewColumns = Sequence[Tuple[int, pyarrow.Field, Callable[[pyarrow.Table], Iterable[Any]]]]
"""Sequence of tuples: (field index, field, generating function)"""


def add_constant_column(
    item: TAnyArrowItem,
    name: str,
    data_type: pyarrow.DataType,
    value: Any = None,
    nullable: bool = True,
    index: int = -1,
    use_dictionary: bool = True,
) -> TAnyArrowItem:
    """Add column with a single value to the table.

    Args:
        item: Arrow table or record batch
        name: The new column name
        data_type: The data type of the new column
        nullable: Whether the new column is nullable
        value: The value to fill the new column with
        index: The index at which to insert the new column. Defaults to -1 (append)
        use_dictionary: When True (default), creates a dictionary-encoded column which is
            memory-efficient for repeated values. Set to False for destinations that don't
            support dictionary types (e.g., ADBC drivers for MSSQL).
    Note:
        When use_dictionary=True, the column is created as a DictionaryArray with int8 indices.
        When use_dictionary=False, a regular array filled with the repeated value is created.
    """
    if use_dictionary:
        dictionary = pyarrow.array([value], type=data_type)
        zero_buffer = pyarrow.allocate_buffer(item.num_rows, resizable=False)
        ctypes.memset(zero_buffer.address, 0, item.num_rows)

        indices = pyarrow.Array.from_buffers(
            pyarrow.int8(),
            item.num_rows,
            [None, zero_buffer],  # None validity bitmap means arrow assumes all entries are valid
        )
        column_array = pyarrow.DictionaryArray.from_arrays(indices, dictionary)
    else:
        # Create a regular array filled with the repeated value
        column_array = pyarrow.repeat(pyarrow.scalar(value, type=data_type), item.num_rows)

    field = pyarrow.field(name, column_array.type, nullable=nullable)
    if index == -1:
        return item.append_column(field, column_array)
    return item.add_column(index, field, column_array)


def pq_stream_with_new_columns(
    parquet_file: TFileOrPath, columns: TNewColumns, row_groups_per_read: int = 1
) -> Iterator[pyarrow.Table]:
    """Add column(s) to the table in batches.

    The table is read from parquet `row_groups_per_read` row groups at a time

    Args:
        parquet_file: path or file object to parquet file
        columns: list of columns to add in the form of (insertion index, `pyarrow.Field`, column_value_callback)
            The callback should accept a `pyarrow.Table` and return an array of values for the column.
        row_groups_per_read: number of row groups to read at a time. Defaults to 1.

    Yields:
        `pyarrow.Table` objects with the new columns added.
    """
    with pyarrow.parquet.ParquetFile(parquet_file) as reader:
        n_groups = reader.num_row_groups
        # Iterate through n row groups at a time
        for i in range(0, n_groups, row_groups_per_read):
            tbl: pyarrow.Table = reader.read_row_groups(
                range(i, min(i + row_groups_per_read, n_groups))
            )
            for idx, field, gen_ in columns:
                if idx == -1:
                    tbl = tbl.append_column(field, gen_(tbl))
                else:
                    tbl = tbl.add_column(idx, field, gen_(tbl))
            yield tbl


def cast_arrow_schema_types(
    schema: pyarrow.Schema,
    type_map: Dict[Callable[[pyarrow.DataType], bool], Callable[..., pyarrow.DataType]],
) -> pyarrow.Schema:
    """Returns type-casted Arrow schema.

    Replaces data types for fields matching a type check in `type_map`.
    Type check functions in `type_map` are assumed to be mutually exclusive, i.e.
    a data type does not match more than one type check function.
    """
    for i, e in enumerate(schema.types):
        for type_check, cast_type in type_map.items():
            if type_check(e):
                if callable(cast_type):
                    cast_type = cast_type(e)
                adjusted_field = schema.field(i).with_type(cast_type)
                schema = schema.set(i, adjusted_field)
                break  # if type matches type check, do not do other type checks
    return schema


def concat_batches_and_tables_in_order(
    tables_or_batches: Iterable[Union[pyarrow.Table, pyarrow.RecordBatch]],
    promote_options: str = "none",
) -> pyarrow.Table:
    """Concatenate iterable of tables and batches into a single table, preserving row order.

    Args:
        promote_options: PyArrow concat_tables promote_options. "none" (default) requires identical
            schemas and enables zero-copy concat. "default" promotes within type families (e.g.
            int32→int64). "permissive" promotes across families (e.g. int64→double).
    """
    batches = []
    tables = []
    for item in tables_or_batches:
        if isinstance(item, pyarrow.RecordBatch):
            batches.append(item)
        elif isinstance(item, pyarrow.Table):
            if batches:
                tables.append(pyarrow.Table.from_batches(batches))
                batches = []
            tables.append(item)
        else:
            raise ValueError(f"Unsupported type: `{type(item)}`")
    if batches:
        tables.append(pyarrow.Table.from_batches(batches))
    # "none" ensures 0 copy concat; "default"/"permissive" allow type promotion
    return pyarrow.concat_tables(tables, promote_options=promote_options)


def transpose_rows_to_columns(
    rows: TDataItems, column_names: Iterable[str]
) -> dict[str, Any]:  # dict[str, np.ndarray]
    """Transpose rows (data items) into columns (numpy arrays). Returns a dictionary of {column_name: column_data}

    Uses pandas if available. Otherwise, use numpy, which is slower
    """
    try:
        from dlt.common.libs.numpy import numpy as np
    except MissingDependencyException:
        raise MissingDependencyException(
            "dlt pyarrow helpers", ["numpy"], "Numpy is required for this pyarrow operation"
        )

    try:
        from pandas._libs import lib

        # NOTE: this is part of public interface now via DataFrame.from_records()
        pivoted_rows = lib.to_object_array_tuples(rows).T
    except ImportError:
        logger.info(
            "Pandas not installed, reverting to numpy.asarray to create a table which is slower"
        )
        pivoted_rows = np.asarray(rows, dtype="object", order="K").T
    return {
        column_name: data.ravel()
        for column_name, data in zip(column_names, np.vsplit(pivoted_rows, len(pivoted_rows)))
    }


def convert_numpy_to_arrow(
    column_data: Any,  # 1-dimensional np.ndarray
    caps: DestinationCapabilitiesContext,
    column_schema: TColumnSchema,
    tz: str,
    safe_arrow_conversion: bool,
) -> Any:  # pyarrow.Array
    """Convert a numpy array to a pyarrow array.

    Args:
        rows: data items
        caps: capabilities of the storage backend
        columns: dlt hints about the table columns (e.g., data type, nullabe)
        tz: time zone identifier
        safe_arrow_conversion: if False, truncation and loss of precision is allowed
            ref: https://arrow.apache.org/docs/python/generated/pyarrow.compute.CastOptions.html#pyarrow.compute.CastOptions

    Returns:
        an arrow Array
    """
    from dlt.common.libs.pyarrow import pyarrow as pa

    dlt_data_type = column_schema.get("data_type")
    inferred_arrow_type = (
        get_py_arrow_datatype(column_schema, caps, tz) if dlt_data_type is not None else None
    )
    inferred_array = None

    # base case (0): allow pyarrow to infer type, or create array of dlt specified type
    try:
        # type=None lets pyarrow infer the type from the data
        inferred_array = pa.array(column_data, type=inferred_arrow_type)
    # detailed error handling should happen in fallback cases
    except (pa.ArrowInvalid, pyarrow.ArrowTypeError):
        logger.warning(
            f"Default conversion to `{inferred_arrow_type}` for `data_type={dlt_data_type}` failed."
            " Using fallback strategies."
        )

    def _first_non_none(types_: Sequence[AnyType]) -> bool:
        # determine the first non-null value to guide fallbacks
        first_non_none = None
        for _v in column_data:
            if _v is not None:
                first_non_none = _v
                break
        return isinstance(first_non_none, types_)  # type: ignore[arg-type]

    # case 1 & 2: pyarrow infers the type (e.g., float, string) THEN cast it to the dlt specified type; less constraints than the base case
    # for example, this handles when backends return decimals as floats or strings
    if inferred_array is None and dlt_data_type is not None:
        try:
            raw_array = pa.array(column_data)
            inferred_array = cast_arrow_array_as_column_schema(
                raw_array, caps, column_schema, tz, safe_arrow_conversion
            )
        except PyToArrowConversionException:
            raise
        except Exception as e:
            if (
                (dlt_data_type in ("text", "json"))
                and pa.types.is_string(inferred_arrow_type)
                and _first_non_none((list, dict))
            ):
                # this is handled by fallback case 2 for text/json with nested values
                logger.warning(
                    f"Received `data_type='{dlt_data_type}'`, data requires serialization to"
                    " string, slowing extraction. Cast the JSON field to STRING in your database"
                    " system to improve performance. For example, create and extract data from an"
                    " SQL VIEW that SELECT with CAST."
                )
            else:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details=f"This conversion is currently unsupported by dlt ({str(e)})",
                ) from e

    # case 2: encode Sequence and Mapping types (list, tuples, set, dict, etc.) to JSON strings
    # This logic needs to be before case 3, otherwise pyarrow might infer the deserialized JSON object as a `pyarrow.struct` instead of `pyarrow.string`
    if inferred_array is None and dlt_data_type in (
        "json",
        "text",
    ):
        # depending on the backend, JSON columns are inferred as data_type="text"
        json_serialized_values: list[Union[bytes, None]] = []
        for value in column_data:
            if value is None:
                json_serialized_values.append(None)
                continue
            try:
                json_serialized_values.append(json.dumpb(value))
            except TypeError as e:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details="dlt failed to a JSON-serializable type.",
                ) from e

        inferred_array = pa.array(json_serialized_values).cast(pa.string())

    # case 3: encode Python types unsupported by Arrow. Simple types are converted to strings and complex types to common structures (dict, list)
    # This catches specialized SQL types like `Ranges`
    if inferred_array is None and dlt_data_type is None:
        try:
            inferred_array = pa.array(column_data)
        except (pa.ArrowInvalid, pyarrow.ArrowTypeError) as e:
            logger.warning(
                f"Type can't be inferred by `pyarrow` {e.args[0]}. Values will be encoded as in a"
                " loop, slowing extraction."
            )
            encoded_values: list[Union[None, Mapping[Any, Any], Sequence[Any], str]] = []
            for value in column_data:
                if value is None:
                    encoded_values.append(None)
                    continue
                try:
                    # the 3 types match those supported by `map_nested_in_place()`
                    if isinstance(value, (tuple, dict, list)):
                        encoded_value = map_nested_values_in_place(custom_encode, value)
                    # convert set to list
                    elif isinstance(value, set):
                        encoded_value = map_nested_values_in_place(custom_encode, list(value))
                    # no nesting
                    else:
                        encoded_value = custom_encode(value)  # type: ignore[assignment]
                    encoded_values.append(encoded_value)
                except TypeError as e:
                    raise PyToArrowConversionException(
                        data_type=dlt_data_type,
                        inferred_arrow_type=inferred_arrow_type,
                        details="dlt failed to encode values to an Arrow-compatible type.",
                    ) from e

            inferred_array = pa.array(encoded_values)

    return inferred_array


def cast_arrow_array_as_column_schema(
    raw_array: pyarrow.Array,
    caps: DestinationCapabilitiesContext,
    column_schema: TColumnSchema,
    tz: str,
    safe_arrow_conversion: bool,
) -> pyarrow.Array:
    from dlt.common.libs.pyarrow import pyarrow as pa

    dlt_data_type = column_schema.get("data_type")
    inferred_arrow_type = (
        get_py_arrow_datatype(column_schema, caps, tz) if dlt_data_type is not None else None
    )
    inferred_array = None
    try:
        inferred_array = raw_array.cast(inferred_arrow_type, safe=safe_arrow_conversion)
    except (pa.ArrowInvalid, pyarrow.ArrowTypeError, pyarrow.ArrowNotImplementedError) as e:
        # TODO add specific error handling as we encounter them
        error_msg = e.args[0]
        if (
            "would cause data loss"
            in error_msg  # specific pyarrow error related to precision loss (i.e., conversion to decimal)
            and dlt_data_type == "decimal"
            and pa.types.is_decimal(inferred_arrow_type)
        ):
            # TODO provide user interface for safe_arrow_conversion=False and include in this error message
            raise PyToArrowConversionException(
                data_type=dlt_data_type,
                inferred_arrow_type=inferred_arrow_type,
                details=(
                    f"Insufficient decimal precision {error_msg}. Consider setting `precision`"
                    " and `scale` hints:"
                    " https://dlthub.com/docs/general-usage/schema/#tables-and-columns"
                ),
            ) from e
        elif dlt_data_type == "timestamp" and "Failed to parse string" in error_msg:
            if "expected a zone offset" in error_msg:
                # expected tz-aware timestamps in column_data
                actual_type = pyarrow.timestamp(inferred_arrow_type.unit, None)

            elif "expected no zone offset" in error_msg:
                # expected naive timestamps in column_data
                actual_type = pyarrow.timestamp(inferred_arrow_type.unit, tz)
            else:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details=f"Timestamp conversion unsupported by dlt ({error_msg})",
                ) from e
            # cast strings to date-times
            raw_array = raw_array.cast(actual_type)
            _, inferred_array = normalize_py_arrow_item_column(
                column_schema, actual_type, raw_array
            )
        elif dlt_data_type == "time" and "function cast_time" in error_msg:
            if "from string to" in error_msg:
                n = len(raw_array)
                is_null = None
                is_null_b = None
                null_count = 0

                data_b = pyarrow.allocate_buffer(n * 8, resizable=False)
                data = memoryview(data_b).cast("q")

                def allocate_lazy_null_mask() -> None:
                    nonlocal is_null, data, is_null_b
                    if is_null is None:
                        nbytes = (n + 7) // 8
                        is_null_b = pa.allocate_buffer(nbytes, resizable=False)
                        ctypes.memset(is_null_b.address, 0xFF, nbytes)  # start all-valid
                        is_null = memoryview(is_null_b).cast("B")

                for i, scalar in enumerate(raw_array):
                    if not scalar.is_valid:
                        # mark null in bitmap
                        allocate_lazy_null_mask()
                        byte = i >> 3
                        bit = i & 7
                        is_null[byte] &= ~(1 << bit)
                        null_count += 1
                        continue

                    s = scalar.as_py()  # Python str
                    t = time.fromisoformat(s)  # HH:MM | HH:MM:SS | HH:MM:SS.ffffff

                    data[i] = (
                        (t.hour * 3600 + t.minute * 60 + t.second) * 1_000_000
                    ) + t.microsecond

                # Zero-copy build of Arrow int64 with mask; then logical view to time64[us]
                arr_i64 = pyarrow.Array.from_buffers(
                    pa.int64(), n, buffers=[is_null_b, data_b], null_count=null_count
                )  # data zero-copies from NumPy
                inferred_array = arr_i64.view(pa.time64("us"))  # zero-copy reinterpret
                # ts  = pyarrow.compute.strptime(raw_array, format="%H:%M:%S", unit="us")
                # inferred_array = pyarrow.compute.cast(ts, pa.time64("us"))

            elif "from duration" in error_msg:
                # duration and time (if wrapping is not needed) have the same representation
                inferred_array = raw_array.view(pa.time64(raw_array.type.unit))

            else:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details=f"Time conversion unsupported by dlt ({error_msg})",
                ) from e
        # fallback: nested arrow -> json strings when casting to text/json
        elif (
            pa.types.is_nested(raw_array.type)
            and pa.types.is_string(inferred_arrow_type)
            and dlt_data_type in ("text", "json")
        ):
            # log a warning similar to numpy case 2, explaining serialization and destination caps
            logger.warning(
                "Received `data_type='%s'`, Arrow nested values require serialization to STRING."
                " They will be JSON-encoded, which may slow extraction. If your destination"
                " supports nested types, enable them (`supports_nested_types=True`) and provide a"
                " 'json' column with 'x-nested-type' hint to load nested values natively."
                " Otherwise, consider casting to STRING upstream (e.g., in a VIEW).",
                dlt_data_type,
            )
            try:
                # convert nested arrow values to python and json-encode. this mirrors the generic
                # json/text fallback (case 2) used for numpy inputs, but works on arrow arrays.
                py_values = raw_array.to_pylist()
                json_bytes = [None if v is None else json.dumpb(v) for v in py_values]
                inferred_array = pa.array(json_bytes).cast(pa.string())
            except TypeError as enc_err:
                raise PyToArrowConversionException(
                    data_type=dlt_data_type,
                    inferred_arrow_type=inferred_arrow_type,
                    details="dlt failed to encode nested values as JSON strings.",
                ) from enc_err
        else:
            raise PyToArrowConversionException(
                data_type=dlt_data_type,
                inferred_arrow_type=inferred_arrow_type,
                details=f"This conversion is currently unsupported by dlt ({error_msg})",
            ) from e
    assert inferred_array
    return inferred_array


def cast_arrow_as_columns_schema(
    item: TAnyArrowItem,
    columns: TTableSchemaColumns,
    caps: DestinationCapabilitiesContext,
    tz: str,
    safe_arrow_conversion: bool = True,
) -> TAnyArrowItem:
    """Cast an Arrow item to match dlt column schema.

    Performs per-column casts using Arrow, with fallbacks equivalent to
    `cast_arrow_array_as_column_schema`. Timestamp timezone normalization is
    applied when required by the dlt schema. The result preserves the input
    item's schema metadata and per-field metadata.

    Args:
        item: Arrow table or record batch to cast.
        columns: dlt table schema columns containing data type hints.
        caps: capabilities of the storage backend.
        tz: time zone identifier used to resolve timestamp hints.
        safe_arrow_conversion: if False, truncation and loss of precision is allowed.

    Returns:
        The same kind of Arrow object (Table or RecordBatch) with columns cast to
        the requested dlt types. Schema and field metadata are preserved.
    """
    from dlt.common.libs.pyarrow import pyarrow as pa

    schema = item.schema
    arrays_out = []
    fields_out = []

    for idx, fld in enumerate(schema):
        name = fld.name
        col_schema = columns[name]
        col_data = item.column(idx)

        # default: keep original
        out_array = col_data
        out_nullable = fld.nullable
        out_type = fld.type

        out_nullable = col_schema.get("nullable", fld.nullable)

        # final cast to the exact target type using fallbacks
        try:
            cast_arr = cast_arrow_array_as_column_schema(
                out_array, caps, col_schema, tz, safe_arrow_conversion
            )
        except PyToArrowConversionException as e:
            e.field_name = name
            raise

        out_array = cast_arr
        out_type = cast_arr.type

        # build new field preserving metadata
        new_field = pa.field(name, out_type, nullable=out_nullable, metadata=fld.metadata)
        fields_out.append(new_field)
        arrays_out.append(out_array)

    # preserve schema-level metadata
    new_schema = pa.schema(fields_out, metadata=schema.metadata)

    return item.__class__.from_arrays(arrays_out, schema=new_schema)


def row_tuples_to_arrow(
    rows: TDataItems,
    caps: DestinationCapabilitiesContext,
    columns: TTableSchemaColumns,
    tz: str,
    safe_arrow_conversion: bool = True,
) -> Any:  # pyarrow.Table
    """Converts the rows to an arrow table using the columns schema.
    1. Pivot rows into columns.
    2. Convert columns to pyarrow arrays; coerce type and nullability; exclude types unsupported by arrow
    3. Create table
    4. Remove columns full of null values

    Args:
        rows: data items
        caps: capabilities of the storage backend
        columns: dlt hints about the table columns (e.g., data type, nullabe)
        tz: time zone identifier
        safe_arrow_conversion: if False, truncation and loss of precision is allowed
            ref: https://arrow.apache.org/docs/python/generated/pyarrow.compute.CastOptions.html#pyarrow.compute.CastOptions

    Returns:
        an arrow Table
    """
    from dlt.common.libs.pyarrow import pyarrow as pa

    columnar = transpose_rows_to_columns(rows, column_names=columns.keys())

    arrow_arrays = []
    arrow_fields = []
    for column_name, column_data in columnar.items():
        column_schema = columns[column_name]

        try:
            arrow_array = convert_numpy_to_arrow(
                column_data, caps, column_schema, tz, safe_arrow_conversion
            )
        # TODO if converting to arrow fail, should we raise or skip column?
        except PyToArrowConversionException as e:
            e.field_name = column_name
            raise

        field = pa.field(
            name=column_name, type=arrow_array.type, nullable=column_schema.get("nullable", True)
        )
        arrow_arrays.append(arrow_array)
        arrow_fields.append(field)

    # NOTE careful when casting, modifying types, or enforcing schemas in place. Arrow issues are common
    # This can corrupt the data when writing to Parquet
    # ref: https://github.com/apache/arrow/issues/43146
    # ref: https://github.com/apache/arrow/issues/41667
    arrow_table = pa.Table.from_arrays(arrow_arrays, schema=pa.schema(arrow_fields))
    return arrow_table


class NameNormalizationCollision(ValueError):
    def __init__(self, reason: str) -> None:
        msg = f"Arrow column name collision after input data normalization. {reason}"
        super().__init__(msg)


def add_arrow_metadata(
    item: Union[pyarrow.Table, pyarrow.RecordBatch], metadata: dict[str, Any]
) -> pyarrow.Table:
    # Get current metadata or initialize empty
    schema = item.schema
    current = schema.metadata or {}

    # Convert new metadata to bytes and merge
    update = {k.encode("utf-8"): v.encode("utf-8") for k, v in metadata.items()}
    merged = current.copy()
    merged.update(update)

    # Apply updated schema
    new_schema = schema.with_metadata(merged)

    # Rebuild the object with updated schema
    if isinstance(item, pyarrow.Table):
        return pyarrow.Table.from_arrays(item.columns, schema=new_schema)
    else:  # RecordBatch
        return pyarrow.RecordBatch.from_arrays(item.columns, schema=new_schema)


def set_plus0000_timezone_to_utc(tbl: pyarrow.Table) -> pyarrow.Table:
    """
    Convert any +00:00 timestamp columns to UTC.
    Returns the original table object if nothing needed fixing.
    """
    arrays, fields = [], []
    changed = False

    for col, fld in zip(tbl.columns, tbl.schema):
        if pyarrow.types.is_timestamp(fld.type) and fld.type.tz == "+00:00":
            changed = True
            new_type = pyarrow.timestamp(fld.type.unit, "UTC")
            arrays.append(pyarrow.compute.cast(col, new_type))
            fields.append(pyarrow.field(fld.name, new_type, fld.nullable, fld.metadata))
        else:
            arrays.append(col)
            fields.append(fld)

    if not changed:
        return tbl

    new_schema = pyarrow.schema(fields, metadata=tbl.schema.metadata)
    return pyarrow.Table.from_arrays(arrays, schema=new_schema)


def cast_date64_columns_to_timestamp(tbl: pyarrow.Table, tz: Optional[str] = None) -> pyarrow.Table:
    """
    Cast any date64 columns to timestamp with microsecond precision, preserving the
    semantic time values. Uses pyarrow.compute.cast on the column (works for chunked arrays)
    to cast from milliseconds (date64) to microseconds (timestamp[us]).

    Args:
        tbl: Input Arrow table.
        tz: Optional timezone to annotate the resulting timestamp with (e.g. "UTC").
            If None (default), produces a naive timestamp.

    Returns:
        A new table with date64 columns cast to timestamp[us] (optionally tz-aware),
        or the original table if no date64 columns were found.
    """
    arrays, fields = [], []
    changed = False

    for col, fld in zip(tbl.columns, tbl.schema):
        if pyarrow.types.is_date64(fld.type):
            changed = True
            unit = "us"
            new_type = pyarrow.timestamp(unit, tz)
            # Rescale from ms (date64) to us (timestamp).
            arrays.append(pyarrow.compute.cast(col, new_type))
            fields.append(pyarrow.field(fld.name, new_type, fld.nullable, fld.metadata))
        else:
            arrays.append(col)
            fields.append(fld)

    if not changed:
        return tbl

    new_schema = pyarrow.schema(fields, metadata=tbl.schema.metadata)
    return pyarrow.Table.from_arrays(arrays, schema=new_schema)


def flatten_struct_column(
    table: TAnyArrowItem,
    col_name: str,
    naming: NamingConvention,
    columns_schema: Dict[str, Any],
    _r_lvl: int,
    existing_names: Optional[Set[str]] = None,
    struct_arr: Optional[pyarrow.Array] = None,
    casefold_identifier: Optional[Callable[[str], str]] = None,
) -> Tuple[pyarrow.Table, Dict[str, Any]]:
    """Flattens a struct column into top-level `parent__child` columns using zero-copy
    PyArrow operations.

    Given a table with a struct column (e.g. `data` containing
    `struct<name: string, age: int32>`), decomposes it into separate columns named
    `data__name` and `data__age`, drops the original struct column, and returns the
    updated table along with a mapping of new column names to their PyArrow data types.

    Args:
        table: A PyArrow Table or RecordBatch containing the struct column.
            RecordBatch input is automatically converted to a Table.
        col_name: The name of the struct column to flatten.
        naming: A `NamingConvention` instance used to build child column names via
            `shorten_fragments`.
        columns_schema: Accumulator dict for collecting `{column_name: pyarrow.DataType}`
            entries for downstream schema updates. Mutated in-place.
        _r_lvl: Recursion depth remaining. Nested structs are flattened only when
            `_r_lvl > 0`. When `_r_lvl <= 0` the struct is kept as-is.
        existing_names: Optional set of existing column names in the table. When
            provided, any child column whose computed name collides is skipped with
            a warning instead of causing the parent struct to be skipped entirely.
        struct_arr: Optional pre-combined struct array. When provided, skips reading
            and combining the column from `table`, avoiding an intermediate table copy.

    Returns:
        Tuple of (updated_table, columns_schema). The updated table has the struct
        column replaced by its top-level children; `columns_schema` contains type
        entries for every newly added column.

    Note:
        - List fields are not unnested — they are added as-is.
        - Null struct rows produce null child columns (via `pyarrow.compute.struct_field`).
        - Uses `combine_chunks()` to handle chunked arrays efficiently.
    """
    if isinstance(table, pyarrow.RecordBatch):
        table = pyarrow.Table.from_batches([table])
    if struct_arr is not None:
        struct_col = struct_arr
    else:
        struct_col = table.column(col_name).combine_chunks()

    # enforce maximum recursion depth to prevent runaway flattening
    if _r_lvl > MAX_RECURSION_DEPTH:
        _r_lvl = MAX_RECURSION_DEPTH

    # flatten struct to {full_path: array} dict (zero-copy field extraction, no temp tables)
    new_columns, column_fields = _flatten_struct_to_columns(
        struct_col, col_name, naming, columns_schema, _r_lvl, existing_names, casefold_identifier
    )

    # build final table: keep all columns except the one being flattened
    # also skip columns that have the same names as flattened output (replace placeholders)
    flat_column_names = set(new_columns.keys())
    existing_arrays: List[pyarrow.Array] = []
    existing_fields: List[pyarrow.Field] = []
    for f in table.schema:
        if f.name == col_name:
            continue
        if f.name in flat_column_names:
            continue
        existing_arrays.append(table.column(f.name))
        existing_fields.append(f)
    for c_name, c_array in new_columns.items():
        c_field = column_fields[c_name]
        existing_fields.append(
            pyarrow.field(
                c_name, c_field.type, nullable=c_field.nullable, metadata=c_field.metadata
            )
        )
        existing_arrays.append(c_array)

    new_schema = pyarrow.schema(existing_fields, metadata=table.schema.metadata)
    table = pyarrow.Table.from_arrays(existing_arrays, schema=new_schema)

    # extract just DataTypes for the downstream columns_schema (integration code puts DataType values)
    new_entries = {name: field.type for name, field in column_fields.items()}
    return table, new_entries


def _resolve_casefold_collisions(
    names: Dict[str, Any],
    fields: Dict[str, Any],
    parent_prefix: str,
    naming: "NamingConvention",
    casefold_identifier: Callable[[str], str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve names that collide under destination casefolding rules.

    For case-insensitive destinations (e.g. BigQuery), two column names that
    differ only by case (e.g. ``request__tuan`` and ``request__Tuan``) collide.
    This function detects such collisions and appends a deterministic
    ``__c_<hash>`` suffix to colliding names.
    """
    # collect all names and check which need renaming
    rename: Dict[str, str] = {}  # old_name → new_name
    name_list = list(names.keys())
    folded: Dict[str, List[str]] = {}
    for name in name_list:
        folded.setdefault(casefold_identifier(name), []).append(name)

    for name in name_list:
        folded_name = casefold_identifier(name)
        if folded_name != name or len(folded.get(folded_name, [])) > 1:
            # name is unsafe under casefolding — derive deterministic suffix
            renamed = _casefold_safe_name(name, naming)
            if renamed != name:
                rename[name] = renamed

    if not rename:
        return names, fields

    new_names: Dict[str, Any] = {}
    new_fields: Dict[str, Any] = {}
    for name, value in names.items():
        new_name = rename.get(name, name)
        new_names[new_name] = value
        new_fields[new_name] = fields[name]
    return new_names, new_fields


def _casefold_safe_name(name: str, naming: "NamingConvention") -> str:
    """Append a deterministic ``__c_<hash>`` suffix within naming max_length."""
    tag = hashlib.shake_128(name.encode("utf-8")).hexdigest(8)
    suffix = f"__c_{tag}"
    return naming.shorten_identifier(name + suffix, name, naming.max_length)


def _flatten_struct_to_columns(
    struct_arr: pyarrow.Array,
    parent_prefix: str,
    naming: NamingConvention,
    columns_schema: Dict[str, Any],
    _r_lvl: int,
    existing_names: Optional[Set[str]] = None,
    casefold_identifier: Optional[Callable[[str], str]] = None,
) -> Tuple[Dict[str, pyarrow.Array], Dict[str, pyarrow.Field]]:
    """Decompose a struct array into `{full_path: array}` flat columns (recursive).

    Operates directly on the struct array — no intermediate table wrapping."""
    new_columns: Dict[str, pyarrow.Array] = {}
    column_fields: Dict[str, pyarrow.Field] = {}
    struct_type = struct_arr.type

    for field_idx, field in enumerate(struct_type):
        norm_field_name = naming.normalize_identifier(field.name)
        child_name = naming.shorten_fragments(parent_prefix, norm_field_name)

        # per-field naming collision check — log warning if replacing existing column
        if existing_names is not None and child_name in existing_names:
            logger.warning(
                f"Flattened column '{child_name}' conflicts with existing column — replacing"
            )

        child_array = pyarrow.compute.struct_field(struct_arr, [field_idx])

        if pyarrow.types.is_struct(field.type):
            if _r_lvl > 0:
                sub_cols, sub_fields = _flatten_struct_to_columns(
                    child_array,
                    child_name,
                    naming,
                    columns_schema,
                    _r_lvl - 1,
                    existing_names,
                    casefold_identifier,
                )
                new_columns.update(sub_cols)
                column_fields.update(sub_fields)
            else:
                new_columns[child_name] = child_array
                column_fields[child_name] = field
        elif pyarrow.types.is_list(field.type):
            new_columns[child_name] = child_array
            column_fields[child_name] = field
        else:
            new_columns[child_name] = child_array
            column_fields[child_name] = field

    # resolve casefold collisions for case-insensitive destinations (e.g. BigQuery)
    if casefold_identifier is not None:
        new_columns, column_fields = _resolve_casefold_collisions(
            new_columns, column_fields, parent_prefix, naming, casefold_identifier
        )

    return new_columns, column_fields


def apply_arrow_path_filter(struct_arr: pyarrow.Array, paths: List[str]) -> pyarrow.Array:
    """Extract specified dot-paths from a struct array, rebuilding a filtered struct.

    Parses dot-separated paths like `"user.name"`, recurses into nested structs via
    `pyarrow.compute.struct_field()`, and collects only the requested leaf fields.
    Intermediate structs are rebuilt to preserve the nesting structure of the paths.

    Args:
        struct_arr: A `pyarrow.StructArray` to filter.
        paths: Dot-separated paths (e.g. `["user.name", "user.email"]`) specifying
            which leaf fields to retain.

    Returns:
        A new `pyarrow.StructArray` containing only the fields reachable via `paths`.
        Fields not referenced in any path are omitted. Field nullability and metadata
        from the original struct are preserved.

    Note:
        Only leaf-level fields are collected; intermediate struct nodes are
        inferred from the path structure. If a path references a non-struct
        intermediate (e.g. `"a.b"` where `a` is an `int32`), that path is
        silently skipped.
    """
    pc = pyarrow.compute
    if isinstance(struct_arr, pyarrow.ChunkedArray):
        struct_arr = struct_arr.combine_chunks()
    st = struct_arr.type

    # group sub-paths by top-level field name
    path_tree: Dict[str, List[str]] = {}
    for path in paths:
        parts = path.split(".")
        if not parts:
            continue
        root = parts[0]
        rest = parts[1:]
        if root not in path_tree:
            path_tree[root] = []
        if rest:
            path_tree[root].append(".".join(rest))

    new_arrays: List[pyarrow.Array] = []
    new_fields: List[pyarrow.Field] = []

    for field_idx in range(st.num_fields):
        field = st.field(field_idx)
        if field.name not in path_tree:
            continue

        sub_paths = path_tree[field.name]
        field_arr = pc.struct_field(struct_arr, field_idx)

        if not sub_paths:
            # leaf field — include as-is
            new_arrays.append(field_arr)
            new_fields.append(field)
        elif pyarrow.types.is_struct(field.type):
            # nested struct — recurse with remaining sub-paths
            filtered_sub = apply_arrow_path_filter(field_arr, sub_paths)
            new_arrays.append(filtered_sub)
            new_fields.append(
                pyarrow.field(field.name, filtered_sub.type, field.nullable, field.metadata)
            )
        # else: non-struct field with sub-paths — silently skip

    if not new_arrays:
        logger.warning(
            f"apply_arrow_path_filter: no fields matched the given paths {paths}; returning"
            " original struct unchanged"
        )
        return struct_arr

    return pyarrow.StructArray.from_arrays(new_arrays, fields=new_fields)


def apply_arrow_depth_limit(struct_arr: pyarrow.Array, max_depth: Optional[int]) -> pyarrow.Array:
    """Limit struct nesting depth by serializing nested struct fields to JSON strings.

    When `max_depth` is reached, struct-typed child fields are serialized to JSON
    strings in-place, preserving the parent struct shape (matching JSON normalizer
    `limit_depth` semantics). Scalar and list fields pass through unchanged.

    Args:
        struct_arr: A `pyarrow.StructArray` whose nesting depth to limit.
        max_depth: Maximum nesting depth before struct children are serialized.
            `None` means no limit (returns struct_arr unchanged).

    Returns:
        A new `pyarrow.StructArray` with the same root shape but struct-typed
        children at the depth boundary replaced by `pyarrow.string()` columns.

    Note:
        Serializing struct children to JSON strings requires a Python roundtrip
        via `to_pylist()` / `json.dumps()` — unavoidable for struct→JSON.
    """
    if max_depth is None:
        return struct_arr
    if max_depth == 0:
        # serialize entire root struct to JSON string (matching JSON normalizer limit_depth(d, 0))
        pylist = struct_arr.to_pylist()
        json_strings = [json.dumps(row) if row is not None else None for row in pylist]
        return pyarrow.array(json_strings, type=pyarrow.string())

    pc = pyarrow.compute
    if isinstance(struct_arr, pyarrow.ChunkedArray):
        struct_arr = struct_arr.combine_chunks()
    st = struct_arr.type

    new_arrays: List[pyarrow.Array] = []
    new_fields: List[pyarrow.Field] = []

    for field_idx in range(st.num_fields):
        field = st.field(field_idx)
        field_arr = pc.struct_field(struct_arr, field_idx)

        if pyarrow.types.is_struct(field.type):
            if max_depth <= 1:
                # at the depth boundary: serialize this struct child to JSON string
                pylist = field_arr.to_pylist()
                json_strings = [json.dumps(row) if row is not None else None for row in pylist]
                new_arrays.append(pyarrow.array(json_strings, type=pyarrow.string()))
                new_fields.append(
                    pyarrow.field(field.name, pyarrow.string(), field.nullable, field.metadata)
                )
            else:
                limited = apply_arrow_depth_limit(field_arr, max_depth - 1)
                new_arrays.append(limited)
                new_fields.append(
                    pyarrow.field(field.name, limited.type, field.nullable, field.metadata)
                )
        else:
            new_arrays.append(field_arr)
            new_fields.append(field)

    return pyarrow.StructArray.from_arrays(new_arrays, fields=new_fields)


def apply_arrow_force_string(struct_arr: pyarrow.Array) -> pyarrow.Array:
    """Coerce all leaf scalar fields in a struct array to `pyarrow.string()`.

    Recurses into nested structs so that every scalar leaf is converted. List
    fields are left entirely as-is — they become child tables during flattening,
    not flattened scalar columns. `None` values remain `None`.

    Args:
        struct_arr: A `pyarrow.StructArray` whose scalar leaves to coerce.

    Returns:
        A new `pyarrow.StructArray` where every leaf scalar field (not struct,
        not list) has been cast to `pyarrow.string()`. Struct fields are
        recursively processed; list fields are preserved unchanged.

    Note:
        Uses `pyarrow.compute.cast` for the coercion, which preserves the
        null bitmap so that null values are not converted to the string `"None"`.
    """
    pc = pyarrow.compute
    if isinstance(struct_arr, pyarrow.ChunkedArray):
        struct_arr = struct_arr.combine_chunks()
    st = struct_arr.type

    new_arrays: List[pyarrow.Array] = []
    new_fields: List[pyarrow.Field] = []

    for field_idx in range(st.num_fields):
        field = st.field(field_idx)
        field_arr = pc.struct_field(struct_arr, field_idx)

        if pyarrow.types.is_struct(field.type):
            coerced = apply_arrow_force_string(field_arr)
            new_arrays.append(coerced)
            new_fields.append(
                pyarrow.field(field.name, coerced.type, field.nullable, field.metadata)
            )
        elif pyarrow.types.is_list(field.type) or pyarrow.types.is_large_list(field.type):
            # leave lists as-is
            new_arrays.append(field_arr)
            new_fields.append(field)
        else:
            # leaf scalar — cast to string, preserving nulls
            casted = pc.cast(field_arr, pyarrow.string())
            new_arrays.append(casted)
            new_fields.append(
                pyarrow.field(field.name, pyarrow.string(), field.nullable, field.metadata)
            )

    return pyarrow.StructArray.from_arrays(new_arrays, fields=new_fields)


def expand_arrow_json_column(
    column: "pyarrow.Array",
    spec: "TJsonColumnExpansionSpec",
) -> Tuple[Optional["pyarrow.Array"], Optional["pyarrow.Table"]]:
    """Expand a pyarrow string column containing JSON values into a flattened Arrow table.

    Parses each string row as JSON, applies filter-depth-string coercion per the
    expansion spec, and rebuilds the expanded dicts as a pyarrow table for the
    flattening engine.

    Args:
        column: pyarrow Array or ChunkedArray of string type with JSON-encoded values.
        spec: Column expansion specification with flatten rules, keep_original flag,
            force_string coercion, and max_depth limit.

    Returns:
        Tuple of (original_column, expanded_table). `original_column` is the input
        array when `spec.keep_original` is True, else None. `expanded_table` is a
        pyarrow Table of the expanded dicts, or None when no rows contained expandable
        JSON.

    Note:
        This function converts Arrow data to Python lists for JSON parsing, then
        rebuilds the result via `pyarrow.Table.from_pylist`. For better performance,
        consider casting JSON columns to STRUCT type upstream so that expansion can
        operate directly on structured Arrow data.
    """

    # combine chunks if ChunkedArray
    if isinstance(column, pyarrow.ChunkedArray):
        column = column.combine_chunks()

    # non-string columns: nothing to expand
    if not (pyarrow.types.is_string(column.type) or pyarrow.types.is_large_string(column.type)):
        return None, None

    # one-time warning about the Python roundtrip cost
    if "string" not in _warned_json_expansion:
        _warned_json_expansion.add("string")
        logger.warning(
            "x-json-flatten on string column requires JSON parsing; for better performance, "
            "cast to STRUCT upstream"
        )

    flatten_spec = spec.flatten_spec
    keep_original = spec.keep_original
    force_string = spec.force_string
    max_depth = spec.max_depth

    # no expansion requested
    if not flatten_spec:
        return (column if keep_original else None, None)

    pylist = column.to_pylist()
    expanded_dicts: List[Dict[str, Any]] = []

    for val in pylist:
        if val is None:
            expanded_dicts.append({})
            continue

        try:
            parsed = parse_json_value(val)
        except Exception:
            logger.warning("error parsing JSON value for column expansion, skipping row")
            expanded_dicts.append({})
            continue

        if parsed is None or not isinstance(parsed, dict):
            expanded_dicts.append({})
            continue

        expanded: Dict[str, Any] = parsed

        if isinstance(flatten_spec, list):
            expanded = filter_by_paths(expanded, flatten_spec)

        if max_depth is not None:
            expanded = limit_depth(expanded, max_depth)

        if force_string:
            expanded = apply_force_string(expanded)

        expanded_dicts.append(expanded)

    # union all keys across all expanded dicts to avoid from_pylist() inferring zero
    # columns when early rows are empty (e.g., first row is None or invalid JSON)
    all_keys: Dict[str, None] = {}
    for d in expanded_dicts:
        for k in d:
            all_keys[k] = None
    all_keys_list = list(all_keys.keys())

    if not all_keys_list:
        return (column if keep_original else None, None)

    arrays: Dict[str, pyarrow.Array] = {}
    for key in all_keys_list:
        values = [d.get(key, None) for d in expanded_dicts]
        arrays[key] = pyarrow.array(values)

    expanded_table = pyarrow.table(arrays)
    return (column if keep_original else None, expanded_table)
