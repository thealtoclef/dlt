import base64
import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Type

from google.api_core import exceptions as api_core_exceptions  # noqa: I250
from google.cloud.bigquery import SchemaField  # noqa: I250
from google.cloud.bigquery.table import TableReference  # noqa: I250
from google.cloud.bigquery_storage_v1 import types as gapic_types  # noqa: I250
from google.cloud.bigquery_storage_v1.services.big_query_write import (
    BigQueryWriteClient,
)  # noqa: I250
from google.cloud.bigquery_storage_v1.writer import AppendRowsStream  # noqa: I250
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # noqa: I250

from dlt.common import logger
from dlt.common.destination.client import RunnableLoadJob
from dlt.common.json import json
from dlt.common.storages import FileStorage
from dlt.common.typing import DictStrAny

from dlt.destinations.exceptions import DestinationTerminalException
from dlt.destinations.impl.bigquery.bigquery_adapter import should_autodetect_schema
from dlt.destinations.impl.bigquery.configuration import BigQueryClientConfiguration

_TYPE_STRING = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_TYPE_INT64 = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
_TYPE_DOUBLE = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
_TYPE_BOOL = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
_TYPE_BYTES = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
_LABEL_REPEATED = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
_LABEL_OPTIONAL = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

_BQ_TO_PROTO: Dict[str, int] = {
    "STRING": _TYPE_STRING,
    "JSON": _TYPE_STRING,
    "INT64": _TYPE_INT64,
    "INTEGER": _TYPE_INT64,
    "FLOAT64": _TYPE_DOUBLE,
    "FLOAT": _TYPE_DOUBLE,
    "BOOL": _TYPE_BOOL,
    "BOOLEAN": _TYPE_BOOL,
    "BYTES": _TYPE_BYTES,
    "TIMESTAMP": _TYPE_STRING,
    "DATE": _TYPE_STRING,
    "DATETIME": _TYPE_STRING,
    "TIME": _TYPE_STRING,
    "NUMERIC": _TYPE_STRING,
    "BIGNUMERIC": _TYPE_STRING,
}


def _build_proto_field(bq_field: SchemaField, number: int) -> Dict[str, Any]:
    """Maps a BigQuery schema field to protobuf descriptor attributes."""
    typ = bq_field.field_type.upper()
    proto_type = _BQ_TO_PROTO.get(typ)
    if proto_type is None:
        raise ValueError(f"Unsupported BigQuery type: {typ} for field '{bq_field.name}'")
    label = _LABEL_REPEATED if bq_field.mode == "REPEATED" else _LABEL_OPTIONAL
    return dict(
        name=bq_field.name, number=number, type=proto_type, label=label, json_name=bq_field.name
    )


def bq_schema_to_proto_class(bq_schema: List[SchemaField]) -> Type[Any]:
    """Dynamically creates a protobuf message class from BigQuery schema fields.

    Uses schema hash for caching so identical schemas return the same class.
    """
    fhash = hashlib.sha1()
    for f in bq_schema:
        fhash.update(hash(f).to_bytes(8, "big", signed=True))
    clsname = f"net.proto2.target_bq.Row_{fhash.hexdigest()}"
    fname = f"row_{fhash.hexdigest()}.proto"

    pool = descriptor_pool.Default()
    try:
        desc = pool.FindMessageTypeByName(clsname)
        return message_factory.GetMessageClass(desc)
    except KeyError:
        pass

    package, name = clsname.rsplit(".", 1)
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = fname
    file_proto.package = package
    desc_proto = file_proto.message_type.add()
    desc_proto.name = name

    for i, bq_field in enumerate(bq_schema):
        field_proto = desc_proto.field.add()
        for k, v in _build_proto_field(bq_field, i + 1).items():
            setattr(field_proto, k, v)

    pool.Add(file_proto)
    desc = pool.FindMessageTypeByName(clsname)
    return message_factory.GetMessageClass(desc)


def build_field_types(proto_class: Type[Any]) -> Dict[str, int]:
    """Returns field name to protobuf type mapping for serialization dispatch."""
    return {fd.name: fd.type for fd in proto_class.DESCRIPTOR.fields}


def _parse_str_value(value: str, proto_type: int) -> Any:
    """Coerces a string value to the Python type expected by protobuf."""
    if proto_type == _TYPE_INT64:
        return int(value)
    if proto_type == _TYPE_DOUBLE:
        return float(value)
    if proto_type == _TYPE_BOOL:
        return value.lower() == "true"
    if proto_type == _TYPE_BYTES:
        encoded = value.encode("utf-8")
        padded = encoded + b"=" * (4 - len(encoded) % 4)
        return base64.urlsafe_b64decode(padded)
    return value


def serialize_row(row: DictStrAny, proto_class: Type[Any], field_types: Dict[str, int]) -> bytes:
    """Serializes a single row dict to protobuf bytes.

    None values are skipped (proto2 default). Non-string values on STRING proto fields
    are JSON-serialized (dict/list) or coerced with str(). String values for non-string
    proto types are parsed. Unknown fields are silently ignored.
    """
    msg = proto_class()
    for key, value in row.items():
        if value is None:
            continue
        proto_type = field_types.get(key)
        if proto_type is None:
            continue
        if proto_type == _TYPE_STRING and not isinstance(value, str):
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            else:
                value = str(value)
        elif not isinstance(value, (str, int, float, bool)):
            value = str(value)
        if isinstance(value, str) and proto_type != _TYPE_STRING:
            value = _parse_str_value(value, proto_type)
        setattr(msg, key, value)
    return msg.SerializeToString()  # type: ignore[no-any-return]


def serialize_rows_to_proto(
    rows: List[DictStrAny],
    proto_class: Type[Any],
    field_types: Dict[str, int],
    max_request_bytes: int,
) -> List[gapic_types.ProtoRows]:
    """Serializes rows into ProtoRows batches respecting the byte limit.

    Each returned ProtoRows batch is ≤ max_request_bytes. Raises ValueError if a
    single row exceeds the limit.
    """
    proto_rows = gapic_types.ProtoRows()
    total_size = 0
    result: List[gapic_types.ProtoRows] = []

    for row in rows:
        serialized = serialize_row(row, proto_class, field_types)
        row_size = len(serialized)

        if row_size > max_request_bytes:
            raise ValueError(
                f"Single row exceeds {max_request_bytes} bytes ({row_size:,}). Cannot send."
            )

        if total_size + row_size > max_request_bytes:
            result.append(proto_rows)
            proto_rows = gapic_types.ProtoRows()
            total_size = 0

        proto_rows.serialized_rows.append(serialized)
        total_size += row_size

    if proto_rows.serialized_rows:
        result.append(proto_rows)

    return result


class BigQueryStorageWriteJob(RunnableLoadJob):
    """Loads a single JSONL file to BigQuery via Storage Write API default stream.

    Uses the _default stream for at-least-once delivery. Bypasses the
    partition_modifications_per_column_partitioned_table quota. Only supports
    write_disposition='append'.
    """

    def __init__(self, file_path: str, config: BigQueryClientConfiguration) -> None:
        super().__init__(file_path)
        self._config = config
        self._write_client: BigQueryWriteClient = None
        self._stream: AppendRowsStream = None
        self._proto_class: Type[Any] = None
        self._field_types: Dict[str, int] = None
        self._stream_name: str = None

    def run(self) -> None:
        # validate constraints
        if self._load_table["write_disposition"] != "append":
            raise DestinationTerminalException(
                "Storage Write API only supports `write_disposition='append'`. "
                f"Got `{self._load_table['write_disposition']}`."
            )
        if should_autodetect_schema(self._load_table):
            raise DestinationTerminalException(
                "Storage Write API does not support autodetect_schema. "
                "Disable it to use `insert_api='storage_write'`."
            )

        table_name = self._load_table["name"]
        sql_client = self._job_client.sql_client  # type: ignore[attr-defined]
        bq_client = sql_client.native_connection
        table_ref = TableReference.from_string(
            f"{sql_client.fully_qualified_dataset_name(quote=False)}.{table_name}"
        )
        table = bq_client.get_table(table_ref)

        self._proto_class = bq_schema_to_proto_class(table.schema)
        self._field_types = build_field_types(self._proto_class)

        self._write_client = BigQueryWriteClient()
        self._stream_name = (
            f"projects/{sql_client.project_id}"
            f"/datasets/{sql_client.dataset_name}"
            f"/tables/{table_name}/_default"
        )
        self._stream = self._init_stream()

        try:
            rows = self._read_jsonl_file()
            proto_batches = serialize_rows_to_proto(
                rows,
                self._proto_class,
                self._field_types,
                self._config.storage_write_max_request_bytes,
            )
        except Exception:
            self._cleanup()
            raise

        try:
            self._send_rows(table_name, proto_batches)
        except api_core_exceptions.Aborted:
            logger.warning(f"Stream expired for {table_name}, reconnecting")
            self._reconnect_stream()
            self._send_rows(table_name, proto_batches)
        finally:
            self._cleanup()

    def _read_jsonl_file(self) -> List[DictStrAny]:
        """Reads the entire JSONL.gz job file into a list of row dicts."""
        rows: List[DictStrAny] = []
        with FileStorage.open_zipsafe_ro(self._file_path) as f:
            for line in f:
                row = json.typed_loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                elif isinstance(row, list):
                    rows.extend(row)
        return rows

    def _init_stream(self) -> AppendRowsStream:
        """Creates an AppendRowsStream with writer schema from the proto class."""
        request_template = gapic_types.AppendRowsRequest()
        request_template.write_stream = self._stream_name

        proto_schema = gapic_types.ProtoSchema()
        proto_descriptor = descriptor_pb2.DescriptorProto()
        self._proto_class.DESCRIPTOR.CopyToProto(proto_descriptor)
        proto_schema.proto_descriptor = proto_descriptor

        proto_data = gapic_types.AppendRowsRequest.ProtoData()
        proto_data.writer_schema = proto_schema
        request_template.proto_rows = proto_data

        return AppendRowsStream(self._write_client, request_template)

    def _send_rows(self, table_name: str, proto_batches: List[gapic_types.ProtoRows]) -> None:
        """Sends proto batches via AppendRowsStream. Raises on row or stream errors."""
        send_futures = []
        for proto_rows in proto_batches:
            request = gapic_types.AppendRowsRequest()
            request.write_stream = self._stream_name
            proto_data = gapic_types.AppendRowsRequest.ProtoData()
            proto_data.rows = proto_rows
            request.proto_rows = proto_data
            send_futures.append(self._stream.send(request))

        for f in send_futures:
            try:
                response = f.result()
            except api_core_exceptions.GoogleAPICallError as exc:
                raise DestinationTerminalException(
                    f"BigQuery Storage Write API error for {table_name}: {exc.message}"
                ) from exc

            if response is not None and response.row_errors:
                for err in response.row_errors:
                    logger.error(
                        "Row %d rejected for table %s: code=%s, message=%s",
                        err.index, table_name, err.code, err.message,
                    )
                raise DestinationTerminalException(
                    f"{len(response.row_errors)} row(s) rejected by BigQuery for {table_name}"
                )

    def _reconnect_stream(self) -> None:
        """Closes the expired stream and opens a new one."""
        try:
            if self._stream and self._stream.is_active:
                self._stream.close()
        except Exception:
            pass

        self._stream = self._init_stream()

    def _cleanup(self) -> None:
        """Closes the stream and write client transport."""
        try:
            if self._stream and self._stream.is_active:
                self._stream.close()
        except Exception:
            logger.debug("Error closing stream", exc_info=True)

        try:
            if self._write_client:
                self._write_client.transport.close()
        except Exception:
            logger.debug("Error closing write client", exc_info=True)
