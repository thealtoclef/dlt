import base64
import math
from typing import List

import pytest
from google.cloud.bigquery import SchemaField
from google.protobuf import descriptor_pb2

from dlt.destinations.impl.bigquery.storage_write_job import (
    bq_schema_to_proto_class,
    build_field_types,
    serialize_row,
    serialize_rows_to_proto,
)


@pytest.mark.parametrize(
    "bq_type, expected_proto_type",
    [
        ("STRING", 9),  # TYPE_STRING
        ("INT64", 3),  # TYPE_INT64
        ("FLOAT64", 1),  # TYPE_DOUBLE
        ("BOOL", 8),  # TYPE_BOOL
        ("BYTES", 12),  # TYPE_BYTES
        ("TIMESTAMP", 9),  # TYPE_STRING
        ("DATE", 9),  # TYPE_STRING
        ("DATETIME", 9),  # TYPE_STRING
        ("TIME", 9),  # TYPE_STRING
        ("NUMERIC", 9),  # TYPE_STRING
        ("BIGNUMERIC", 9),  # TYPE_STRING
        ("JSON", 9),  # TYPE_STRING
    ],
    ids=[
        "string",
        "int64",
        "float64",
        "bool",
        "bytes",
        "timestamp",
        "date",
        "datetime",
        "time",
        "numeric",
        "bignumeric",
        "json",
    ],
)
def test_proto_type_mapping(bq_type: str, expected_proto_type: int) -> None:
    """BigQuery types map to correct protobuf field types."""
    schema = [SchemaField("col", bq_type)]
    proto_cls = bq_schema_to_proto_class(schema)
    assert proto_cls.DESCRIPTOR.fields[0].type == expected_proto_type


def test_unsupported_type_raises() -> None:
    """GEOGRAPHY type raises ValueError (not in BQ_TO_PROTO map)."""
    schema = [SchemaField("geo", "GEOGRAPHY")]
    with pytest.raises(ValueError, match="Unsupported BigQuery type"):
        bq_schema_to_proto_class(schema)


def test_schema_caching() -> None:
    """Identical schemas return the same proto class (cached by hash)."""
    schema = [SchemaField("x", "STRING")]
    cls1 = bq_schema_to_proto_class(schema)
    cls2 = bq_schema_to_proto_class(schema)
    assert cls1 is cls2


def test_repeated_field() -> None:
    """REPEATED mode yields LABEL_REPEATED."""
    schema = [SchemaField("tags", "STRING", mode="REPEATED")]
    proto_cls = bq_schema_to_proto_class(schema)
    assert (
        proto_cls.DESCRIPTOR.fields[0].label == descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    )


def test_nullable_field_defaults_to_optional() -> None:
    """NULLABLE (default) mode yields LABEL_OPTIONAL."""
    schema = [SchemaField("name", "STRING")]
    proto_cls = bq_schema_to_proto_class(schema)
    assert (
        proto_cls.DESCRIPTOR.fields[0].label == descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    )


# ---------------------------------------------------------------------------
# Row serialization tests
# ---------------------------------------------------------------------------


def _make_proto(bq_schema: List[SchemaField]):
    proto_cls = bq_schema_to_proto_class(bq_schema)
    fc = build_field_types(proto_cls)
    return proto_cls, fc


def _deserialize(data: bytes, proto_class):
    msg = proto_class()
    msg.ParseFromString(data)
    return msg


def test_serialize_row_basic() -> None:
    """Int, string, and float values serialize correctly."""
    proto_cls, fc = _make_proto(
        [
            SchemaField("id", "INT64"),
            SchemaField("name", "STRING"),
            SchemaField("score", "FLOAT64"),
        ]
    )
    row = {"id": 42, "name": "Alice", "score": 3.14}
    msg = _deserialize(serialize_row(row, proto_cls, fc), proto_cls)
    assert msg.id == 42
    assert msg.name == "Alice"
    assert msg.score == 3.14


def test_serialize_row_int_from_string() -> None:
    """String values for INT columns are coerced to int (CDC pattern)."""
    proto_cls, fc = _make_proto([SchemaField("id", "INT64")])
    msg = _deserialize(serialize_row({"id": "42"}, proto_cls, fc), proto_cls)
    assert msg.id == 42


def test_serialize_row_float_from_string() -> None:
    """String values for FLOAT columns are coerced to float."""
    proto_cls, fc = _make_proto([SchemaField("val", "FLOAT64")])
    msg = _deserialize(serialize_row({"val": "3.14"}, proto_cls, fc), proto_cls)
    assert msg.val == 3.14


def test_serialize_row_bool_from_string() -> None:
    """String 'true'/'false' is coerced to bool."""
    proto_cls, fc = _make_proto([SchemaField("active", "BOOL")])
    msg = _deserialize(serialize_row({"active": "true"}, proto_cls, fc), proto_cls)
    assert msg.active is True
    msg = _deserialize(serialize_row({"active": "false"}, proto_cls, fc), proto_cls)
    assert msg.active is False


def test_serialize_row_bool_native() -> None:
    """Native bool values pass through unchanged."""
    proto_cls, fc = _make_proto([SchemaField("flag", "BOOL")])
    msg = _deserialize(serialize_row({"flag": True}, proto_cls, fc), proto_cls)
    assert msg.flag is True


def test_serialize_row_none_skipped() -> None:
    """None values are omitted (proto2 default = unset)."""
    proto_cls, fc = _make_proto(
        [
            SchemaField("id", "INT64"),
            SchemaField("name", "STRING"),
        ]
    )
    row = {"id": 1, "name": None}
    msg = _deserialize(serialize_row(row, proto_cls, fc), proto_cls)
    assert msg.id == 1
    assert msg.name == ""


def test_serialize_row_missing_column_skipped() -> None:
    """Columns not in the proto schema are silently ignored."""
    proto_cls, fc = _make_proto([SchemaField("id", "INT64")])
    msg = _deserialize(serialize_row({"id": 1, "extra": "ignored"}, proto_cls, fc), proto_cls)
    assert msg.id == 1


def test_serialize_row_unknown_field_ignored() -> None:
    """Rows with extra keys serialize same as rows without."""
    proto_cls, fc = _make_proto([SchemaField("id", "INT64")])
    new_bytes = serialize_row({"id": 1, "unknown_col": "foo"}, proto_cls, fc)
    expected = serialize_row({"id": 1}, proto_cls, fc)
    assert new_bytes == expected


def test_serialize_row_bytes_base64() -> None:
    """BYTES field with base64-encoded string."""
    proto_cls, fc = _make_proto([SchemaField("data", "BYTES")])
    raw = b"hello world"
    encoded = base64.urlsafe_b64encode(raw).decode()
    msg = _deserialize(serialize_row({"data": encoded}, proto_cls, fc), proto_cls)
    assert msg.data == raw


def test_serialize_row_timestamp_date_as_string() -> None:
    """TIMESTAMP and DATE are sent as strings."""
    proto_cls, fc = _make_proto(
        [
            SchemaField("ts", "TIMESTAMP"),
            SchemaField("d", "DATE"),
        ]
    )
    row = {"ts": "2026-06-21T10:00:00Z", "d": "2026-06-21"}
    msg = _deserialize(serialize_row(row, proto_cls, fc), proto_cls)
    assert msg.ts == "2026-06-21T10:00:00Z"
    assert msg.d == "2026-06-21"


def test_serialize_row_empty_string() -> None:
    """Empty string is serialized as empty."""
    proto_cls, fc = _make_proto([SchemaField("name", "STRING")])
    msg = _deserialize(serialize_row({"name": ""}, proto_cls, fc), proto_cls)
    assert msg.name == ""


def test_serialize_row_decimal_as_string() -> None:
    """NUMERIC values are sent as strings."""
    proto_cls, fc = _make_proto([SchemaField("amount", "NUMERIC")])
    row = {"amount": "999.99"}
    msg = _deserialize(serialize_row(row, proto_cls, fc), proto_cls)
    assert msg.amount == "999.99"


def test_serialize_row_non_primitive_coerced() -> None:
    """Non-primitive values (datetime, Decimal) are coerced to str."""
    from datetime import datetime
    from decimal import Decimal

    proto_cls, fc = _make_proto([SchemaField("ts", "TIMESTAMP")])
    dt = datetime(2026, 6, 21, 10, 30, 0)
    msg = _deserialize(serialize_row({"ts": dt}, proto_cls, fc), proto_cls)
    assert msg.ts != ""  # str() representation


# ---------------------------------------------------------------------------
# Batch serialization tests
# ---------------------------------------------------------------------------


def test_serialize_rows_single_batch() -> None:
    """All rows fit in one ProtoRows batch."""
    proto_cls, fc = _make_proto(
        [
            SchemaField("id", "INT64"),
            SchemaField("name", "STRING"),
        ]
    )
    rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    batches = serialize_rows_to_proto(rows, proto_cls, fc, max_request_bytes=10 * 1024 * 1024)
    assert len(batches) == 1
    assert len(batches[0].serialized_rows) == 2


def test_serialize_rows_splits_at_limit() -> None:
    """Rows are chunked when exceeding max_request_bytes."""
    proto_cls, fc = _make_proto([SchemaField("data", "STRING")])
    large_value = "x" * 1000
    rows = [{"data": large_value} for _ in range(50)]
    # 50 rows × ~1KB each = ~50KB, limit to 10KB
    batches = serialize_rows_to_proto(rows, proto_cls, fc, max_request_bytes=10 * 1024)
    assert len(batches) >= 2


def test_serialize_rows_single_too_large_raises() -> None:
    """Single row exceeding max_request_bytes raises ValueError."""
    proto_cls, fc = _make_proto([SchemaField("data", "STRING")])
    huge_value = "x" * (11 * 1024 * 1024)
    rows = [{"data": huge_value}]
    with pytest.raises(ValueError, match="exceeds"):
        serialize_rows_to_proto(rows, proto_cls, fc, max_request_bytes=10 * 1024 * 1024)


def test_serialize_rows_empty_list() -> None:
    """Empty list returns empty result."""
    proto_cls, fc = _make_proto([SchemaField("id", "INT64")])
    batches = serialize_rows_to_proto([], proto_cls, fc, max_request_bytes=10 * 1024 * 1024)
    assert batches == []
