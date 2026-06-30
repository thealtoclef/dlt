"""Integration test for BigQuery Storage Write API — all data types."""
import base64
import time
from datetime import date, datetime, time as dtime, timezone
from decimal import Decimal

import dlt
from tests.pipeline.utils import assert_load_info

DATASET_PREFIX = "dlt_sw_all_types"
TABLE_NAME = "test_all_bq_types"


def test_storage_write_all_types() -> None:
    """Verifies Storage Write API handles all BigQuery-mapped data types."""
    @dlt.resource(write_disposition="append")
    def test_resource():
        now = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        yield {
            "col_text": "hello world",
            "col_double": 3.14159265358979,
            "col_bool": True,
            "col_bigint": 9223372036854775807,
            "col_date": date(2026, 6, 21),
            "col_time": dtime(14, 30, 45),
            "col_timestamp": now,
            "col_decimal": Decimal("12345678901234567890.123456789"),
            "col_json": '{"key": "value", "nested": {"a": 1}}',
            "col_bytes": base64.b64encode(b"binary data"),
        }
        yield {
            "col_text": "second row",
            "col_double": 2.71828,
            "col_bool": False,
            "col_bigint": -42,
            "col_date": date(2025, 1, 1),
            "col_time": dtime(9, 0, 0),
            "col_timestamp": datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "col_decimal": Decimal("-9999999999.999999999"),
            "col_json": "null",
            "col_bytes": b"",
        }

    test_resource.apply_hints(
        columns={
            "col_decimal": {
                "name": "col_decimal", "data_type": "decimal", "precision": 38, "scale": 9
            },
        }
    )

    dlt.destinations.impl.bigquery.bigquery_adapter.bigquery_adapter(
        test_resource, insert_api="storage_write"
    )

    ts = int(time.time())
    dest = dlt.destinations.bigquery(location="asia-southeast1")
    pipe = dlt.pipeline(
        pipeline_name=f"storage_write_all_types_{ts}",
        destination=dest,
        dataset_name=f"{DATASET_PREFIX}_{ts}",
        dev_mode=True,
    )
    pack = pipe.run(test_resource, table_name=TABLE_NAME)
    assert_load_info(pack)

    actual_dataset = pipe.dataset_name
    print(f"\n✅ Load completed. Dataset: {actual_dataset}")

    with pipe.sql_client() as client:
        with client.execute_query(
            f"SELECT * FROM {actual_dataset}.{TABLE_NAME} ORDER BY col_text;"
        ) as cursor:
            rows = cursor.fetchall()
            assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
            col_names = [d[0] for d in cursor.description]

        print(f"✅ Row count: {len(rows)}")
        print(f"   Columns: {col_names}")

        assert rows[0].col_text == "hello world"
        assert abs(rows[0].col_double - 3.14159265358979) < 1e-10
        assert rows[0].col_bool is True
        assert rows[0].col_bigint == 9223372036854775807
        assert isinstance(rows[0].col_date, date)
        assert str(rows[0].col_date) == "2026-06-21"
        assert rows[0].col_time is not None
        assert rows[0].col_timestamp is not None
        assert rows[0].col_json is not None
        assert rows[0].col_bytes is not None

        assert rows[1].col_bool is False
        assert rows[1].col_bigint == -42
        assert str(rows[1].col_decimal) == "-9999999999.999999999"

    print("\n🎉 All types integration test PASSED!")


if __name__ == "__main__":
    test_storage_write_all_types()
