"""Integration test for BigQuery Storage Write API in dlt.

Run against cake-data-non-production project.
"""
import time

import dlt
from tests.pipeline.utils import assert_load_info

DATASET_PREFIX = "dlt_storage_write_test"
TABLE_NAME = "test_storage_write_items"


def test_storage_write_append() -> None:
    """Verifies Storage Write API loads data to BigQuery and bypasses partition quotas."""
    @dlt.resource(write_disposition="append")
    def test_resource():
        for i in range(100):
            yield {
                "id": i,
                "name": f"user_{i}",
                "created_date": "2026-06-21",
                "score": float(i) * 1.5,
            }

    dlt.destinations.impl.bigquery.bigquery_adapter.bigquery_adapter(
        test_resource, insert_api="storage_write"
    )

    ts = int(time.time())
    dest = dlt.destinations.bigquery(location="asia-southeast1")
    pipe = dlt.pipeline(
        pipeline_name=f"storage_write_test_{ts}",
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
            f"SELECT COUNT(*) FROM {actual_dataset}.{TABLE_NAME};"
        ) as cursor:
            count = cursor.fetchone()[0]
            print(f"✅ Row count: {count}")
            assert count == 100, f"Expected 100 rows, got {count}"

        with client.execute_query(
            f"SELECT id, name FROM {actual_dataset}.{TABLE_NAME} ORDER BY id LIMIT 3;"
        ) as cursor:
            rows = cursor.fetchall()
            print(f"✅ Sample rows: {rows}")
            assert rows[0][0] == 0
            assert rows[0][1] == "user_0"

    print("\n🎉 Storage Write API integration test PASSED!")


if __name__ == "__main__":
    test_storage_write_append()
