from typing import Any

from dlt import version
from dlt.common.exceptions import MissingDependencyException

try:
    import duckdb
except ModuleNotFoundError:
    raise MissingDependencyException(
        "dlt duckdb helpers",
        [f"{version.DLT_PKG_NAME}[duckdb]"],
        "Install duckdb to use the DuckDB-backed JSON flatten engine.",
    )


def make_connection(memory_limit: str = "2GB", threads: int = 1) -> Any:
    """Creates an in-memory DuckDB connection preconfigured for vectorized JSON ops.

    The connection has the JSON extension loaded and resource caps applied so it can be
    safely instantiated inside a normalize worker without contending with the rest of
    the process.

    Args:
        memory_limit (str): DuckDB `memory_limit` PRAGMA value (e.g. `"2GB"`).
        threads (int): DuckDB `threads` PRAGMA value.

    Returns:
        Any: A new `duckdb.DuckDBPyConnection` ready for JSON queries.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute(f"PRAGMA threads={int(threads)}")
    conn.execute("INSTALL json")
    conn.execute("LOAD json")
    return conn
