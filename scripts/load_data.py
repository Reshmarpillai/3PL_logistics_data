"""Bulk CSV → PostgreSQL loader using COPY FROM STDIN.

Loads all 9 tables in dependency order with index management
for faster bulk loading.
"""

import io
import re
import sys
import time
from pathlib import Path

import psycopg2

# Add parent dir so db_config is importable when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_config import get_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "scripts" / "schema.sql"
DATA_DIR = PROJECT_ROOT / "data"

# Load order respects table dependencies
LOAD_ORDER: list[str] = [
    "warehouses",
    "products",
    "customers",
    "inventory",
    "orders",
    "shipments",
    "order_shipments",
    "shipment_items",
    "delivery_events",
]


def execute_schema(conn: psycopg2.extensions.connection) -> None:
    """Execute schema.sql to drop and recreate all tables + indexes."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("Schema created (tables + indexes).")


def get_index_definitions(conn: psycopg2.extensions.connection) -> list[str]:
    """Retrieve CREATE INDEX statements for all user-created indexes."""
    query = """
        SELECT indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname NOT LIKE '%_pkey'
        ORDER BY indexname;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]


def drop_indexes(conn: psycopg2.extensions.connection) -> list[str]:
    """Drop all non-PK indexes, return their CREATE statements for later."""
    index_defs = get_index_definitions(conn)
    if not index_defs:
        return []

    # Extract index names from definitions
    query = """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname NOT LIKE '%_pkey'
        ORDER BY indexname;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        index_names = [row[0] for row in cur.fetchall()]
        for name in index_names:
            cur.execute(f'DROP INDEX IF EXISTS "{name}";')
    conn.commit()
    print(f"Dropped {len(index_names)} indexes for faster loading.")
    return index_defs


def recreate_indexes(
    conn: psycopg2.extensions.connection, index_defs: list[str]
) -> None:
    """Recreate indexes from saved definitions."""
    if not index_defs:
        return
    with conn.cursor() as cur:
        for defn in index_defs:
            cur.execute(defn + ";")
    conn.commit()
    print(f"Recreated {len(index_defs)} indexes.")


def load_table(
    conn: psycopg2.extensions.connection, table_name: str
) -> int:
    """Load a single CSV into the given table using COPY FROM STDIN.

    Returns the number of rows loaded.
    """
    csv_path = DATA_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        print(f"  SKIP {table_name}: {csv_path.name} not found")
        return 0

    # Read header to get column names (handles quoted camelCase columns)
    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    columns = [c.strip().strip('"') for c in header_line.split(",")]

    # Quote all column names to handle camelCase safely
    col_list = ", ".join(f'"{c}"' for c in columns)
    copy_sql = (
        f'COPY {table_name} ({col_list}) '
        f"FROM STDIN WITH (FORMAT CSV, HEADER, DELIMITER ',')"
    )

    # Preprocess: fix pandas float-int issue (e.g. "181.0" → "181")
    # This happens when integer columns contain NaN — pandas promotes to float64
    _float_int_re = re.compile(r"(?<![.\d])(\d+)\.0(?![.\d])")

    start = time.perf_counter()
    with open(csv_path, "r", encoding="utf-8") as f:
        cleaned = io.StringIO()
        for line in f:
            cleaned.write(_float_int_re.sub(r"\1", line))
        cleaned.seek(0)
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, cleaned)
    conn.commit()
    elapsed = time.perf_counter() - start

    # Get row count
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cur.fetchone()[0]

    print(f"  {table_name}: {count:,} rows loaded ({elapsed:.1f}s)")
    return count


def print_summary(row_counts: dict[str, int]) -> None:
    """Print a summary table of all loaded tables."""
    print("\n" + "=" * 45)
    print(f"{'Table':<22} {'Rows':>12}")
    print("-" * 45)
    total = 0
    for table, count in row_counts.items():
        print(f"  {table:<20} {count:>12,}")
        total += count
    print("-" * 45)
    print(f"  {'TOTAL':<20} {total:>12,}")
    print("=" * 45)


def main() -> None:
    """Main entry point: create schema, load CSVs, rebuild indexes."""
    print("Connecting to PostgreSQL...")
    try:
        conn = get_connection()
    except psycopg2.OperationalError as e:
        print(f"\nConnection failed: {e}")
        print("Make sure PostgreSQL is running and .env is configured.")
        sys.exit(1)

    print("Connected.\n")

    # Step 1: Execute schema (drop + create tables + indexes)
    print("[1/4] Creating schema...")
    execute_schema(conn)

    # Step 2: Drop indexes for faster bulk loading
    print("\n[2/4] Dropping indexes for bulk load...")
    index_defs = drop_indexes(conn)

    # Step 3: Load CSVs in dependency order
    print("\n[3/4] Loading CSVs...")
    row_counts: dict[str, int] = {}
    total_start = time.perf_counter()

    for table in LOAD_ORDER:
        count = load_table(conn, table)
        row_counts[table] = count

    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll tables loaded in {total_elapsed:.1f}s.")

    # Step 4: Recreate indexes
    print("\n[4/4] Recreating indexes...")
    idx_start = time.perf_counter()
    recreate_indexes(conn, index_defs)
    idx_elapsed = time.perf_counter() - idx_start
    print(f"Indexes rebuilt in {idx_elapsed:.1f}s.")

    # Summary
    print_summary(row_counts)

    conn.close()
    print("\nDone. Connection closed.")


if __name__ == "__main__":
    main()
