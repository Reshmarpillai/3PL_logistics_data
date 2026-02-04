"""Data validation — runs SQL checks against PostgreSQL and prints PASS/FAIL summary.

Validates row counts and all intentional data-quality issues against
target percentages defined in the project spec.
"""

import sys
from pathlib import Path

import psycopg2

# Allow import of db_config from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_config import get_connection


# ── Expected row counts ──────────────────────────────────────
EXPECTED_ROWS: dict[str, int] = {
    "warehouses": 50,
    "products": 10_000,
    "customers": 50_000,
    "inventory": 25_000,
    "orders": 500_000,
    "shipments": 800_000,
    "order_shipments": 500_000,
    "shipment_items": 2_000_000,
    "delivery_events": 1_500_000,
}

# ── Data-quality checks ─────────────────────────────────────
# Each tuple: (display_name, sql_returning_one_numeric, target_min%, target_max%)
QUALITY_CHECKS: list[tuple[str, str, float, float]] = [
    # --- Null percentages (target 3-5%) ---
    (
        'Null "orderPriority" (orders)',
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE "orderPriority" IS NULL)
                / COUNT(*), 2)
           FROM orders""",
        2.0, 6.0,
    ),
    (
        "Null fulfillment_wh_id (orders)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE fulfillment_warehouse_id IS NULL)
                / COUNT(*), 2)
           FROM orders""",
        2.0, 6.0,
    ),
    (
        "Null quantity_on_hand (inventory)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE quantity_on_hand IS NULL)
                / COUNT(*), 2)
           FROM inventory""",
        2.0, 6.0,
    ),
    (
        "Null reorder_level (inventory)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE reorder_level IS NULL)
                / COUNT(*), 2)
           FROM inventory""",
        2.0, 6.0,
    ),
    (
        "Null carrier (shipments)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE carrier IS NULL)
                / COUNT(*), 2)
           FROM shipments""",
        2.0, 6.0,
    ),
    (
        "Null shipment_cost (shipments)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE shipment_cost IS NULL)
                / COUNT(*), 2)
           FROM shipments""",
        2.0, 6.0,
    ),

    # --- Duplicate SKUs (target ~2% of distinct SKUs) ---
    (
        "Duplicate SKUs (products)",
        """SELECT ROUND(100.0 * dup_count / NULLIF(total_distinct, 0), 2)
           FROM (SELECT COUNT(*) AS dup_count
                 FROM (SELECT sku FROM products
                       GROUP BY sku HAVING COUNT(*) > 1) d) a,
                (SELECT COUNT(DISTINCT sku) AS total_distinct
                 FROM products) b""",
        1.0, 3.0,
    ),

    # --- Orphaned product_ids (target 8-10%) ---
    (
        "Orphaned product_ids (ship_items)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE p.product_id IS NULL)
                / COUNT(*), 2)
           FROM shipment_items si
           LEFT JOIN products p ON si.product_id = p.product_id""",
        7.0, 11.0,
    ),

    # --- Impossible dates (target ~5%) ---
    (
        "Impossible dates (shipments)",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (
                    WHERE "actual_delivery" IS NOT NULL
                      AND "actual_delivery" != ''
                      AND "actual_delivery"::DATE < shipment_date)
                / NULLIF(COUNT(*) FILTER (
                    WHERE "actual_delivery" IS NOT NULL
                      AND "actual_delivery" != ''), 0), 2)
           FROM shipments""",
        3.5, 6.5,
    ),

    # --- Missing delivery events (target ~15%) ---
    (
        "Missing delivery events",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE de.shipment_id IS NULL)
                / COUNT(*), 2)
           FROM shipments s
           LEFT JOIN (SELECT DISTINCT shipment_id
                      FROM delivery_events) de
               ON s.shipment_id = de.shipment_id""",
        12.0, 18.0,
    ),

    # --- Weight mismatches (target ~10%) ---
    (
        "Weight mismatches (shipments)",
        """WITH calc AS (
               SELECT si.shipment_id,
                      SUM(si.quantity * p."weightKg") AS cw
               FROM shipment_items si
               INNER JOIN products p ON si.product_id = p.product_id
               GROUP BY si.shipment_id)
           SELECT ROUND(100.0
                * COUNT(*) FILTER (
                    WHERE ABS(s."totalWeight" - c.cw)
                        / GREATEST(c.cw, 0.001) > 0.01)
                / NULLIF(COUNT(*), 0), 2)
           FROM shipments s
           INNER JOIN calc c ON s.shipment_id = c.shipment_id
           WHERE s."totalWeight" IS NOT NULL AND c.cw > 0""",
        7.0, 13.0,
    ),

    # --- Negative inventory (target ~3%) ---
    (
        "Negative inventory",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (WHERE quantity_on_hand < 0)
                / NULLIF(COUNT(*) FILTER (
                    WHERE quantity_on_hand IS NOT NULL), 0), 2)
           FROM inventory""",
        1.5, 4.5,
    ),

    # --- Past requested_delivery_date (target ~5%) ---
    (
        "Past requested_delivery_date",
        """SELECT ROUND(100.0
                * COUNT(*) FILTER (
                    WHERE "requested_delivery_date" IS NOT NULL
                      AND "requested_delivery_date" != ''
                      AND "requested_delivery_date"::DATE < order_date)
                / NULLIF(COUNT(*) FILTER (
                    WHERE "requested_delivery_date" IS NOT NULL
                      AND "requested_delivery_date" != ''), 0), 2)
           FROM orders""",
        3.5, 6.5,
    ),
]


# ── Helpers ──────────────────────────────────────────────────

def check_row_counts(conn: psycopg2.extensions.connection) -> list[tuple[str, int, int]]:
    """Return (table, actual_count, expected_count) for every table."""
    results: list[tuple[str, int, int]] = []
    with conn.cursor() as cur:
        for table, expected in EXPECTED_ROWS.items():
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            actual: int = cur.fetchone()[0]
            results.append((table, actual, expected))
    return results


def check_quality(conn: psycopg2.extensions.connection) -> list[tuple[str, float, float, float, bool]]:
    """Return (name, actual_pct, min_target, max_target, passed) per check."""
    results: list[tuple[str, float, float, float, bool]] = []
    with conn.cursor() as cur:
        for name, sql, lo, hi in QUALITY_CHECKS:
            cur.execute(sql)
            raw = cur.fetchone()[0]
            value = float(raw) if raw is not None else 0.0
            passed = lo <= value <= hi
            results.append((name, value, lo, hi, passed))
    return results


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    print("Connecting to PostgreSQL...")
    try:
        conn = get_connection()
    except psycopg2.OperationalError as e:
        print(f"\nConnection failed: {e}")
        print("Make sure PostgreSQL is running and .env is configured.")
        sys.exit(1)
    print("Connected.\n")

    # ── 1. Row counts ────────────────────────────────────────
    print("=" * 65)
    print("  ROW COUNTS")
    print("=" * 65)
    print(f"  {'Table':<22} {'Actual':>12} {'Expected':>12}")
    print("-" * 65)

    row_results = check_row_counts(conn)
    total_actual = 0
    total_expected = 0
    for table, actual, expected in row_results:
        flag = "" if actual == expected else "  *"
        print(f"  {table:<22} {actual:>12,} {expected:>12,}{flag}")
        total_actual += actual
        total_expected += expected

    print("-" * 65)
    print(f"  {'TOTAL':<22} {total_actual:>12,} {total_expected:>12,}")
    print("  (* = differs from expected)\n")

    # ── 2. Data-quality checks ───────────────────────────────
    print("=" * 65)
    print("  DATA QUALITY CHECKS")
    print("=" * 65)
    print(f"  {'Check':<40} {'Actual':>7} {'Target':>10} {'':>6}")
    print("-" * 65)

    quality_results = check_quality(conn)
    pass_count = 0
    for name, value, lo, hi, passed in quality_results:
        status = "PASS" if passed else "FAIL"
        target_str = f"{lo:.0f}-{hi:.0f}%"
        print(f"  {name:<40} {value:>6.2f}% {target_str:>10}  [{status}]")
        pass_count += int(passed)

    print("-" * 65)
    total_checks = len(quality_results)
    print(f"  {pass_count}/{total_checks} quality checks passed\n")

    # ── 3. Summary ───────────────────────────────────────────
    print("=" * 65)
    if pass_count == total_checks:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        failed = total_checks - pass_count
        print(f"  RESULT: {failed} CHECK(S) FAILED — review above for details")
    print("=" * 65)

    conn.close()


if __name__ == "__main__":
    main()
