-- ============================================================
-- Data Validation Queries
-- Supply Chain & Logistics Database
--
-- 9 checks: row counts, nulls, duplicates, orphans,
-- impossible dates, missing events, weight mismatches,
-- negative inventory, past delivery dates.
--
-- Each query is self-contained — run individually in any
-- SQL client (psql, pgAdmin, DBeaver, Autonmis, etc.)
-- ============================================================


-- ---------------------------------------------------------
-- 1. Row Counts per Table
--    Expected: 50 + 10K + 50K + 25K + 500K + 800K + 500K
--              + 2M + 1.5M ≈ 5.4M total
-- ---------------------------------------------------------
SELECT 'warehouses'      AS table_name, COUNT(*) AS row_count, 50        AS expected FROM warehouses
UNION ALL
SELECT 'products',                      COUNT(*),              10000              FROM products
UNION ALL
SELECT 'customers',                     COUNT(*),              50000              FROM customers
UNION ALL
SELECT 'inventory',                     COUNT(*),              25000              FROM inventory
UNION ALL
SELECT 'orders',                        COUNT(*),              500000             FROM orders
UNION ALL
SELECT 'shipments',                     COUNT(*),              800000             FROM shipments
UNION ALL
SELECT 'order_shipments',               COUNT(*),              500000             FROM order_shipments
UNION ALL
SELECT 'shipment_items',                COUNT(*),              2000000            FROM shipment_items
UNION ALL
SELECT 'delivery_events',               COUNT(*),              1500000            FROM delivery_events
ORDER BY table_name;


-- ---------------------------------------------------------
-- 2. Null Percentages in Critical Columns (target: 3-5%)
-- ---------------------------------------------------------
SELECT 'orders."orderPriority"' AS column_check,
       ROUND(100.0 * COUNT(*) FILTER (WHERE "orderPriority" IS NULL)
                   / COUNT(*), 2) AS null_pct,
       '3-5%' AS target
FROM orders

UNION ALL
SELECT 'orders.fulfillment_warehouse_id',
       ROUND(100.0 * COUNT(*) FILTER (WHERE fulfillment_warehouse_id IS NULL)
                   / COUNT(*), 2),
       '3-5%'
FROM orders

UNION ALL
SELECT 'inventory.quantity_on_hand',
       ROUND(100.0 * COUNT(*) FILTER (WHERE quantity_on_hand IS NULL)
                   / COUNT(*), 2),
       '3-5%'
FROM inventory

UNION ALL
SELECT 'inventory.reorder_level',
       ROUND(100.0 * COUNT(*) FILTER (WHERE reorder_level IS NULL)
                   / COUNT(*), 2),
       '3-5%'
FROM inventory

UNION ALL
SELECT 'shipments.carrier',
       ROUND(100.0 * COUNT(*) FILTER (WHERE carrier IS NULL)
                   / COUNT(*), 2),
       '3-5%'
FROM shipments

UNION ALL
SELECT 'shipments.shipment_cost',
       ROUND(100.0 * COUNT(*) FILTER (WHERE shipment_cost IS NULL)
                   / COUNT(*), 2),
       '3-5%'
FROM shipments;


-- ---------------------------------------------------------
-- 3. Duplicate SKUs (target: ~2% of distinct SKUs)
-- ---------------------------------------------------------
SELECT dup_skus                                              AS duplicate_sku_count,
       total_distinct                                        AS total_distinct_skus,
       ROUND(100.0 * dup_skus / NULLIF(total_distinct, 0), 2) AS duplicate_pct,
       '~2%'                                                 AS target
FROM (SELECT COUNT(*) AS dup_skus
      FROM (SELECT sku FROM products GROUP BY sku HAVING COUNT(*) > 1) d) a,
     (SELECT COUNT(DISTINCT sku) AS total_distinct FROM products) b;


-- ---------------------------------------------------------
-- 4. Orphaned Product IDs in shipment_items (target: 8-10%)
--    Items referencing product_ids that don't exist
-- ---------------------------------------------------------
SELECT COUNT(*) FILTER (WHERE p.product_id IS NULL) AS orphaned_count,
       COUNT(*)                                     AS total_items,
       ROUND(100.0 * COUNT(*) FILTER (WHERE p.product_id IS NULL)
                   / COUNT(*), 2)                   AS orphaned_pct,
       '8-10%'                                      AS target
FROM shipment_items si
LEFT JOIN products p ON si.product_id = p.product_id;


-- ---------------------------------------------------------
-- 5. Impossible Dates: actual_delivery < shipment_date
--    (target: ~5% of shipments with an actual_delivery)
-- ---------------------------------------------------------
SELECT COUNT(*) FILTER (
           WHERE "actual_delivery" IS NOT NULL
             AND "actual_delivery" != ''
             AND "actual_delivery"::DATE < shipment_date
       ) AS impossible_count,
       COUNT(*) FILTER (
           WHERE "actual_delivery" IS NOT NULL
             AND "actual_delivery" != ''
       ) AS total_with_actual,
       ROUND(100.0 *
           COUNT(*) FILTER (
               WHERE "actual_delivery" IS NOT NULL
                 AND "actual_delivery" != ''
                 AND "actual_delivery"::DATE < shipment_date)
           / NULLIF(COUNT(*) FILTER (
               WHERE "actual_delivery" IS NOT NULL
                 AND "actual_delivery" != ''), 0), 2
       ) AS impossible_pct,
       '~5%' AS target
FROM shipments;


-- ---------------------------------------------------------
-- 6. Missing Delivery Events (target: ~15%)
--    Shipments with zero rows in delivery_events
-- ---------------------------------------------------------
SELECT COUNT(*) FILTER (WHERE de.shipment_id IS NULL) AS missing_count,
       COUNT(*)                                       AS total_shipments,
       ROUND(100.0 * COUNT(*) FILTER (WHERE de.shipment_id IS NULL)
                   / COUNT(*), 2)                     AS missing_pct,
       '~15%'                                         AS target
FROM shipments s
LEFT JOIN (SELECT DISTINCT shipment_id FROM delivery_events) de
    ON s.shipment_id = de.shipment_id;


-- ---------------------------------------------------------
-- 7. Weight Mismatches (target: ~10%)
--    totalWeight vs SUM(shipment_items.quantity * products.weightKg)
--    Only comparable shipments (both have weight data)
-- ---------------------------------------------------------
WITH calculated_weights AS (
    SELECT si.shipment_id,
           SUM(si.quantity * p."weightKg") AS calc_weight
    FROM shipment_items si
    INNER JOIN products p ON si.product_id = p.product_id
    GROUP BY si.shipment_id
)
SELECT COUNT(*) FILTER (
           WHERE ABS(s."totalWeight" - cw.calc_weight)
               / GREATEST(cw.calc_weight, 0.001) > 0.01
       ) AS mismatched_count,
       COUNT(*) AS total_comparable,
       ROUND(100.0 *
           COUNT(*) FILTER (
               WHERE ABS(s."totalWeight" - cw.calc_weight)
                   / GREATEST(cw.calc_weight, 0.001) > 0.01)
           / NULLIF(COUNT(*), 0), 2
       ) AS mismatch_pct,
       '~10%' AS target
FROM shipments s
INNER JOIN calculated_weights cw ON s.shipment_id = cw.shipment_id
WHERE s."totalWeight" IS NOT NULL
  AND cw.calc_weight > 0;


-- ---------------------------------------------------------
-- 8. Negative Inventory (target: ~3%)
--    quantity_on_hand < 0 among non-null values
-- ---------------------------------------------------------
SELECT COUNT(*) FILTER (WHERE quantity_on_hand < 0)          AS negative_count,
       COUNT(*) FILTER (WHERE quantity_on_hand IS NOT NULL)   AS total_with_qty,
       ROUND(100.0 *
           COUNT(*) FILTER (WHERE quantity_on_hand < 0)
           / NULLIF(COUNT(*) FILTER (WHERE quantity_on_hand IS NOT NULL), 0), 2
       ) AS negative_pct,
       '~3%' AS target
FROM inventory;


-- ---------------------------------------------------------
-- 9. Past Requested Delivery Date (target: ~5%)
--    requested_delivery_date < order_date
-- ---------------------------------------------------------
SELECT COUNT(*) FILTER (
           WHERE "requested_delivery_date" IS NOT NULL
             AND "requested_delivery_date" != ''
             AND "requested_delivery_date"::DATE < order_date
       ) AS past_count,
       COUNT(*) FILTER (
           WHERE "requested_delivery_date" IS NOT NULL
             AND "requested_delivery_date" != ''
       ) AS total_with_rdd,
       ROUND(100.0 *
           COUNT(*) FILTER (
               WHERE "requested_delivery_date" IS NOT NULL
                 AND "requested_delivery_date" != ''
                 AND "requested_delivery_date"::DATE < order_date)
           / NULLIF(COUNT(*) FILTER (
               WHERE "requested_delivery_date" IS NOT NULL
                 AND "requested_delivery_date" != ''), 0), 2
       ) AS past_rdd_pct,
       '~5%' AS target
FROM orders;
