-- ============================================================
-- Supply Chain & Logistics — Analysis Queries
-- 10 Metrics for Autonmis Dashboard
--
-- SCHEMA NOTES:
--   - camelCase columns require double quotes ("columnName")
--   - Several DATE/TIMESTAMP columns are stored as VARCHAR
--     and must be cast: "actual_delivery"::DATE
--   - 3 redundant cost columns in shipments table
--   - No foreign keys — orphaned records exist by design
-- ============================================================


-- ============================================================
-- METRIC 1: On-Time Delivery Rate
--
-- Monthly trend of on-time vs late deliveries.
-- Casts VARCHAR actual_delivery to DATE.
-- Excludes impossible dates (actual < shipment) for clean metric.
-- Tables: shipments
-- ============================================================

WITH delivered AS (
    SELECT
        shipment_id,
        shipment_date,
        estimated_delivery,
        "actual_delivery"::DATE AS actual_date,
        DATE_TRUNC('month', shipment_date) AS month
    FROM shipments
    WHERE shipment_status = 'delivered'
      AND "actual_delivery" IS NOT NULL
      AND estimated_delivery IS NOT NULL
      AND "actual_delivery"::DATE >= shipment_date  -- exclude impossible dates
)
SELECT
    month,
    COUNT(*) AS total_delivered,
    COUNT(*) FILTER (WHERE actual_date <= estimated_delivery) AS on_time,
    COUNT(*) FILTER (WHERE actual_date > estimated_delivery) AS late,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE actual_date <= estimated_delivery) / COUNT(*),
        2
    ) AS on_time_pct
FROM delivered
GROUP BY month
ORDER BY month;


-- ============================================================
-- METRIC 2: Average Delivery Time by Carrier & Route
--
-- 4-table join: shipments + warehouses(origin) + warehouses(dest)
--   + order_shipments
-- Cross-border flag derived from origin vs dest country.
-- Tables: shipments, warehouses (x2), order_shipments
-- ============================================================

WITH delivery_times AS (
    SELECT
        s.shipment_id,
        s.carrier,
        s.shipment_date,
        s."actual_delivery"::DATE AS actual_date,
        (s."actual_delivery"::DATE - s.shipment_date) AS delivery_days,
        wo.country AS origin_country,
        wo.region AS origin_region,
        wo.warehouse_type AS origin_type,
        wd.country AS dest_country,
        wd.region AS dest_region,
        CASE
            WHEN wo.country != wd.country THEN 'Cross-Border'
            ELSE 'Domestic'
        END AS route_type
    FROM shipments s
    JOIN warehouses wo ON s.origin_warehouse_id = wo.warehouse_id
    JOIN warehouses wd ON s.dest_warehouse_id = wd.warehouse_id
    JOIN order_shipments os ON s.shipment_id = os.shipment_id
    WHERE s.shipment_status = 'delivered'
      AND s."actual_delivery" IS NOT NULL
      AND s.carrier IS NOT NULL
      AND s."actual_delivery"::DATE >= s.shipment_date
)
SELECT
    carrier,
    route_type,
    origin_region,
    dest_region,
    COUNT(*) AS shipment_count,
    ROUND(AVG(delivery_days), 1) AS avg_delivery_days,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delivery_days)::NUMERIC,
        1
    ) AS median_delivery_days,
    MIN(delivery_days) AS min_days,
    MAX(delivery_days) AS max_days
FROM delivery_times
GROUP BY carrier, route_type, origin_region, dest_region
ORDER BY carrier, route_type, avg_delivery_days;


-- ============================================================
-- METRIC 3: Shipment Delay Analysis
--
-- 4-table join: shipments + delivery_events + warehouses(origin)
--   + warehouses(dest)
-- Breakdown by delay_reason with monthly trend.
-- Tables: shipments, delivery_events, warehouses (x2)
-- ============================================================

-- 3A: Delay breakdown by reason and route type
WITH delayed_shipments AS (
    SELECT
        s.shipment_id,
        s.shipment_date,
        s.carrier,
        de.delay_reason,
        de."eventType",
        wo.country AS origin_country,
        wd.country AS dest_country,
        CASE
            WHEN wo.country != wd.country THEN 'Cross-Border'
            ELSE 'Domestic'
        END AS route_type,
        DATE_TRUNC('month', s.shipment_date) AS month
    FROM shipments s
    JOIN delivery_events de ON s.shipment_id = de.shipment_id
    JOIN warehouses wo ON s.origin_warehouse_id = wo.warehouse_id
    JOIN warehouses wd ON s.dest_warehouse_id = wd.warehouse_id
    WHERE de.delay_reason IS NOT NULL
      AND de.delay_reason != 'none'
)
SELECT
    delay_reason,
    route_type,
    COUNT(*) AS event_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM delayed_shipments
GROUP BY delay_reason, route_type
ORDER BY event_count DESC;

-- 3B: Monthly delay trend
SELECT
    DATE_TRUNC('month', s.shipment_date) AS month,
    COUNT(DISTINCT s.shipment_id) AS total_shipments,
    COUNT(DISTINCT s.shipment_id) FILTER (
        WHERE de.delay_reason IS NOT NULL AND de.delay_reason != 'none'
    ) AS delayed_shipments,
    ROUND(
        100.0 * COUNT(DISTINCT s.shipment_id) FILTER (
            WHERE de.delay_reason IS NOT NULL AND de.delay_reason != 'none'
        ) / NULLIF(COUNT(DISTINCT s.shipment_id), 0),
        2
    ) AS delay_pct
FROM shipments s
LEFT JOIN delivery_events de ON s.shipment_id = de.shipment_id
GROUP BY DATE_TRUNC('month', s.shipment_date)
ORDER BY month;


-- ============================================================
-- METRIC 4: Failed Delivery Rate & Reasons
--
-- Percentage of shipments with a 'failed' event, top reasons,
-- and monthly trend.
-- Tables: delivery_events, shipments
-- ============================================================

-- 4A: Overall failed rate by reason
WITH failed AS (
    SELECT
        de.shipment_id,
        de.delay_reason,
        s.shipment_date,
        s.carrier
    FROM delivery_events de
    JOIN shipments s ON de.shipment_id = s.shipment_id
    WHERE de."eventType" = 'failed'
)
SELECT
    delay_reason,
    COUNT(*) AS failed_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_failures
FROM failed
GROUP BY delay_reason
ORDER BY failed_count DESC;

-- 4B: Monthly failed delivery trend
SELECT
    DATE_TRUNC('month', s.shipment_date) AS month,
    COUNT(DISTINCT s.shipment_id) AS total_shipments,
    COUNT(DISTINCT CASE
        WHEN de."eventType" = 'failed' THEN s.shipment_id
    END) AS failed_shipments,
    ROUND(
        100.0 * COUNT(DISTINCT CASE
            WHEN de."eventType" = 'failed' THEN s.shipment_id
        END) / NULLIF(COUNT(DISTINCT s.shipment_id), 0),
        2
    ) AS failed_pct
FROM shipments s
LEFT JOIN delivery_events de ON s.shipment_id = de.shipment_id
GROUP BY DATE_TRUNC('month', s.shipment_date)
ORDER BY month;


-- ============================================================
-- METRIC 5: Warehouse Utilization
--
-- Proxy utilization: SUM(quantity_on_hand * weightKg) / capacitySqft
-- Grouped by warehouse_type, flags over/under-utilized.
-- Tables: warehouses, inventory, products
-- ============================================================

-- 5A: Per-warehouse utilization
WITH warehouse_load AS (
    SELECT
        w.warehouse_id,
        w.warehouse_name,
        w.warehouse_type,
        w.country,
        w."capacitySqft",
        COALESCE(SUM(i.quantity_on_hand * p."weightKg"), 0) AS total_weight_stored,
        COUNT(DISTINCT i.product_id) AS unique_products,
        COALESCE(SUM(i.quantity_on_hand), 0) AS total_units
    FROM warehouses w
    LEFT JOIN inventory i ON w.warehouse_id = i.warehouse_id
    LEFT JOIN products p ON i.product_id = p.product_id
    WHERE i.quantity_on_hand IS NOT NULL OR i.inventory_id IS NULL
    GROUP BY w.warehouse_id, w.warehouse_name, w.warehouse_type,
             w.country, w."capacitySqft"
)
SELECT
    warehouse_id,
    warehouse_name,
    warehouse_type,
    country,
    "capacitySqft",
    unique_products,
    total_units,
    ROUND(total_weight_stored, 2) AS total_weight_kg,
    ROUND(total_weight_stored / NULLIF("capacitySqft", 0), 4) AS weight_per_sqft,
    CASE
        WHEN total_weight_stored / NULLIF("capacitySqft", 0) > 0.5 THEN 'Over-utilized'
        WHEN total_weight_stored / NULLIF("capacitySqft", 0) < 0.05 THEN 'Under-utilized'
        ELSE 'Normal'
    END AS utilization_flag
FROM warehouse_load
ORDER BY warehouse_type, weight_per_sqft DESC;

-- 5B: Summary by warehouse type
SELECT
    w.warehouse_type,
    COUNT(*) AS warehouse_count,
    ROUND(AVG(w."capacitySqft")) AS avg_capacity_sqft,
    ROUND(AVG(COALESCE(agg.total_units, 0))) AS avg_units_stored,
    ROUND(AVG(COALESCE(agg.total_weight, 0)), 2) AS avg_weight_stored
FROM warehouses w
LEFT JOIN (
    SELECT
        i.warehouse_id,
        SUM(i.quantity_on_hand) AS total_units,
        SUM(i.quantity_on_hand * p."weightKg") AS total_weight
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    WHERE i.quantity_on_hand IS NOT NULL
    GROUP BY i.warehouse_id
) agg ON w.warehouse_id = agg.warehouse_id
GROUP BY w.warehouse_type
ORDER BY w.warehouse_type;


-- ============================================================
-- METRIC 6: Inventory Turnover & Stock-Out Frequency
--
-- Turnover ratio = total units shipped / avg quantity_on_hand
-- Stock-out = quantity_on_hand <= 0
-- Tables: inventory, products, shipment_items
-- ============================================================

-- 6A: Turnover by product category
WITH shipped_units AS (
    SELECT
        si.product_id,
        SUM(si.quantity) AS total_shipped
    FROM shipment_items si
    WHERE si.product_id IN (SELECT product_id FROM products)
    GROUP BY si.product_id
),
inventory_avg AS (
    SELECT
        i.product_id,
        AVG(i.quantity_on_hand) AS avg_on_hand
    FROM inventory i
    WHERE i.quantity_on_hand IS NOT NULL
    GROUP BY i.product_id
)
SELECT
    p."productCategory",
    COUNT(DISTINCT p.product_id) AS product_count,
    ROUND(AVG(ia.avg_on_hand), 1) AS avg_inventory,
    ROUND(AVG(su.total_shipped), 1) AS avg_shipped,
    ROUND(
        SUM(su.total_shipped)::NUMERIC / NULLIF(SUM(ia.avg_on_hand), 0),
        2
    ) AS turnover_ratio
FROM products p
LEFT JOIN shipped_units su ON p.product_id = su.product_id
LEFT JOIN inventory_avg ia ON p.product_id = ia.product_id
GROUP BY p."productCategory"
ORDER BY turnover_ratio DESC NULLS LAST;

-- 6B: Stock-out frequency by category and warehouse type
SELECT
    p."productCategory",
    w.warehouse_type,
    COUNT(*) AS inventory_records,
    COUNT(*) FILTER (WHERE i.quantity_on_hand <= 0) AS stockout_count,
    COUNT(*) FILTER (WHERE i.quantity_on_hand < 0) AS negative_inventory,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE i.quantity_on_hand <= 0)
        / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_pct
FROM inventory i
JOIN products p ON i.product_id = p.product_id
JOIN warehouses w ON i.warehouse_id = w.warehouse_id
WHERE i.quantity_on_hand IS NOT NULL
GROUP BY p."productCategory", w.warehouse_type
ORDER BY stockout_pct DESC;


-- ============================================================
-- METRIC 7: Cost Per Shipment by Carrier & Route
--
-- Average cost grouped by carrier and domestic/cross-border.
-- Surfaces the 3 redundant cost columns and their discrepancies.
-- Tables: shipments, warehouses (x2)
-- ============================================================

-- 7A: Avg cost by carrier and route type
SELECT
    s.carrier,
    CASE
        WHEN wo.country != wd.country THEN 'Cross-Border'
        ELSE 'Domestic'
    END AS route_type,
    COUNT(*) AS shipment_count,
    ROUND(AVG(s.shipment_cost), 2) AS avg_shipment_cost,
    ROUND(AVG(s.total_cost), 2) AS avg_total_cost,
    ROUND(AVG(s.cost_amount), 2) AS avg_cost_amount,
    -- Surface the redundancy: avg absolute difference between cost columns
    ROUND(AVG(ABS(s.shipment_cost - s.total_cost)), 2) AS avg_diff_cost_vs_total,
    ROUND(AVG(ABS(s.shipment_cost - s.cost_amount)), 2) AS avg_diff_cost_vs_amount
FROM shipments s
JOIN warehouses wo ON s.origin_warehouse_id = wo.warehouse_id
JOIN warehouses wd ON s.dest_warehouse_id = wd.warehouse_id
WHERE s.carrier IS NOT NULL
  AND s.shipment_cost IS NOT NULL
GROUP BY s.carrier,
         CASE WHEN wo.country != wd.country THEN 'Cross-Border' ELSE 'Domestic' END
ORDER BY s.carrier, route_type;

-- 7B: Monthly cost trend
SELECT
    DATE_TRUNC('month', s.shipment_date) AS month,
    COUNT(*) AS shipment_count,
    ROUND(AVG(s.shipment_cost), 2) AS avg_cost,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.shipment_cost)::NUMERIC,
        2
    ) AS median_cost,
    ROUND(SUM(s.shipment_cost), 2) AS total_cost_sum
FROM shipments s
WHERE s.shipment_cost IS NOT NULL
GROUP BY DATE_TRUNC('month', s.shipment_date)
ORDER BY month;


-- ============================================================
-- METRIC 8: Order Fulfillment Time
--
-- 4-table join: orders + order_shipments + shipments + customers
-- Days from order_date to actual_delivery, by priority.
-- Shows whether express/critical actually arrive faster.
-- Tables: orders, order_shipments, shipments, customers
-- ============================================================

-- 8A: Fulfillment time by priority and customer type
WITH fulfillment AS (
    SELECT
        o.order_id,
        o.order_date,
        o."orderPriority",
        c."customerType",
        s."actual_delivery"::DATE AS actual_date,
        (s."actual_delivery"::DATE - o.order_date) AS fulfillment_days
    FROM orders o
    JOIN order_shipments os ON o.order_id = os.order_id
    JOIN shipments s ON os.shipment_id = s.shipment_id
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE s.shipment_status = 'delivered'
      AND s."actual_delivery" IS NOT NULL
      AND s."actual_delivery"::DATE >= o.order_date
      AND o."orderPriority" IS NOT NULL
)
SELECT
    "orderPriority",
    "customerType",
    COUNT(*) AS order_count,
    ROUND(AVG(fulfillment_days), 1) AS avg_fulfillment_days,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fulfillment_days)::NUMERIC,
        1
    ) AS median_days,
    ROUND(
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fulfillment_days)::NUMERIC,
        1
    ) AS p95_days,
    MIN(fulfillment_days) AS min_days,
    MAX(fulfillment_days) AS max_days
FROM fulfillment
GROUP BY "orderPriority", "customerType"
ORDER BY "orderPriority", "customerType";

-- 8B: Monthly fulfillment trend by priority
WITH fulfillment AS (
    SELECT
        o.order_date,
        o."orderPriority",
        (s."actual_delivery"::DATE - o.order_date) AS fulfillment_days,
        DATE_TRUNC('month', o.order_date) AS month
    FROM orders o
    JOIN order_shipments os ON o.order_id = os.order_id
    JOIN shipments s ON os.shipment_id = s.shipment_id
    WHERE s.shipment_status = 'delivered'
      AND s."actual_delivery" IS NOT NULL
      AND s."actual_delivery"::DATE >= o.order_date
      AND o."orderPriority" IS NOT NULL
)
SELECT
    month,
    "orderPriority",
    COUNT(*) AS order_count,
    ROUND(AVG(fulfillment_days), 1) AS avg_fulfillment_days
FROM fulfillment
GROUP BY month, "orderPriority"
ORDER BY month, "orderPriority";


-- ============================================================
-- METRIC 9: Customer Order Volume & Segmentation
--
-- B2B vs B2C breakdown, top customers (validates Zipf),
-- and monthly trend.
-- Tables: orders, customers
-- ============================================================

-- 9A: Volume by customer type
SELECT
    c."customerType",
    COUNT(DISTINCT c.customer_id) AS customer_count,
    COUNT(o.order_id) AS total_orders,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    ROUND(SUM(o.total_amount), 2) AS total_revenue
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c."customerType"
ORDER BY total_orders DESC;

-- 9B: Top 20 customers by order count (validates Zipf distribution)
SELECT
    c.customer_id,
    c.customer_name,
    c."customerType",
    COUNT(o.order_id) AS order_count,
    ROUND(SUM(o.total_amount), 2) AS total_spent,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name, c."customerType"
ORDER BY order_count DESC
LIMIT 20;

-- 9C: Monthly order volume trend by customer type
SELECT
    DATE_TRUNC('month', o.order_date) AS month,
    c."customerType",
    COUNT(o.order_id) AS order_count,
    COUNT(DISTINCT o.customer_id) AS active_customers,
    ROUND(SUM(o.total_amount), 2) AS monthly_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY DATE_TRUNC('month', o.order_date), c."customerType"
ORDER BY month, c."customerType";

-- 9D: Order distribution percentiles (Zipf shape validation)
WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY customer_id
)
SELECT
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY order_count)::NUMERIC, 1) AS p50_orders,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY order_count)::NUMERIC, 1) AS p75_orders,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY order_count)::NUMERIC, 1) AS p90_orders,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY order_count)::NUMERIC, 1) AS p95_orders,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY order_count)::NUMERIC, 1) AS p99_orders,
    MAX(order_count) AS max_orders,
    ROUND(AVG(order_count), 1) AS avg_orders
FROM customer_orders;


-- ============================================================
-- METRIC 10: Data Quality Dashboard
--
-- Surfaces all intentional data issues with counts,
-- percentages, and target comparisons.
-- Tables: all 9 tables
-- ============================================================

-- 10A: Orphaned product_ids in shipment_items — target 8-10%
SELECT
    'Orphaned Product IDs in shipment_items' AS issue,
    COUNT(*) AS total_items,
    COUNT(*) FILTER (
        WHERE si.product_id NOT IN (SELECT product_id FROM products)
    ) AS orphaned_count,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE si.product_id NOT IN (SELECT product_id FROM products)
        ) / COUNT(*),
        2
    ) AS orphaned_pct,
    '8-10%' AS target
FROM shipment_items si;

-- 10B: Weight mismatches (totalWeight vs SUM of items) — target 10%
WITH actual_weights AS (
    SELECT
        si.shipment_id,
        SUM(si.quantity * p."weightKg") AS calculated_weight
    FROM shipment_items si
    JOIN products p ON si.product_id = p.product_id
    GROUP BY si.shipment_id
)
SELECT
    'Weight Mismatches in shipments' AS issue,
    COUNT(*) AS total_checked,
    COUNT(*) FILTER (
        WHERE ABS(s."totalWeight" - aw.calculated_weight)
              / NULLIF(aw.calculated_weight, 0) > 0.15
    ) AS mismatch_count,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE ABS(s."totalWeight" - aw.calculated_weight)
                  / NULLIF(aw.calculated_weight, 0) > 0.15
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS mismatch_pct,
    '10%' AS target
FROM shipments s
JOIN actual_weights aw ON s.shipment_id = aw.shipment_id
WHERE s."totalWeight" IS NOT NULL;

-- 10C: Impossible delivery dates (actual_delivery < shipment_date) — target 5%
SELECT
    'Impossible Delivery Dates' AS issue,
    COUNT(*) AS total_with_actual,
    COUNT(*) FILTER (
        WHERE "actual_delivery"::DATE < shipment_date
    ) AS impossible_count,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE "actual_delivery"::DATE < shipment_date
        ) / COUNT(*),
        2
    ) AS impossible_pct,
    '5%' AS target
FROM shipments
WHERE "actual_delivery" IS NOT NULL;

-- 10D: Shipments missing delivery events entirely — target 15%
SELECT
    'Shipments Missing Delivery Events' AS issue,
    (SELECT COUNT(*) FROM shipments) AS total_shipments,
    COUNT(*) AS missing_count,
    ROUND(
        100.0 * COUNT(*) / (SELECT COUNT(*) FROM shipments),
        2
    ) AS missing_pct,
    '15%' AS target
FROM shipments s
WHERE NOT EXISTS (
    SELECT 1 FROM delivery_events de WHERE de.shipment_id = s.shipment_id
);

-- 10E: Orders with past requested_delivery_date — target 5%
SELECT
    'Past Requested Delivery Date' AS issue,
    COUNT(*) AS total_orders_checked,
    COUNT(*) FILTER (
        WHERE "requested_delivery_date"::DATE < order_date
    ) AS past_date_count,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE "requested_delivery_date"::DATE < order_date
        ) / COUNT(*),
        2
    ) AS past_date_pct,
    '5%' AS target
FROM orders
WHERE "requested_delivery_date" IS NOT NULL;

-- 10F: Negative inventory — target ~3%
SELECT
    'Negative Inventory' AS issue,
    COUNT(*) AS total_records,
    COUNT(*) FILTER (WHERE quantity_on_hand < 0) AS negative_count,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE quantity_on_hand < 0) / COUNT(*),
        2
    ) AS negative_pct,
    '~3%' AS target
FROM inventory
WHERE quantity_on_hand IS NOT NULL;

-- 10G: Duplicate SKUs — target 2%
SELECT
    'Duplicate SKUs' AS issue,
    COUNT(*) AS total_products,
    COUNT(*) - COUNT(DISTINCT sku) AS duplicate_sku_count,
    ROUND(
        100.0 * (COUNT(*) - COUNT(DISTINCT sku)) / COUNT(*),
        2
    ) AS duplicate_pct,
    '2%' AS target
FROM products;

-- 10H: Null values in critical fields — target 3-5%
SELECT
    'Nulls in orders.orderPriority' AS field,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE "orderPriority" IS NULL) AS null_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "orderPriority" IS NULL) / COUNT(*), 2) AS null_pct,
    '3-5%' AS target
FROM orders
UNION ALL
SELECT
    'Nulls in orders.fulfillment_warehouse_id',
    COUNT(*),
    COUNT(*) FILTER (WHERE fulfillment_warehouse_id IS NULL),
    ROUND(100.0 * COUNT(*) FILTER (WHERE fulfillment_warehouse_id IS NULL) / COUNT(*), 2),
    '3-5%'
FROM orders
UNION ALL
SELECT
    'Nulls in shipments.carrier',
    COUNT(*),
    COUNT(*) FILTER (WHERE carrier IS NULL),
    ROUND(100.0 * COUNT(*) FILTER (WHERE carrier IS NULL) / COUNT(*), 2),
    '3-5%'
FROM shipments
UNION ALL
SELECT
    'Nulls in shipments.shipment_cost',
    COUNT(*),
    COUNT(*) FILTER (WHERE shipment_cost IS NULL),
    ROUND(100.0 * COUNT(*) FILTER (WHERE shipment_cost IS NULL) / COUNT(*), 2),
    '3-5%'
FROM shipments
UNION ALL
SELECT
    'Nulls in inventory.quantity_on_hand',
    COUNT(*),
    COUNT(*) FILTER (WHERE quantity_on_hand IS NULL),
    ROUND(100.0 * COUNT(*) FILTER (WHERE quantity_on_hand IS NULL) / COUNT(*), 2),
    '3-5%'
FROM inventory
UNION ALL
SELECT
    'Nulls in inventory.reorder_level',
    COUNT(*),
    COUNT(*) FILTER (WHERE reorder_level IS NULL),
    ROUND(100.0 * COUNT(*) FILTER (WHERE reorder_level IS NULL) / COUNT(*), 2),
    '3-5%'
FROM inventory;

-- 10I: Combined data quality summary
SELECT issue, actual_pct, target,
    CASE
        WHEN ABS(actual_pct - target_num) <= 3 THEN 'PASS'
        ELSE 'REVIEW'
    END AS status
FROM (
    SELECT 'Orphaned Product IDs' AS issue,
        ROUND(100.0 * COUNT(*) FILTER (
            WHERE product_id NOT IN (SELECT product_id FROM products)
        ) / COUNT(*), 2) AS actual_pct,
        '8-10%' AS target, 9.0 AS target_num
    FROM shipment_items
    UNION ALL
    SELECT 'Impossible Delivery Dates',
        ROUND(100.0 * COUNT(*) FILTER (
            WHERE "actual_delivery"::DATE < shipment_date
        ) / COUNT(*), 2),
        '5%', 5.0
    FROM shipments WHERE "actual_delivery" IS NOT NULL
    UNION ALL
    SELECT 'Shipments Missing Events',
        ROUND(100.0 * (
            SELECT COUNT(*) FROM shipments s2
            WHERE NOT EXISTS (
                SELECT 1 FROM delivery_events de WHERE de.shipment_id = s2.shipment_id
            )
        ) / COUNT(*)::NUMERIC, 2),
        '15%', 15.0
    FROM shipments
    UNION ALL
    SELECT 'Past Requested Delivery Date',
        ROUND(100.0 * COUNT(*) FILTER (
            WHERE "requested_delivery_date"::DATE < order_date
        ) / COUNT(*), 2),
        '5%', 5.0
    FROM orders WHERE "requested_delivery_date" IS NOT NULL
    UNION ALL
    SELECT 'Negative Inventory',
        ROUND(100.0 * COUNT(*) FILTER (WHERE quantity_on_hand < 0) / COUNT(*), 2),
        '~3%', 3.0
    FROM inventory WHERE quantity_on_hand IS NOT NULL
    UNION ALL
    SELECT 'Duplicate SKUs',
        ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT sku)) / COUNT(*), 2),
        '2%', 2.0
    FROM products
) quality
ORDER BY issue;
