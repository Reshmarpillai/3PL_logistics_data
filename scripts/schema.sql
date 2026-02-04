-- ============================================================
-- Supply Chain & Logistics Database Schema
-- PostgreSQL 16
--
-- INTENTIONAL DESIGN CHOICES:
--   - Mixed snake_case and camelCase column names
--   - Some DATE/TIMESTAMP columns stored as VARCHAR
--   - 3 redundant cost columns in shipments
--   - NO foreign key constraints (allows orphaned records)
--   - Only PRIMARY KEYs and performance indexes
-- ============================================================

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS delivery_events;
DROP TABLE IF EXISTS shipment_items;
DROP TABLE IF EXISTS order_shipments;
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS warehouses;

-- ============================================================
-- 1. warehouses
--    Naming: snake_case + capacitySqft (camelCase)
--    Type issue: operational_since stored as VARCHAR
-- ============================================================
CREATE TABLE warehouses (
    warehouse_id    INTEGER PRIMARY KEY,
    warehouse_name  VARCHAR(100) NOT NULL,
    warehouse_type  VARCHAR(30)  NOT NULL,     -- hub, regional, micro-fulfillment
    country         VARCHAR(50)  NOT NULL,
    city            VARCHAR(80)  NOT NULL,
    region          VARCHAR(30)  NOT NULL,     -- Americas, Europe, Asia-Pacific
    "capacitySqft"  INTEGER      NOT NULL,     -- camelCase (intentional)
    "operational_since" VARCHAR(20),           -- VARCHAR, not DATE (intentional)
    is_active       BOOLEAN DEFAULT TRUE
);

-- ============================================================
-- 2. products
--    Naming: mixed — snake_case + camelCase
--    product_id, sku, product_name (snake)
--    productCategory, weightKg, unitCost, supplierID (camel)
-- ============================================================
CREATE TABLE products (
    product_id          INTEGER PRIMARY KEY,
    sku                 VARCHAR(20)  NOT NULL,  -- CAT-XXXXX format; 2% duplicates
    product_name        VARCHAR(150) NOT NULL,
    "productCategory"   VARCHAR(50)  NOT NULL,  -- camelCase (intentional)
    "weightKg"          NUMERIC(8,2),           -- camelCase (intentional)
    "unitCost"          NUMERIC(10,2),          -- camelCase (intentional)
    fragile_flag        BOOLEAN,
    "supplierID"        INTEGER,                -- camelCase (intentional)
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 3. customers
--    Naming: customer_name (snake) + customerType (camel)
--    Type issue: onboarded_date is DATE, create_date is VARCHAR
-- ============================================================
CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    customer_name   VARCHAR(120) NOT NULL,
    "customerType"  VARCHAR(10)  NOT NULL,     -- B2B, B2C (camelCase)
    email           VARCHAR(150),
    country         VARCHAR(50)  NOT NULL,
    city            VARCHAR(80),
    account_status  VARCHAR(20)  DEFAULT 'active',  -- active, suspended, closed
    onboarded_date  DATE,                      -- proper DATE type
    "create_date"   VARCHAR(30)                -- VARCHAR, not TIMESTAMP (intentional mismatch)
);

-- ============================================================
-- 4. inventory
--    Naming: snake_case + stockStatus (camelCase)
--    Type issue: last_updated stored as VARCHAR
-- ============================================================
CREATE TABLE inventory (
    inventory_id        INTEGER PRIMARY KEY,
    warehouse_id        INTEGER NOT NULL,       -- no FK constraint
    product_id          INTEGER NOT NULL,       -- no FK constraint
    quantity_on_hand    INTEGER,                -- 3-5% nulls, ~3% negative values
    reorder_level       INTEGER,                -- 3-5% nulls
    "stockStatus"       VARCHAR(20),            -- camelCase: in_stock, low_stock, out_of_stock
    "last_updated"      VARCHAR(30)             -- VARCHAR, not TIMESTAMP (intentional)
);

-- ============================================================
-- 5. orders
--    Naming: order_id, order_date (snake) + orderPriority (camel)
--    Type issue: requested_delivery_date stored as VARCHAR
-- ============================================================
CREATE TABLE orders (
    order_id                    INTEGER PRIMARY KEY,
    customer_id                 INTEGER NOT NULL,       -- no FK constraint
    order_date                  DATE    NOT NULL,
    "orderPriority"             VARCHAR(20),            -- camelCase: standard, express, critical
    "requested_delivery_date"   VARCHAR(20),            -- VARCHAR, not DATE (intentional); 5% before order_date
    fulfillment_warehouse_id    INTEGER,                -- no FK; 3-5% nulls
    order_status                VARCHAR(20) DEFAULT 'processing',  -- delivered, shipped, processing, cancelled
    total_amount                NUMERIC(12,2)
);

-- ============================================================
-- 6. shipments
--    Naming: 3 redundant cost cols, totalWeight (camel)
--    Type issue: actual_delivery stored as VARCHAR
-- ============================================================
CREATE TABLE shipments (
    shipment_id         INTEGER PRIMARY KEY,
    origin_warehouse_id INTEGER NOT NULL,       -- no FK constraint
    dest_warehouse_id   INTEGER,                -- no FK constraint
    shipment_date       DATE    NOT NULL,
    carrier             VARCHAR(30),            -- 3-5% nulls
    estimated_delivery  DATE,
    "actual_delivery"   VARCHAR(20),            -- VARCHAR, not DATE (intentional); 5% impossible dates
    shipment_status     VARCHAR(20) DEFAULT 'in_transit',  -- delivered, in_transit, failed, returned
    "totalWeight"       NUMERIC(10,2),          -- camelCase; NULL until Pass 2 backfill
    shipment_cost       NUMERIC(10,2),          -- redundant cost col 1; 3-5% nulls
    total_cost          NUMERIC(10,2),          -- redundant cost col 2 (= shipment_cost * ~1.0)
    cost_amount         NUMERIC(10,2)           -- redundant cost col 3 (= shipment_cost * ~1.0)
);

-- ============================================================
-- 7. order_shipments
--    Naming: clean snake_case (intentionally the only clean table)
-- ============================================================
CREATE TABLE order_shipments (
    order_shipment_id   INTEGER PRIMARY KEY,
    order_id            INTEGER NOT NULL,       -- no FK constraint
    shipment_id         INTEGER NOT NULL,       -- no FK constraint
    fulfillment_date    DATE
);

-- ============================================================
-- 8. shipment_items
--    Naming: conditionOnArrival (camelCase)
-- ============================================================
CREATE TABLE shipment_items (
    shipment_item_id        INTEGER PRIMARY KEY,
    shipment_id             INTEGER NOT NULL,       -- no FK constraint
    product_id              INTEGER NOT NULL,       -- no FK; 8-10% orphaned (non-existent product_id)
    quantity                INTEGER NOT NULL DEFAULT 1,
    "conditionOnArrival"    VARCHAR(20)             -- camelCase: good, damaged, missing
);

-- ============================================================
-- 9. delivery_events
--    Naming: eventType (camelCase), event_timestamp as VARCHAR
-- ============================================================
CREATE TABLE delivery_events (
    event_id            INTEGER PRIMARY KEY,
    shipment_id         INTEGER NOT NULL,       -- no FK constraint
    "eventType"         VARCHAR(30) NOT NULL,   -- camelCase: picked, in_transit, out_for_delivery, delivered, failed
    "event_timestamp"   VARCHAR(30),            -- VARCHAR, not TIMESTAMP (intentional)
    location            VARCHAR(100),
    delay_reason        VARCHAR(50)             -- weather, customs, capacity, address_issue, none
);

-- ============================================================
-- Performance Indexes (on ID/lookup columns only)
-- ============================================================

-- warehouses
CREATE INDEX idx_warehouses_type ON warehouses (warehouse_type);
CREATE INDEX idx_warehouses_country ON warehouses (country);

-- products
CREATE INDEX idx_products_sku ON products (sku);
CREATE INDEX idx_products_category ON products ("productCategory");

-- customers
CREATE INDEX idx_customers_type ON customers ("customerType");
CREATE INDEX idx_customers_country ON customers (country);
CREATE INDEX idx_customers_status ON customers (account_status);

-- inventory
CREATE INDEX idx_inventory_warehouse ON inventory (warehouse_id);
CREATE INDEX idx_inventory_product ON inventory (product_id);
CREATE INDEX idx_inventory_status ON inventory ("stockStatus");

-- orders
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date ON orders (order_date);
CREATE INDEX idx_orders_warehouse ON orders (fulfillment_warehouse_id);
CREATE INDEX idx_orders_status ON orders (order_status);

-- shipments
CREATE INDEX idx_shipments_origin ON shipments (origin_warehouse_id);
CREATE INDEX idx_shipments_dest ON shipments (dest_warehouse_id);
CREATE INDEX idx_shipments_date ON shipments (shipment_date);
CREATE INDEX idx_shipments_carrier ON shipments (carrier);
CREATE INDEX idx_shipments_status ON shipments (shipment_status);

-- order_shipments
CREATE INDEX idx_order_shipments_order ON order_shipments (order_id);
CREATE INDEX idx_order_shipments_shipment ON order_shipments (shipment_id);

-- shipment_items
CREATE INDEX idx_shipment_items_shipment ON shipment_items (shipment_id);
CREATE INDEX idx_shipment_items_product ON shipment_items (product_id);

-- delivery_events
CREATE INDEX idx_delivery_events_shipment ON delivery_events (shipment_id);
CREATE INDEX idx_delivery_events_type ON delivery_events ("eventType");
