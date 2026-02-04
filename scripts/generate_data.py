"""Supply Chain & Logistics Data Generation Script

Generates ~5.4M rows across 9 tables with intentional data quality issues.
Uses NumPy for all numeric/categorical generation, Faker for text fields only.
Chunked writing for large tables (>100K rows).

Usage: python scripts/generate_data.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_SIZE = 100_000

ROW_COUNTS = {
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

# Initialize generators
rng = np.random.default_rng(SEED)
fake = Faker()
Faker.seed(SEED)

# ============================================================
# Geographic Data (10 countries, 3 regions)
# ============================================================

REGIONS_DATA = {
    "Americas": {
        "United States": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Dallas", "Miami", "Seattle"],
        "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary"],
        "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador"],
        "Mexico": ["Mexico City", "Guadalajara", "Monterrey", "Puebla"],
    },
    "Europe": {
        "United Kingdom": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"],
        "Germany": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Cologne"],
        "France": ["Paris", "Lyon", "Marseille", "Toulouse"],
    },
    "Asia-Pacific": {
        "Japan": ["Tokyo", "Osaka", "Yokohama", "Nagoya"],
        "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth"],
        "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
    },
}

# Build lookup structures
COUNTRIES: list[str] = []
COUNTRY_CITIES: dict[str, list[str]] = {}
COUNTRY_REGION: dict[str, str] = {}

for _region_name, _countries in REGIONS_DATA.items():
    for _country, _cities in _countries.items():
        COUNTRIES.append(_country)
        COUNTRY_CITIES[_country] = _cities
        COUNTRY_REGION[_country] = _region_name

ALL_CITIES = [city for cities in COUNTRY_CITIES.values() for city in cities]

# ============================================================
# Product Category Correlation Matrix
# ============================================================

CATEGORIES = {
    "electronics": {
        "weight_min": 1.0, "weight_max": 10.0,
        "cost_min": 50.0, "cost_max": 500.0,
        "fragile_pct": 0.80,
        "log_s_weight": 0.8, "log_scale_weight": 3.0,
        "log_s_cost": 0.8, "log_scale_cost": 150.0,
    },
    "furniture": {
        "weight_min": 10.0, "weight_max": 50.0,
        "cost_min": 100.0, "cost_max": 1000.0,
        "fragile_pct": 0.40,
        "log_s_weight": 0.6, "log_scale_weight": 20.0,
        "log_s_cost": 0.7, "log_scale_cost": 300.0,
    },
    "clothing": {
        "weight_min": 0.1, "weight_max": 2.0,
        "cost_min": 10.0, "cost_max": 100.0,
        "fragile_pct": 0.05,
        "log_s_weight": 0.5, "log_scale_weight": 0.5,
        "log_s_cost": 0.6, "log_scale_cost": 30.0,
    },
    "food": {
        "weight_min": 0.2, "weight_max": 5.0,
        "cost_min": 1.0, "cost_max": 20.0,
        "fragile_pct": 0.10,
        "log_s_weight": 0.6, "log_scale_weight": 1.0,
        "log_s_cost": 0.5, "log_scale_cost": 5.0,
    },
    "automotive": {
        "weight_min": 2.0, "weight_max": 30.0,
        "cost_min": 20.0, "cost_max": 500.0,
        "fragile_pct": 0.30,
        "log_s_weight": 0.7, "log_scale_weight": 8.0,
        "log_s_cost": 0.8, "log_scale_cost": 80.0,
    },
    "health": {
        "weight_min": 0.1, "weight_max": 3.0,
        "cost_min": 5.0, "cost_max": 80.0,
        "fragile_pct": 0.30,
        "log_s_weight": 0.6, "log_scale_weight": 0.5,
        "log_s_cost": 0.6, "log_scale_cost": 15.0,
    },
    "toys": {
        "weight_min": 0.2, "weight_max": 5.0,
        "cost_min": 5.0, "cost_max": 50.0,
        "fragile_pct": 0.50,
        "log_s_weight": 0.5, "log_scale_weight": 1.0,
        "log_s_cost": 0.5, "log_scale_cost": 15.0,
    },
    "office": {
        "weight_min": 0.3, "weight_max": 10.0,
        "cost_min": 5.0, "cost_max": 200.0,
        "fragile_pct": 0.40,
        "log_s_weight": 0.6, "log_scale_weight": 2.0,
        "log_s_cost": 0.7, "log_scale_cost": 30.0,
    },
}

CATEGORY_NAMES = list(CATEGORIES.keys())

# ============================================================
# Carrier Data
# ============================================================

CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "LocalExpress", "RegionalPost"]
CARRIER_COST_MULT = {
    "FedEx": 1.20, "UPS": 1.20, "DHL": 1.15,
    "USPS": 1.0, "LocalExpress": 1.0, "RegionalPost": 1.0,
}
CARRIER_WEIGHTS = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.05])

# ============================================================
# Product Name Templates
# ============================================================

PRODUCT_ADJECTIVES = {
    "electronics": ["Wireless", "Smart", "Digital", "Pro", "Ultra", "Mini", "Portable", "HD", "Premium", "Compact"],
    "furniture": ["Modern", "Classic", "Ergonomic", "Adjustable", "Foldable", "Premium", "Compact", "Executive", "Rustic", "Industrial"],
    "clothing": ["Cotton", "Silk", "Casual", "Premium", "Classic", "Slim", "Oversized", "Athletic", "Organic", "Vintage"],
    "food": ["Organic", "Premium", "Natural", "Fresh", "Artisan", "Gourmet", "Bulk", "Family", "Gluten-Free", "Vegan"],
    "automotive": ["Heavy-Duty", "Universal", "Performance", "OEM", "Premium", "Standard", "Chrome", "Sport", "Racing", "All-Weather"],
    "health": ["Natural", "Clinical", "Premium", "Gentle", "Advanced", "Daily", "Ultra", "Pure", "Herbal", "Extra-Strength"],
    "toys": ["Interactive", "Creative", "Educational", "Classic", "Deluxe", "Mini", "Giant", "Super", "Magnetic", "Wooden"],
    "office": ["Professional", "Deluxe", "Standard", "Compact", "Heavy-Duty", "Ergonomic", "Premium", "Budget", "Recycled", "Executive"],
}

PRODUCT_NOUNS = {
    "electronics": ["Headphones", "Speaker", "Camera", "Monitor", "Keyboard", "Mouse", "Charger", "Cable", "Tablet Stand", "SSD Drive"],
    "furniture": ["Desk", "Chair", "Bookshelf", "Cabinet", "Table", "Sofa", "Bed Frame", "TV Stand", "Shoe Rack", "Drawer Unit"],
    "clothing": ["T-Shirt", "Jacket", "Pants", "Sweater", "Dress", "Shorts", "Hoodie", "Coat", "Vest", "Button Shirt"],
    "food": ["Snack Pack", "Coffee Beans", "Tea Set", "Spice Mix", "Protein Bar", "Cereal Box", "Hot Sauce", "Olive Oil", "Mixed Nuts", "Dried Fruit"],
    "automotive": ["Oil Filter", "Brake Pad", "Spark Plug", "Battery", "Drive Belt", "Gasket Set", "O2 Sensor", "Water Pump", "Valve Cover", "Radiator Hose"],
    "health": ["Multivitamin", "Fish Oil", "Face Cream", "Hair Serum", "Bandage Kit", "Thermometer", "Face Mask", "Hand Sanitizer", "Body Lotion", "Eye Drops"],
    "toys": ["Jigsaw Puzzle", "Block Set", "Action Figure", "Board Game", "Plush Doll", "Robot Kit", "Car Set", "Art Kit", "Bouncy Ball", "Stuffed Animal"],
    "office": ["Pen Set", "Notebook", "Stapler", "Binder Set", "Copy Paper", "Marker Set", "Packing Tape", "Scissors", "Clipboard", "File Folder"],
}

# ============================================================
# Helper Functions
# ============================================================


def ensure_dirs() -> None:
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, filepath: Path, mode: str = "w", header: bool = True) -> None:
    """Write DataFrame to CSV with LF line endings."""
    with open(filepath, mode, encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, header=header, lineterminator="\n")


def introduce_nulls_inplace(df: pd.DataFrame, column: str, pct: float) -> None:
    """Set ~pct fraction of values to NaN/None in a DataFrame column."""
    n_nulls = int(len(df) * pct)
    if n_nulls == 0:
        return
    null_idx = rng.choice(len(df), size=n_nulls, replace=False)
    df.loc[df.index[null_idx], column] = None


def temporal_weight(dt: date) -> float:
    """Calculate temporal weight for a given date (order/shipment volume)."""
    weekday_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.4]
    weekday_mult = weekday_weights[dt.weekday()]

    month = dt.month
    if month >= 10:      # Q4 spike
        month_mult = 1.5 + (month - 10) * 0.25
    elif month <= 3:     # Q1 dip
        month_mult = 0.8
    else:                # Q2-Q3 baseline
        month_mult = 1.0

    year_mult = 1.15 ** (dt.year - 2021)  # 15% YoY growth
    return weekday_mult * month_mult * year_mult


def build_date_pool(start: date, end: date) -> tuple[np.ndarray, np.ndarray]:
    """Build arrays of all dates and their temporal weights in [start, end]."""
    n_days = (end - start).days + 1
    py_dates = [start + timedelta(days=i) for i in range(n_days)]
    dates_arr = np.array(py_dates, dtype="datetime64[D]")
    weights = np.array([temporal_weight(d) for d in py_dates], dtype=np.float64)
    weights /= weights.sum()
    return dates_arr, weights


def sample_weighted_dates(n: int, date_pool: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sample n dates from the weighted date pool."""
    indices = rng.choice(len(date_pool), size=n, p=weights)
    return date_pool[indices]


def zipf_customer_orders(customer_ids: np.ndarray, target_orders: int, zipf_a: float) -> np.ndarray:
    """Assign orders to customers using Zipf distribution. Returns array of customer_ids."""
    n_customers = len(customer_ids)
    raw = rng.zipf(zipf_a, size=n_customers).astype(np.float64)
    raw = raw / raw.sum() * target_orders
    counts = np.maximum(np.round(raw).astype(np.int64), 1)

    # Adjust to hit exact target
    diff = target_orders - counts.sum()
    if diff > 0:
        add_idx = rng.choice(n_customers, size=int(diff), replace=True)
        for idx in add_idx:
            counts[idx] += 1
    elif diff < 0:
        candidates = np.where(counts > 1)[0]
        remove_idx = rng.choice(candidates, size=min(int(-diff), len(candidates)), replace=True)
        for idx in remove_idx:
            if counts[idx] > 1:
                counts[idx] -= 1

    result = np.repeat(customer_ids, counts)
    if len(result) > target_orders:
        result = result[:target_orders]
    elif len(result) < target_orders:
        extra = rng.choice(customer_ids, size=target_orders - len(result))
        result = np.concatenate([result, extra])

    rng.shuffle(result)
    return result


def get_warehouse_weights(wh_types: np.ndarray, wh_capacities: np.ndarray) -> np.ndarray:
    """Throughput weights: hub=60%, regional=30%, micro=10%, weighted by capacity within type."""
    type_budget = {"hub": 0.60, "regional": 0.30, "micro-fulfillment": 0.10}
    weights = np.zeros(len(wh_types), dtype=np.float64)
    for wh_type, budget in type_budget.items():
        mask = wh_types == wh_type
        if not mask.any():
            continue
        caps = wh_capacities[mask].astype(np.float64)
        weights[mask] = caps / caps.sum() * budget
    weights /= weights.sum()
    return weights


# ============================================================
# Generator: warehouses (50 rows)
# ============================================================


def generate_warehouses() -> dict:
    """Generate warehouses.csv and return metadata for downstream tables."""
    n = ROW_COUNTS["warehouses"]
    filepath = DATA_DIR / "warehouses.csv"
    print(f"\n[1/9] Generating warehouses ({n} rows)...")

    # Types: 10 hubs, 20 regional, 20 micro-fulfillment
    types = np.array(["hub"] * 10 + ["regional"] * 20 + ["micro-fulfillment"] * 20)

    # Distribute across 10 countries (5 per country), shuffle type assignment
    country_idx = np.tile(np.arange(len(COUNTRIES)), 5)
    rng.shuffle(country_idx)
    countries = np.array([COUNTRIES[i] for i in country_idx])
    regions = np.array([COUNTRY_REGION[c] for c in countries])
    cities = np.array([rng.choice(COUNTRY_CITIES[c]) for c in countries])

    # Capacity per type
    capacity = np.zeros(n, dtype=np.int32)
    hub_mask = types == "hub"
    reg_mask = types == "regional"
    mic_mask = types == "micro-fulfillment"
    capacity[hub_mask] = rng.integers(100_000, 500_001, size=hub_mask.sum())
    capacity[reg_mask] = rng.integers(20_000, 100_001, size=reg_mask.sum())
    capacity[mic_mask] = rng.integers(2_000, 20_001, size=mic_mask.sum())

    # operational_since as VARCHAR, spread by type
    op_years = np.zeros(n, dtype=np.int32)
    op_years[hub_mask] = rng.integers(2015, 2020, size=hub_mask.sum())
    op_years[reg_mask] = rng.integers(2016, 2022, size=reg_mask.sum())
    op_years[mic_mask] = rng.integers(2020, 2025, size=mic_mask.sum())
    op_months = rng.integers(1, 13, size=n)
    op_days = rng.integers(1, 29, size=n)
    op_since = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(op_years, op_months, op_days)]

    # Names
    names = [f"{cities[i]} {types[i].replace('-', ' ').title()} WH-{i+1:03d}" for i in range(n)]

    # is_active: mostly True, 3 inactive
    is_active = np.ones(n, dtype=bool)
    is_active[rng.choice(n, size=3, replace=False)] = False

    # Urban flag (derived for downstream use, not stored)
    is_urban = np.ones(n, dtype=bool)
    reg_indices = np.where(reg_mask)[0]
    if len(reg_indices) > 0:
        rural_count = len(reg_indices) // 2
        is_urban[rng.choice(reg_indices, size=rural_count, replace=False)] = False

    df = pd.DataFrame({
        "warehouse_id": np.arange(1, n + 1),
        "warehouse_name": names,
        "warehouse_type": types,
        "country": countries,
        "city": cities,
        "region": regions,
        "capacitySqft": capacity,
        "operational_since": op_since,
        "is_active": is_active,
    })
    write_csv(df, filepath)
    print(f"  Done: {filepath.name} ({n} rows)")

    return {
        "ids": np.arange(1, n + 1),
        "types": types,
        "countries": countries,
        "cities": cities,
        "capacities": capacity,
        "is_urban": is_urban,
    }


# ============================================================
# Generator: products (10K rows)
# ============================================================


def generate_products() -> dict:
    """Generate products.csv and return metadata for downstream tables."""
    n = ROW_COUNTS["products"]
    filepath = DATA_DIR / "products.csv"
    print(f"\n[2/9] Generating products ({n:,} rows)...")

    product_ids = np.arange(1, n + 1)

    # Assign categories (roughly equal distribution)
    categories = np.array(rng.choice(CATEGORY_NAMES, size=n))

    # SKU generation: CAT-XXXXX format, with 2% duplicates (~200)
    sku_prefixes = {
        "electronics": "ELC", "furniture": "FRN", "clothing": "CLT",
        "food": "FOD", "automotive": "AUT", "health": "HLT",
        "toys": "TOY", "office": "OFC",
    }
    skus = np.array([f"{sku_prefixes[cat]}-{rng.integers(10000, 99999)}" for cat in categories])

    # Introduce ~2% duplicate SKUs (~200 products share a SKU with another)
    # Natural collisions from 5-digit range add ~1%, so only add ~0.5% deliberate
    n_dupes = int(n * 0.005)
    dupe_sources = rng.choice(n, size=n_dupes, replace=False)
    dupe_targets = rng.choice(np.setdiff1d(np.arange(n), dupe_sources), size=n_dupes, replace=False)
    skus[dupe_targets] = skus[dupe_sources]

    # Product names
    names = []
    for cat in categories:
        adj = rng.choice(PRODUCT_ADJECTIVES[cat])
        noun = rng.choice(PRODUCT_NOUNS[cat])
        names.append(f"{adj} {noun}")
    names = np.array(names)

    # Weight and cost per category (lognormal, clipped)
    weights = np.zeros(n, dtype=np.float32)
    costs = np.zeros(n, dtype=np.float32)
    fragile = np.zeros(n, dtype=bool)

    for cat_name, cat_info in CATEGORIES.items():
        mask = categories == cat_name
        count = mask.sum()
        if count == 0:
            continue

        w = rng.lognormal(mean=np.log(cat_info["log_scale_weight"]),
                          sigma=cat_info["log_s_weight"], size=count).astype(np.float32)
        weights[mask] = np.clip(w, cat_info["weight_min"], cat_info["weight_max"])

        c = rng.lognormal(mean=np.log(cat_info["log_scale_cost"]),
                          sigma=cat_info["log_s_cost"], size=count).astype(np.float32)
        costs[mask] = np.clip(c, cat_info["cost_min"], cat_info["cost_max"])

        fragile[mask] = rng.random(count) < cat_info["fragile_pct"]

    # supplierID: 1-500
    supplier_ids = rng.integers(1, 501, size=n)

    # created_at: 80% in 2019-2022, 20% in 2022-2024
    created_dates = np.empty(n, dtype="datetime64[s]")
    early_mask = rng.random(n) < 0.80
    n_early = early_mask.sum()
    n_late = n - n_early

    early_start = np.datetime64("2019-01-01", "s")
    early_range = int((np.datetime64("2022-06-30", "s") - early_start) / np.timedelta64(1, "s"))
    created_dates[early_mask] = early_start + (rng.integers(0, early_range, size=n_early) * np.timedelta64(1, "s"))

    late_start = np.datetime64("2022-07-01", "s")
    late_range = int((np.datetime64("2024-12-31", "s") - late_start) / np.timedelta64(1, "s"))
    created_dates[~early_mask] = late_start + (rng.integers(0, late_range, size=n_late) * np.timedelta64(1, "s"))

    created_str = pd.Series(created_dates).dt.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame({
        "product_id": product_ids,
        "sku": skus,
        "product_name": names,
        "productCategory": categories,
        "weightKg": np.round(weights, 2),
        "unitCost": np.round(costs, 2),
        "fragile_flag": fragile,
        "supplierID": supplier_ids,
        "created_at": created_str,
    })
    write_csv(df, filepath)
    print(f"  Done: {filepath.name} ({n:,} rows)")

    return {
        "ids": product_ids,
        "categories": categories,
        "weights": weights.astype(np.float64),
    }


# ============================================================
# Generator: customers (50K rows)
# ============================================================


def generate_customers() -> dict:
    """Generate customers.csv and return metadata for downstream tables."""
    n = ROW_COUNTS["customers"]
    filepath = DATA_DIR / "customers.csv"
    print(f"\n[3/9] Generating customers ({n:,} rows)...")

    customer_ids = np.arange(1, n + 1)

    # customerType: 70% B2C, 30% B2B
    types = np.where(rng.random(n) < 0.30, "B2B", "B2C")

    # Countries matching warehouse regions
    countries = np.array(rng.choice(COUNTRIES, size=n))
    cities = np.array([rng.choice(COUNTRY_CITIES[c]) for c in countries])

    # Names and emails via Faker
    print("  Generating names and emails (Faker)...")
    names = np.array([fake.name() for _ in tqdm(range(n), desc="  Names", leave=False)])
    emails = np.array([fake.email() for _ in tqdm(range(n), desc="  Emails", leave=False)])

    # account_status: active 85%, suspended 10%, closed 5%
    status_choices = ["active", "suspended", "closed"]
    status_probs = [0.85, 0.10, 0.05]
    statuses = rng.choice(status_choices, size=n, p=status_probs)

    # onboarded_date: 2018-2024, gradual growth curve
    onboard_pool, onboard_weights = build_date_pool(date(2018, 1, 1), date(2024, 12, 31))
    onboard_dates = sample_weighted_dates(n, onboard_pool, onboard_weights)

    # create_date as VARCHAR (intentional type mismatch with onboarded_date)
    # create_date is close to onboarded_date but stored as string
    create_offsets = rng.integers(-5, 6, size=n) * np.timedelta64(1, "D")
    create_timestamps = onboard_dates.astype("datetime64[s]") + create_offsets
    # Add random time component
    create_timestamps = create_timestamps + (rng.integers(0, 86400, size=n) * np.timedelta64(1, "s"))
    create_date_str = pd.Series(create_timestamps).dt.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_name": names,
        "customerType": types,
        "email": emails,
        "country": countries,
        "city": cities,
        "account_status": statuses,
        "onboarded_date": pd.Series(onboard_dates).dt.strftime("%Y-%m-%d"),
        "create_date": create_date_str,
    })
    write_csv(df, filepath)
    print(f"  Done: {filepath.name} ({n:,} rows)")

    return {
        "ids": customer_ids,
        "types": types,
    }


# ============================================================
# Generator: inventory (25K rows)
# ============================================================


def generate_inventory(wh_data: dict, prod_data: dict) -> None:
    """Generate inventory.csv (warehouse x product subset)."""
    n = ROW_COUNTS["inventory"]
    filepath = DATA_DIR / "inventory.csv"
    print(f"\n[4/9] Generating inventory ({n:,} rows)...")

    inventory_ids = np.arange(1, n + 1)

    # Assign warehouse-product pairs
    warehouse_ids = rng.choice(wh_data["ids"], size=n)
    product_ids = rng.choice(prod_data["ids"], size=n)

    # quantity_on_hand: normal distribution, mean=200, std=80 (tighter spread)
    qty = rng.normal(loc=200, scale=80, size=n).astype(np.int32)
    # Target ~3% negative — force exactly 3%
    neg_idx = np.where(qty < 0)[0]
    pos_idx = np.where(qty > 0)[0]
    target_neg = int(n * 0.03)
    if len(neg_idx) > target_neg:
        # Too many negatives — convert excess back to positive
        fix_idx = rng.choice(neg_idx, size=len(neg_idx) - target_neg, replace=False)
        qty[fix_idx] = rng.integers(1, 100, size=len(fix_idx))
    elif len(neg_idx) < target_neg:
        # Too few negatives — convert some positives
        make_neg = rng.choice(pos_idx, size=target_neg - len(neg_idx), replace=False)
        qty[make_neg] = -rng.integers(1, 50, size=len(make_neg))

    # reorder_level: based on category demand (50-300 range)
    product_cats = prod_data["categories"][product_ids - 1]
    cat_reorder_base = {
        "electronics": 80, "furniture": 30, "clothing": 150,
        "food": 200, "automotive": 50, "health": 120,
        "toys": 100, "office": 180,
    }
    reorder = np.array([cat_reorder_base.get(c, 100) for c in product_cats], dtype=np.int32)
    reorder = reorder + rng.integers(-20, 21, size=n).astype(np.int32)

    # stockStatus derived from quantity vs reorder
    stock_status = np.where(
        qty <= 0, "out_of_stock",
        np.where(qty < reorder, "low_stock", "in_stock")
    )

    # last_updated as VARCHAR
    update_pool, update_weights = build_date_pool(date(2023, 1, 1), date(2024, 12, 31))
    update_dates = sample_weighted_dates(n, update_pool, update_weights)
    update_timestamps = update_dates.astype("datetime64[s]") + (
        rng.integers(0, 86400, size=n) * np.timedelta64(1, "s")
    )
    last_updated_str = pd.Series(update_timestamps).dt.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame({
        "inventory_id": inventory_ids,
        "warehouse_id": warehouse_ids,
        "product_id": product_ids,
        "quantity_on_hand": qty,
        "reorder_level": reorder,
        "stockStatus": stock_status,
        "last_updated": last_updated_str,
    })

    # Introduce 3-5% nulls in quantity_on_hand and reorder_level
    introduce_nulls_inplace(df, "quantity_on_hand", 0.04)
    introduce_nulls_inplace(df, "reorder_level", 0.04)

    write_csv(df, filepath)
    print(f"  Done: {filepath.name} ({n:,} rows)")


# ============================================================
# Generator: orders (500K rows, chunked)
# ============================================================


def generate_orders(cust_data: dict, wh_data: dict) -> dict:
    """Generate orders.csv using Zipf distribution for customer assignment."""
    n = ROW_COUNTS["orders"]
    filepath = DATA_DIR / "orders.csv"
    print(f"\n[5/9] Generating orders ({n:,} rows)...")

    order_ids = np.arange(1, n + 1)

    # --- Zipf customer assignment ---
    b2b_mask = cust_data["types"] == "B2B"
    b2b_ids = cust_data["ids"][b2b_mask]
    b2c_ids = cust_data["ids"][~b2b_mask]

    b2b_orders = int(n * 0.70)  # 350K
    b2c_orders = n - b2b_orders  # 150K

    print("  Assigning customers (Zipf distribution)...")
    b2b_assignments = zipf_customer_orders(b2b_ids, b2b_orders, 1.5)
    b2c_assignments = zipf_customer_orders(b2c_ids, b2c_orders, 1.2)

    customer_ids = np.concatenate([b2b_assignments, b2c_assignments])
    rng.shuffle(customer_ids)

    # --- Order dates (temporal pattern) ---
    print("  Generating order dates (temporal weights)...")
    date_pool, date_weights = build_date_pool(date(2021, 1, 1), date(2024, 12, 31))
    order_dates = sample_weighted_dates(n, date_pool, date_weights)

    # Sort by date for realistic ordering
    sort_idx = np.argsort(order_dates)
    order_dates = order_dates[sort_idx]
    customer_ids = customer_ids[sort_idx]

    # --- orderPriority: standard 70%, express 25%, critical 5% ---
    priorities = rng.choice(
        ["standard", "express", "critical"], size=n, p=[0.70, 0.25, 0.05]
    )

    # --- requested_delivery_date as VARCHAR ---
    # Normal: order_date + 3-14 days
    # 5% intentional error: before order_date
    req_offsets = rng.integers(3, 15, size=n)
    bad_date_mask = rng.random(n) < 0.05
    req_offsets[bad_date_mask] = -rng.integers(1, 10, size=bad_date_mask.sum())
    req_dates = order_dates + (req_offsets * np.timedelta64(1, "D"))
    req_date_str = pd.Series(req_dates).dt.strftime("%Y-%m-%d").values

    # --- fulfillment_warehouse_id (weighted toward hubs) ---
    wh_weights = get_warehouse_weights(wh_data["types"], wh_data["capacities"])
    fulfillment_wh = rng.choice(wh_data["ids"], size=n, p=wh_weights)

    # --- order_status: delivered 70%, shipped 15%, processing 10%, cancelled 5% ---
    statuses = rng.choice(
        ["delivered", "shipped", "processing", "cancelled"],
        size=n, p=[0.70, 0.15, 0.10, 0.05],
    )

    # --- total_amount (lognormal based on priority) ---
    total_amount = np.zeros(n, dtype=np.float32)
    for priority, scale in [("standard", 100), ("express", 250), ("critical", 500)]:
        mask = priorities == priority
        total_amount[mask] = rng.lognormal(
            mean=np.log(scale), sigma=0.7, size=mask.sum()
        ).astype(np.float32)
    total_amount = np.clip(total_amount, 5.0, 50000.0)

    # --- Write in chunks ---
    print("  Writing chunks...")
    for start in tqdm(range(0, n, CHUNK_SIZE), desc="  Orders", leave=False):
        end = min(start + CHUNK_SIZE, n)
        s = slice(start, end)

        chunk_df = pd.DataFrame({
            "order_id": order_ids[s],
            "customer_id": customer_ids[s],
            "order_date": pd.Series(order_dates[s]).dt.strftime("%Y-%m-%d").values,
            "orderPriority": priorities[s],
            "requested_delivery_date": req_date_str[s],
            "fulfillment_warehouse_id": fulfillment_wh[s],
            "order_status": statuses[s],
            "total_amount": np.round(total_amount[s], 2),
        })

        # Introduce nulls in first chunk's priority and warehouse
        introduce_nulls_inplace(chunk_df, "orderPriority", 0.04)
        introduce_nulls_inplace(chunk_df, "fulfillment_warehouse_id", 0.04)

        write_csv(chunk_df, filepath, mode="a" if start > 0 else "w", header=(start == 0))

    print(f"  Done: {filepath.name} ({n:,} rows)")

    return {
        "ids": order_ids,
        "dates": order_dates,
        "statuses": statuses,
    }


# ============================================================
# Generator: shipments pass 1 (800K rows, chunked)
# ============================================================


def generate_shipments(wh_data: dict) -> dict:
    """Generate shipments.csv (pass 1 — totalWeight left NULL)."""
    n = ROW_COUNTS["shipments"]
    filepath = DATA_DIR / "shipments.csv"
    print(f"\n[6/9] Generating shipments ({n:,} rows)...")

    shipment_ids = np.arange(1, n + 1)

    # --- Origin warehouse (throughput-weighted) ---
    wh_weights = get_warehouse_weights(wh_data["types"], wh_data["capacities"])
    origin_wh = rng.choice(wh_data["ids"], size=n, p=wh_weights)

    # --- Destination warehouse (any, ~20% NULL for direct-to-customer) ---
    dest_wh = rng.choice(wh_data["ids"], size=n).astype(object)
    null_dest_mask = rng.random(n) < 0.20
    dest_wh[null_dest_mask] = None

    # --- Shipment dates (temporal pattern, same as orders) ---
    print("  Generating shipment dates...")
    date_pool, date_weights = build_date_pool(date(2021, 1, 1), date(2024, 12, 31))
    ship_dates = sample_weighted_dates(n, date_pool, date_weights)
    sort_idx = np.argsort(ship_dates)
    ship_dates = ship_dates[sort_idx]
    origin_wh = origin_wh[sort_idx]
    dest_wh = dest_wh[sort_idx]

    # --- Carrier assignment ---
    carriers = rng.choice(CARRIERS, size=n, p=CARRIER_WEIGHTS)

    # --- Delivery time calculation ---
    print("  Calculating delivery estimates...")
    origin_countries = np.array([wh_data["countries"][wid - 1] for wid in origin_wh])

    # For dest: use dest warehouse country if not null, else origin country
    dest_countries = np.empty(n, dtype=object)
    for i in range(n):
        if dest_wh[i] is not None:
            dest_countries[i] = wh_data["countries"][int(dest_wh[i]) - 1]
        else:
            dest_countries[i] = origin_countries[i]

    cross_border = origin_countries != dest_countries
    origin_urban = np.array([wh_data["is_urban"][wid - 1] for wid in origin_wh])

    base_days = rng.integers(3, 6, size=n)  # 3-5 days base
    rural_mult = np.where(origin_urban, 1.0, 1.3)
    xborder_mult = np.where(
        cross_border & origin_urban, 2.0,
        np.where(cross_border & ~origin_urban, 2.5, 1.0)
    )
    transit_days = np.round(base_days * rural_mult * xborder_mult).astype(int)

    estimated_del = ship_dates + (transit_days * np.timedelta64(1, "D"))

    # --- Actual delivery ---
    noise_days = rng.integers(-2, 4, size=n)
    actual_del = estimated_del + (noise_days * np.timedelta64(1, "D"))

    # 5% impossible dates (actual < shipment_date)
    impossible_mask = rng.random(n) < 0.05
    impossible_offset = rng.integers(1, 10, size=impossible_mask.sum())
    actual_del[impossible_mask] = ship_dates[impossible_mask] - (impossible_offset * np.timedelta64(1, "D"))

    actual_del_str = pd.Series(actual_del).dt.strftime("%Y-%m-%d").values

    # --- Shipment status ---
    statuses = rng.choice(
        ["delivered", "in_transit", "failed", "returned"],
        size=n, p=[0.75, 0.15, 0.07, 0.03],
    )

    # --- Cost calculation (using temporary weight, NOT stored as totalWeight) ---
    print("  Calculating costs...")
    temp_weight = np.clip(
        rng.lognormal(mean=np.log(5.0), sigma=1.0, size=n), 0.5, 200.0
    )
    rate_per_kg = 2.50
    carrier_mult = np.array([CARRIER_COST_MULT[c] for c in carriers])
    xborder_cost_mult = np.where(cross_border, 1.8, 1.0)
    cost_noise = rng.uniform(0.9, 1.1, size=n)

    shipment_cost = np.round(
        temp_weight * rate_per_kg * carrier_mult * xborder_cost_mult * cost_noise, 2
    )
    total_cost = np.round(shipment_cost * rng.uniform(0.95, 1.05, size=n), 2)
    cost_amount = np.round(shipment_cost * rng.uniform(0.98, 1.02, size=n), 2)

    # --- Write in chunks ---
    print("  Writing chunks...")
    for start in tqdm(range(0, n, CHUNK_SIZE), desc="  Shipments", leave=False):
        end = min(start + CHUNK_SIZE, n)
        s = slice(start, end)

        chunk_df = pd.DataFrame({
            "shipment_id": shipment_ids[s],
            "origin_warehouse_id": origin_wh[s],
            "dest_warehouse_id": pd.array(dest_wh[s], dtype=pd.Int32Dtype()),
            "shipment_date": pd.Series(ship_dates[s]).dt.strftime("%Y-%m-%d").values,
            "carrier": carriers[s],
            "estimated_delivery": pd.Series(estimated_del[s]).dt.strftime("%Y-%m-%d").values,
            "actual_delivery": actual_del_str[s],
            "shipment_status": statuses[s],
            "totalWeight": np.nan,  # NULL — filled in pass 2
            "shipment_cost": shipment_cost[s],
            "total_cost": total_cost[s],
            "cost_amount": cost_amount[s],
        })

        # Introduce nulls in carrier and shipment_cost
        introduce_nulls_inplace(chunk_df, "carrier", 0.04)
        introduce_nulls_inplace(chunk_df, "shipment_cost", 0.04)

        write_csv(chunk_df, filepath, mode="a" if start > 0 else "w", header=(start == 0))

    print(f"  Done: {filepath.name} ({n:,} rows)")

    return {
        "ids": shipment_ids,
        "dates": ship_dates,
        "statuses": statuses,
        "origin_countries": origin_countries,
        "dest_countries": dest_countries,
    }


# ============================================================
# Generator: order_shipments (500K rows)
# ============================================================


def generate_order_shipments(order_data: dict, ship_data: dict) -> None:
    """Generate order_shipments.csv — links orders to shipments."""
    n = ROW_COUNTS["order_shipments"]
    filepath = DATA_DIR / "order_shipments.csv"
    print(f"\n[7/9] Generating order_shipments ({n:,} rows)...")

    # Orders eligible for shipment links: delivered + shipped (~85%)
    eligible_mask = np.isin(order_data["statuses"], ["delivered", "shipped"])
    eligible_order_ids = order_data["ids"][eligible_mask]
    eligible_order_dates = order_data["dates"][eligible_mask]

    # Remove ~10% for data quality (orders with no shipment link)
    n_linked = int(len(eligible_order_ids) * 0.90)
    linked_idx = rng.choice(len(eligible_order_ids), size=n_linked, replace=False)
    linked_idx.sort()
    linked_order_ids = eligible_order_ids[linked_idx]
    linked_order_dates = eligible_order_dates[linked_idx]

    # Assign link counts: target total = 500K
    # Average = 500K / n_linked links per order
    link_counts = np.ones(n_linked, dtype=np.int32)
    n_extra_needed = n - n_linked
    if n_extra_needed > 0:
        # Give some orders extra shipments (2 or 3)
        extra_idx = rng.choice(n_linked, size=n_extra_needed, replace=True)
        for idx in extra_idx:
            link_counts[idx] += 1

    # Trim to exact target
    total_links = link_counts.sum()
    if total_links > n:
        # Remove extra links from random orders with count > 1
        excess = total_links - n
        multi_idx = np.where(link_counts > 1)[0]
        remove_idx = rng.choice(multi_idx, size=min(excess, len(multi_idx)), replace=True)
        for idx in remove_idx:
            if link_counts[idx] > 1 and link_counts.sum() > n:
                link_counts[idx] -= 1

    total_links = min(link_counts.sum(), n)

    # Expand order_ids and dates by link counts
    expanded_order_ids = np.repeat(linked_order_ids, link_counts)[:n]
    expanded_order_dates = np.repeat(linked_order_dates, link_counts)[:n]

    # Assign shipment_ids (sorted by date for rough alignment)
    ship_sort_idx = np.argsort(ship_data["dates"])
    sorted_ship_ids = ship_data["ids"][ship_sort_idx]
    sorted_ship_dates = ship_data["dates"][ship_sort_idx]

    # Sample shipment_ids sequentially (date-aligned)
    # Use evenly spaced indices into the sorted shipments
    ship_indices = np.linspace(0, len(sorted_ship_ids) - 1, n, dtype=int)
    assigned_ship_ids = sorted_ship_ids[ship_indices]
    assigned_ship_dates = sorted_ship_dates[ship_indices]

    # fulfillment_date: between order_date and shipment_date
    order_dt = expanded_order_dates.astype("datetime64[D]")
    ship_dt = assigned_ship_dates.astype("datetime64[D]")
    # Use the earlier of the two as the base, add small offset
    base_dates = np.minimum(order_dt, ship_dt)
    date_range = np.abs((ship_dt - order_dt) / np.timedelta64(1, "D")).astype(int)
    date_range = np.maximum(date_range, 1)
    offsets = np.array([rng.integers(0, max(dr, 1)) for dr in date_range])
    fulfillment_dates = base_dates + (offsets * np.timedelta64(1, "D"))

    # Write in chunks
    print("  Writing chunks...")
    for start in tqdm(range(0, n, CHUNK_SIZE), desc="  OrderShipments", leave=False):
        end = min(start + CHUNK_SIZE, n)
        s = slice(start, end)

        chunk_df = pd.DataFrame({
            "order_shipment_id": np.arange(start + 1, end + 1),
            "order_id": expanded_order_ids[s],
            "shipment_id": assigned_ship_ids[s],
            "fulfillment_date": pd.Series(fulfillment_dates[s]).dt.strftime("%Y-%m-%d").values,
        })
        write_csv(chunk_df, filepath, mode="a" if start > 0 else "w", header=(start == 0))

    print(f"  Done: {filepath.name} ({n:,} rows)")


# ============================================================
# Generator: shipment_items (2M rows, chunked)
# ============================================================


def generate_shipment_items(ship_data: dict, prod_data: dict) -> None:
    """Generate shipment_items.csv with 8-10% orphaned product_ids."""
    n = ROW_COUNTS["shipment_items"]
    filepath = DATA_DIR / "shipment_items.csv"
    print(f"\n[8/9] Generating shipment_items ({n:,} rows)...")

    n_shipments = ROW_COUNTS["shipments"]
    max_product_id = prod_data["ids"].max()

    # Determine items per shipment: average ~2.5 items per shipment
    # 800K shipments * 2.5 = 2M items
    items_per_ship = rng.choice([1, 2, 3, 4, 5], size=n_shipments, p=[0.15, 0.35, 0.30, 0.15, 0.05])

    # Adjust to hit target
    total_items = items_per_ship.sum()
    if total_items > n:
        # Reduce some counts
        excess = total_items - n
        multi_idx = np.where(items_per_ship > 1)[0]
        reduce_idx = rng.choice(multi_idx, size=min(excess, len(multi_idx)), replace=True)
        for idx in reduce_idx:
            if items_per_ship[idx] > 1 and items_per_ship.sum() > n:
                items_per_ship[idx] -= 1
    elif total_items < n:
        deficit = n - total_items
        add_idx = rng.choice(n_shipments, size=deficit, replace=True)
        for idx in add_idx:
            items_per_ship[idx] += 1

    total_items = items_per_ship.sum()

    # Generate all item data
    print("  Pre-generating arrays...")
    shipment_ids_expanded = np.repeat(ship_data["ids"], items_per_ship)[:n]

    # Product IDs: valid products, with 8-10% orphaned (non-existent IDs)
    product_ids = rng.choice(prod_data["ids"], size=n)
    orphan_mask = rng.random(n) < 0.09  # ~9%
    # Orphaned IDs: above max valid product_id
    product_ids[orphan_mask] = rng.integers(max_product_id + 1, max_product_id + 5000, size=orphan_mask.sum())

    # Quantity: 1-20 per item
    quantities = rng.integers(1, 21, size=n)

    # conditionOnArrival: good 90%, damaged 7%, missing 3%
    conditions = rng.choice(
        ["good", "damaged", "missing"], size=n, p=[0.90, 0.07, 0.03]
    )

    # Write in chunks
    print("  Writing chunks...")
    for start in tqdm(range(0, n, CHUNK_SIZE), desc="  ShipmentItems", leave=False):
        end = min(start + CHUNK_SIZE, n)
        s = slice(start, end)

        chunk_df = pd.DataFrame({
            "shipment_item_id": np.arange(start + 1, end + 1),
            "shipment_id": shipment_ids_expanded[s],
            "product_id": product_ids[s],
            "quantity": quantities[s],
            "conditionOnArrival": conditions[s],
        })
        write_csv(chunk_df, filepath, mode="a" if start > 0 else "w", header=(start == 0))

    print(f"  Done: {filepath.name} ({n:,} rows)")


# ============================================================
# Generator: delivery_events (1.5M rows)
# ============================================================


def generate_delivery_events(ship_data: dict, wh_data: dict) -> None:
    """Generate delivery_events.csv — 15% of shipments have no events."""
    n_target = ROW_COUNTS["delivery_events"]
    filepath = DATA_DIR / "delivery_events.csv"
    n_shipments = ROW_COUNTS["shipments"]
    print(f"\n[9/9] Generating delivery_events (~{n_target:,} rows)...")

    # 85% of shipments get events
    has_events_mask = rng.random(n_shipments) < 0.85
    event_ship_ids = ship_data["ids"][has_events_mask]
    event_ship_dates = ship_data["dates"][has_events_mask]
    event_ship_statuses = ship_data["statuses"][has_events_mask]
    n_with_events = len(event_ship_ids)

    # Event counts per shipment: avg ~2.21 to hit ~1.5M total
    # P(2)=0.85, P(3)=0.10, P(4)=0.04, P(5)=0.01
    event_counts = rng.choice(
        [2, 3, 4, 5], size=n_with_events, p=[0.85, 0.10, 0.04, 0.01]
    )
    total_events = event_counts.sum()
    print(f"  Shipments with events: {n_with_events:,}, total events: {total_events:,}")

    # Event chain templates (last event = FINAL based on status)
    chain_templates = {
        2: ["picked", "FINAL"],
        3: ["picked", "in_transit", "FINAL"],
        4: ["picked", "in_transit", "out_for_delivery", "FINAL"],
        5: ["picked", "in_transit", "in_transit", "out_for_delivery", "FINAL"],
    }

    # Time offsets between events (in hours)
    time_offset_ranges = {
        0: (0, 4),          # picked: 0-4 hours after shipment_date
        1: (6, 48),         # in_transit: 6-48 hours after picked
        2: (12, 72),        # next stage: 12-72 hours
        3: (1, 24),         # next stage: 1-24 hours
        4: (1, 24),         # final: 1-24 hours
    }

    # Generate events grouped by count
    all_chunks = []
    event_id_counter = 1

    for count_val in [2, 3, 4, 5]:
        group_mask = event_counts == count_val
        n_group = group_mask.sum()
        if n_group == 0:
            continue

        g_ship_ids = event_ship_ids[group_mask]
        g_ship_dates = event_ship_dates[group_mask].astype("datetime64[s]")
        g_ship_statuses = event_ship_statuses[group_mask]
        template = chain_templates[count_val]

        # Build cumulative timestamps
        cumulative_ts = g_ship_dates.copy()
        position_timestamps = []

        for pos in range(count_val):
            lo, hi = time_offset_ranges.get(pos, (1, 24))
            offset_hours = rng.integers(lo, hi + 1, size=n_group)
            cumulative_ts = cumulative_ts + (offset_hours * np.timedelta64(1, "h"))
            position_timestamps.append(cumulative_ts.copy())

        # Determine event types
        is_failed_ship = np.isin(g_ship_statuses, ["failed", "returned"])

        for pos in range(count_val):
            event_type_label = template[pos]
            if event_type_label == "FINAL":
                event_types = np.where(is_failed_ship, "failed", "delivered")
            else:
                event_types = np.full(n_group, event_type_label)

            ts_strings = pd.Series(position_timestamps[pos]).dt.strftime("%Y-%m-%d %H:%M:%S").values

            # Location: random cities
            locations = np.array(rng.choice(ALL_CITIES, size=n_group))

            # Delay reason
            delay_reasons = np.full(n_group, "none", dtype=object)
            # Get month from timestamp for clustering
            months = (position_timestamps[pos].astype("datetime64[M]") - position_timestamps[pos].astype("datetime64[Y]")).astype(int) + 1

            # Weather delays: Dec-Jan
            weather_mask = ((months == 12) | (months == 1)) & (rng.random(n_group) < 0.20)
            delay_reasons[weather_mask] = "weather"

            # Capacity delays: Nov-Dec
            capacity_mask = ((months == 11) | (months == 12)) & (rng.random(n_group) < 0.15) & (delay_reasons == "none")
            delay_reasons[capacity_mask] = "capacity"

            # Customs: random 5%
            customs_mask = (rng.random(n_group) < 0.05) & (delay_reasons == "none")
            delay_reasons[customs_mask] = "customs"

            # Address issue: for failed events
            if event_type_label == "FINAL":
                addr_mask = is_failed_ship & (rng.random(n_group) < 0.30) & (delay_reasons == "none")
                delay_reasons[addr_mask] = "address_issue"

            n_events_here = n_group
            event_ids_here = np.arange(event_id_counter, event_id_counter + n_events_here)
            event_id_counter += n_events_here

            chunk_df = pd.DataFrame({
                "event_id": event_ids_here,
                "shipment_id": g_ship_ids,
                "eventType": event_types,
                "event_timestamp": ts_strings,
                "location": locations,
                "delay_reason": delay_reasons,
            })
            all_chunks.append(chunk_df)

    # Concatenate and sort by shipment_id, event_timestamp
    print("  Concatenating and sorting events...")
    events_df = pd.concat(all_chunks, ignore_index=True)
    events_df.sort_values(["shipment_id", "event_timestamp"], inplace=True)
    events_df["event_id"] = np.arange(1, len(events_df) + 1)
    events_df.reset_index(drop=True, inplace=True)

    # Write in chunks
    total = len(events_df)
    print(f"  Writing {total:,} events in chunks...")
    for start in tqdm(range(0, total, CHUNK_SIZE), desc="  Events", leave=False):
        end = min(start + CHUNK_SIZE, total)
        chunk = events_df.iloc[start:end]
        write_csv(chunk, filepath, mode="a" if start > 0 else "w", header=(start == 0))

    print(f"  Done: {filepath.name} ({total:,} rows)")


# ============================================================
# Shipments Pass 2: backfill totalWeight
# ============================================================


def backfill_shipment_weights(prod_data: dict) -> None:
    """Read shipment_items + products, calculate and backfill totalWeight in shipments.csv."""
    filepath = DATA_DIR / "shipments.csv"
    items_path = DATA_DIR / "shipment_items.csv"
    print("\n[Pass 2] Backfilling shipment totalWeight...")

    # Build product weight lookup
    prod_weights = pd.DataFrame({
        "product_id": prod_data["ids"],
        "weightKg": prod_data["weights"],
    })

    # Read shipment_items in chunks, compute weight per shipment
    print("  Reading shipment_items and computing weights...")
    weight_accum: dict[int, float] = {}

    for chunk in tqdm(
        pd.read_csv(items_path, chunksize=CHUNK_SIZE),
        desc="  Items chunks",
        total=ROW_COUNTS["shipment_items"] // CHUNK_SIZE + 1,
        leave=False,
    ):
        merged = chunk.merge(prod_weights, on="product_id", how="left")
        merged["item_weight"] = merged["weightKg"] * merged["quantity"]
        group = merged.groupby("shipment_id")["item_weight"].sum()
        for sid, w in group.items():
            weight_accum[sid] = weight_accum.get(sid, 0.0) + w

    # Read shipments and update totalWeight
    print("  Updating shipments.csv...")
    shipments_df = pd.read_csv(filepath)

    actual_weights = shipments_df["shipment_id"].map(weight_accum)

    # 90% get actual weight, 10% get mismatched weight
    n = len(shipments_df)
    mismatch_mask = rng.random(n) < 0.10
    total_weight = actual_weights.copy()
    total_weight[mismatch_mask] = actual_weights[mismatch_mask] * rng.uniform(0.5, 1.8, size=mismatch_mask.sum())
    shipments_df["totalWeight"] = np.round(total_weight, 2)

    write_csv(shipments_df, filepath)
    print(f"  Done: totalWeight backfilled ({mismatch_mask.sum():,} mismatches)")


# ============================================================
# Summary
# ============================================================


def print_summary() -> None:
    """Print row counts for all generated CSVs."""
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)

    total = 0
    for table in ROW_COUNTS:
        csv_path = DATA_DIR / f"{table}.csv"
        if csv_path.exists():
            # Count lines (subtract 1 for header)
            with open(csv_path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f) - 1
            status = "OK" if count >= ROW_COUNTS[table] * 0.95 else "LOW"
            print(f"  {table:25s} {count:>12,} rows  (target: {ROW_COUNTS[table]:>12,})  [{status}]")
            total += count
        else:
            print(f"  {table:25s} {'MISSING':>12s}")

    print(f"  {'TOTAL':25s} {total:>12,} rows")
    print("=" * 50)


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run full data generation pipeline."""
    print("=" * 50)
    print("Supply Chain & Logistics Data Generation")
    print(f"Target: {sum(ROW_COUNTS.values()):,} total rows across {len(ROW_COUNTS)} tables")
    print(f"Seed: {SEED}")
    print("=" * 50)

    ensure_dirs()

    # Phase 1: Independent tables
    wh_data = generate_warehouses()
    prod_data = generate_products()
    cust_data = generate_customers()

    # Phase 2: Dependent tables
    generate_inventory(wh_data, prod_data)
    order_data = generate_orders(cust_data, wh_data)
    ship_data = generate_shipments(wh_data)

    # Phase 3: Linking / detail tables
    generate_order_shipments(order_data, ship_data)
    generate_shipment_items(ship_data, prod_data)
    generate_delivery_events(ship_data, wh_data)

    # Phase 4: Backfill
    backfill_shipment_weights(prod_data)

    # Summary
    print_summary()
    print("\nData generation complete.")


if __name__ == "__main__":
    main()
