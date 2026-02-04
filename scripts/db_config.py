"""Database configuration — reads .env and exposes get_connection() helper."""

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


def _load_env() -> None:
    """Load environment variables from .env file in project root."""
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)


def get_connection_params() -> dict[str, str | int]:
    """Return database connection parameters from environment."""
    _load_env()
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "supply_chain_db"),
        "user": os.getenv("DB_USER", "logistics_admin"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def get_connection() -> psycopg2.extensions.connection:
    """Create and return a new PostgreSQL connection."""
    params = get_connection_params()
    return psycopg2.connect(**params)


if __name__ == "__main__":
    params = get_connection_params()
    print("Database connection parameters:")
    for key, value in params.items():
        if key == "password":
            print(f"  {key}: {'*' * len(str(value))}")
        else:
            print(f"  {key}: {value}")

    try:
        conn = get_connection()
        print("\nConnection successful!")
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"\nConnection failed (expected if PostgreSQL isn't running yet): {e}")
