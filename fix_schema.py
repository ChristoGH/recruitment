#!/usr/bin/env python3
"""
Fix database schema by adding missing columns and ensuring consistency.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_schema")

def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get all column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def add_missing_columns(conn: sqlite3.Connection, table_name: str, columns: List[Dict[str, Any]]) -> None:
    """Add missing columns to a table if they don't exist."""
    cursor = conn.cursor()
    existing_columns = get_table_columns(conn, table_name)
    
    for column in columns:
        if column['name'] not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column['name']} {column['type']}")
                logger.info(f"Added column {column['name']} to table {table_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise

def fix_urls_table(conn: sqlite3.Connection) -> None:
    """Fix the urls table schema."""
    columns = [
        {'name': 'created_at', 'type': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'},
        {'name': 'updated_at', 'type': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'},
        {'name': 'processing_status', 'type': 'TEXT DEFAULT "pending"'},
        {'name': 'error_count', 'type': 'INTEGER DEFAULT 0'},
        {'name': 'recruitment_flag', 'type': 'INTEGER DEFAULT -1'},
        {'name': 'error_message', 'type': 'TEXT'}
    ]
    
    add_missing_columns(conn, 'urls', columns)
    
    # Update existing records with current timestamp if created_at is NULL
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE urls 
        SET created_at = CURRENT_TIMESTAMP 
        WHERE created_at IS NULL
    """)
    cursor.execute("""
        UPDATE urls 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE updated_at IS NULL
    """)
    conn.commit()

def fix_url_processing_status_table(conn: sqlite3.Connection) -> None:
    """Fix the url_processing_status table schema."""
    columns = [
        {'name': 'created_at', 'type': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'},
        {'name': 'updated_at', 'type': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'},
        {'name': 'last_processed_at', 'type': 'TIMESTAMP'},
        {'name': 'error_count', 'type': 'INTEGER DEFAULT 0'}
    ]
    
    add_missing_columns(conn, 'url_processing_status', columns)
    
    # Update existing records with current timestamp if created_at is NULL
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE url_processing_status 
        SET created_at = CURRENT_TIMESTAMP 
        WHERE created_at IS NULL
    """)
    cursor.execute("""
        UPDATE url_processing_status 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE updated_at IS NULL
    """)
    conn.commit()

def main():
    """Main function to fix the database schema."""
    db_path = "databases/recruitment.db"
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Fix tables
        fix_urls_table(conn)
        fix_url_processing_status_table(conn)
        
        # Create indexes if they don't exist
        cursor = conn.cursor()
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_urls_domain 
            ON urls(domain_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_url_processing_status 
            ON url_processing_status(status)
        """)
        
        conn.commit()
        logger.info("Successfully fixed database schema")
        
    except Exception as e:
        logger.error(f"Error fixing database schema: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main() 