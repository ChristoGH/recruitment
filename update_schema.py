#!/usr/bin/env python3
"""
Database Schema Update Script

This script updates the existing recruitment.db file with the new schema.
It adds any missing columns and tables without creating a new database file.
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger('update_schema')

# Database path
DB_PATH = "databases/recruitment.db"

def connect_to_db():
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def add_missing_columns(conn):
    """Add missing columns to existing tables."""
    logger.info("Adding missing columns to existing tables...")
    
    # Define tables and their columns to add
    table_columns = {
        'urls': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'companies': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'agencies': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'jobs': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'job_adverts': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'skills': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'qualifications': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'attributes': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'industries': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'duties': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'phones': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'emails': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ],
        'benefits': [
            ('created_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP')
        ]
    }

    cursor = conn.cursor()
    
    for table, columns in table_columns.items():
        try:
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if not cursor.fetchone():
                logger.warning(f"Table {table} does not exist, skipping column updates")
                continue

            # Get existing columns
            cursor.execute(f"PRAGMA table_info({table})")
            existing_columns = [row[1] for row in cursor.fetchall()]

            # Add missing columns
            for column_name, column_def in columns:
                if column_name not in existing_columns:
                    logger.info(f"Adding column {column_name} to table {table}")
                    try:
                        # For columns with non-constant defaults, we need to add them in two steps
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_def.split('DEFAULT')[0].strip()}")
                        if 'DEFAULT' in column_def:
                            cursor.execute(f"UPDATE {table} SET {column_name} = {column_def.split('DEFAULT')[1].strip()}")
                    except sqlite3.Error as e:
                        logger.error(f"Error updating table {table}: {e}")
                        continue

        except sqlite3.Error as e:
            logger.error(f"Error processing table {table}: {e}")
            continue

    conn.commit()
    logger.info("Finished adding missing columns")

def create_missing_tables(conn):
    """Create any missing tables from the schema."""
    logger.info("Creating missing tables...")
    
    # Read the schema file
    schema_path = Path("create_new_db.sql")
    if not schema_path.exists():
        logger.error("Schema file not found: create_new_db.sql")
        return False

    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    # Split the schema into individual CREATE TABLE statements
    create_statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip().upper().startswith('CREATE TABLE')]

    cursor = conn.cursor()
    
    for stmt in create_statements:
        try:
            # Extract table name from CREATE TABLE statement
            table_name = stmt.split('CREATE TABLE')[1].split('(')[0].strip()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if cursor.fetchone():
                logger.info(f"Table {table_name} already exists, skipping")
                continue

            logger.info(f"Creating table {table_name}")
            cursor.execute(stmt)
            conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error creating table {table_name}: {e}")
            continue

    logger.info("Finished creating missing tables")

def update_schema():
    """Update the database schema."""
    logger.info("Starting database schema update")
    
    try:
        conn = connect_to_db()
        
        # Add missing columns to existing tables
        add_missing_columns(conn)
        
        # Create any missing tables
        create_missing_tables(conn)
        
        # Verify tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Database now has the following tables: {tables}")
        
        conn.close()
        logger.info("Database schema update completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if update_schema() else 1) 