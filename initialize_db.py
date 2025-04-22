#!/usr/bin/env python3
"""
Database Initialization Script

This script initializes the recruitment database with all required tables.
It ensures that the database is properly set up before running other operations.
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path
from recruitment_db import RecruitmentDatabase
from logging_config import setup_logging

# Set up logging
logger = setup_logging("initialize_db")

def initialize_database():
    """Initialize the database with all required tables."""
    try:
        logger.info("Starting database initialization")
        
        # Ensure the database directory exists
        db_dir = Path("databases")
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Set database path
        db_path = os.getenv("RECRUITMENT_PATH", "databases/recruitment.db")
        
        # Read the SQL schema
        with open('create_new_db.sql', 'r') as f:
            schema_sql = f.read()
        
        # Create and initialize the database using direct SQLite connection
        conn = sqlite3.connect(db_path)
        try:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Execute the schema SQL
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("Schema executed successfully")
            
            # Verify tables using the same connection
            tables_to_check = [
                "urls", "companies", "agencies", "jobs", "adverts", "skills",
                "qualifications", "attributes", "industries", "duties",
                "phones", "emails", "benefits", "locations"
            ]
            
            cursor = conn.cursor()
            for table in tables_to_check:
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    logger.info(f"Table '{table}' exists with columns: {columns}")
                except Exception as e:
                    logger.error(f"Error checking table '{table}': {e}")
            
            logger.info("Database initialization completed successfully")
            return 0
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(initialize_database()) 