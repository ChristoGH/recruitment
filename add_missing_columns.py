#!/usr/bin/env python3
"""
Database Migration Script to Add Missing Columns

This script adds missing columns to various tables in the recruitment database
to fix foreign key constraints and ensure proper table relationships.
All tables should reference job_adverts through job_advert_id, except for
job_adverts itself which links to urls through url_id.
"""

import os
import sqlite3
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()  # This will still show logs in console
    ]
)
logger = logging.getLogger("column_migration")

class ColumnMigration:
    """Handles the addition of missing columns to database tables."""
    
    def __init__(self, db_path: str) -> None:
        """
        Initialize the migration handler.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self) -> None:
        """Connect to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")
            
    def backup_database(self) -> str:
        """
        Create a backup of the database before migration.
        
        Returns:
            str: Path to the backup file.
        """
        backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with sqlite3.connect(self.db_path) as src_conn:
                with sqlite3.connect(backup_path) as dest_conn:
                    src_conn.backup(dest_conn)
            logger.info(f"Created database backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            raise

    def add_column_if_not_exists(self, table: str, column: str, definition: str) -> None:
        """
        Add a column to a table if it doesn't already exist.
        
        Args:
            table: Name of the table to modify.
            column: Name of the column to add.
            definition: SQL definition of the column.
        """
        try:
            # Check if column exists
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if column not in columns:
                self.cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                logger.info(f"Added column '{column}' to table '{table}'")
            else:
                logger.info(f"Column '{column}' already exists in table '{table}'")
                
        except Exception as e:
            logger.error(f"Error adding column '{column}' to table '{table}': {e}")
            raise

    def create_table_if_not_exists(self, table: str, schema: str) -> None:
        """
        Create a table if it doesn't exist.
        
        Args:
            table: Name of the table to create.
            schema: SQL schema definition for the table.
        """
        try:
            self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} {schema}")
            logger.info(f"Created or verified table '{table}'")
        except Exception as e:
            logger.error(f"Error creating table '{table}': {e}")
            raise

    def run_migration(self) -> None:
        """Run the complete migration process to add missing columns."""
        try:
            # Connect to the database
            self.connect()
            
            # Create a backup
            self.backup_database()
            
            # Start a transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            # Add missing columns to job_adverts
            self.add_column_if_not_exists(
                "job_adverts",
                "description",
                "TEXT"
            )
            
            # Add job_advert_id to various tables
            tables_needing_job_advert_id = [
                "recruitment_evidence",
                "benefits",
                "attributes",
                "duties",
                "skills",
                "qualifications"
            ]
            
            for table in tables_needing_job_advert_id:
                self.add_column_if_not_exists(
                    table,
                    "job_advert_id",
                    "INTEGER REFERENCES job_adverts(id) ON DELETE CASCADE"
                )
            
            # Create emails table
            self.create_table_if_not_exists(
                "emails",
                """(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(email, entity_type, entity_id)
                )"""
            )
            
            # Commit the transaction
            self.conn.commit()
            logger.info("Successfully completed adding missing columns")
            
        except Exception as e:
            # Rollback the transaction in case of error
            if self.conn:
                self.conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
            
        finally:
            # Disconnect from the database
            self.disconnect()

def main():
    """Main function to run the migration."""
    # Get the database path from environment variable or use default
    db_path = os.getenv("RECRUITMENT_DB_PATH", "databases/recruitment.db")
    
    # Run the migration
    migration = ColumnMigration(db_path)
    migration.run_migration()

if __name__ == "__main__":
    main()