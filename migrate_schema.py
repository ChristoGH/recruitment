#!/usr/bin/env python3
"""
Database Schema Migration Script

This script migrates the database schema to establish the correct relationship between
job_adverts and other tables, where job_adverts relates to URLs and all other tables
should reference job_adverts instead of URLs directly.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
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
logger = logging.getLogger("schema_migration")

class SchemaMigration:
    """Handles the migration of the database schema."""
    
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

    def migrate_table(self, table_name: str) -> None:
        """
        Migrate a single table from url_id to job_advert_id.
        
        Args:
            table_name: Name of the table to migrate.
        """
        logger.info(f"Migrating {table_name} table...")
        
        # Check if table exists
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not self.cursor.fetchone():
            logger.info(f"{table_name} table does not exist, skipping migration")
            return

        # Get the current table schema
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        
        # Check if table already has job_advert_id
        has_job_advert_id = any(col[1] == 'job_advert_id' for col in columns)
        if has_job_advert_id:
            logger.info(f"{table_name} table already has job_advert_id, skipping migration")
            return
            
        # Check if table has url_id
        has_url_id = any(col[1] == 'url_id' for col in columns)
        if not has_url_id:
            logger.info(f"{table_name} table has neither url_id nor job_advert_id, skipping migration")
            return
        
        # Create the new table schema
        create_temp_table_sql = f"CREATE TABLE {table_name}_temp ("
        column_defs = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            col_notnull = "NOT NULL" if col[3] else ""
            col_pk = "PRIMARY KEY" if col[5] else ""
            
            # Replace url_id with job_advert_id
            if col_name == 'url_id':
                col_name = 'job_advert_id'
            
            column_defs.append(f"{col_name} {col_type} {col_notnull} {col_pk}")
        
        create_temp_table_sql += ", ".join(column_defs)
        create_temp_table_sql += ", FOREIGN KEY (job_advert_id) REFERENCES job_adverts (id) ON DELETE CASCADE"
        create_temp_table_sql += ")"
        
        # Create the temporary table
        self.cursor.execute(create_temp_table_sql)
        
        # Get column names for the old and new tables
        old_columns = [col[1] for col in columns]
        new_columns = [col[1] for col in columns]
        new_columns[new_columns.index('url_id')] = 'job_advert_id'
        
        # Build the SELECT part of the query
        select_parts = []
        for col in old_columns:
            if col == 'url_id':
                select_parts.append('ja.id as job_advert_id')
            else:
                select_parts.append(f't.{col}')
        
        # Copy data from old table to new table, mapping url_id to job_advert_id
        insert_sql = f"""
            INSERT INTO {table_name}_temp ({', '.join(new_columns)})
            SELECT {', '.join(select_parts)}
            FROM {table_name} t
            JOIN job_adverts ja ON ja.url_id = t.url_id
        """
        
        self.cursor.execute(insert_sql)
        
        # Drop the old table and rename the new one
        self.cursor.execute(f"DROP TABLE {table_name}")
        self.cursor.execute(f"ALTER TABLE {table_name}_temp RENAME TO {table_name}")
        
        logger.info(f"Successfully migrated {table_name} table")
            
    def run_migration(self) -> None:
        """Run the complete migration process."""
        try:
            # Connect to the database
            self.connect()
            
            # Create a backup
            self.backup_database()
            
            # Start a transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            # List of tables to migrate
            tables_to_migrate = [
                'agency', 'company', 'company_address', 'company_email', 'company_phone',
                'contact_person', 'email', 'emails', 'hiring_company',
                'hiring_company_address', 'hiring_company_email', 'hiring_company_phone',
                'phone_numbers', 'recruiter', 'skills', 'attributes', 'benefits', 
                'duties', 'location', 'qualifications', 'recruitment_evidence', 'industry'
            ]
            
            # Migrate each table
            for table in tables_to_migrate:
                self.migrate_table(table)
            
            # Commit the transaction
            self.conn.commit()
            logger.info("Migration completed successfully")
            
        except Exception as e:
            # Rollback the transaction in case of error
            if self.conn:
                self.conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
            
        finally:
            # Disconnect from the database
            self.disconnect()
            
if __name__ == "__main__":
    # Get the database path from environment variable or use default
    db_path = os.getenv("RECRUITMENT_DB_PATH", "recruitment.db")
    
    # Run the migration
    migration = SchemaMigration(db_path)
    migration.run_migration() 