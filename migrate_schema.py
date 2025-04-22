#!/usr/bin/env python3
"""
Migration script to update the database schema from using url_id to job_advert_id.
This script handles the transition of data while maintaining referential integrity.
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchemaMigration:
    """Handles the migration of the database schema from url_id to job_advert_id."""
    
    def __init__(self, db_path: str = None):
        """Initialize the migration handler."""
        self.db_path = db_path or os.getenv("RECRUITMENT_PATH")
        if not self.db_path:
            raise ValueError("Database path not set. Check RECRUITMENT_PATH environment variable.")
        
        self.conn = None
        self.cursor = None
        self.tables_to_migrate = [
            'company',
            'company_address',
            'company_email',
            'company_phone',
            'agency',
            'links',
            'email',
            'contact_person',
            'duties',
            'benefits',
            'industry',
            'location'
        ]

    def connect(self):
        """Establish connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.cursor.execute("PRAGMA foreign_keys = ON;")
            logger.info("Connected to database successfully")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")

    def backup_database(self):
        """Create a backup of the database before migration."""
        backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with sqlite3.connect(self.db_path) as src, sqlite3.connect(backup_path) as dst:
                src.backup(dst)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        except sqlite3.Error as e:
            logger.error(f"Error backing up database: {e}")
            raise

    def migrate_table(self, table_name: str):
        """
        Migrate a single table from using url_id to job_advert_id.
        
        Args:
            table_name: Name of the table to migrate
        """
        logger.info(f"Starting migration of table: {table_name}")
        
        try:
            # Check if table exists
            self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not self.cursor.fetchone():
                logger.warning(f"Table {table_name} does not exist, skipping")
                return

            # Get current schema
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = self.cursor.fetchall()
            
            # Check if table already has job_advert_id
            has_job_advert_id = any(col[1] == 'job_advert_id' for col in columns)
            if has_job_advert_id:
                logger.info(f"Table {table_name} already has job_advert_id column, skipping")
                return

            # Check if table has url_id column
            has_url_id = any(col[1] == 'url_id' for col in columns)
            if not has_url_id:
                logger.warning(f"Table {table_name} does not have url_id column, skipping")
                return

            # Create temporary table with new schema
            temp_table = f"{table_name}_temp"
            
            # Build column definitions for the new table
            column_defs = []
            non_id_columns = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                col_notnull = "NOT NULL" if col[3] else ""
                col_pk = "PRIMARY KEY AUTOINCREMENT" if col[5] else ""
                
                # Replace url_id with job_advert_id
                if col_name == 'url_id':
                    col_name = 'job_advert_id'
                elif col_name not in ['id', 'url_id']:
                    non_id_columns.append(col_name)
                
                column_defs.append(f"{col_name} {col_type} {col_notnull} {col_pk}")
            
            # Add foreign key constraint to job_advert_id
            create_temp_table = f"""
                CREATE TABLE {temp_table} (
                    {', '.join(column_defs)},
                    FOREIGN KEY (job_advert_id) REFERENCES job_adverts(id) ON DELETE CASCADE ON UPDATE CASCADE
                )
            """
            
            # Create the temporary table
            self.cursor.execute(create_temp_table)
            
            # Check for records with invalid url_id (no corresponding job_advert_id)
            check_invalid_sql = f"""
                SELECT COUNT(*) FROM {table_name} t
                LEFT JOIN job_adverts ja ON ja.url_id = t.url_id
                WHERE ja.id IS NULL
            """
            self.cursor.execute(check_invalid_sql)
            invalid_count = self.cursor.fetchone()[0]
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} records in {table_name} with url_id that doesn't correspond to a job_advert_id")
                
                # Log the invalid records for reference
                log_invalid_sql = f"""
                    SELECT t.id, t.url_id FROM {table_name} t
                    LEFT JOIN job_adverts ja ON ja.url_id = t.url_id
                    WHERE ja.id IS NULL
                    LIMIT 10
                """
                self.cursor.execute(log_invalid_sql)
                invalid_records = self.cursor.fetchall()
                for record in invalid_records:
                    logger.warning(f"Invalid record: id={record[0]}, url_id={record[1]}")
                
                # Create a table to store invalid records
                invalid_table = f"{table_name}_invalid"
                create_invalid_sql = f"""
                    CREATE TABLE {invalid_table} AS
                    SELECT t.* FROM {table_name} t
                    LEFT JOIN job_adverts ja ON ja.url_id = t.url_id
                    WHERE ja.id IS NULL
                """
                self.cursor.execute(create_invalid_sql)
                logger.info(f"Created {invalid_table} table with {invalid_count} invalid records")
            
            # Copy data from old table to new table, only for valid records
            column_list = ['ja.id as job_advert_id'] + [f't.{col}' for col in non_id_columns]
            insert_sql = f"""
                INSERT INTO {temp_table} (job_advert_id, {', '.join(non_id_columns)})
                SELECT {', '.join(column_list)}
                FROM {table_name} t
                INNER JOIN job_adverts ja ON ja.url_id = t.url_id
            """
            
            self.cursor.execute(insert_sql)
            
            # Drop old table and rename new one
            self.cursor.execute(f"DROP TABLE {table_name}")
            self.cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
            
            self.conn.commit()
            logger.info(f"Successfully migrated table: {table_name}")
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Error migrating table {table_name}: {e}")
            raise

    def run_migration(self):
        """Execute the complete migration process."""
        try:
            self.connect()
            backup_path = self.backup_database()
            
            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            # Migrate each table
            for table in self.tables_to_migrate:
                self.migrate_table(table)
            
            # Commit transaction
            self.conn.commit()
            logger.info("Migration completed successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            self.disconnect()

if __name__ == "__main__":
    try:
        migration = SchemaMigration()
        migration.run_migration()
    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        exit(1) 