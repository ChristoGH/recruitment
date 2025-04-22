#!/usr/bin/env python3
"""
Verification script to check the success of the database migration
from url_id to job_advert_id.
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
        logging.FileHandler('verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationVerifier:
    """Verifies the success of the database migration."""
    
    def __init__(self, db_path: str = None):
        """Initialize the verifier."""
        self.db_path = db_path or os.getenv("RECRUITMENT_PATH")
        if not self.db_path:
            raise ValueError("Database path not set. Check RECRUITMENT_PATH environment variable.")
        
        self.conn = None
        self.cursor = None
        self.tables_to_verify = [
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

    def verify_table_schema(self, table_name: str):
        """Verify that the table has the correct schema after migration."""
        try:
            # Check if table exists
            self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not self.cursor.fetchone():
                logger.error(f"Table {table_name} does not exist!")
                return False

            # Get table schema
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = {col[1]: col for col in self.cursor.fetchall()}

            # Verify required columns
            if 'job_advert_id' not in columns:
                logger.error(f"Table {table_name} is missing job_advert_id column!")
                return False

            if 'url_id' in columns:
                logger.error(f"Table {table_name} still has url_id column!")
                return False

            # Verify foreign key constraint
            self.cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = self.cursor.fetchall()
            has_job_advert_fk = any(fk[2] == 'job_adverts' and fk[3] == 'id' and fk[4] == 'job_advert_id' for fk in foreign_keys)
            
            if not has_job_advert_fk:
                logger.error(f"Table {table_name} is missing foreign key constraint on job_advert_id!")
                return False

            logger.info(f"Table {table_name} schema verification passed")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error verifying schema for table {table_name}: {e}")
            return False

    def verify_table_data(self, table_name: str):
        """Verify that the table data was migrated correctly."""
        try:
            # Count total records
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = self.cursor.fetchone()[0]

            # Count records with valid job_advert_id (foreign key constraint should ensure this)
            self.cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name} t
                INNER JOIN job_adverts ja ON t.job_advert_id = ja.id
            """)
            valid_count = self.cursor.fetchone()[0]

            # They should be equal due to foreign key constraint
            if total_count != valid_count:
                logger.error(f"Table {table_name} has {total_count} records but only {valid_count} have valid job_advert_id!")
                return False

            # Check for any NULL job_advert_id
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE job_advert_id IS NULL")
            null_count = self.cursor.fetchone()[0]
            if null_count > 0:
                logger.error(f"Table {table_name} has {null_count} records with NULL job_advert_id!")
                return False

            # Check for invalid records table
            invalid_table = f"{table_name}_invalid"
            self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{invalid_table}'")
            if self.cursor.fetchone():
                self.cursor.execute(f"SELECT COUNT(*) FROM {invalid_table}")
                invalid_count = self.cursor.fetchone()[0]
                logger.info(f"Found {invalid_count} invalid records in {invalid_table}")

            logger.info(f"Table {table_name} data verification passed: {total_count} records")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error verifying data for table {table_name}: {e}")
            return False

    def verify_migration(self):
        """Verify the entire migration."""
        try:
            self.connect()
            
            all_passed = True
            for table in self.tables_to_verify:
                logger.info(f"\nVerifying table: {table}")
                schema_ok = self.verify_table_schema(table)
                data_ok = self.verify_table_data(table)
                
                if not (schema_ok and data_ok):
                    all_passed = False
                    logger.error(f"Verification failed for table: {table}")
                else:
                    logger.info(f"Verification passed for table: {table}")

            if all_passed:
                logger.info("\nAll tables verified successfully!")
            else:
                logger.error("\nVerification failed for some tables!")

            return all_passed

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
        finally:
            self.disconnect()

if __name__ == "__main__":
    try:
        verifier = MigrationVerifier()
        success = verifier.verify_migration()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Verification script failed: {e}")
        exit(1) 