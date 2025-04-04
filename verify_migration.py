import sqlite3
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_db(db_path):
    """Create a connection to the database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        raise

def verify_table_count(old_conn, new_conn, table_name, old_table_name=None):
    """Verify that the number of records in a table matches between old and new databases."""
    if old_table_name is None:
        old_table_name = table_name

    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get counts from both databases
        old_cursor.execute(f"SELECT COUNT(*) as count FROM {old_table_name}")
        old_count = old_cursor.fetchone()['count']

        new_cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        new_count = new_cursor.fetchone()['count']

        # Log the results
        if old_count == new_count:
            logger.info(f"✓ {table_name}: {old_count} records match")
            return True
        else:
            logger.warning(f"✗ {table_name}: {old_count} records in old DB, {new_count} records in new DB")
            return False
    except sqlite3.Error as e:
        logger.error(f"Error verifying {table_name}: {e}")
        return False

def verify_relationships(old_conn, new_conn, relationship_table, old_table_name=None):
    """Verify that the relationships between tables are preserved."""
    if old_table_name is None:
        old_table_name = relationship_table

    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get counts from both databases
        old_cursor.execute(f"SELECT COUNT(*) as count FROM {old_table_name}")
        old_count = old_cursor.fetchone()['count']

        new_cursor.execute(f"SELECT COUNT(*) as count FROM {relationship_table}")
        new_count = new_cursor.fetchone()['count']

        # Log the results
        if old_count == new_count:
            logger.info(f"✓ {relationship_table}: {old_count} relationships match")
            return True
        else:
            logger.warning(f"✗ {relationship_table}: {old_count} relationships in old DB, {new_count} in new DB")
            return False
    except sqlite3.Error as e:
        logger.error(f"Error verifying {relationship_table}: {e}")
        return False

def verify_data_integrity(old_conn, new_conn):
    """Verify the integrity of the migrated data."""
    logger.info("Verifying data integrity...")
    
    # Verify core tables
    core_tables = [
        ('urls', 'urls'),
        ('companies', 'company'),
        ('agencies', 'agency'),
        ('skills', 'skills'),
        ('qualifications', 'qualifications'),
        ('jobs', 'job_adverts')
    ]

    # Verify relationship tables
    relationship_tables = [
        ('job_urls', 'job_advert_forms'),
        ('job_skills', 'job_advert_forms'),
        ('job_qualifications', 'job_advert_forms'),
        ('job_companies', 'job_advert_forms'),
        ('job_agencies', 'job_advert_forms')
    ]

    # Check core tables
    core_results = []
    for new_table, old_table in core_tables:
        result = verify_table_count(old_conn, new_conn, new_table, old_table)
        core_results.append(result)

    # Check relationship tables
    relationship_results = []
    for new_table, old_table in relationship_tables:
        result = verify_relationships(old_conn, new_conn, new_table, old_table)
        relationship_results.append(result)

    # Calculate overall success rate
    total_checks = len(core_results) + len(relationship_results)
    successful_checks = sum(core_results) + sum(relationship_results)
    success_rate = (successful_checks / total_checks) * 100

    logger.info(f"\nVerification Summary:")
    logger.info(f"Total checks: {total_checks}")
    logger.info(f"Successful checks: {successful_checks}")
    logger.info(f"Success rate: {success_rate:.2f}%")

    return success_rate == 100

def main():
    """Main function to handle the verification process."""
    try:
        # Connect to databases
        old_conn = connect_to_db('databases/recruitment.db')
        new_conn = connect_to_db('databases/recruitment_new.db')

        # Verify data integrity
        is_valid = verify_data_integrity(old_conn, new_conn)

        if is_valid:
            logger.info("\n✓ Migration verification completed successfully")
        else:
            logger.warning("\n⚠ Migration verification completed with warnings")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise
    finally:
        # Close connections
        if old_conn:
            old_conn.close()
        if new_conn:
            new_conn.close()

if __name__ == "__main__":
    main() 