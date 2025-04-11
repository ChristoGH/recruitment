import sqlite3
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def migrate_job_adverts():
    """Add company_id and recruiter_id columns to job_adverts table if they don't exist."""
    try:
        # Connect to the database
        conn = sqlite3.connect('recruitment.db')
        cursor = conn.cursor()
        
        # Check if columns exist
        cursor.execute("PRAGMA table_info(job_adverts)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add company_id column if it doesn't exist
        if 'company_id' not in columns:
            logging.info("Adding company_id column to job_adverts table")
            cursor.execute("""
                ALTER TABLE job_adverts
                ADD COLUMN company_id INTEGER
                REFERENCES company(id)
            """)
        
        # Add recruiter_id column if it doesn't exist
        if 'recruiter_id' not in columns:
            logging.info("Adding recruiter_id column to job_adverts table")
            cursor.execute("""
                ALTER TABLE job_adverts
                ADD COLUMN recruiter_id INTEGER
                REFERENCES recruiter(id)
            """)
        
        # Commit the changes
        conn.commit()
        logging.info("Migration completed successfully")
        
    except Exception as e:
        logging.error(f"Error during migration: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_job_adverts() 