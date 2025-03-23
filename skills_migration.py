#!/usr/bin/env python3
"""
Migration script to update the skills table structure in the recruitment database.
This script will:
1. Create a backup of the current skills data
2. Drop the old skills table
3. Create a new skills table with proper constraints
4. Restore the data from the backup
"""

import os
import logging
import sqlite3
from pathlib import Path
from logging_config import setup_logging


from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging

# Create module-specific logger
logger = setup_logging("skills_migration")

def get_db_path():
    """Get the database path from environment variables or use a default."""
    db_path = os.getenv("RECRUITMENT_PATH")
    if not db_path:
        logger.warning("Database path not set in environment. Using default path.")
        db_path = "recruitment.db"
    return db_path


def backup_skills_data(conn):
    """Backup all data from the skills table."""
    logger.info("Backing up skills data...")

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skills")
    skills_data = cursor.fetchall()

    logger.info(f"Backed up {len(skills_data)} skill records.")
    return skills_data


def drop_skills_table(conn):
    """Drop the existing skills table."""
    logger.info("Dropping skills table...")

    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS skills")
    logger.info("Skills table dropped.")


def create_new_skills_table(conn):
    """Create a new skills table with proper constraints."""
    logger.info("Creating new skills table...")

    cursor = conn.cursor()

    # Option 1: Include experience in the uniqueness constraint
    create_table_query = """
    CREATE TABLE IF NOT EXISTS skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url_id INTEGER NOT NULL,
        skill TEXT NOT NULL,
        experience TEXT,
        UNIQUE (url_id, skill, experience),
        FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
    )
    """

    # Option 2: Remove the uniqueness constraint altogether
    # Uncomment this and comment out Option 1 if you want to allow duplicates
    """
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url_id INTEGER NOT NULL,
        skill TEXT NOT NULL,
        experience TEXT,
        FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
    )
    '''
    """

    cursor.execute(create_table_query)
    logger.info("New skills table created.")


def restore_skills_data(conn, skills_data):
    """Restore the skills data to the new table."""
    logger.info("Restoring skills data...")

    cursor = conn.cursor()

    for skill_record in skills_data:
        # Skip the id (first column) as it will be auto-generated
        # Format: id, url_id, skill, experience
        url_id = skill_record[1]
        skill = skill_record[2]
        experience = skill_record[3]

        insert_query = "INSERT INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
        try:
            cursor.execute(insert_query, (url_id, skill, experience))
        except sqlite3.IntegrityError as e:
            logger.warning(f"Could not insert skill '{skill}' for URL ID {url_id}: {e}")
            continue

    conn.commit()
    logger.info("Skills data restored.")


def verify_migration(conn):
    """Verify that the migration was successful."""
    cursor = conn.cursor()

    # Verify table structure
    cursor.execute("PRAGMA table_info(skills)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    logger.info(f"Skills table columns: {column_names}")

    # Verify foreign key and uniqueness constraints
    cursor.execute("PRAGMA foreign_key_list(skills)")
    fk_constraints = cursor.fetchall()
    logger.info(f"Foreign key constraints: {fk_constraints}")

    cursor.execute("PRAGMA index_list(skills)")
    indexes = cursor.fetchall()
    logger.info(f"Indexes: {indexes}")

    # Count records
    cursor.execute("SELECT COUNT(*) FROM skills")
    count = cursor.fetchone()[0]
    logger.info(f"Total skills records after migration: {count}")


def run_migration():
    """Execute the full migration process."""
    db_path = get_db_path()
    logger.info(f"Starting skills table migration for database: {db_path}")

    try:
        # Ensure the database file exists
        if not os.path.exists(db_path):
            logger.error(f"Database file does not exist: {db_path}")
            return False

        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        # Backup the data
        skills_data = backup_skills_data(conn)

        # Drop the old table
        drop_skills_table(conn)

        # Create the new table
        create_new_skills_table(conn)

        # Restore the data
        restore_skills_data(conn, skills_data)

        # Verify the migration
        verify_migration(conn)

        conn.close()
        logger.info("Migration completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if run_migration():
        print("Skills table migration completed successfully.")
    else:
        print("Skills table migration failed. See logs for details.")