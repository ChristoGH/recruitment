#!/usr/bin/env python3
"""
Database Confusion Cleanup Script

This script:
1. Backs up existing database files
2. Consolidates schemas
3. Renames files
4. Updates code references
5. Updates Docker configuration
"""

import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cleanup.log')
    ]
)
logger = logging.getLogger(__name__)

def backup_database(db_path):
    """Create a backup of a database file."""
    if not os.path.exists(db_path):
        logger.warning(f"Database {db_path} does not exist, skipping backup")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}.backup_{timestamp}"
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup {db_path}: {e}")
        return None

def get_table_schema(db_path, table_name):
    """Get the schema of a specific table."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to get schema for {table_name} from {db_path}: {e}")
        return None

def add_missing_columns(db_path):
    """Add missing created_at and updated_at columns if they don't exist."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if columns exist
            cursor.execute("PRAGMA table_info(urls)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # Add missing columns
            if 'created_at' not in columns:
                cursor.execute("ALTER TABLE urls ADD COLUMN created_at DATETIME")
                logger.info("Added created_at column")
            
            if 'updated_at' not in columns:
                cursor.execute("ALTER TABLE urls ADD COLUMN updated_at DATETIME")
                logger.info("Added updated_at column")
            
            conn.commit()
            logger.info(f"Successfully updated schema for {db_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to add missing columns to {db_path}: {e}")
        return False

def update_file_content(file_path, old_text, new_text):
    """Update content in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        updated_content = content.replace(old_text, new_text)
        
        if content != updated_content:
            with open(file_path, 'w') as f:
                f.write(updated_content)
            logger.info(f"Updated {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to update {file_path}: {e}")
        return False

def update_code_references():
    """Update all code references to use the new naming."""
    python_files = Path('.').rglob('*.py')
    updates_made = False
    
    for file_path in python_files:
        file_path_str = str(file_path)
        if 'cleanup_db_confusion.py' in file_path_str:
            continue
            
        try:
            made_changes = False
            # Update imports
            made_changes |= update_file_content(
                file_path_str,
                'from new_recruitment_db',
                'from recruitment_db'
            )
            made_changes |= update_file_content(
                file_path_str,
                'import new_recruitment_db',
                'import recruitment_db'
            )
            
            # Update class references
            made_changes |= update_file_content(
                file_path_str,
                'NewRecruitmentDatabase',
                'RecruitmentDatabase'
            )
            
            # Update environment variable references
            made_changes |= update_file_content(
                file_path_str,
                'RECRUITMENT_NEW_PATH',
                'RECRUITMENT_PATH'
            )
            
            if made_changes:
                updates_made = True
                logger.info(f"Updated references in {file_path_str}")
        except Exception as e:
            logger.error(f"Failed to update {file_path_str}: {e}")
    
    return updates_made

def update_docker_config():
    """Update Docker configuration files."""
    try:
        # Update docker-compose.yml
        update_file_content(
            'docker-compose.yml',
            'RECRUITMENT_NEW_PATH',
            'RECRUITMENT_PATH'
        )
        
        # Update Dockerfile.processing
        update_file_content(
            'Dockerfile.processing',
            'new_recruitment_db.py',
            'recruitment_db.py'
        )
        
        logger.info("Updated Docker configuration files")
        return True
    except Exception as e:
        logger.error(f"Failed to update Docker configuration: {e}")
        return False

def main():
    """Main execution function."""
    try:
        logger.info("Starting database confusion cleanup")
        
        # 1. Create backups
        logger.info("Creating database backups...")
        backup_database('databases/recruitment.db')
        backup_database('databases/recruitment_new.db')
        
        # 2. Add missing columns to recruitment.db if needed
        logger.info("Updating database schema...")
        if not add_missing_columns('databases/recruitment.db'):
            logger.error("Failed to update database schema")
            return False
        
        # 3. Rename new_recruitment_db.py to recruitment_db.py
        logger.info("Renaming database module...")
        if os.path.exists('new_recruitment_db.py'):
            if os.path.exists('recruitment_db.py'):
                backup_database('recruitment_db.py')
            shutil.move('new_recruitment_db.py', 'recruitment_db.py')
            logger.info("Renamed new_recruitment_db.py to recruitment_db.py")
        
        # 4. Update code references
        logger.info("Updating code references...")
        if not update_code_references():
            logger.warning("No code references needed updating")
        
        # 5. Update Docker configuration
        logger.info("Updating Docker configuration...")
        if not update_docker_config():
            logger.error("Failed to update Docker configuration")
            return False
        
        logger.info("Cleanup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 