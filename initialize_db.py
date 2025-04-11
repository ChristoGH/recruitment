#!/usr/bin/env python3
"""
Database Initialization Script

This script initializes the recruitment database with all required tables.
It ensures that the database is properly set up before running other operations.
"""

import os
import sys
import logging
from recruitment_db_lib import RecruitmentDatabase
from logging_config import setup_logging

# Set up logging
logger = setup_logging("initialize_db")

def initialize_database():
    """Initialize the database with all required tables."""
    try:
        logger.info("Starting database initialization")
        
        # Initialize database (this will create all tables)
        db = RecruitmentDatabase()
        
        # Verify that key tables exist
        tables_to_check = [
            "urls", 
            "company", 
            "recruiter", 
            "agency", 
            "job_adverts", 
            "job_advert_forms"
        ]
        
        for table in tables_to_check:
            try:
                columns = db.get_column_names(table)
                logger.info(f"Table '{table}' exists with columns: {columns}")
            except Exception as e:
                logger.error(f"Error checking table '{table}': {e}")
        
        logger.info("Database initialization completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(initialize_database()) 