#!/usr/bin/env python3
"""
Update Job Adverts Script

This script updates all existing job adverts to link them with the correct company records.
It uses a smarter approach that links jobs to companies based on name relationships,
not just URL associations.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from recruitment_db import RecruitmentDatabase
from logging_config import setup_logging

# Set up logging
logger = setup_logging("update_job_adverts")

def main():
    """Run the update job adverts process."""
    try:
        logger.info("Starting job adverts update process")
        
        # Initialize database with the correct path
        db_path = os.path.join("data", "recruitment.db")
        db = RecruitmentDatabase(db_path=db_path)
        
        # Option 1: Run the basic update that links jobs to companies on the same URL
        logger.info("Running basic job-company linking based on URL")
        db.update_all_job_advert_relations()
        
        # Option 2: Run the smarter update that analyzes job titles and company names
        logger.info("Running smarter job-company linking based on name analysis")
        db.update_all_job_company_relations()
        
        logger.info("Job adverts update completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error updating job adverts: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 