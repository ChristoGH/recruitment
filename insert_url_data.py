#!/usr/bin/env python3
"""
Database Insertion Script for Processed URL Data

This script takes a JSON file containing processed URL data and inserts it into
the recruitment database following the normalized schema.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sqlite3
from recruitment_db import RecruitmentDatabase, DatabaseError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_date(date_str: Optional[str]) -> Optional[str]:
    """Parse date string to database format."""
    if not date_str:
        return None
    try:
        # Assuming date string is in format YYYYMMDD_HHMMSS
        dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def safe_get(data: Dict[str, Any], *keys, default=None) -> Any:
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

def safe_get_list(data: Dict[str, Any], *keys, default=None) -> list:
    """Safely get nested dictionary values that should be lists."""
    value = safe_get(data, *keys)
    if value is None:
        return default if default is not None else []
    return value

def insert_url_data(db: RecruitmentDatabase, json_file: str) -> None:
    """
    Insert data from a processed URL JSON file into the recruitment database.
    
    Args:
        db: RecruitmentDatabase instance
        json_file: Path to the JSON file containing processed URL data
    """
    try:
        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        url = data.get('url')
        if not url:
            raise ValueError("URL not found in JSON data")
            
        # Extract domain name from URL
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Start transaction
        with db._get_connection() as conn:
            try:
                # Insert URL
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO urls (url, domain_name, source, created_at, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (url, domain, 'crawler'))
                
                # Get URL ID
                cursor.execute("SELECT id FROM urls WHERE url = ?", (url,))
                url_id = cursor.fetchone()[0]
                
                responses = data.get('responses', {})
                
                # Process job data
                job_data = responses.get('job_prompt', {})
                cursor.execute("""
                    INSERT INTO jobs (
                        title, description, salary_min, salary_max, salary_currency,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    safe_get(job_data, 'title'),
                    safe_get(job_data, 'description'),
                    safe_get(job_data, 'salary_min'),
                    safe_get(job_data, 'salary_max'),
                    safe_get(job_data, 'salary_currency'),
                    'active'
                ))
                job_id = cursor.lastrowid
                logger.info(f"Created job with ID: {job_id}")
                
                # Process job advert data
                advert_data = responses.get('jobadvert_prompt', {})
                logger.info(f"Processing job advert data: {advert_data}")
                
                # Always create an advert record for the job
                try:
                    cursor.execute("""
                        INSERT INTO adverts (
                            job_id,
                            posted_date,
                            application_deadline,
                            is_remote,
                            is_hybrid,
                            created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (
                        job_id,                                           # job_id
                        safe_get(advert_data, 'posted_date'),            # posted_date
                        safe_get(advert_data, 'application_deadline'),    # application_deadline
                        False,                                           # is_remote
                        False                                            # is_hybrid
                    ))
                    advert_id = cursor.lastrowid
                    logger.info(f"Created advert record with ID: {advert_id}")

                    # If we have a description in the advert data, update the job description
                    description = safe_get(advert_data, 'description')
                    if description:
                        cursor.execute("""
                            UPDATE jobs 
                            SET description = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (description, job_id))
                        logger.info(f"Updated job {job_id} with description from advert")

                except Exception as e:
                    logger.error(f"Error creating advert record: {str(e)}")
                    raise
                
                # Link job to URL
                cursor.execute("""
                    INSERT INTO job_urls (job_id, url_id, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (job_id, url_id))
                logger.info(f"Linked job {job_id} to URL {url_id}")
                
                # Process company
                company_data = responses.get('company_prompt', {})
                if company_data.get('company'):
                    cursor.execute("""
                        INSERT OR IGNORE INTO companies (name, created_at, updated_at)
                        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (company_data['company'],))
                    cursor.execute("SELECT id FROM companies WHERE name = ?", (company_data['company'],))
                    company_id = cursor.fetchone()[0]
                    cursor.execute("""
                        INSERT INTO job_companies (job_id, company_id, created_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (job_id, company_id))
                
                # Process agency
                agency_data = responses.get('agency_prompt', {})
                if agency_data.get('agency'):
                    cursor.execute("""
                        INSERT OR IGNORE INTO agencies (name, created_at, updated_at)
                        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (agency_data['agency'],))
                    cursor.execute("SELECT id FROM agencies WHERE name = ?", (agency_data['agency'],))
                    agency_id = cursor.fetchone()[0]
                    cursor.execute("""
                        INSERT INTO job_agencies (job_id, agency_id, created_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (job_id, agency_id))
                
                # Process skills
                skills_data = safe_get_list(responses, 'skills_prompt', 'skills', default=[])
                for skill_data in skills_data:
                    skill_name = safe_get(skill_data, 'skill')
                    if skill_name:
                        cursor.execute("""
                            INSERT OR IGNORE INTO skills (name, created_at, updated_at)
                            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (skill_name,))
                        cursor.execute("SELECT id FROM skills WHERE name = ?", (skill_name,))
                        skill_id = cursor.fetchone()[0]
                        cursor.execute("""
                            INSERT INTO job_skills (job_id, skill_id, experience, created_at)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (job_id, skill_id, safe_get(skill_data, 'experience')))
                
                # Process attributes
                attributes = safe_get_list(responses, 'attributes_prompt', 'attributes', default=[])
                for attr in attributes:
                    if attr:
                        cursor.execute("""
                            INSERT OR IGNORE INTO attributes (name, created_at, updated_at)
                            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (attr,))
                        cursor.execute("SELECT id FROM attributes WHERE name = ?", (attr,))
                        attr_id = cursor.fetchone()[0]
                        cursor.execute("""
                            INSERT INTO job_attributes (job_id, attribute_id, created_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        """, (job_id, attr_id))
                
                # Process duties
                duties = safe_get_list(responses, 'duties_prompt', 'duties', default=[])
                for duty in duties:
                    if duty:
                        cursor.execute("""
                            INSERT OR IGNORE INTO duties (description, created_at, updated_at)
                            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (duty,))
                        cursor.execute("SELECT id FROM duties WHERE description = ?", (duty,))
                        duty_id = cursor.fetchone()[0]
                        cursor.execute("""
                            INSERT INTO job_duties (job_id, duty_id, created_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        """, (job_id, duty_id))
                
                # Process location
                location_data = responses.get('location_prompt', {})
                if any([location_data.get(k) for k in ['country', 'province', 'city']]):
                    cursor.execute("""
                        INSERT INTO locations (country, province, city, created_at, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (
                        location_data.get('country'),
                        location_data.get('province'),
                        location_data.get('city')
                    ))
                    location_id = cursor.lastrowid
                    cursor.execute("""
                        INSERT INTO job_locations (job_id, location_id, created_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (job_id, location_id))
                
                # Process qualifications
                qualifications = safe_get_list(responses, 'qualifications_prompt', 'qualifications', default=[])
                for qual in qualifications:
                    if qual:
                        cursor.execute("""
                            INSERT OR IGNORE INTO qualifications (name, created_at, updated_at)
                            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (qual,))
                        cursor.execute("SELECT id FROM qualifications WHERE name = ?", (qual,))
                        qual_id = cursor.fetchone()[0]
                        cursor.execute("""
                            INSERT INTO job_qualifications (job_id, qualification_id, created_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        """, (job_id, qual_id))
                
                # Process benefits
                benefits = safe_get_list(responses, 'benefits_prompt', 'benefits', default=[])
                for benefit in benefits:
                    if benefit:
                        cursor.execute("""
                            INSERT OR IGNORE INTO benefits (description, created_at, updated_at)
                            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (benefit,))
                
                # Process industry with explicit transaction handling
                try:
                    industry_data = responses.get('industry_prompt', {})
                    logger.info(f"Industry data from responses: {industry_data}")
                    
                    # Get industry name directly from the dictionary
                    industry_name = industry_data.get('industry')
                    logger.info(f"Extracted industry name: {industry_name}")
                    
                    if industry_name:
                        logger.info(f"Attempting to insert industry: {industry_name}")
                        
                        # First, ensure the industries table exists
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS industries (
                                id INTEGER PRIMARY KEY,
                                name TEXT NOT NULL UNIQUE,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                        
                        # Insert the industry
                        cursor.execute("""
                            INSERT OR IGNORE INTO industries (name)
                            VALUES (?)
                        """, (industry_name,))
                        
                        # Get the industry ID
                        cursor.execute("SELECT id FROM industries WHERE name = ?", (industry_name,))
                        result = cursor.fetchone()
                        if result:
                            industry_id = result[0]
                            logger.info(f"Industry ID retrieved: {industry_id}")
                            
                            # Ensure the job_industries table exists
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS job_industries (
                                    job_id INTEGER NOT NULL,
                                    industry_id INTEGER NOT NULL,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    PRIMARY KEY (job_id, industry_id),
                                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                                    FOREIGN KEY (industry_id) REFERENCES industries(id) ON DELETE CASCADE
                                )
                            """)
                            
                            # Link job with industry
                            cursor.execute("""
                                INSERT OR IGNORE INTO job_industries (job_id, industry_id)
                                VALUES (?, ?)
                            """, (job_id, industry_id))
                            
                            logger.info(f"Linked job {job_id} with industry {industry_id}")
                        else:
                            logger.error("Failed to retrieve industry ID after insertion")
                    else:
                        logger.warning("No industry name found to process")
                        
                except Exception as e:
                    logger.error(f"Error processing industry: {str(e)}")
                    raise
                
                # Update URL processing status
                cursor.execute("""
                    INSERT OR REPLACE INTO url_processing_status 
                    (url_id, status, last_processed_at, created_at, updated_at)
                    VALUES (?, 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (url_id,))
                
                conn.commit()
                logger.info(f"Successfully inserted data for URL: {url}")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting data: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Error processing file {json_file}: {e}")
        raise

def main():
    """Main function to process command line arguments and insert data."""
    if len(sys.argv) != 2:
        print("Usage: python insert_url_data.py <json_file>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found")
        sys.exit(1)
        
    try:
        db = RecruitmentDatabase()
        insert_url_data(db, json_file)
        print("Data insertion completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
