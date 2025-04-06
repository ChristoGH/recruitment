#!/usr/bin/env python3
"""
New Recruitment Database Handler

This module provides a database handler for the new recruitment database schema.
It replaces the old RecruitmentDatabase class with a new implementation that
works with the normalized schema in recruitment_new.db.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union, Tuple
from logging_config import setup_logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class NewRecruitmentDatabase:
    """
    Handler for the new recruitment database operations with logging, error handling,
    and robust connection management.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize the recruitment database handler.

        Args:
            db_path: Optional path to the database file. If None, uses RECRUITMENT_NEW_PATH from environment.
        """
        self.db_path = db_path or os.getenv("RECRUITMENT_NEW_PATH", "databases/recruitment_new.db")
        if not self.db_path:
            raise DatabaseError("Database path not set. Check RECRUITMENT_NEW_PATH environment variable.")

        # Ensure the parent directory exists.
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure a rotating file logger for database operations."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.logger = setup_logging("new_recruitment_db", log_level=logging.INFO)

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager for database connections.
        Ensures that foreign keys are enabled.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Database connection error: {e}")
        finally:
            conn.close()

    @contextmanager
    def _execute_query(self, query: str, params: tuple = ()) -> Iterator[sqlite3.Cursor]:
        """
        Helper context manager to execute a query with standardized error handling.

        Args:
            query: The SQL query to execute.
            params: Parameters for the query.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                yield cursor
        except sqlite3.Error as e:
            error_msg = f"Query failed: {query} with params {params} - {e}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager for database transactions.
        Ensures that multiple operations either all succeed or all fail.

        Yields:
            sqlite3.Connection: An active database connection

        Raises:
            DatabaseError: If there's an issue with the database connection or transaction
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
            self.logger.info("Transaction committed successfully")
        except sqlite3.Error as e:
            conn.rollback()
            error_msg = f"Transaction failed and was rolled back: {e}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
        except Exception as e:
            conn.rollback()
            error_msg = f"Unexpected error during transaction, rolled back: {e}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
        finally:
            conn.close()

    def _execute_in_transaction(self, queries_and_params: List[tuple]) -> None:
        """
        Execute multiple queries in a single transaction.

        Args:
            queries_and_params: List of (query, params) tuples to execute

        Raises:
            DatabaseError: If any query fails
        """
        if not queries_and_params:
            return

        with self._transaction() as conn:
            cursor = conn.cursor()
            for query, params in queries_and_params:
                try:
                    self.logger.info(f"Executing query: {query}")
                    self.logger.info(f"With params: {params}")
                    cursor.execute(query, params)
                    self.logger.info(f"Query execution successful")
                except Exception as e:
                    self.logger.error(f"Query execution failed: {query}")
                    self.logger.error(f"Params: {params}")
                    self.logger.error(f"Error: {e}")
                    raise

    # URL Processing Status Methods
    def update_url_processing_status(self, url_id: int, status: str, error_count: int = 0) -> None:
        """
        Update the processing status of a URL.

        Args:
            url_id: The ID of the URL
            status: The new status ('pending', 'processing', 'completed', 'failed')
            error_count: The number of errors encountered (default: 0)
        """
        query = """
            INSERT OR REPLACE INTO url_processing_status 
            (url_id, status, last_processed_at, error_count, updated_at) 
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (url_id, status, error_count)):
            self.logger.info(f"Updated URL {url_id} processing status to {status}")

    def get_url_processing_status(self, url_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the processing status of a URL.

        Args:
            url_id: The ID of the URL

        Returns:
            Dictionary with status information or None if not found
        """
        query = "SELECT * FROM url_processing_status WHERE url_id = ?"
        with self._execute_query(query, (url_id,)) as cursor:
            row = cursor.fetchone()
            if row:
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
            return None

    def get_pending_urls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get URLs that are pending processing.

        Args:
            limit: Maximum number of URLs to return

        Returns:
            List of dictionaries with URL information
        """
        query = """
            SELECT u.* FROM urls u
            LEFT JOIN url_processing_status ups ON u.id = ups.url_id
            WHERE ups.url_id IS NULL OR ups.status = 'pending'
            ORDER BY u.id
            LIMIT ?
        """
        with self._execute_query(query, (limit,)) as cursor:
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Core Entity Methods
    def insert_url(self, url: str, domain_name: str, source: str) -> int:
        """
        Insert a URL into the database.

        Args:
            url: The URL string
            domain_name: The domain name
            source: The source of the URL

        Returns:
            The ID of the inserted URL
        """
        query = """
            INSERT OR IGNORE INTO urls (url, domain_name, source, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (url, domain_name, source)) as cursor:
            if cursor.rowcount > 0:
                url_id = cursor.lastrowid
                self.logger.info(f"Inserted URL: {url} with ID: {url_id}")
                return url_id
            else:
                # URL already exists, get its ID
                query = "SELECT id FROM urls WHERE url = ?"
                with self._execute_query(query, (url,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve URL: {url}")

    def insert_agency(self, name: str) -> int:
        """
        Insert an agency into the database.

        Args:
            name: The agency name

        Returns:
            The ID of the inserted agency
        """
        query = """
            INSERT OR IGNORE INTO agencies (name, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (name,)) as cursor:
            if cursor.rowcount > 0:
                agency_id = cursor.lastrowid
                self.logger.info(f"Inserted agency: {name} with ID: {agency_id}")
                return agency_id
            else:
                # Agency already exists, get its ID
                query = "SELECT id FROM agencies WHERE name = ?"
                with self._execute_query(query, (name,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve agency: {name}")

    def insert_company(self, name: str) -> int:
        """
        Insert a company into the database.

        Args:
            name: The company name

        Returns:
            The ID of the inserted company
        """
        query = """
            INSERT OR IGNORE INTO companies (name, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (name,)) as cursor:
            if cursor.rowcount > 0:
                company_id = cursor.lastrowid
                self.logger.info(f"Inserted company: {name} with ID: {company_id}")
                return company_id
            else:
                # Company already exists, get its ID
                query = "SELECT id FROM companies WHERE name = ?"
                with self._execute_query(query, (name,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve company: {name}")

    def insert_job(self, title: str, description: Optional[str] = None, 
                  salary_min: Optional[float] = None, salary_max: Optional[float] = None,
                  salary_currency: Optional[str] = None, status: str = 'active') -> int:
        """
        Insert a job into the database.

        Args:
            title: The job title
            description: The job description (optional)
            salary_min: Minimum salary (optional)
            salary_max: Maximum salary (optional)
            salary_currency: Salary currency (optional)
            status: Job status (default: 'active')

        Returns:
            The ID of the inserted job
        """
        query = """
            INSERT INTO jobs (title, description, salary_min, salary_max, salary_currency, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (title, description, salary_min, salary_max, salary_currency, status)) as cursor:
            job_id = cursor.lastrowid
            self.logger.info(f"Inserted job: {title} with ID: {job_id}")
            return job_id

    # Related Entity Methods
    def insert_skill(self, name: str) -> int:
        """
        Insert a skill into the database.

        Args:
            name: The skill name

        Returns:
            The ID of the inserted skill
        """
        query = """
            INSERT OR IGNORE INTO skills (name, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (name,)) as cursor:
            if cursor.rowcount > 0:
                skill_id = cursor.lastrowid
                self.logger.info(f"Inserted skill: {name} with ID: {skill_id}")
                return skill_id
            else:
                # Skill already exists, get its ID
                query = "SELECT id FROM skills WHERE name = ?"
                with self._execute_query(query, (name,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve skill: {name}")

    def insert_qualification(self, name: str) -> int:
        """
        Insert a qualification into the database.

        Args:
            name: The qualification name

        Returns:
            The ID of the inserted qualification
        """
        query = """
            INSERT OR IGNORE INTO qualifications (name, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (name,)) as cursor:
            if cursor.rowcount > 0:
                qualification_id = cursor.lastrowid
                self.logger.info(f"Inserted qualification: {name} with ID: {qualification_id}")
                return qualification_id
            else:
                # Qualification already exists, get its ID
                query = "SELECT id FROM qualifications WHERE name = ?"
                with self._execute_query(query, (name,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve qualification: {name}")

    def insert_attribute(self, name: str) -> int:
        """
        Insert an attribute into the database.

        Args:
            name: The attribute name

        Returns:
            The ID of the inserted attribute
        """
        query = """
            INSERT OR IGNORE INTO attributes (name, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (name,)) as cursor:
            if cursor.rowcount > 0:
                attribute_id = cursor.lastrowid
                self.logger.info(f"Inserted attribute: {name} with ID: {attribute_id}")
                return attribute_id
            else:
                # Attribute already exists, get its ID
                query = "SELECT id FROM attributes WHERE name = ?"
                with self._execute_query(query, (name,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve attribute: {name}")

    def insert_duty(self, description: str) -> int:
        """
        Insert a duty into the database.

        Args:
            description: The duty description

        Returns:
            The ID of the inserted duty
        """
        query = """
            INSERT OR IGNORE INTO duties (description, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (description,)) as cursor:
            if cursor.rowcount > 0:
                duty_id = cursor.lastrowid
                self.logger.info(f"Inserted duty with ID: {duty_id}")
                return duty_id
            else:
                # Duty already exists, get its ID
                query = "SELECT id FROM duties WHERE description = ?"
                with self._execute_query(query, (description,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve duty: {description}")

    def insert_location(self, country: Optional[str] = None, province: Optional[str] = None,
                       city: Optional[str] = None) -> int:
        """
        Insert a location into the database.

        Args:
            country: The country name (optional)
            province: The province/state name (optional)
            city: The city name (optional)

        Returns:
            The ID of the inserted location
        """
        query = """
            INSERT OR IGNORE INTO locations (country, province, city, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (country, province, city)) as cursor:
            if cursor.rowcount > 0:
                location_id = cursor.lastrowid
                self.logger.info(f"Inserted location with ID: {location_id}")
                return location_id
            else:
                # Location already exists, get its ID
                query = "SELECT id FROM locations WHERE country = ? AND province = ? AND city = ?"
                with self._execute_query(query, (country, province, city)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve location")

    def insert_phone(self, number: str, type: str = 'mobile') -> int:
        """
        Insert a phone number into the database.

        Args:
            number: The phone number
            type: The phone type ('mobile', 'landline', 'fax')

        Returns:
            The ID of the inserted phone
        """
        query = """
            INSERT OR IGNORE INTO phones (number, type, created_at, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (number, type)) as cursor:
            if cursor.rowcount > 0:
                phone_id = cursor.lastrowid
                self.logger.info(f"Inserted phone: {number} with ID: {phone_id}")
                return phone_id
            else:
                # Phone already exists, get its ID
                query = "SELECT id FROM phones WHERE number = ? AND type = ?"
                with self._execute_query(query, (number, type)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve phone: {number}")

    def insert_email(self, email: str, url_id: Optional[int] = None, type: str = 'primary') -> int:
        """
        Insert an email into the database.

        Args:
            email: The email address
            url_id: The associated URL ID (optional)
            type: The email type ('primary', 'secondary', 'work')

        Returns:
            The ID of the inserted email
        """
        query = """
            INSERT OR IGNORE INTO emails (email, url_id, type, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (email, url_id, type)) as cursor:
            if cursor.rowcount > 0:
                email_id = cursor.lastrowid
                self.logger.info(f"Inserted email: {email} with ID: {email_id}")
                return email_id
            else:
                # Email already exists, get its ID
                query = "SELECT id FROM emails WHERE email = ?"
                with self._execute_query(query, (email,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve email: {email}")

    def insert_benefit(self, description: str) -> int:
        """
        Insert a benefit into the database.

        Args:
            description: The benefit description

        Returns:
            The ID of the inserted benefit
        """
        query = """
            INSERT OR IGNORE INTO benefits (description, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (description,)) as cursor:
            if cursor.rowcount > 0:
                benefit_id = cursor.lastrowid
                self.logger.info(f"Inserted benefit with ID: {benefit_id}")
                return benefit_id
            else:
                # Benefit already exists, get its ID
                query = "SELECT id FROM benefits WHERE description = ?"
                with self._execute_query(query, (description,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                    raise DatabaseError(f"Failed to insert or retrieve benefit: {description}")

    # Relationship Methods
    def link_job_agency(self, job_id: int, agency_id: int) -> None:
        """
        Link a job with an agency.

        Args:
            job_id: The job ID
            agency_id: The agency ID
        """
        query = """
            INSERT OR IGNORE INTO job_agencies (job_id, agency_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, agency_id)):
            self.logger.info(f"Linked job {job_id} with agency {agency_id}")

    def link_job_company(self, job_id: int, company_id: int) -> None:
        """
        Link a job with a company.

        Args:
            job_id: The job ID
            company_id: The company ID
        """
        query = """
            INSERT OR IGNORE INTO job_companies (job_id, company_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, company_id)):
            self.logger.info(f"Linked job {job_id} with company {company_id}")

    def link_job_skill(self, job_id: int, skill_id: int, experience: Optional[str] = None) -> None:
        """
        Link a job with a skill.

        Args:
            job_id: The job ID
            skill_id: The skill ID
            experience: The experience level ('entry', 'intermediate', 'senior', 'expert')
        """
        query = """
            INSERT OR IGNORE INTO job_skills (job_id, skill_id, experience, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, skill_id, experience)):
            self.logger.info(f"Linked job {job_id} with skill {skill_id}")

    def link_job_qualification(self, job_id: int, qualification_id: int) -> None:
        """
        Link a job with a qualification.

        Args:
            job_id: The job ID
            qualification_id: The qualification ID
        """
        query = """
            INSERT OR IGNORE INTO job_qualifications (job_id, qualification_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, qualification_id)):
            self.logger.info(f"Linked job {job_id} with qualification {qualification_id}")

    def link_job_attribute(self, job_id: int, attribute_id: int) -> None:
        """
        Link a job with an attribute.

        Args:
            job_id: The job ID
            attribute_id: The attribute ID
        """
        query = """
            INSERT OR IGNORE INTO job_attributes (job_id, attribute_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, attribute_id)):
            self.logger.info(f"Linked job {job_id} with attribute {attribute_id}")

    def link_job_duty(self, job_id: int, duty_id: int) -> None:
        """
        Link a job with a duty.

        Args:
            job_id: The job ID
            duty_id: The duty ID
        """
        query = """
            INSERT OR IGNORE INTO job_duties (job_id, duty_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, duty_id)):
            self.logger.info(f"Linked job {job_id} with duty {duty_id}")

    def link_job_location(self, job_id: int, location_id: int) -> None:
        """
        Link a job with a location.

        Args:
            job_id: The job ID
            location_id: The location ID
        """
        query = """
            INSERT OR IGNORE INTO job_locations (job_id, location_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, location_id)):
            self.logger.info(f"Linked job {job_id} with location {location_id}")

    def link_company_phone(self, company_id: int, phone_id: int) -> None:
        """
        Link a company with a phone number.

        Args:
            company_id: The company ID
            phone_id: The phone ID
        """
        query = """
            INSERT OR IGNORE INTO company_phones (company_id, phone_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (company_id, phone_id)):
            self.logger.info(f"Linked company {company_id} with phone {phone_id}")

    def link_agency_phone(self, agency_id: int, phone_id: int) -> None:
        """
        Link an agency with a phone number.

        Args:
            agency_id: The agency ID
            phone_id: The phone ID
        """
        query = """
            INSERT OR IGNORE INTO agency_phones (agency_id, phone_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (agency_id, phone_id)):
            self.logger.info(f"Linked agency {agency_id} with phone {phone_id}")

    def link_company_email(self, company_id: int, email_id: int) -> None:
        """
        Link a company with an email.

        Args:
            company_id: The company ID
            email_id: The email ID
        """
        query = """
            INSERT OR IGNORE INTO company_emails (company_id, email_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (company_id, email_id)):
            self.logger.info(f"Linked company {company_id} with email {email_id}")

    def link_agency_email(self, agency_id: int, email_id: int) -> None:
        """
        Link an agency with an email.

        Args:
            agency_id: The agency ID
            email_id: The email ID
        """
        query = """
            INSERT OR IGNORE INTO agency_emails (agency_id, email_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (agency_id, email_id)):
            self.logger.info(f"Linked agency {agency_id} with email {email_id}")

    def link_job_url(self, job_id: int, url_id: int) -> None:
        """
        Link a job with a URL.

        Args:
            job_id: The job ID
            url_id: The URL ID
        """
        query = """
            INSERT OR IGNORE INTO job_urls (job_id, url_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """
        with self._execute_query(query, (job_id, url_id)):
            self.logger.info(f"Linked job {job_id} with URL {url_id}")

    # Batch Processing Methods
    def process_job_data(self, url_id: int, job_data: Dict[str, Any]) -> int:
        """
        Process job data and insert it into the database.

        Args:
            url_id: The URL ID
            job_data: Dictionary containing job data

        Returns:
            The ID of the inserted job
        """
        # Extract job data
        title = job_data.get('title', '')
        description = job_data.get('description')
        salary_min = job_data.get('salary_min')
        salary_max = job_data.get('salary_max')
        salary_currency = job_data.get('salary_currency')
        status = job_data.get('status', 'active')

        # Insert job
        job_id = self.insert_job(
            title=title,
            description=description,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            status=status
        )

        # Link job with URL
        self.link_job_url(job_id, url_id)

        # Process related entities
        self._process_job_related_entities(job_id, job_data)

        return job_id

    def _process_job_related_entities(self, job_id: int, job_data: Dict[str, Any]) -> None:
        """
        Process job-related entities and create relationships.

        Args:
            job_id: The job ID
            job_data: Dictionary containing job data
        """
        # Process skills
        if 'skills' in job_data:
            for skill_data in job_data['skills']:
                skill_name = skill_data.get('skill')
                experience = skill_data.get('experience')
                if skill_name:
                    skill_id = self.insert_skill(skill_name)
                    self.link_job_skill(job_id, skill_id, experience)

        # Process qualifications
        if 'qualifications' in job_data:
            for qualification_name in job_data['qualifications']:
                if qualification_name:
                    qualification_id = self.insert_qualification(qualification_name)
                    self.link_job_qualification(job_id, qualification_id)

        # Process attributes
        if 'attributes' in job_data:
            for attribute_name in job_data['attributes']:
                if attribute_name:
                    attribute_id = self.insert_attribute(attribute_name)
                    self.link_job_attribute(job_id, attribute_id)

        # Process duties
        if 'duties' in job_data:
            for duty_description in job_data['duties']:
                if duty_description:
                    duty_id = self.insert_duty(duty_description)
                    self.link_job_duty(job_id, duty_id)

        # Process locations
        if 'locations' in job_data:
            for location_data in job_data['locations']:
                country = location_data.get('country')
                province = location_data.get('province')
                city = location_data.get('city')
                if country or province or city:
                    location_id = self.insert_location(country, province, city)
                    self.link_job_location(job_id, location_id)

        # Process benefits
        if 'benefits' in job_data:
            for benefit_description in job_data['benefits']:
                if benefit_description:
                    benefit_id = self.insert_benefit(benefit_description)
                    # Note: There's no direct link between jobs and benefits in the new schema

    def process_company_data(self, company_data: Dict[str, Any]) -> int:
        """
        Process company data and insert it into the database.

        Args:
            company_data: Dictionary containing company data

        Returns:
            The ID of the inserted company
        """
        # Extract company data
        name = company_data.get('name', '')
        if not name:
            raise ValueError("Company name is required")

        # Insert company
        company_id = self.insert_company(name)

        # Process related entities
        self._process_company_related_entities(company_id, company_data)

        return company_id

    def _process_company_related_entities(self, company_id: int, company_data: Dict[str, Any]) -> None:
        """
        Process company-related entities and create relationships.

        Args:
            company_id: The company ID
            company_data: Dictionary containing company data
        """
        # Process phones
        if 'phones' in company_data:
            for phone_data in company_data['phones']:
                number = phone_data.get('number')
                phone_type = phone_data.get('type', 'mobile')
                if number:
                    phone_id = self.insert_phone(number, phone_type)
                    self.link_company_phone(company_id, phone_id)

        # Process emails
        if 'emails' in company_data:
            for email_data in company_data['emails']:
                email = email_data.get('email')
                email_type = email_data.get('type', 'primary')
                if email:
                    email_id = self.insert_email(email, type=email_type)
                    self.link_company_email(company_id, email_id)

    def process_agency_data(self, agency_data: Dict[str, Any]) -> int:
        """
        Process agency data and insert it into the database.

        Args:
            agency_data: Dictionary containing agency data

        Returns:
            The ID of the inserted agency
        """
        # Extract agency data
        name = agency_data.get('name', '')
        if not name:
            raise ValueError("Agency name is required")

        # Insert agency
        agency_id = self.insert_agency(name)

        # Process related entities
        self._process_agency_related_entities(agency_id, agency_data)

        return agency_id

    def _process_agency_related_entities(self, agency_id: int, agency_data: Dict[str, Any]) -> None:
        """
        Process agency-related entities and create relationships.

        Args:
            agency_id: The agency ID
            agency_data: Dictionary containing agency data
        """
        # Process phones
        if 'phones' in agency_data:
            for phone_data in agency_data['phones']:
                number = phone_data.get('number')
                phone_type = phone_data.get('type', 'mobile')
                if number:
                    phone_id = self.insert_phone(number, phone_type)
                    self.link_agency_phone(agency_id, phone_id)

        # Process emails
        if 'emails' in agency_data:
            for email_data in agency_data['emails']:
                email = email_data.get('email')
                email_type = email_data.get('type', 'primary')
                if email:
                    email_id = self.insert_email(email, type=email_type)
                    self.link_agency_email(agency_id, email_id)

    # Transaction-based batch processing
    def process_url_data_in_transaction(self, url_id: int, data: Dict[str, Any]) -> None:
        """
        Process all data for a URL in a single transaction.

        Args:
            url_id: The URL ID
            data: Dictionary containing all data for the URL
        """
        operations = []

        # Update URL processing status to 'processing'
        operations.append((
            "UPDATE url_processing_status SET status = 'processing', last_processed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE url_id = ?",
            (url_id,)
        ))

        # Process job data
        if 'job' in data:
            job_data = data['job']
            title = job_data.get('title', '')
            description = job_data.get('description')
            salary_min = job_data.get('salary_min')
            salary_max = job_data.get('salary_max')
            salary_currency = job_data.get('salary_currency')
            status = job_data.get('status', 'active')

            # Insert job
            operations.append((
                "INSERT INTO jobs (title, description, salary_min, salary_max, salary_currency, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (title, description, salary_min, salary_max, salary_currency, status)
            ))

            # Get job ID (will be the last inserted ID)
            operations.append((
                "SELECT last_insert_rowid()",
                ()
            ))

            # Link job with URL
            operations.append((
                "INSERT OR IGNORE INTO job_urls (job_id, url_id, created_at) VALUES (last_insert_rowid(), ?, CURRENT_TIMESTAMP)",
                (url_id,)
            ))

            # Process skills
            if 'skills' in job_data:
                for skill_data in job_data['skills']:
                    skill_name = skill_data.get('skill')
                    experience = skill_data.get('experience')
                    if skill_name:
                        # Insert skill
                        operations.append((
                            "INSERT OR IGNORE INTO skills (name, created_at, updated_at) VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                            (skill_name,)
                        ))
                        # Get skill ID
                        operations.append((
                            "SELECT id FROM skills WHERE name = ?",
                            (skill_name,)
                        ))
                        # Link job with skill
                        operations.append((
                            "INSERT OR IGNORE INTO job_skills (job_id, skill_id, experience, created_at) VALUES (last_insert_rowid(), (SELECT id FROM skills WHERE name = ?), ?, CURRENT_TIMESTAMP)",
                            (skill_name, experience)
                        ))

            # Process other related entities similarly...

        # Update URL processing status to 'completed'
        operations.append((
            "UPDATE url_processing_status SET status = 'completed', last_processed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE url_id = ?",
            (url_id,)
        ))

        # Execute all operations in a single transaction
        self._execute_in_transaction(operations) 