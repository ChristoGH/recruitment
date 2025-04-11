# recruitment_db_lib.py

import logging
import logging.handlers
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Union
from logging_config import setup_logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class LinkRecord(BaseModel):
    """Model for link records."""
    url_id: int
    link_url: str
    link_text: Optional[str] = None
    link_type: str
    source_page: Optional[str] = None

@dataclass
class URLRecord:
    """Data class representing a URL record used as the base for job postings."""
    url: str
    domain_name: str
    source: str
    id: Optional[int] = None
    extracted_date: Optional[str] = None
    content: Optional[str] = None
    recruitment_flag: Optional[int] = 0  # formerly actual_incident/actual_flag
    author: Optional[str] = None
    published_date: Optional[str] = None
    title: Optional[str] = None
    error_message: Optional[str] = None


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class RecruitmentDatabase:
    """
    Handler for recruitment database operations with logging, error handling,
    and robust connection management.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize the recruitment database handler.

        Args:
            db_path: Optional path to the database file. If None, uses RECRUITMENT_PATH from environment.
        """
        self.db_path = db_path or os.getenv("RECRUITMENT_PATH")
        if not self.db_path:
            raise DatabaseError("Database path not set. Check RECRUITMENT_PATH environment variable.")

        # Ensure the parent directory exists.
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self._initialize_schema()

    def test_skill_insertion(self, url_id: int):
        """Test direct skill insertion with experience data using multiple scenarios."""
        test_cases = [
            {"skill": "Test Skill 1", "experience": "5 years"},
            {"skill": "Test Skill 2", "experience": None},
            {"skill": "Test Skill 3", "experience": "Entry level"},
            {"skill": "Test Skill 3", "experience": "Advanced level"},  # Same skill, different experience
        ]

        success_count = 0
        for test_case in test_cases:
            try:
                query = "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
                with self._execute_query(query, (url_id, test_case["skill"], test_case["experience"])) as cursor:
                    if cursor.rowcount > 0:
                        self.logger.info(
                            f"Successfully inserted skill '{test_case['skill']}' with experience '{test_case['experience']}'")
                        success_count += 1
                    else:
                        self.logger.warning(
                            f"Skill '{test_case['skill']}' with experience '{test_case['experience']}' not inserted (possibly duplicate)")

                # Verify insertion
                verify_query = "SELECT skill, experience FROM skills WHERE url_id = ? AND skill = ? AND (experience = ? OR (experience IS NULL AND ? IS NULL))"
                with self._execute_query(verify_query, (
                url_id, test_case["skill"], test_case["experience"], test_case["experience"])) as cursor:
                    result = cursor.fetchone()
                    if result:
                        self.logger.info(f"Retrieved skill: {result[0]}, experience: {result[1]}")
                    else:
                        self.logger.warning(
                            f"Could not retrieve skill '{test_case['skill']}' with experience '{test_case['experience']}'")
            except Exception as e:
                self.logger.error(f"Test skill insertion failed for '{test_case['skill']}': {e}")

        self.logger.info(f"Test skill insertion: {success_count} out of {len(test_cases)} successful insertions")

    def check_skills_table_schema(self):
        """Verify that the skills table has the necessary columns."""
        query = "PRAGMA table_info(skills)"
        with self._execute_query(query) as cursor:
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            self.logger.info(f"Skills table columns: {column_names}")
            if 'experience' not in column_names:
                self.logger.warning("Skills table is missing the 'experience' column!")

    # Create module-specific logger

    def _setup_logging(self) -> None:
        """Configure a rotating file logger for database operations."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.logger = setup_logging("recruitment_db_lib", log_level=logging.INFO)


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

    def _initialize_schema(self) -> None:
        """Create (or upgrade) all tables for the recruitment database."""
        self._create_urls_table()
        self._create_company_table()
        self._create_recruiter_table()
        self._create_agency_table()
        self._create_contact_person_table()
        self._create_benefits_table()
        self._create_skills_table()
        self._create_attributes_table()
        self._create_job_table()
        self._create_job_advert_table()
        self._create_links_table()
        self._create_emails_table()
        self._create_company_phone_numbers_table()
        self._create_location_table()
        self._create_duties_table()
        self._create_url_links()
        self._create_qualifications_table()
        self._create_recruitment_evidence_table()


    def _create_urls_table(self) -> None:
        """Create the base 'urls' table."""
        query = """
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                extracted_date TEXT,
                content TEXT,
                domain_name TEXT,
                source TEXT,
                recruitment_flag INTEGER DEFAULT -1,
                accessible INTEGER DEFAULT 1,
                author TEXT,
                published_date TEXT,
                title TEXT,
                error_message TEXT
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'urls' created or verified.")

    # Fix for _create_company_table method
    def _create_company_table(self) -> None:
        """Create tables related to hiring companies."""
        # Change table name from hiring_company to company
        query_company = """
            CREATE TABLE IF NOT EXISTS company (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (name),
                UNIQUE (url_id, name),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_company):
            self.logger.info("Table 'company' created or verified.")

        # Update related table names to be consistent
        query_address = """
            CREATE TABLE IF NOT EXISTS company_address (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                address TEXT NOT NULL,
                UNIQUE (url_id, address),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_address):
            self.logger.info("Table 'company_address' created or verified.")

        # Update other company-related tables similarly
        query_email = """
            CREATE TABLE IF NOT EXISTS company_email (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                UNIQUE (url_id, email),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_email):
            self.logger.info("Table 'company_email' created or verified.")

        # Update phone table
        query_phone = """
            CREATE TABLE IF NOT EXISTS company_phone (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                phone TEXT NOT NULL,
                UNIQUE (url_id, phone),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_phone):
            self.logger.info("Table 'company_phone' created or verified.")

    # Fix for _create_job_table method to use correct table references
    def _create_job_table(self) -> None:
        """Create the 'job_adverts' table with correct foreign key references."""
        query = """
            CREATE TABLE IF NOT EXISTS job_adverts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                company_id INTEGER, 
                recruiter_id INTEGER,
                title TEXT,
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (company_id) REFERENCES company (id) ON DELETE SET NULL ON UPDATE CASCADE,
                FOREIGN KEY (recruiter_id) REFERENCES recruiter (id) ON DELETE SET NULL ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'job_adverts' created or verified.")


    def _create_agency_table(self) -> None:
        """Create the 'agency' table."""
        query = """
            CREATE TABLE IF NOT EXISTS agency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                agency TEXT NOT NULL,
                UNIQUE (agency, url_id),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'agency' created or verified.")

    # Fix 4: Add the missing update_field_by_id method
    def update_field_by_id(self, url_id: int, field: str, value: Any) -> None:
        """
        Update a specific field for a URL by its ID in the database.

        Args:
            url_id: The URL ID to update.
            field: The field name.
            value: The new value.
        """
        valid_fields = {"content", "domain_name", "source", "extracted_date",
                        "recruitment_flag", "author", "published_date", "title", "error_message"}
        if field not in valid_fields:
            raise ValueError(f"Invalid field name: {field}. Valid fields: {', '.join(valid_fields)}")

        query = f"UPDATE urls SET {field} = ? WHERE id = ?"
        with self._execute_query(query, (value, url_id)) as cursor:
            if cursor.rowcount == 0:
                self.logger.warning(f"No record found for URL ID: {url_id}")
            else:
                self.logger.info(f"Updated field '{field}' for URL ID: {url_id}")

    def _create_links_table(self) -> None:
        """Create the 'links' table."""
        query = """
            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                link TEXT NOT NULL,
                UNIQUE (url_id, link),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'links' created or verified.")

    def _create_emails_table(self) -> None:
        """Create the 'emails' table."""
        # Update other company-related tables similarly
        query_email = """
            CREATE TABLE IF NOT EXISTS email (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                UNIQUE (url_id, email),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_email):
            self.logger.info("Table 'company_email' created or verified.")

    def _create_company_phone_numbers_table(self) -> None:
        """Create the 'skills' table."""
        query_phone = """
            CREATE TABLE IF NOT EXISTS company_phone (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                phone TEXT NOT NULL,
                UNIQUE (url_id, phone),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_phone):
            self.logger.info("Table 'company_phone' created or verified.")

    def _create_contact_person_table(self) -> None:
        """Create the 'contact_person' table."""
        query = """CREATE TABLE IF NOT EXISTS contact_person (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    UNIQUE (url_id, name),
    FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
)

        """
        with self._execute_query(query):
            self.logger.info("Table 'contact_person' created or verified.")

    def _create_duties_table(self) -> None:
        """Create the 'duties' table."""
        query = """
            CREATE TABLE IF NOT EXISTS duties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                duty TEXT NOT NULL,
                UNIQUE (url_id, duty),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'duties' created or verified.")

    def _create_benefits_table(self) -> None:
        """Create the 'benefits' table."""
        query = """
            CREATE TABLE IF NOT EXISTS benefits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                benefit TEXT NOT NULL,
                UNIQUE (url_id, benefit),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'benefits' created or verified.")

    def _create_skills_table(self) -> None:
        """
        Create the 'skills' table with columns for both skill name and experience.
        This updated table structure supports storing multiple experience levels for the same skill.
        """
        query = """
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                skill TEXT NOT NULL,
                experience TEXT,
                UNIQUE (url_id, skill, experience),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'skills' created or verified with experience column and proper constraints.")

    def _create_location_table(self) -> None:
        """Create the 'location' table."""
        query = """CREATE TABLE IF NOT EXISTS location (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_id INTEGER NOT NULL,
    country TEXT,
    province TEXT,
    city TEXT,
    street_address TEXT,
    UNIQUE (url_id),
    FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
)

        """
        with self._execute_query(query):
            self.logger.info("Table 'location' created or verified.")

    def _create_attributes_table(self) -> None:
        """Create the 'attributes' table."""
        query = """
            CREATE TABLE IF NOT EXISTS attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                attribute TEXT NOT NULL,
                UNIQUE (url_id, attribute),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'attributes' created or verified.")


    def _create_url_links(self):
        """Create the 'url_links' table to store links extracted from webpages."""
        # Create the table first
        table_query = """
            CREATE TABLE IF NOT EXISTS url_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                link_url TEXT NOT NULL,
                link_text TEXT,
                link_type TEXT NOT NULL,
                source_page TEXT,
                UNIQUE (url_id, link_url, link_type),
                FOREIGN KEY (url_id) REFERENCES urls(id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(table_query):
            self.logger.info("Table 'url_links' created or verified.")

        # Create the indexes separately
        index_query1 = """
            CREATE INDEX IF NOT EXISTS idx_url_links_url_id ON url_links(url_id)
        """
        with self._execute_query(index_query1):
            self.logger.info("Index 'idx_url_links_url_id' created or verified.")

        index_query2 = """
            CREATE INDEX IF NOT EXISTS idx_url_links_link_type ON url_links(link_type)
        """
        with self._execute_query(index_query2):
            self.logger.info("Index 'idx_url_links_link_type' created or verified.")

    def _create_job_advert_table(self) -> None:
        """Create the 'job_advert_forms' table to store job posting details."""
        query = """
            CREATE TABLE IF NOT EXISTS job_advert_forms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                job_advert_id INTEGER,
                description TEXT,
                salary TEXT,
                duration TEXT,
                start_date TEXT,
                end_date TEXT,
                posted_date TEXT,
                application_deadline TEXT,
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (job_advert_id) REFERENCES job_adverts (id) ON DELETE SET NULL ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'job_advert_forms' created or verified.")

    def get_column_names(self, table_name: str) -> List[str]:
        """
        Retrieve all column names from a specified table.

        Args:
            table_name: The name of the table.
        """
        query = f"PRAGMA table_info({table_name})"
        with self._execute_query(query) as cursor:
            rows = cursor.fetchall()
            columns = [row[1] for row in rows]
            if not columns:
                error_msg = f"Table '{table_name}' does not exist or has no columns."
                self.logger.error(error_msg)
                raise DatabaseError(error_msg)
            return columns

    def update_field(self, url: str, field: str, value: Any) -> None:
        """
        Update a specific field for a URL in the database.

        Args:
            url: The URL to update.
            field: The field name.
            value: The new value.
        """
        valid_fields = {"content", "domain_name", "source", "extracted_date",
                        "recruitment_flag", "author", "published_date", "title"}
        if field not in valid_fields:
            raise ValueError(f"Invalid field name: {field}. Valid fields: {', '.join(valid_fields)}")

        query = f"UPDATE urls SET {field} = ? WHERE url = ?"
        with self._execute_query(query, (value, url)) as cursor:
            if cursor.rowcount == 0:
                self.logger.warning(f"No record found for URL: {url}")
            else:
                self.logger.info(f"Updated field '{field}' for URL: {url}")

    def insert_url(self, record: Dict[str, Any]) -> None:
        """
        Insert a URL record into the database.

        The record should include:
          - 'url'
          - 'domain_name'
          - 'source'
          - 'recruitment_flag'
          - optionally, 'accessible' and 'content'
        """
        try:
            url_record = URLRecord(
                url=record["url"],
                domain_name=record["domain_name"],
                source=record["source"],
                extracted_date=datetime.now().isoformat(),
                recruitment_flag=record["recruitment_flag"]
            )
        except KeyError as e:
            error_msg = f"Missing required field in record: {e}"
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        query = """
            INSERT OR IGNORE INTO urls
            (url, domain_name, source, extracted_date, recruitment_flag, accessible, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._execute_query(query, (
                url_record.url,
                url_record.domain_name,
                url_record.source,
                url_record.extracted_date,
                url_record.recruitment_flag,
                record.get("accessible", 1),
                record.get("content")
        )):
            self.logger.info(f"Inserted URL: {url_record.url}")

    def insert_url_links(self, url_id: int, links: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Insert links extracted from a webpage into the database.

        Args:
            url_id: The ID of the URL in the database
            links: Dictionary of links organized by type (internal/external)
        """
        try:
            for link_type, link_list in links.items():
                for link in link_list:
                    link_record = LinkRecord(
                        url_id=url_id,
                        link_url=link.get('href', ''),
                        link_text=link.get('text'),
                        link_type=link_type,
                        source_page=link.get('source_page')
                    )

                    query = """
                        INSERT OR IGNORE INTO url_links
                        (url_id, link_url, link_text, link_type, source_page)
                        VALUES (?, ?, ?, ?, ?)
                    """
                    with self._execute_query(query, (
                            link_record.url_id,
                            link_record.link_url,
                            link_record.link_text,
                            link_record.link_type,
                            link_record.source_page
                    )):
                        pass  # The context manager handles commits and rollbacks

            self.logger.info(
                f"Inserted {sum(len(link_list) for link_list in links.values())} links for URL ID: {url_id}")
        except Exception as e:
            self.logger.error(f"Error inserting links for URL ID {url_id}: {e}")
            raise DatabaseError(f"Error inserting links: {e}")

    def update_content(self, url: str, content: str) -> None:
        """
        Update the content field for a specific URL.

        Args:
            url: The URL whose content is updated.
            content: The new content.
        """
        query = "UPDATE urls SET content = ? WHERE url = ?"
        with self._execute_query(query, (content, url)) as cursor:
            if cursor.rowcount == 0:
                self.logger.warning(f"No record found for URL: {url}")
            else:
                self.logger.info(f"Updated content for URL: {url}")

    def get_url_by_id(self, url_id: int) -> Optional[URLRecord]:
        """
        Retrieve a URL record by its ID.

        Args:
            url_id: The URL record ID.
        """
        query = """
            SELECT id, url, extracted_date, content, domain_name, source,
                   author, recruitment_flag, published_date, title, error_message
            FROM urls
            WHERE id = ?
        """
        with self._execute_query(query, (url_id,)) as cursor:
            row = cursor.fetchone()
            if row:
                return URLRecord(
                    id=row[0],
                    url=row[1],
                    extracted_date=row[2],
                    content=row[3],
                    domain_name=row[4],
                    source=row[5],
                    author=row[6],
                    recruitment_flag=row[7],
                    published_date=row[8],
                    title=row[9],
                    error_message=row[10]
                )
            return None

    def search_urls(
            self,
            domain: Optional[str] = None,
            source: Optional[str] = None,
            date_from: Optional[Union[str, datetime]] = None,
            date_to: Optional[Union[str, datetime]] = None,
            limit: int = 100
    ) -> List[URLRecord]:
        """
        Search URLs based on given criteria.

        Args:
            domain: Filter by domain.
            source: Filter by source.
            date_from: Start date.
            date_to: End date.
            limit: Maximum records returned.
        """
        query_parts = ["SELECT * FROM urls WHERE 1=1"]
        params: List[Any] = []

        if domain:
            query_parts.append("AND domain_name LIKE ?")
            params.append(f"%{domain}%")
        if source:
            query_parts.append("AND source = ?")
            params.append(source)
        if date_from:
            if isinstance(date_from, datetime):
                date_from = date_from.isoformat()
            query_parts.append("AND extracted_date >= ?")
            params.append(date_from)
        if date_to:
            if isinstance(date_to, datetime):
                date_to = date_to.isoformat()
            query_parts.append("AND extracted_date <= ?")
            params.append(date_to)

        query_parts.append("ORDER BY extracted_date DESC LIMIT ?")
        params.append(limit)
        query = " ".join(query_parts)
        self.logger.info(f"Executing search query: {query} with params: {params}")

        with self._execute_query(query, tuple(params)) as cursor:
            rows = cursor.fetchall()
            return [
                URLRecord(
                    id=row[0],
                    url=row[1],
                    extracted_date=row[2],
                    content=row[3],
                    domain_name=row[4],
                    source=row[5],
                    author=row[6],
                    recruitment_flag=row[7],
                    published_date=row[8],
                    title=row[9],
                    error_message=row[10]
                )
                for row in rows
            ]

    def get_recent_urls(self, days: int = 7, limit: int = 50) -> List[URLRecord]:
        """
        Retrieve URLs extracted within the past given number of days.

        Args:
            days: Number of days to look back.
            limit: Maximum records returned.
        """
        date_from = (datetime.now() - timedelta(days=days)).isoformat()
        return self.search_urls(date_from=date_from, limit=limit)

    def get_urls_by_domain(self, domain: str) -> List[URLRecord]:
        """
        Retrieve all URLs for a specific domain.

        Args:
            domain: The domain name.
        """
        return self.search_urls(domain=domain)

    def insert_jobskill(self, url_id: int, skill: str) -> None:
        """
        Insert a job skill record.

        Args:
            url_id: Associated URL ID.
            skill: Job skill name.
        """
        query = "INSERT INTO jobskills (url_id, name) VALUES (?, ?)"
        with self._execute_query(query, (url_id, skill)):
            self.logger.info(f"Inserted job skill for URL ID {url_id}")

    def insert_qualification(self, url_id: int, qualification: str) -> None:
        """
        Insert a qualification record.

        Args:
            url_id: Associated URL ID.
            qualification: Job qualification.
        """
        query = "INSERT INTO qualifications (url_id, qualification) VALUES (?, ?)"
        with self._execute_query(query, (url_id, qualification)):
            self.logger.info(f"Inserted qualification for URL ID {url_id}")


    def insert_candidate(self, url_id: int, name: str) -> None:
        """
        Insert a candidate record.

        Args:
            url_id: Associated URL ID.
            name: Candidate name.
        """
        query = "INSERT INTO candidates (url_id, name) VALUES (?, ?)"
        with self._execute_query(query, (url_id, name)):
            self.logger.info(f"Inserted candidate for URL ID {url_id}")

    def get_job_advert_id(self, url_id: int, title: str) -> Optional[int]:
        """
        Retrieve the job advert ID for a given URL and title.

        Args:
            url_id: URL record ID.
            title: Job advert title.
        """
        query = "SELECT id FROM job_adverts WHERE url_id = ? AND title = ?"
        with self._execute_query(query, (url_id, title)) as cursor:
            row = cursor.fetchone()
            return row[0] if row else None

    def get_candidate_id(self, url_id: int, name: str) -> Optional[int]:
        """
        Retrieve the candidate ID for a given URL and candidate name.

        Args:
            url_id: URL record ID.
            name: Candidate name.
        """
        query = "SELECT id FROM candidates WHERE url_id = ? AND name = ?"
        with self._execute_query(query, (url_id, name)) as cursor:
            row = cursor.fetchone()
            return row[0] if row else None

    def insert_job_advert(self, url_id: int, title: str, company_id: Optional[int] = None, recruiter_id: Optional[int] = None) -> None:
        """
        Insert a job advert record with company and recruiter information.

        Args:
            url_id: Associated URL ID.
            title: Job advert title.
            company_id: ID of the company offering the job (optional).
            recruiter_id: ID of the recruiter for the job (optional).
        """
        query = "INSERT INTO job_adverts (url_id, title, company_id, recruiter_id) VALUES (?, ?, ?, ?)"
        with self._execute_query(query, (url_id, title, company_id, recruiter_id)):
            self.logger.info(f"Inserted job advert for URL ID {url_id} with company_id {company_id} and recruiter_id {recruiter_id}")

    def update_job_advert_relations(self, job_advert_id: int, company_id: Optional[int] = None, recruiter_id: Optional[int] = None) -> None:
        """
        Update company_id and recruiter_id for an existing job advert.

        Args:
            job_advert_id: ID of the job advert to update.
            company_id: ID of the company to associate with the job advert (optional).
            recruiter_id: ID of the recruiter to associate with the job advert (optional).
        """
        # Build the query dynamically based on which IDs are provided
        set_clauses = []
        params = []
        
        if company_id is not None:
            set_clauses.append("company_id = ?")
            params.append(company_id)
        
        if recruiter_id is not None:
            set_clauses.append("recruiter_id = ?")
            params.append(recruiter_id)
        
        if not set_clauses:
            self.logger.warning(f"No company_id or recruiter_id provided to update job advert {job_advert_id}")
            return
        
        query = f"UPDATE job_adverts SET {', '.join(set_clauses)} WHERE id = ?"
        params.append(job_advert_id)
        
        with self._execute_query(query, tuple(params)):
            self.logger.info(f"Updated relations for job advert ID {job_advert_id}: company_id={company_id}, recruiter_id={recruiter_id}")

    def get_url_id(self, url: str) -> Optional[int]:
        """
        Retrieve the ID of a URL given the URL string.

        Args:
            url: The URL string.
        """
        query = "SELECT id FROM urls WHERE url = ?"
        with self._execute_query(query, (url,)) as cursor:
            row = cursor.fetchone()
            return row[0] if row else None

    def _create_recruiter_table(self) -> None:
        """Create the 'recruiter' table for recruitment agencies."""
        query = """
            CREATE TABLE IF NOT EXISTS recruiter (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (url_id, name),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'recruiter' created or verified.")


# Insert methods for all the different prompt types


    def insert_agency(self, url_id: int, agency_name: str) -> None:
        """
        Insert a recruitment agency record.

        Args:
            url_id: Associated URL ID.
            agency_name: Agency name.
        """
        query = "INSERT OR IGNORE INTO agency (url_id, agency) VALUES (?, ?)"
        with self._execute_query(query, (url_id, agency_name)):
            self.logger.info(f"Inserted agency '{agency_name}' for URL ID {url_id}")


    def insert_job_title(self, url_id: int, title: str) -> None:
        """
        Insert a job title record.

        Args:
            url_id: Associated URL ID.
            title: Job title.
        """
        query = "INSERT OR IGNORE INTO job_adverts (url_id, title) VALUES (?, ?)"
        with self._execute_query(query, (url_id, title)):
            self.logger.info(f"Inserted job title '{title}' for URL ID {url_id}")

    def insert_skill(self, url_id: int, skill: str, experience: str = None) -> None:
        """
        Insert a skill record with optional experience information.

        Args:
            url_id: Associated URL ID.
            skill: Required skill.
            experience: Experience requirement for the skill (optional).
        """
        query = "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
        with self._execute_query(query, (url_id, skill, experience)):
            self.logger.info(f"Inserted skill '{skill}' with experience '{experience}' for URL ID {url_id}")

    def insert_company_phone_number(self, url_id: int, number: str) -> None:
        """
        Insert a phone number record.

        Args:
            url_id: Associated URL ID.
            number: Phone number.
        """
        query = "INSERT OR IGNORE INTO company_phone (url_id, number) VALUES (?, ?)"
        with self._execute_query(query, (url_id, number)):
            self.logger.info(f"Inserted phone number for URL ID {url_id}")


    def insert_email(self, url_id: int, email: str) -> None:
        """
        Insert an email record.

        Args:
            url_id: Associated URL ID.
            email: Email address.
        """
        query = "INSERT OR IGNORE INTO emails (url_id, email) VALUES (?, ?)"
        with self._execute_query(query, (url_id, email)):
            self.logger.info(f"Inserted email for URL ID {url_id}")


    def insert_link(self, url_id: int, link: str) -> None:
        """
        Insert a link record.

        Args:
            url_id: Associated URL ID.
            link: URL link.
        """
        query = "INSERT OR IGNORE INTO links (url_id, link) VALUES (?, ?)"
        with self._execute_query(query, (url_id, link)):
            self.logger.info(f"Inserted link for URL ID {url_id}")


    def insert_benefit(self, url_id: int, benefit: str) -> None:
        """
        Insert a benefit record.

        Args:
            url_id: Associated URL ID.
            benefit: Job benefit.
        """
        query = "INSERT OR IGNORE INTO benefits (url_id, benefit) VALUES (?, ?)"
        with self._execute_query(query, (url_id, benefit)):
            self.logger.info(f"Inserted benefit '{benefit}' for URL ID {url_id}")

    def insert_duty(self, url_id: int, duty: str) -> None:
        """
        Insert a duty record.

        Args:
            url_id: Associated URL ID.
            duty: Job duty.
        """
        query = "INSERT OR IGNORE INTO duties (url_id, duty) VALUES (?, ?)"
        with self._execute_query(query, (url_id, duty)):
            self.logger.info(f"Inserted duty '{duty}' for URL ID {url_id}")


    def insert_attribute(self, url_id: int, attribute: str) -> None:
        """
        Insert an attribute record.

        Args:
            url_id: Associated URL ID.
            attribute: Required attribute.
        """
        query = "INSERT OR IGNORE INTO attributes (url_id, attribute) VALUES (?, ?)"
        with self._execute_query(query, (url_id, attribute)):
            self.logger.info(f"Inserted attribute '{attribute}' for URL ID {url_id}")


    def insert_contact_person(self, url_id: int, name: str) -> None:
        """
        Insert a contact person record.

        Args:
            url_id: Associated URL ID.
            name: Contact person name.
        """
        query = "INSERT OR IGNORE INTO contact_person (url_id, name) VALUES (?, ?)"
        with self._execute_query(query, (url_id, name)):
            self.logger.info(f"Inserted contact person '{name}' for URL ID {url_id}")


    def insert_location(self, url_id: int, country: str = None, province: str = None,
                        city: str = None, street_address: str = None) -> None:
        """
        Insert a location record.

        Args:
            url_id: Associated URL ID.
            country: Country name (optional).
            province: Province/state name (optional).
            city: City name (optional).
            street_address: Street address (optional).
        """
        query = """
            INSERT OR IGNORE INTO location 
            (url_id, country, province, city, street_address) 
            VALUES (?, ?, ?, ?, ?)
        """
        with self._execute_query(query, (url_id, country, province, city, street_address)):
            self.logger.info(f"Inserted location information for URL ID {url_id}")


    def insert_job_advert_details(self, url_id: int, description: str = None, salary: str = None,
                                 duration: str = None, start_date: str = None, end_date: str = None,
                                 posted_date: str = None, application_deadline: str = None) -> None:
        """
        Insert job advertisement details.

        Args:
            url_id: Associated URL ID.
            description: Job description (optional).
            salary: Salary information (optional).
            duration: Job duration (optional).
            start_date: Start date (optional).
            end_date: End date (optional).
            posted_date: When the job was posted (optional).
            application_deadline: Application deadline (optional).
        """
        # First, check if there's a corresponding job_advert for this url_id
        query_job_advert = "SELECT id FROM job_adverts WHERE url_id = ? LIMIT 1"
        job_advert_id = None
        
        try:
            with self._execute_query(query_job_advert, (url_id,)) as cursor:
                row = cursor.fetchone()
                if row:
                    job_advert_id = row[0]
                    self.logger.info(f"Found job_advert_id {job_advert_id} for URL ID {url_id}")
                else:
                    # Instead of proceeding with NULL job_advert_id, we should insert a new job_advert record
                    insert_job_advert_query = "INSERT INTO job_adverts (url_id) VALUES (?)"
                    with self._execute_query(insert_job_advert_query, (url_id,)) as cursor:
                        job_advert_id = cursor.lastrowid
                        self.logger.info(f"Created new job_advert with ID {job_advert_id} for URL ID {url_id}")
        except Exception as e:
            self.logger.error(f"Error finding or creating job_advert_id for URL ID {url_id}: {e}")
            return  # Exit early if we can't establish the job_advert_id
        
        # Now insert the job advert details with the job_advert_id
        query = """
            INSERT OR IGNORE INTO job_advert_forms 
            (url_id, job_advert_id, description, salary, duration, start_date, end_date, posted_date, application_deadline) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._execute_query(query, (
            url_id, 
            job_advert_id,
            description, 
            salary, 
            duration,
            start_date, 
            end_date, 
            posted_date, 
            application_deadline
        )):
            self.logger.info(f"Inserted job advert details for URL ID {url_id} with job_advert_id {job_advert_id}")


    def _create_recruitment_evidence_table(self) -> None:
        """Create the 'recruitment_evidence' table."""
        query = """
            CREATE TABLE IF NOT EXISTS recruitment_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                evidence TEXT NOT NULL,
                UNIQUE (url_id, evidence),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'recruitment_evidence' created or verified.")
            
    def insert_recruitment_evidence(self, url_id: int, evidence: str) -> None:
        """
        Insert recruitment evidence record.

        Args:
            url_id: Associated URL ID.
            evidence: Evidence text.
        """
        # Make sure the table exists
        self._create_recruitment_evidence_table()
        
        # Insert the evidence
        insert_query = "INSERT OR IGNORE INTO recruitment_evidence (url_id, evidence) VALUES (?, ?)"
        with self._execute_query(insert_query, (url_id, evidence)):
            self.logger.info(f"Inserted recruitment evidence for URL ID {url_id}")


    # Methods to handle bulk inserts for list-returning prompts

    def insert_skills_list(self, url_id: int, skills_data: list) -> None:
        """
        Insert multiple skills with their associated experience requirements for a URL.
        Handles skills in tuple, list, dict, or string format.

        Args:
            url_id: Associated URL ID.
            skills_data: List of skill items in various formats:
                - Tuples: [(skill1, exp1), (skill2, exp2), (skill3, None)]
                - Lists from converted tuples: [["skill1", "exp1"], ["skill2", "exp2"]]
                - SkillExperience objects from Pydantic model
                - Strings (for skills without experience data)
                - Dictionaries with skill and experience keys
        """
        if not skills_data:
            self.logger.warning(f"No skills data provided for URL ID {url_id}")
            return

        self.logger.info(f"Inserting {len(skills_data)} skills for URL ID {url_id}")

        # For debugging purposes
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Raw skills data: {skills_data}")

        operations = []
        successful_skills = []
        failed_skills = []

        for i, skill_item in enumerate(skills_data):
            try:
                skill = None
                experience = None

                # Handle different input formats
                if isinstance(skill_item, tuple):
                    # Handle tuple format (skill, experience)
                    if len(skill_item) >= 2:
                        skill, experience = skill_item
                    else:
                        skill, experience = skill_item[0], None

                    self.logger.debug(f"Processed tuple format: ({skill}, {experience})")

                elif isinstance(skill_item, list):
                    # Handle list format (converted from tuple)
                    if len(skill_item) >= 2:
                        skill, experience = skill_item[0], skill_item[1]
                    else:
                        skill, experience = skill_item[0], None

                    self.logger.debug(f"Processed list format: {skill_item} -> ({skill}, {experience})")

                elif isinstance(skill_item, dict) and 'skill' in skill_item:
                    # Handle dictionary format
                    skill = skill_item['skill']
                    experience = skill_item.get('experience')

                    self.logger.debug(f"Processed dict format: {skill_item} -> ({skill}, {experience})")

                elif hasattr(skill_item, 'skill') and hasattr(skill_item, 'experience'):
                    # Handle SkillExperience objects from Pydantic model
                    skill = skill_item.skill
                    experience = skill_item.experience

                    self.logger.debug(f"Processed object format with attributes -> ({skill}, {experience})")

                elif hasattr(skill_item, 'model_dump'):
                    # Handle full Pydantic model object
                    data = skill_item.model_dump()
                    skill = data.get('skill')
                    experience = data.get('experience')

                    self.logger.debug(f"Processed model_dump format: {data} -> ({skill}, {experience})")

                elif isinstance(skill_item, str):
                    # Handle string format (for skills without experience data)
                    skill = skill_item
                    experience = None

                    self.logger.debug(f"Processed string format: {skill_item} -> ({skill}, None)")

                else:
                    # Try to convert to string as last resort
                    self.logger.warning(
                        f"Unrecognized skill format at index {i}: {type(skill_item).__name__} - {skill_item}")
                    try:
                        skill = str(skill_item)
                        experience = None
                        self.logger.debug(f"Converted unknown format to string: {skill}")
                    except:
                        failed_skills.append((str(skill_item), "Unrecognized format"))
                        continue

                # Ensure skill is not empty
                if not skill or not isinstance(skill, str) or not skill.strip():
                    self.logger.warning(f"Skipping empty skill at index {i}")
                    failed_skills.append(("", "Empty skill"))
                    continue

                # Normalize "not_listed" to None
                if experience == "not_listed" or (isinstance(experience, str) and not experience.strip()):
                    experience = None

                # Clean the skill name
                skill = skill.strip()

                # Add operation to insert the skill
                operations.append((
                    "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)",
                    (url_id, skill, experience)
                ))

                successful_skills.append((skill, experience))

            except Exception as e:
                self.logger.error(f"Error processing skill at index {i}: {e}")
                failed_skills.append((str(skill_item), str(e)))

        # Execute all operations in a single transaction for efficiency
        if operations:
            try:
                self._execute_in_transaction(operations)
                self.logger.info(f"Successfully inserted {len(operations)} skills for URL ID {url_id}")

                # Verify if skills were actually inserted
                query = f"SELECT COUNT(*) FROM skills WHERE url_id = ?"
                with self._execute_query(query, (url_id,)) as cursor:
                    count = cursor.fetchone()[0]
                    self.logger.info(f"Verification: {count} skills now exist for URL ID {url_id}")

            except Exception as e:
                self.logger.error(f"Error in transaction when inserting skills for URL ID {url_id}: {e}")
                self.logger.error(f"Failed skills: {failed_skills}")
                raise
        else:
            self.logger.warning(f"No valid skills to insert for URL ID {url_id}")

    def insert_attributes_list(self, url_id: int, attributes: list) -> None:
        """
        Insert multiple attributes for a URL.

        Args:
            url_id: Associated URL ID.
            attributes: List of attributes.
        """
        if not attributes:
            return

        for attribute in attributes:
            self.insert_attribute(url_id, attribute)


    def insert_contact_persons_list(self, url_id: int, contacts: list) -> None:
        """
        Insert multiple contact persons for a URL.

        Args:
            url_id: Associated URL ID.
            contacts: List of contact persons.
        """
        if not contacts:
            return

        for contact in contacts:
            self.insert_contact_person(url_id, contact)


    def insert_benefits_list(self, url_id: int, benefits: list) -> None:
        """
        Insert multiple benefits for a URL.

        Args:
            url_id: Associated URL ID.
            benefits: List of benefits.
        """
        if not benefits:
            return

        for benefit in benefits:
            self.insert_benefit(url_id, benefit)

    def insert_duties_list(self, url_id: int, duties: list) -> None:
        """
        Insert multiple duties for a URL.

        Args:
            url_id: Associated URL ID.
            duties: List of duties.
        """
        if not duties:
            return

        for duty in duties:
            self.insert_duty(url_id, duty)

    def insert_recruitment_evidence_list(self, url_id: int, evidence_list: list) -> None:
        """
        Insert multiple evidence items for recruitment flagging.

        Args:
            url_id: Associated URL ID.
            evidence_list: List of evidence items.
        """
        if not evidence_list:
            return

        for evidence in evidence_list:
            self.insert_recruitment_evidence(url_id, evidence)

    # Fix _create_qualifications_table to use consistent table name in comment
    def _create_qualifications_table(self) -> None:
        """Create the 'qualifications' table."""
        query = """
            CREATE TABLE IF NOT EXISTS qualifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                qualification TEXT NOT NULL,
                UNIQUE (url_id, qualification),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'qualifications' created or verified.")


    def insert_qualifications_list(self, url_id: int, qualifications: list) -> None:
        """
        Insert multiple qualifications for a URL.

        Args:
            url_id: Associated URL ID.
            qualifications: List of qualifications.
        """
        if not qualifications:
            return

        for qualification in qualifications:
            self.insert_qualification(url_id, qualification)


    # Ensure that company-related methods have consistent naming
    def insert_company(self, url_id: int, company_name: str) -> None:
        """
        Insert a company record.

        Args:
            url_id: Associated URL ID.
            company_name: Company name.
        """
        query = "INSERT OR IGNORE INTO company (url_id, name) VALUES (?, ?)"
        with self._execute_query(query, (url_id, company_name)):
            self.logger.info(f"Inserted company '{company_name}' for URL ID {url_id}")


    # Add missing method to get or create a company ID
    def get_company_id(self, url_id: int, name: str) -> Optional[int]:
        """
        Retrieve the company ID for a given URL and company name.

        Args:
            url_id: URL record ID.
            name: Company name.
        """
        query = "SELECT id FROM company WHERE url_id = ? AND name = ?"
        with self._execute_query(query, (url_id, name)) as cursor:
            row = cursor.fetchone()
            if row:
                return row[0]

            # If company doesn't exist, create it
            insert_query = "INSERT INTO company (url_id, name) VALUES (?, ?)"
            with self._execute_query(insert_query, (url_id, name)) as cursor:
                return cursor.lastrowid

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
        if not queries_and_params:
            return

        with self._transaction() as conn:
            cursor = conn.cursor()
            for i, (query, params) in enumerate(queries_and_params):
                self.logger.info(f"Execute data: {(i, query, params)}")
                try:
                    self.logger.info(f"Executing query ({i + 1}/{len(queries_and_params)}): {query}")
                    self.logger.info(f"With params: {params}")
                    cursor.execute(query, params)
                    self.logger.info(f"Query execution successful")
                except Exception as e:
                    self.logger.error(f"Failed to execute query ({i + 1}/{len(queries_and_params)}): {query}")
                    self.logger.error(f"With params: {params}")
                    self.logger.error(f"Error: {e}")
                    raise

    def insert_job_with_details(self, url_id: int, job_title: str,
                                company_name: Optional[str] = None,
                                description: Optional[str] = None,
                                salary: Optional[str] = None) -> None:
        """
        Insert a job posting with its details in a single transaction.

        Args:
            url_id: Associated URL ID.
            job_title: Job title.
            company_name: Company name (optional).
            description: Job description (optional).
            salary: Salary information (optional).
        """
        operations = []

        # Add job title insert operation
        operations.append((
            "INSERT OR IGNORE INTO job_adverts (url_id, title) VALUES (?, ?)",
            (url_id, job_title)
        ))

        # Add company name if provided
        if company_name:
            operations.append((
                "INSERT OR IGNORE INTO company (url_id, name) VALUES (?, ?)",
                (url_id, company_name)
            ))

            # If we have both title and company, link them
            operations.append((
                """UPDATE job_adverts SET company_id = 
                   (SELECT id FROM company WHERE url_id = ? AND name = ?) 
                   WHERE url_id = ? AND title = ?""",
                (url_id, company_name, url_id, job_title)
            ))

        # Add job details if provided
        if description or salary:
            operations.append((
                """INSERT OR IGNORE INTO job_advert_forms 
                   (url_id, job_advert_id, description, salary) 
                   VALUES (?, 
                          (SELECT id FROM job_adverts WHERE url_id = ? AND title = ?), 
                          ?, ?)""",
                (url_id, url_id, job_title, description, salary)
            ))

        # Execute all operations in a single transaction
        self._execute_in_transaction(operations)
        self.logger.info(f"Inserted job '{job_title}' with related details for URL ID {url_id}")


    def check_skills_table_constraints(self):
        """Check the unique constraints on the skills table."""
        query = "SELECT sql FROM sqlite_master WHERE type='table' AND name='skills'"
        with self._execute_query(query) as cursor:
            result = cursor.fetchone()
            if result:
                self.logger.info(f"Skills table definition: {result[0]}")
                # Check for UNIQUE constraint
                constraint_info = result[0]
                if "UNIQUE" in constraint_info:
                    unique_constraint = constraint_info[constraint_info.find("UNIQUE"):]
                    unique_constraint = unique_constraint[:unique_constraint.find(")") + 1]
                    self.logger.info(f"Unique constraint: {unique_constraint}")
                else:
                    self.logger.info("No UNIQUE constraint found on skills table")

    def get_company_id_by_url(self, url_id: int) -> Optional[int]:
        """
        Get the company ID for a given URL ID.
        
        Args:
            url_id: The URL ID to search.
            
        Returns:
            The company ID if found, None otherwise.
        """
        try:
            query = "SELECT id FROM company WHERE url_id = ? LIMIT 1"
            with self._execute_query(query, (url_id,)) as cursor:
                row = cursor.fetchone()
                if row:
                    company_id = row[0]
                    self.logger.info(f"Found company ID {company_id} for URL ID {url_id}")
                    return company_id
                else:
                    self.logger.info(f"No company found for URL ID {url_id}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting company ID for URL ID {url_id}: {e}")
            return None

    def get_recruiter_id_by_url(self, url_id: int) -> Optional[int]:
        """
        Get the recruiter ID for a given URL ID.
        
        Args:
            url_id: The URL ID to search.
            
        Returns:
            The recruiter ID if found, None otherwise.
        """
        try:
            # First check recruiter table
            query = "SELECT id FROM recruiter WHERE url_id = ? LIMIT 1"
            with self._execute_query(query, (url_id,)) as cursor:
                row = cursor.fetchone()
                if row:
                    recruiter_id = row[0]
                    self.logger.info(f"Found recruiter ID {recruiter_id} for URL ID {url_id}")
                    return recruiter_id
                
            # If no recruiter found, check agency table and create a recruiter record if needed
            query = "SELECT id, agency FROM agency WHERE url_id = ? LIMIT 1"
            with self._execute_query(query, (url_id,)) as cursor:
                row = cursor.fetchone()
                if row:
                    agency_id = row[0]
                    agency_name = row[1]
                    self.logger.info(f"Found agency {agency_name} (ID: {agency_id}) for URL ID {url_id}")
                    
                    # Insert into recruiter table and get ID
                    insert_query = "INSERT OR IGNORE INTO recruiter (url_id, name) VALUES (?, ?)"
                    with self._execute_query(insert_query, (url_id, agency_name)) as cursor:
                        # Get the ID of the inserted or existing recruiter
                        select_query = "SELECT id FROM recruiter WHERE url_id = ? AND name = ?"
                        with self._execute_query(select_query, (url_id, agency_name)) as cursor2:
                            row2 = cursor2.fetchone()
                            if row2:
                                recruiter_id = row2[0]
                                self.logger.info(f"Created/found recruiter ID {recruiter_id} based on agency for URL ID {url_id}")
                                return recruiter_id
            
            self.logger.info(f"No recruiter or agency found for URL ID {url_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting recruiter ID for URL ID {url_id}: {e}")
            return None

    def update_all_job_advert_relations(self) -> None:
        """
        Update all existing job adverts to link them with company and recruiter records.
        This is a utility method to fix existing data.
        """
        try:
            # Get all job adverts without company_id or recruiter_id
            query = """
                SELECT ja.id, ja.url_id 
                FROM job_adverts ja 
                WHERE ja.company_id IS NULL OR ja.recruiter_id IS NULL
            """
            
            with self._execute_query(query) as cursor:
                job_adverts = cursor.fetchall()
                self.logger.info(f"Found {len(job_adverts)} job adverts that need company or recruiter relations")
                
                for job_advert_id, url_id in job_adverts:
                    # Get company ID for this URL
                    company_id = self.get_company_id_by_url(url_id)
                    
                    # Get recruiter ID for this URL
                    recruiter_id = self.get_recruiter_id_by_url(url_id)
                    
                    # Update job advert relations if we found any IDs
                    if company_id is not None or recruiter_id is not None:
                        self.update_job_advert_relations(job_advert_id, company_id, recruiter_id)
                        self.logger.info(f"Updated job advert ID {job_advert_id} with company_id {company_id} and recruiter_id {recruiter_id}")
                    
            self.logger.info("Completed updating job advert relations")
        except Exception as e:
            self.logger.error(f"Error updating job advert relations: {e}")
            raise

    def link_job_advert_to_company(self, job_advert_id: int, company_name: Optional[str] = None) -> None:
        """
        Link a job advert to its company by name, not just by URL.
        
        Args:
            job_advert_id: ID of the job advert
            company_name: Name of the company to link with (optional)
        """
        try:
            # If company name provided, find or create company record
            if company_name:
                # First get the url_id for this job advert
                query_url_id = "SELECT url_id FROM job_adverts WHERE id = ?"
                with self._execute_query(query_url_id, (job_advert_id,)) as cursor:
                    row = cursor.fetchone()
                    if row:
                        url_id = row[0]
                        
                        # Now find or create company record for this name
                        company_query = "SELECT id FROM company WHERE name = ? LIMIT 1"
                        with self._execute_query(company_query, (company_name,)) as cursor:
                            company_row = cursor.fetchone()
                            
                            # If company exists, use its ID
                            if company_row:
                                company_id = company_row[0]
                                self.logger.info(f"Found existing company '{company_name}' with ID {company_id}")
                            else:
                                # Create new company record
                                insert_query = "INSERT INTO company (url_id, name) VALUES (?, ?)"
                                with self._execute_query(insert_query, (url_id, company_name)) as cursor:
                                    company_id = cursor.lastrowid
                                    self.logger.info(f"Created new company '{company_name}' with ID {company_id}")
                            
                            # Link job advert to company
                            update_query = "UPDATE job_adverts SET company_id = ? WHERE id = ?"
                            with self._execute_query(update_query, (company_id, job_advert_id)) as cursor:
                                self.logger.info(f"Linked job advert ID {job_advert_id} with company ID {company_id}")
                    else:
                        self.logger.error(f"Job advert ID {job_advert_id} not found")
        except Exception as e:
            self.logger.error(f"Error linking job advert to company: {e}")

    def update_all_job_company_relations(self) -> None:
        """
        Update job-company relationships by analyzing job and company data.
        This is a smarter approach compared to update_all_job_advert_relations.
        """
        try:
            # Get all job adverts that have no company_id
            job_query = """
                SELECT ja.id, ja.url_id, ja.title 
                FROM job_adverts ja 
                WHERE ja.company_id IS NULL
            """
            
            with self._execute_query(job_query) as cursor:
                jobs = cursor.fetchall()
                self.logger.info(f"Found {len(jobs)} job adverts without company relations")
                
                for job_id, url_id, job_title in jobs:
                    # Get company names for this URL
                    company_query = "SELECT id, name FROM company WHERE url_id = ?"
                    with self._execute_query(company_query, (url_id,)) as cursor:
                        companies = cursor.fetchall()
                        
                        if not companies:
                            self.logger.info(f"No companies found for URL ID {url_id}, job ID {job_id}")
                            continue
                        
                        # If only one company, use it
                        if len(companies) == 1:
                            company_id, company_name = companies[0]
                            self.update_job_advert_relations(job_id, company_id, None)
                            self.logger.info(f"Linked job ID {job_id} '{job_title}' with company '{company_name}' (ID: {company_id})")
                        else:
                            # Multiple companies - try to match by looking for company name in job title
                            matched = False
                            for company_id, company_name in companies:
                                # Simple check - is company name in job title?
                                if job_title and company_name and company_name.lower() in job_title.lower():
                                    self.update_job_advert_relations(job_id, company_id, None)
                                    self.logger.info(f"Matched job '{job_title}' with company '{company_name}' based on title match")
                                    matched = True
                                    break
                            
                            if not matched:
                                # Use the first company as fallback
                                company_id = companies[0][0]
                                self.update_job_advert_relations(job_id, company_id, None)
                                self.logger.info(f"Linked job ID {job_id} with first available company ID {company_id} (fallback)")
                
                self.logger.info("Completed updating job-company relations")
        except Exception as e:
            self.logger.error(f"Error updating job-company relations: {e}")
            raise