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
                        INSERT INTO url_links
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

    def insert_job_advert(self, url_id: int, title: str) -> None:
        """
        Insert a job advert record.

        Args:
            url_id: Associated URL ID.
            title: Job advert title.
        """
        query = "INSERT INTO job_adverts (url_id, title) VALUES (?, ?)"
        with self._execute_query(query, (url_id, title)):
            self.logger.info(f"Inserted job advert for URL ID {url_id}")


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
        query = """
            INSERT OR IGNORE INTO job_advert_forms 
            (url_id, description, salary, duration, start_date, end_date, posted_date, application_deadline) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._execute_query(query, (url_id, description, salary, duration,
                                         start_date, end_date, posted_date, application_deadline)):
            self.logger.info(f"Inserted job advert details for URL ID {url_id}")


    def insert_recruitment_evidence(self, url_id: int, evidence: str) -> None:
        """
        Insert recruitment evidence record.

        Args:
            url_id: Associated URL ID.
            evidence: Evidence text.
        """
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
    # Example refactored method using transactions
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
        if not any([country, province, city, street_address]):
            return

        query = """
            INSERT OR IGNORE INTO location 
            (url_id, country, province, city, street_address) 
            VALUES (?, ?, ?, ?, ?)
        """

        # Execute as a single operation
        with self._execute_query(query, (url_id, country, province, city, street_address)):
            self.logger.info(f"Inserted location information for URL ID {url_id}")

    # Example of a method handling multiple related inserts in a transaction
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

    # Refactored version of process_all_prompt_responses with transaction support
    def process_prompt_responses_in_transaction(self, url_id: int, responses: dict) -> None:
        """
        Process multiple prompt responses in a single transaction.
        Focus on correctly handling skills formatted as tuples.
        """
        operations = []

        # Build operations list based on response types
        for response_type, data in responses.items():
            if response_type == "recruitment_prompt" and data.get("answer") == "yes":
                operations.append((
                    "UPDATE urls SET recruitment_flag = 1 WHERE id = ?",
                    (url_id,)
                ))

                # Add evidence if provided
                evidence = data.get("evidence", [])
                for item in evidence:
                    operations.append((
                        "INSERT OR IGNORE INTO recruitment_evidence (url_id, evidence) VALUES (?, ?)",
                        (url_id, item)
                    ))

            elif response_type == "recruitment_prompt" and data.get("answer") == "no":
                operations.append((
                    "UPDATE urls SET recruitment_flag = 0 WHERE id = ?",
                    (url_id,)
                ))

            elif response_type == "company_prompt" and data.get("company"):
                operations.append((
                    "INSERT OR IGNORE INTO company (url_id, name) VALUES (?, ?)",
                    (url_id, data["company"])
                ))

            elif response_type == "agency_prompt" and data.get("agency"):
                operations.append((
                    "INSERT OR IGNORE INTO agency (url_id, agency) VALUES (?, ?)",
                    (url_id, data["agency"])
                ))

            elif response_type == "job_prompt" and data.get("title"):
                operations.append((
                    "INSERT OR IGNORE INTO job_adverts (url_id, title) VALUES (?, ?)",
                    (url_id, data["title"])
                ))

            elif response_type == "company_phone_number_prompt" and data.get("number"):
                operations.append((
                    "INSERT OR IGNORE INTO company_phone (url_id, phone) VALUES (?, ?)",
                    (url_id, data["number"])
                ))

            elif response_type == "email_prompt" and data.get("email"):
                operations.append((
                    "INSERT OR IGNORE INTO email (url_id, email) VALUES (?, ?)",
                    (url_id, data["email"])
                ))

            elif response_type == "link_prompt" and data.get("link"):
                operations.append((
                    "INSERT OR IGNORE INTO links (url_id, link) VALUES (?, ?)",
                    (url_id, data["link"])
                ))

            elif response_type == "benefits_prompt" and data.get("benefits"):
                for benefit in data["benefits"]:
                    operations.append((
                        "INSERT OR IGNORE INTO benefits (url_id, benefit) VALUES (?, ?)",
                        (url_id, benefit)
                    ))

            # CRITICAL FIX FOR SKILLS
            # CRITICAL FIX FOR SKILLS - ADD THIS CODE
            elif response_type == "skills_prompt" and data.get("skills"):
                skills_data = data["skills"]
                self.logger.info(f"Processing skills data: {skills_data}, type: {type(skills_data)}")

                # Handle null skills
                if skills_data is None:
                    self.logger.warning(f"Skills data is None for URL ID {url_id}")
                    continue

                # Handle skills list
                if isinstance(skills_data, list):
                    for skill_item in skills_data:
                        try:
                            # Handle tuple format
                            if isinstance(skill_item, tuple):
                                if len(skill_item) >= 2:
                                    skill, experience = skill_item
                                    # Convert "not_listed" to None
                                    if experience == "not_listed":
                                        experience = None

                                    self.logger.info(f"Adding skill from tuple: ({skill}, {experience})")
                                    operations.append((
                                        "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)",
                                        (url_id, skill, experience)
                                    ))

                            # Handle SkillExperience objects
                            elif hasattr(skill_item, 'skill'):
                                skill = skill_item.skill
                                experience = getattr(skill_item, 'experience', None)
                                # Convert "not_listed" to None
                                if experience == "not_listed":
                                    experience = None

                                self.logger.info(f"Adding skill from object: ({skill}, {experience})")
                                operations.append((
                                    "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)",
                                    (url_id, skill, experience)
                                ))

                            # Handle dictionary format
                            elif isinstance(skill_item, dict) and 'skill' in skill_item:
                                skill = skill_item['skill']
                                experience = skill_item.get('experience')
                                # Convert "not_listed" to None
                                if experience == "not_listed":
                                    experience = None

                                self.logger.info(f"Adding skill from dict: ({skill}, {experience})")
                                operations.append((
                                    "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)",
                                    (url_id, skill, experience)
                                ))

                            # Handle string items
                            elif isinstance(skill_item, str):
                                self.logger.info(f"Adding skill from string: ({skill_item}, None)")
                                operations.append((
                                    "INSERT OR IGNORE INTO skills (url_id, skill, experience) VALUES (?, ?, ?)",
                                    (url_id, skill_item, None)
                                ))
                        except Exception as e:
                            self.logger.error(f"Error processing skill item {skill_item}: {e}")

                    skill_ops_count = sum(1 for op in operations if "INSERT OR IGNORE INTO skills" in op[0])
                    self.logger.info(f"Added {skill_ops_count} skill operations for URL ID {url_id}")

            elif response_type == "attributes_prompt" and data.get("attributes"):
                for attribute in data["attributes"]:
                    operations.append((
                        "INSERT OR IGNORE INTO attributes (url_id, attribute) VALUES (?, ?)",
                        (url_id, attribute)
                    ))

            elif response_type == "contacts_prompt" and data.get("contacts"):
                for contact in data["contacts"]:
                    operations.append((
                        "INSERT OR IGNORE INTO contact_person (url_id, name) VALUES (?, ?)",
                        (url_id, contact)
                    ))

            elif response_type == "location_prompt":
                if any([data.get("country"), data.get("province"), data.get("city"), data.get("street_address")]):
                    operations.append((
                        """INSERT OR IGNORE INTO location 
                           (url_id, country, province, city, street_address) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (url_id, data.get("country"), data.get("province"),
                         data.get("city"), data.get("street_address"))
                    ))

            elif response_type == "jobadvert_prompt":
                operations.append((
                    """INSERT OR IGNORE INTO job_advert_forms 
                       (url_id, description, salary, duration, start_date, end_date, posted_date, application_deadline) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (url_id, data.get("description"), data.get("salary"), data.get("duration"),
                     data.get("start_date"), data.get("end_date"), data.get("posted_date"),
                     data.get("application_deadline"))
                ))

            elif response_type == "qualifications_prompt" and data.get("qualifications"):
                for qualification in data["qualifications"]:
                    operations.append((
                        "INSERT OR IGNORE INTO qualifications (url_id, qualification) VALUES (?, ?)",
                        (url_id, qualification)
                    ))

            elif response_type == "duties_prompt" and data.get("duties"):
                for duty in data["duties"]:
                    operations.append((
                        "INSERT OR IGNORE INTO duties (url_id, duty) VALUES (?, ?)",
                        (url_id, duty)
                    ))

        # Execute all operations in a single transaction
        if operations:
            # Log skill operations specifically for debugging
            skill_ops = [op for op in operations if op[0].startswith('INSERT OR IGNORE INTO skills')]
            if skill_ops:
                self.logger.info(f"Will execute {len(skill_ops)} skill insert operations in transaction")
                for i, (query, params) in enumerate(skill_ops):
                    self.logger.info(f"Skill operation {i + 1}: {params}")

            self._execute_in_transaction(operations)
            self.logger.info(f"Processed {len(responses)} prompt responses for URL ID {url_id}")

            # Verify skills were inserted
            query = "SELECT COUNT(*) FROM skills WHERE url_id = ?"
            with self._execute_query(query, (url_id,)) as cursor:
                count = cursor.fetchone()[0]
                self.logger.info(f"After transaction: {count} skills exist for URL ID {url_id}")

                if count > 0:
                    query = "SELECT skill, experience FROM skills WHERE url_id = ?"
                    with self._execute_query(query, (url_id,)) as cursor:
                        results = cursor.fetchall()
                        for skill, exp in results:
                            self.logger.info(f"Inserted skill: '{skill}' with experience: '{exp}'")
        else:
            self.logger.warning(f"No database operations generated for responses: {list(responses.keys())}")

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

    # 3. Enhanced logging for _execute_in_transaction() in recruitment_db_lib.py
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