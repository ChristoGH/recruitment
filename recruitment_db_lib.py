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

    def _setup_logging(self) -> None:
        """Configure a rotating file logger for database operations."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        log_file = self.log_dir / "recruitment_db.log"

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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
        self._create_phone_numbers_table()
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

    def _create_company_table(self) -> None:
        """Create tables related to hiring companies."""
        # Hiring Company table
        query_company = """
            CREATE TABLE IF NOT EXISTS hiring_company (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (name),
                UNIQUE (url_id, name),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_company):
            self.logger.info("Table 'hiring_company' created or verified.")

        # Address table
        query_address = """
            CREATE TABLE IF NOT EXISTS hiring_company_address (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                address TEXT NOT NULL,
                UNIQUE (url_id, address),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_address):
            self.logger.info("Table 'hiring_company_address' created or verified.")

        # Email table
        query_email = """
            CREATE TABLE IF NOT EXISTS hiring_company_email (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                UNIQUE (url_id, email),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_email):
            self.logger.info("Table 'hiring_company_email' created or verified.")

        # Phone table
        query_phone = """
            CREATE TABLE IF NOT EXISTS hiring_company_phone (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                phone TEXT NOT NULL,
                UNIQUE (url_id, phone),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query_phone):
            self.logger.info("Table 'hiring_company_phone' created or verified.")

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
        query = """
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                UNIQUE (url_id, email),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'emails' created or verified.")

    def _create_phone_numbers_table(self) -> None:
        """Create the 'skills' table."""
        query = """
            CREATE TABLE IF NOT EXISTS phone_numbers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                number TEXT NOT NULL,
                UNIQUE (url_id, number),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'phone_numbers' created or verified.")

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
        """Create the 'skills' table."""
        query = """
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                skill TEXT NOT NULL,
                UNIQUE (url_id, skill),
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'skills' created or verified.")

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
            self.logger.info("Table 'qualification' created or verified.")

    def _create_job_table(self) -> None:
        """Create the 'job_adverts' table."""
        query = """
            CREATE TABLE IF NOT EXISTS job_adverts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL,
                hiring_company_id INTEGER,
                recruiter_id INTEGER,
                title TEXT,
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (hiring_company_id) REFERENCES company (id) ON DELETE SET NULL ON UPDATE CASCADE,
                FOREIGN KEY (recruiter_id) REFERENCES recruiter (id) ON DELETE SET NULL ON UPDATE CASCADE
            )
        """
        with self._execute_query(query):
            self.logger.info("Table 'job_adverts' created or verified.")

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
        self.logger.debug(f"Executing search query: {query} with params: {params}")

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


    def insert_skill(self, url_id: int, skill: str) -> None:
        """
        Insert a skill record.

        Args:
            url_id: Associated URL ID.
            skill: Required skill.
        """
        query = "INSERT OR IGNORE INTO skills (url_id, skill) VALUES (?, ?)"
        with self._execute_query(query, (url_id, skill)):
            self.logger.info(f"Inserted skill '{skill}' for URL ID {url_id}")


    def insert_phone_number(self, url_id: int, number: str) -> None:
        """
        Insert a phone number record.

        Args:
            url_id: Associated URL ID.
            number: Phone number.
        """
        query = "INSERT OR IGNORE INTO phone_numbers (url_id, number) VALUES (?, ?)"
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

    def insert_skills_list(self, url_id: int, skills: list) -> None:
        """
        Insert multiple skills for a URL.

        Args:
            url_id: Associated URL ID.
            skills: List of skills.
        """
        if not skills:
            return

        for skill in skills:
            self.insert_skill(url_id, skill)


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