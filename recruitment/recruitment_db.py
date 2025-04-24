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
import threading

# Load environment variables
load_dotenv()

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class RecruitmentDatabase:
    """
    Handler for the new recruitment database operations with logging, error handling,
    and robust connection management.
    """

    def __init__(self, db_path: str = "databases/recruitment.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.logger = setup_logging("recruitment_db")
        self._setup_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise DatabaseError(f"Failed to connect to database: {str(e)}")

    def _execute_query(self, query: str, params: tuple = (), commit: bool = True) -> sqlite3.Cursor:
        """Execute a database query with proper error handling."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
            return cursor
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise DatabaseError(f"Database error: {str(e)}")
        finally:
            if conn and not commit:  # Only close if we're not committing (for transactions)
                conn.close()

    def _setup_database(self):
        """Set up the database tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS urls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE,
                        domain_name TEXT,
                        source TEXT,
                        processing_status TEXT DEFAULT 'pending',
                        error_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS url_processing_status (
                        url_id INTEGER PRIMARY KEY,
                        status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
                        last_processed_at TIMESTAMP,
                        error_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (url_id) REFERENCES urls(id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        description TEXT,
                        posted_date TEXT,
                        type TEXT,
                        location TEXT,
                        url_id INTEGER,
                        FOREIGN KEY (url_id) REFERENCES urls (id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS skills (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_skills (
                        job_id INTEGER,
                        skill_id INTEGER,
                        FOREIGN KEY (job_id) REFERENCES jobs (id),
                        FOREIGN KEY (skill_id) REFERENCES skills (id),
                        PRIMARY KEY (job_id, skill_id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS qualifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_qualifications (
                        job_id INTEGER,
                        qualification_id INTEGER,
                        FOREIGN KEY (job_id) REFERENCES jobs (id),
                        FOREIGN KEY (qualification_id) REFERENCES qualifications (id),
                        PRIMARY KEY (job_id, qualification_id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attributes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_attributes (
                        job_id INTEGER,
                        attribute_id INTEGER,
                        FOREIGN KEY (job_id) REFERENCES jobs (id),
                        FOREIGN KEY (attribute_id) REFERENCES attributes (id),
                        PRIMARY KEY (job_id, attribute_id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS duties (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        description TEXT UNIQUE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_duties (
                        job_id INTEGER,
                        duty_id INTEGER,
                        FOREIGN KEY (job_id) REFERENCES jobs (id),
                        FOREIGN KEY (duty_id) REFERENCES duties (id),
                        PRIMARY KEY (job_id, duty_id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS benefits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        description TEXT UNIQUE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_benefits (
                        job_id INTEGER,
                        benefit_id INTEGER,
                        FOREIGN KEY (job_id) REFERENCES jobs (id),
                        FOREIGN KEY (benefit_id) REFERENCES benefits (id),
                        PRIMARY KEY (job_id, benefit_id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS companies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        website TEXT,
                        UNIQUE(name, website)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agencies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        website TEXT,
                        UNIQUE(name, website)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS locations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        city TEXT,
                        province TEXT,
                        country TEXT,
                        UNIQUE(city, province, country)
                    )
                """)

                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to set up database: {str(e)}")
            raise DatabaseError(f"Failed to set up database: {str(e)}")

    def insert_url(self, url: str, domain: str, source: str) -> int:
        """Insert a URL into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO urls (url, domain_name, source) VALUES (?, ?, ?)",
                        (url, domain, source)
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    cursor.execute(
                        "SELECT id FROM urls WHERE url = ?",
                        (url,)
                    )
                    return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert URL: {str(e)}")

    def update_url_processing_status(self, url_id: int, status: str, error_count: int = 0) -> None:
        """Update the processing status of a URL."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE urls SET processing_status = ?, error_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, error_count, url_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update URL status: {str(e)}")

    def get_unprocessed_urls(self) -> List[Dict[str, str]]:
        """Get unprocessed URLs from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, url, domain, source FROM urls WHERE processing_status = 'pending'"
                )
                return [
                    {
                        "id": row[0],
                        "url": row[1],
                        "domain": row[2],
                        "source": row[3]
                    }
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get unprocessed URLs: {str(e)}")

    def insert_job(self, title: str, description: str = None, posted_date: str = None,
                  type: str = None, location: str = None, url_id: int = None) -> int:
        """Insert a job into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO jobs (title, description, posted_date, type, location, url_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (title, description, posted_date, type, location, url_id)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert job: {str(e)}")

    def insert_advert(self, job_id: int, posted_date: str = None, application_deadline: str = None,
                     is_remote: bool = False, is_hybrid: bool = False) -> int:
        """Insert an advert into the database."""
        cursor = self._execute_query(
            """INSERT INTO adverts (job_id, posted_date, application_deadline, is_remote, is_hybrid)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, posted_date, application_deadline, is_remote, is_hybrid)
        )
        return cursor.lastrowid

    def insert_skill(self, name: str) -> int:
        """Insert a skill into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO skills (name) VALUES (?)",
                        (name,)
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    cursor.execute(
                        "SELECT id FROM skills WHERE name = ?",
                        (name,)
                    )
                    return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert skill: {str(e)}")

    def link_job_skill(self, job_id: int, skill_id: int) -> None:
        """Link a job to a skill."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO job_skills (job_id, skill_id) VALUES (?, ?)",
                    (job_id, skill_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to link job and skill: {str(e)}")

    def insert_qualification(self, name: str) -> int:
        """Insert a qualification into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO qualifications (name) VALUES (?)",
                        (name,)
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    cursor.execute(
                        "SELECT id FROM qualifications WHERE name = ?",
                        (name,)
                    )
                    return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert qualification: {str(e)}")

    def link_job_qualification(self, job_id: int, qualification_id: int) -> None:
        """Link a job to a qualification."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO job_qualifications (job_id, qualification_id) VALUES (?, ?)",
                    (job_id, qualification_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to link job and qualification: {str(e)}")

    def insert_attribute(self, name: str) -> int:
        """Insert an attribute into the database."""
        try:
            cursor = self._execute_query(
                "INSERT INTO attributes (name) VALUES (?)",
                (name,)
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor = self._execute_query(
                "SELECT id FROM attributes WHERE name = ?",
                (name,)
            )
            return cursor.fetchone()[0]

    def link_job_attribute(self, job_id: int, attribute_id: int) -> None:
        """Link a job to an attribute."""
        self._execute_query(
            "INSERT INTO job_attributes (job_id, attribute_id) VALUES (?, ?)",
            (job_id, attribute_id)
        )

    def insert_duty(self, description: str) -> int:
        """Insert a duty into the database."""
        try:
            cursor = self._execute_query(
                "INSERT INTO duties (description) VALUES (?)",
                (description,)
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor = self._execute_query(
                "SELECT id FROM duties WHERE description = ?",
                (description,)
            )
            return cursor.fetchone()[0]

    def link_job_duty(self, job_id: int, duty_id: int) -> None:
        """Link a job to a duty."""
        self._execute_query(
            "INSERT INTO job_duties (job_id, duty_id) VALUES (?, ?)",
            (job_id, duty_id)
        )

    def insert_benefit(self, name: str) -> int:
        """Insert a benefit into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO benefits (name) VALUES (?)",
                        (name,)
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    cursor.execute(
                        "SELECT id FROM benefits WHERE name = ?",
                        (name,)
                    )
                    return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert benefit: {str(e)}")

    def link_job_benefit(self, job_id: int, benefit_id: int) -> None:
        """Link a job to a benefit."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO job_benefits (job_id, benefit_id) VALUES (?, ?)",
                    (job_id, benefit_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to link job and benefit: {str(e)}")

    def insert_company(self, name: str, website: str = None) -> int:
        """Insert a company into the database."""
        try:
            cursor = self._execute_query(
                "INSERT INTO companies (name, website) VALUES (?, ?)",
                (name, website)
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor = self._execute_query(
                "SELECT id FROM companies WHERE name = ? AND website = ?",
                (name, website)
            )
            return cursor.fetchone()[0]

    def insert_agency(self, name: str, website: str = None) -> int:
        """Insert an agency into the database."""
        try:
            cursor = self._execute_query(
                "INSERT INTO agencies (name, website) VALUES (?, ?)",
                (name, website)
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor = self._execute_query(
                "SELECT id FROM agencies WHERE name = ? AND website = ?",
                (name, website)
            )
            return cursor.fetchone()[0]

    def insert_location(self, city: str, province: str, country: str) -> int:
        """Insert a location into the database."""
        try:
            cursor = self._execute_query(
                "INSERT INTO locations (city, province, country) VALUES (?, ?, ?)",
                (city, province, country)
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor = self._execute_query(
                "SELECT id FROM locations WHERE city = ? AND province = ? AND country = ?",
                (city, province, country)
            )
            return cursor.fetchone()[0]

    def link_job_location(self, job_id: int, location_id: int) -> None:
        """Link a job to a location."""
        self._execute_query(
            "INSERT INTO job_locations (job_id, location_id) VALUES (?, ?)",
            (job_id, location_id)
        )

    def insert_industry(self, name: str) -> int:
        """Insert an industry into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO industries (name) VALUES (?)",
                        (name,)
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    cursor.execute(
                        "SELECT id FROM industries WHERE name = ?",
                        (name,)
                    )
                    return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert industry: {str(e)}")

    def link_job_industry(self, job_id: int, industry_id: int) -> None:
        """Link a job to an industry."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO job_industries (job_id, industry_id) VALUES (?, ?)",
                    (job_id, industry_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to link job and industry: {str(e)}") 