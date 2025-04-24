"""
Test module for URL processing.
"""
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from recruitment.url_processing_service import (
    process_url,
    extract_job_details,
    extract_skills,
    extract_qualifications,
    extract_attributes,
    extract_duties,
    extract_benefits,
    extract_company_details,
    extract_agency_details,
    extract_location_details,
    extract_industries,
)
from recruitment.recruitment_db import DatabaseError, RecruitmentDatabase

@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn

@pytest.fixture
def db():
    """Create a RecruitmentDatabase instance with a mocked connection."""
    with patch('sqlite3.connect') as mock_connect:
        mock_connect.return_value = MagicMock()
        db = RecruitmentDatabase()
        yield db

def test_process_url(db, mock_db_connection):
    """Test processing a URL."""
    # Mock the connection
    db.conn = mock_db_connection
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Mock the URL content
    url = "http://example.com/job"
    content = """
    <html>
        <body>
            <h1>Software Engineer</h1>
            <p>We are looking for a Python developer.</p>
            <ul>
                <li>3+ years of experience</li>
                <li>Bachelor's degree required</li>
            </ul>
        </body>
    </html>
    """

    # Test processing the URL
    with patch('requests.get') as mock_get:
        mock_get.return_value.text = content
        mock_get.return_value.status_code = 200
        process_url(url, db)

    # Verify that the job was inserted
    mock_cursor.execute.assert_called()
    mock_db_connection.commit.assert_called()

def test_extract_job_details():
    """Test extracting job details from HTML content."""
    content = """
    <html>
        <body>
            <h1>Software Engineer</h1>
            <p>We are looking for a Python developer.</p>
            <div>Salary: $100,000 - $150,000</div>
            <div>Posted: 2024-03-01</div>
            <div>Type: Full-time</div>
            <div>Location: Remote</div>
        </body>
    </html>
    """

    details = extract_job_details(content)
    assert details["title"] == "Software Engineer"
    assert "Python developer" in details["description"]
    assert details["posted_date"] == "2024-03-01"
    assert details["type"] == "Full-time"
    assert details["location"] == "Remote"

def test_extract_skills():
    """Test extracting skills from HTML content."""
    content = """
    <html>
        <body>
            <h2>Required Skills:</h2>
            <ul>
                <li>Python</li>
                <li>SQL</li>
                <li>Git</li>
            </ul>
        </body>
    </html>
    """

    skills = extract_skills(content)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Git" in skills

def test_extract_qualifications():
    """Test extracting qualifications from HTML content."""
    content = """
    <html>
        <body>
            <h2>Required Qualifications:</h2>
            <ul>
                <li>Bachelor's degree in Computer Science</li>
                <li>Master's degree preferred</li>
            </ul>
        </body>
    </html>
    """

    qualifications = extract_qualifications(content)
    assert "Bachelor's degree in Computer Science" in qualifications
    assert "Master's degree preferred" in qualifications

def test_extract_attributes():
    """Test extracting attributes from HTML content."""
    content = """
    <html>
        <body>
            <h2>Required Attributes:</h2>
            <ul>
                <li>Team player</li>
                <li>Self-motivated</li>
                <li>Problem solver</li>
            </ul>
        </body>
    </html>
    """

    attributes = extract_attributes(content)
    assert "Team player" in attributes
    assert "Self-motivated" in attributes
    assert "Problem solver" in attributes

def test_extract_duties():
    """Test extracting duties from HTML content."""
    content = """
    <html>
        <body>
            <h2>Job Duties:</h2>
            <ul>
                <li>Write clean code</li>
                <li>Review pull requests</li>
                <li>Mentor junior developers</li>
            </ul>
        </body>
    </html>
    """

    duties = extract_duties(content)
    assert "Write clean code" in duties
    assert "Review pull requests" in duties
    assert "Mentor junior developers" in duties

def test_extract_benefits():
    """Test extracting benefits from HTML content."""
    content = """
    <html>
        <body>
            <h2>Benefits:</h2>
            <ul>
                <li>Health insurance</li>
                <li>401(k) matching</li>
                <li>Flexible hours</li>
            </ul>
        </body>
    </html>
    """

    benefits = extract_benefits(content)
    assert "Health insurance" in benefits
    assert "401(k) matching" in benefits
    assert "Flexible hours" in benefits

def test_extract_company_details():
    """Test extracting company details from HTML content."""
    content = """
    <html>
        <body>
            <h2>About the Company:</h2>
            <div>
                <h3>Example Corp</h3>
                <p>A leading technology company</p>
                <a href="http://example.com">Visit our website</a>
            </div>
        </body>
    </html>
    """

    details = extract_company_details(content)
    assert details["name"] == "Example Corp"
    assert details["website"] == "http://example.com"

def test_extract_agency_details():
    """Test extracting agency details from HTML content."""
    content = """
    <html>
        <body>
            <h2>Posted by:</h2>
            <div>
                <h3>Example Agency</h3>
                <p>Leading recruitment agency</p>
                <a href="http://agency.com">Visit our website</a>
            </div>
        </body>
    </html>
    """

    details = extract_agency_details(content)
    assert details["name"] == "Example Agency"
    assert details["website"] == "http://agency.com"

def test_extract_location_details():
    """Test extracting location details from HTML content."""
    content = """
    <html>
        <body>
            <h2>Location:</h2>
            <div>
                <p>Cape Town, Western Cape, South Africa</p>
            </div>
        </body>
    </html>
    """

    details = extract_location_details(content)
    assert details["city"] == "Cape Town"
    assert details["province"] == "Western Cape"
    assert details["country"] == "South Africa"

def test_extract_industries():
    """Test extracting industries from HTML content."""
    # Test with a list format
    content_list = """
    <html>
        <body>
            <h2>Industry</h2>
            <ul>
                <li>Technology</li>
                <li>Software Development</li>
                <li>Information Technology</li>
            </ul>
        </body>
    </html>
    """
    industries = extract_industries(content_list)
    assert "Technology" in industries
    assert "Software Development" in industries
    assert "Information Technology" in industries
    assert len(industries) == 3

    # Test with a paragraph format
    content_para = """
    <html>
        <body>
            <h2>Industry/Sector</h2>
            <p>Finance, Banking, Insurance</p>
        </body>
    </html>
    """
    industries = extract_industries(content_para)
    assert "Finance" in industries
    assert "Banking" in industries
    assert "Insurance" in industries
    assert len(industries) == 3

    # Test with no industry section
    content_empty = """
    <html>
        <body>
            <h2>Other Section</h2>
            <p>Some content</p>
        </body>
    </html>
    """
    industries = extract_industries(content_empty)
    assert len(industries) == 0 