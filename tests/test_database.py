"""
Test module for the RecruitmentDatabase class.
"""
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from recruitment.recruitment_db import RecruitmentDatabase, DatabaseError

@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.lastrowid = 1
    return mock_conn

@pytest.fixture
def db(mock_db_connection):
    """Create a RecruitmentDatabase instance with a mock connection."""
    with patch('sqlite3.connect', return_value=mock_db_connection):
        db = RecruitmentDatabase(':memory:')
        return db

def test_insert_url(db, mock_db_connection):
    """Test inserting a URL into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a URL
    url_id = db.insert_url("http://example.com", "example.com", "test")
    assert url_id == 1
    mock_cursor.execute.assert_called_once()

def test_update_url_processing_status(db, mock_db_connection):
    """Test updating the processing status of a URL."""
    mock_cursor = mock_db_connection.cursor()

    # Test updating the status
    db.update_url_processing_status(1, "completed")
    mock_cursor.execute.assert_called_once()

def test_insert_job(db, mock_db_connection):
    """Test inserting a job into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a job
    job_id = db.insert_job("Software Engineer", "A great job", "2024-03-01", "Full-time", "Remote", 1)
    assert job_id == 1
    mock_cursor.execute.assert_called_once()

def test_insert_advert(db, mock_db_connection):
    """Test inserting an advert into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting an advert
    advert_id = db.insert_advert("Software Engineer", "A great job", "2024-03-01", "Full-time", "Remote")
    assert advert_id == 1
    mock_cursor.execute.assert_called_once()

def test_insert_skill(db, mock_db_connection):
    """Test inserting a skill into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a skill
    skill_id = db.insert_skill("Python")
    assert skill_id == 1
    mock_cursor.execute.assert_called_once()

def test_link_job_skill(db, mock_db_connection):
    """Test linking a job to a skill."""
    mock_cursor = mock_db_connection.cursor()

    # Test linking a job to a skill
    db.link_job_skill(1, 1)
    mock_cursor.execute.assert_called_once()

def test_insert_qualification(db, mock_db_connection):
    """Test inserting a qualification into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a qualification
    qualification_id = db.insert_qualification("Bachelor's Degree")
    assert qualification_id == 1
    mock_cursor.execute.assert_called_once()

def test_link_job_qualification(db, mock_db_connection):
    """Test linking a job to a qualification."""
    mock_cursor = mock_db_connection.cursor()

    # Test linking a job to a qualification
    db.link_job_qualification(1, 1)
    mock_cursor.execute.assert_called_once()

def test_insert_attribute(db, mock_db_connection):
    """Test inserting an attribute into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting an attribute
    attribute_id = db.insert_attribute("Team Player")
    assert attribute_id == 1
    mock_cursor.execute.assert_called_once()

def test_link_job_attribute(db, mock_db_connection):
    """Test linking a job to an attribute."""
    mock_cursor = mock_db_connection.cursor()

    # Test linking a job to an attribute
    db.link_job_attribute(1, 1)
    mock_cursor.execute.assert_called_once()

def test_insert_duty(db, mock_db_connection):
    """Test inserting a duty into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a duty
    duty_id = db.insert_duty("Write code")
    assert duty_id == 1
    mock_cursor.execute.assert_called_once()

def test_link_job_duty(db, mock_db_connection):
    """Test linking a job to a duty."""
    mock_cursor = mock_db_connection.cursor()

    # Test linking a job to a duty
    db.link_job_duty(1, 1)
    mock_cursor.execute.assert_called_once()

def test_insert_benefit(db, mock_db_connection):
    """Test inserting a benefit into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a benefit
    benefit_id = db.insert_benefit("Health Insurance")
    assert benefit_id == 1
    mock_cursor.execute.assert_called_once()

def test_link_job_benefit(db, mock_db_connection):
    """Test linking a job to a benefit."""
    mock_cursor = mock_db_connection.cursor()

    # Test linking a job to a benefit
    db.link_job_benefit(1, 1)
    mock_cursor.execute.assert_called_once()

def test_insert_company(db, mock_db_connection):
    """Test inserting a company into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a company
    company_id = db.insert_company("Example Corp", "http://example.com")
    assert company_id == 1
    mock_cursor.execute.assert_called_once()

def test_insert_agency(db, mock_db_connection):
    """Test inserting an agency into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting an agency
    agency_id = db.insert_agency("Example Agency", "http://example.com")
    assert agency_id == 1
    mock_cursor.execute.assert_called_once()

def test_insert_location(db, mock_db_connection):
    """Test inserting a location into the database."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.lastrowid = 1

    # Test inserting a location
    location_id = db.insert_location("Cape Town", "Western Cape", "South Africa")
    assert location_id == 1
    mock_cursor.execute.assert_called_once()

def test_link_job_location(db, mock_db_connection):
    """Test linking a job to a location."""
    mock_cursor = mock_db_connection.cursor()

    # Test linking a job to a location
    db.link_job_location(1, 1)
    mock_cursor.execute.assert_called_once()

def test_database_error_handling(db, mock_db_connection):
    """Test that database errors are handled correctly."""
    mock_cursor = mock_db_connection.cursor()
    mock_cursor.execute.side_effect = sqlite3.Error("Test error")

    # Test that the error is caught and re-raised as DatabaseError
    with pytest.raises(DatabaseError):
        db.insert_url("http://example.com", "example.com", "test") 