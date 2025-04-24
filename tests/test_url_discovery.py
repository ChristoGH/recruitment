"""Test module for URL discovery service."""
import pytest
import asyncio
from unittest.mock import Mock, patch
from fastapi import FastAPI
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from url_discovery_service import (
    SearchConfig,
    RecruitmentAdSearch,
    is_valid_url,
    get_rabbitmq_connection,
    publish_urls_to_queue
)

@pytest.fixture
def app():
    return FastAPI()

@pytest.fixture
def search_config():
    return SearchConfig(
        id="test_search",
        days_back=7,
        excluded_domains=["example.com"],
        academic_suffixes=[".edu"],
        recruitment_terms=[
            '"recruitment advert"',
            '"job vacancy"',
            '"hiring now"'
        ]
    )

@pytest.fixture
def mock_rabbitmq():
    with patch('pika.BlockingConnection') as mock_conn:
        mock_channel = Mock()
        mock_conn.return_value.channel.return_value = mock_channel
        yield mock_conn, mock_channel

def test_is_valid_url():
    """Test URL validation function."""
    # Valid URLs
    assert is_valid_url("https://example.com") is True
    assert is_valid_url("http://example.com/path") is True
    assert is_valid_url("https://example.com?query=param") is True
    
    # Invalid URLs
    assert is_valid_url("ftp://example.com") is False  # Invalid scheme
    assert is_valid_url("example.com") is False  # Missing scheme
    assert is_valid_url("https://") is False  # Missing netloc
    assert is_valid_url("https://example.com/" + "a" * 2001) is False  # Too long
    assert is_valid_url("https://example.com/\u2603") is False  # Invalid character
    assert is_valid_url("") is False  # Empty string
    assert is_valid_url(None) is False  # None value

def test_search_config_validation(search_config):
    assert search_config.id == "test_search"
    assert search_config.days_back == 7
    assert len(search_config.excluded_domains) == 1
    assert len(search_config.academic_suffixes) == 1
    assert len(search_config.recruitment_terms) == 3

def test_recruitment_ad_search_initialization(search_config):
    searcher = RecruitmentAdSearch(search_config)
    assert searcher.search_name == "test_search"
    assert searcher.days_back == 7
    assert len(searcher.excluded_domains) == 1
    assert len(searcher.academic_suffixes) == 1
    assert len(searcher.recruitment_terms) == 3

def test_is_valid_recruitment_site(search_config):
    searcher = RecruitmentAdSearch(search_config)
    assert searcher.is_valid_recruitment_site("https://example.com") is False
    assert searcher.is_valid_recruitment_site("https://valid-job-site.com") is True
    assert searcher.is_valid_recruitment_site("https://university.edu") is False

def test_construct_query(search_config):
    searcher = RecruitmentAdSearch(search_config)
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    query = searcher.construct_query(start_date, end_date)
    assert "recruitment advert" in query
    assert "job vacancy" in query
    assert "hiring now" in query
    assert "South Africa" in query
    assert start_date in query
    assert end_date in query

@pytest.mark.asyncio
async def test_publish_urls_to_queue(mock_rabbitmq):
    mock_conn, mock_channel = mock_rabbitmq
    urls = ["https://example1.com", "https://example2.com"]
    search_id = "test_search"
    
    await publish_urls_to_queue(urls, search_id)
    
    assert mock_channel.basic_publish.call_count == len(urls)
    for call in mock_channel.basic_publish.call_args_list:
        args, kwargs = call
        assert kwargs['exchange'] == ""
        assert kwargs['routing_key'] == "recruitment_urls"
        assert kwargs['properties'].delivery_mode == 2

@pytest.mark.asyncio
async def test_perform_search(search_config, mock_rabbitmq):
    with patch('url_discovery_service.search') as mock_search:
        mock_search.return_value = [
            "https://valid-job-site.com/job1",
            "https://valid-job-site.com/job2"
        ]
        
        from url_discovery_service import perform_search
        response = await perform_search(search_config, Mock())
        
        assert response.search_id.startswith("test_search_")
        assert response.urls_found == 2
        assert len(response.urls) == 2
        assert "valid-job-site.com" in response.urls[0]

@pytest.mark.asyncio
async def test_get_search_status():
    from url_discovery_service import get_search_status, search_results
    
    # Setup test data
    search_id = "test_search_20240101"
    search_results[search_id] = {
        "status": "completed",
        "urls_found": 5,
        "timestamp": "2024-01-01T00:00:00"
    }
    
    response = await get_search_status(search_id)
    assert response.search_id == search_id
    assert response.status == "completed"
    assert response.urls_found == 5

@pytest.mark.asyncio
async def test_get_search_urls():
    from url_discovery_service import get_search_urls, search_results
    
    # Setup test data
    search_id = "test_search_20240101"
    test_urls = ["https://example1.com", "https://example2.com"]
    search_results[search_id] = {
        "urls": test_urls
    }
    
    response = await get_search_urls(search_id)
    assert response == test_urls

@pytest.mark.asyncio
async def test_health_check():
    from url_discovery_service import health_check
    
    with patch('pika.BlockingConnection') as mock_conn:
        mock_conn.return_value.is_closed = False
        response = await health_check()
        assert response["status"] == "healthy"
        assert response["rabbitmq"] == "connected"
        
        mock_conn.side_effect = Exception("Connection failed")
        response = await health_check()
        assert response["status"] == "unhealthy"
        assert "error" in response 