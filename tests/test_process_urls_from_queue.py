import pytest
import asyncio
from unittest.mock import MagicMock, patch
from recruitment.process_urls_from_queue import process_urls_from_queue
from recruitment.models import URL

@pytest.fixture
def mock_db():
    with patch('recruitment.process_urls_from_queue.RecruitmentDatabase') as mock:
        db = mock.return_value
        db.get_unprocessed_urls.return_value = [
            URL(id=1, url="http://example.com", domain="example.com", source="test"),
            URL(id=2, url="http://example.org", domain="example.org", source="test")
        ]
        yield db

@pytest.fixture
def mock_processor():
    with patch('recruitment.process_urls_from_queue.URLProcessor') as mock:
        processor = mock.return_value
        # Create a mock for process_url that returns a coroutine
        async def mock_coro(_):
            return None
        process_url_mock = MagicMock(side_effect=mock_coro)
        processor.process_url = process_url_mock
        yield processor

@pytest.mark.asyncio
async def test_process_urls_from_queue(mock_db, mock_processor):
    """Test the process_urls_from_queue function."""
    # Call the function
    await process_urls_from_queue()
    
    # Verify that get_unprocessed_urls was called
    mock_db.get_unprocessed_urls.assert_called_once()
    
    # Verify that process_url was called for each URL
    assert mock_processor.process_url.call_count == 2
    
    # Verify that update_url_processing_status was called for each URL
    assert mock_db.update_url_processing_status.call_count == 2
    mock_db.update_url_processing_status.assert_any_call(1, "completed")
    mock_db.update_url_processing_status.assert_any_call(2, "completed")

@pytest.mark.asyncio
async def test_process_urls_from_queue_with_error(mock_db, mock_processor):
    """Test the process_urls_from_queue function with processing errors."""
    # Make the processor raise an exception
    async def mock_error(_):
        raise Exception("Test error")
    error_mock = MagicMock(side_effect=mock_error)
    mock_processor.process_url = error_mock
    
    # Call the function
    await process_urls_from_queue()
    
    # Verify that get_unprocessed_urls was called
    mock_db.get_unprocessed_urls.assert_called_once()
    
    # Verify that process_url was called for each URL
    assert mock_processor.process_url.call_count == 2
    
    # Verify that update_url_processing_status was called with error status
    assert mock_db.update_url_processing_status.call_count == 2
    mock_db.update_url_processing_status.assert_any_call(1, "error")
    mock_db.update_url_processing_status.assert_any_call(2, "error") 