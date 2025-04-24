"""
Module for processing URLs from a queue.
"""
from typing import List
from recruitment.models import URL
from recruitment.recruitment_db import RecruitmentDatabase
from recruitment.url_processor import URLProcessor

async def process_urls_from_queue() -> None:
    """Process URLs from the queue."""
    db = RecruitmentDatabase()
    processor = URLProcessor()
    
    # Get unprocessed URLs
    urls: List[URL] = db.get_unprocessed_urls()
    
    # Process each URL
    for url in urls:
        try:
            await processor.process_url(url)
            db.update_url_processing_status(url.id, "completed")
        except Exception as e:
            print(f"Error processing URL {url.url}: {str(e)}")
            db.update_url_processing_status(url.id, "error") 