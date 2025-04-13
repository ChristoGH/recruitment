#!/usr/bin/env python3
"""
URL Discovery Service

This FastAPI service searches for recruitment URLs and publishes them to a RabbitMQ queue.
It's based on the recruitment_ad_search.py script.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import tldextract
from googlesearch import search
from functools import lru_cache
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pika
import asyncio
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import uuid

from logging_config import setup_logging
from libraries.config_validator import ConfigValidator

# Load environment variables
load_dotenv()

# Create module-specific logger
logger = setup_logging("url_discovery_service")

# Pydantic models for API
class SearchConfig(BaseModel):
    id: str
    days_back: int = 7
    excluded_domains: List[str] = []
    academic_suffixes: List[str] = []
    recruitment_terms: List[str] = [
        '"recruitment advert"',
        '"job vacancy"',
        '"hiring now"',
        '"employment opportunity"',
        '"career opportunity"',
        '"job advertisement"',
        '"recruitment drive"'
    ]

class SearchResponse(BaseModel):
    search_id: str
    urls_found: int
    urls: List[str]
    timestamp: str

class SearchStatus(BaseModel):
    search_id: str
    status: str
    urls_found: int
    timestamp: str

# Global variables
rabbitmq_connection = None
rabbitmq_channel = None
search_results: Dict[str, Dict] = {}

def get_rabbitmq_connection():
    """Get a connection to RabbitMQ."""
    try:
        # Get RabbitMQ connection parameters from environment variables
        rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
        
        # Create connection parameters
        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=rabbitmq_host,
            port=rabbitmq_port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        
        # Create connection
        connection = pika.BlockingConnection(parameters)
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to RabbitMQ: {str(e)}")

def get_or_create_rabbitmq_connection():
    """Get existing RabbitMQ connection or create a new one."""
    global rabbitmq_connection, rabbitmq_channel
    
    try:
        # Check if connection is closed or doesn't exist
        if not rabbitmq_connection or rabbitmq_connection.is_closed:
            # Create new connection
            rabbitmq_connection = get_rabbitmq_connection()
            rabbitmq_channel = rabbitmq_connection.channel()
            
            # Declare queue
            queue_name = "recruitment_urls"
            rabbitmq_channel.queue_declare(queue=queue_name, durable=True)
            
        return rabbitmq_connection, rabbitmq_channel
        
    except Exception as e:
        logger.error(f"Error creating RabbitMQ connection: {e}")
        # Close connection if it exists
        if rabbitmq_connection and not rabbitmq_connection.is_closed:
            rabbitmq_connection.close()
        rabbitmq_connection = None
        rabbitmq_channel = None
        raise

def publish_urls_to_queue(urls: List[str], search_id: str):
    """Publish URLs to RabbitMQ queue."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get or create RabbitMQ connection
            connection, channel = get_or_create_rabbitmq_connection()
            
            # Publish each URL to the queue
            queue_name = "recruitment_urls"
            for url in urls:
                message = {
                    "url": url,
                    "search_id": search_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    )
                )
                
            logger.info(f"Published {len(urls)} URLs to RabbitMQ queue")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to RabbitMQ queue (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)  # Wait before retrying
                continue
            raise

# URL validation
def is_valid_url(url: str) -> bool:
    """Validate URL format and structure."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# RecruitmentAdSearch class adapted for the service
class RecruitmentAdSearch:
    """Encapsulates search logic for recruitment advertisements."""

    def __init__(self, search_config: SearchConfig) -> None:
        """Initialize the search handler."""
        self.search_name = search_config.id
        self.days_back = search_config.days_back
        self.excluded_domains = search_config.excluded_domains
        self.academic_suffixes = search_config.academic_suffixes
        self.recruitment_terms = search_config.recruitment_terms

    def is_valid_recruitment_site(self, url: str) -> bool:
        """Filter out non-relevant domains if necessary."""
        try:
            if not is_valid_url(url):
                logger.debug(f"Invalid URL format: {url}")
                return False

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if any(excluded in domain for excluded in self.excluded_domains):
                logger.debug(f"Excluded domain: {url}")
                return False

            if any(domain.endswith(suffix) for suffix in self.academic_suffixes):
                logger.debug(f"Academic domain excluded: {url}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking domain validity for {url}: {e}")
            return False

    def construct_query(self, start_date: str, end_date: str) -> str:
        """Construct a targeted search query for recruitment adverts."""
        recruitment_part = f"({' OR '.join(self.recruitment_terms)})"
        base_query = f'{recruitment_part} AND "South Africa"'
        final_query = f"{base_query} after:{start_date} before:{end_date}"
        logger.debug(f"Constructed query: {final_query}")
        return final_query

    def get_date_range(self) -> tuple[str, str]:
        """Get the date range for the search."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    def get_recent_ads(self) -> List[str]:
        """Retrieve recent recruitment advertisements based on days_back in the configuration."""
        start_date, end_date = self.get_date_range()
        query = self.construct_query(start_date=start_date, end_date=end_date)
        logger.info(f"Searching for recruitment ads with query: {query}")
        
        ads = []
        try:
            # Try with tld parameter
            try:
                results = search(query, tld="com", lang="en", num=10, start=0, stop=200, pause=2)
            except TypeError:
                # Fall back to version without tld parameter
                results = search(query, lang="en", num_results=100, sleep_interval=5)

            # Validate and filter results
            for url in results:
                if self.is_valid_recruitment_site(url):
                    ads.append(url)
                    logger.debug(f"Found valid URL: {url}")
            
            # Log results per search term
            for term in self.recruitment_terms:
                term_results = [url for url in ads if term.lower() in url.lower()]
                logger.info(f"Found {len(term_results)} results for term: {term}")
            
            return ads
            
        except Exception as e:
            logger.error(f"Error fetching recruitment ads: {e}")
            return []

# Core search functionality
async def perform_search(search_config: SearchConfig, background_tasks: BackgroundTasks) -> SearchResponse:
    """Perform a search for recruitment URLs."""
    try:
        # Create search ID
        search_id = f"{search_config.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize search status
        search_results[search_id] = {
            "status": "pending",
            "urls_found": 0,
            "urls": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Create search instance
        searcher = RecruitmentAdSearch(search_config)
        
        # Get URLs
        urls = searcher.get_recent_ads()
        
        # Update search results
        search_results[search_id]["urls"] = urls
        search_results[search_id]["urls_found"] = len(urls)
        
        # Add background task to publish URLs to queue
        background_tasks.add_task(publish_urls_to_queue, urls, search_id)
        
        return SearchResponse(
            search_id=search_id,
            urls_found=len(urls),
            urls=urls,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to periodically trigger search
async def periodic_search(background_tasks: BackgroundTasks, search_config: dict) -> dict:
    """
    Perform a periodic search and publish URLs to RabbitMQ queue.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
        search_config: Search configuration dictionary
        
    Returns:
        dict: Search results including search_id and status
    """
    try:
        # Convert dict to SearchConfig object
        config = SearchConfig(**search_config)
        
        # Generate unique search ID
        search_id = str(uuid.uuid4())
        
        # Initialize search results
        search_results[search_id] = {
            "status": "in_progress",
            "urls_found": 0,
            "error": None,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None
        }
        
        # Perform search
        try:
            response = await perform_search(config, background_tasks)
            
            # Update search results with found URLs
            search_results[search_id]["urls_found"] = response.urls_found
            
            # Publish URLs to queue
            try:
                await publish_urls_to_queue(response.urls, search_id)
                search_results[search_id]["status"] = "completed"
            except Exception as e:
                logger.error(f"Error publishing URLs to queue: {str(e)}")
                search_results[search_id]["status"] = "failed"
                search_results[search_id]["error"] = f"Queue publishing error: {str(e)}"
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to publish URLs to queue: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            search_results[search_id]["status"] = "failed"
            search_results[search_id]["error"] = f"Search error: {str(e)}"
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )
            
        finally:
            # Update end time
            search_results[search_id]["end_time"] = datetime.utcnow().isoformat()
            
        return {
            "search_id": search_id,
            "status": search_results[search_id]["status"],
            "urls_found": search_results[search_id]["urls_found"]
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in periodic_search: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Create background tasks instance
    background_tasks = BackgroundTasks()
    
    # Default search configuration
    default_config = {
        "id": "periodic_search",
        "days_back": 7,
        "excluded_domains": [],
        "academic_suffixes": [],
        "recruitment_terms": [
            '"recruitment advert"',
            '"job vacancy"',
            '"hiring now"',
            '"employment opportunity"',
            '"career opportunity"',
            '"job advertisement"',
            '"recruitment drive"'
        ]
    }
    
    # Startup: Start background tasks
    background_task = asyncio.create_task(periodic_search(background_tasks, default_config))
    
    yield  # This is where the application runs
    
    # Shutdown: Cancel background tasks
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="URL Discovery Service",
    description="Service for discovering recruitment URLs and publishing them to a queue",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
@app.post("/search", response_model=SearchResponse)
async def create_search(search_config: SearchConfig, background_tasks: BackgroundTasks):
    """Create a new search for recruitment URLs."""
    return await perform_search(search_config=search_config, background_tasks=background_tasks)

@app.get("/search/{search_id}", response_model=SearchStatus)
async def get_search_status(search_id: str):
    """Get the status of a search."""
    if search_id not in search_results:
        raise HTTPException(status_code=404, detail=f"Search ID {search_id} not found")
    
    return SearchStatus(
        search_id=search_id,
        status=search_results[search_id]["status"],
        urls_found=search_results[search_id]["urls_found"],
        timestamp=search_results[search_id]["timestamp"]
    )

@app.get("/search/{search_id}/urls", response_model=List[str])
async def get_search_urls(search_id: str):
    """Get the URLs found by a search."""
    if search_id not in search_results:
        raise HTTPException(status_code=404, detail=f"Search ID {search_id} not found")
    
    return search_results[search_id]["urls"]

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck."""
    try:
        # Check RabbitMQ connection
        connection = get_rabbitmq_connection()
        if connection and not connection.is_closed:
            connection.close()
            return {"status": "healthy", "rabbitmq": "connected"}
        return {"status": "unhealthy", "rabbitmq": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 