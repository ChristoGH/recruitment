#!/usr/bin/env python3
"""
Module for targeted recruitment advertisement search in South Africa.

Refactored to exclude async usage and concurrency references.

Usage:
    python recruitment_ad_search.py --days_back 7
"""
import csv
import argparse
import logging
from datetime import datetime, timedelta
from time import sleep
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import tldextract
from googlesearch import search
import json
import os
from functools import lru_cache
from logging_config import setup_logging
from libraries.neo4j_lib import execute_neo4j_query
from libraries.storage import RecruitmentStorage
from libraries.config_validator import ConfigValidator

logger = setup_logging("recruitment_ad_search")

# ------------------------------
# Configuration Loading & Logger Setup
# ------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Loaded configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        json.JSONDecodeError: If configuration file contains invalid JSON
        ValueError: If configuration validation fails
    """
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, config_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    try:
        with open(full_path, "r") as f:
            config = json.load(f)
            
        # Validate configuration
        if not ConfigValidator.validate_config(config):
            raise ValueError("Invalid configuration structure")
            
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {str(e)}", e.doc, e.pos)

# ------------------------------
# URL Validation
# ------------------------------
def is_valid_url(url: str) -> bool:
    """Validate URL format and structure.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# ------------------------------
# RecruitmentAdSearch Class Definition
# ------------------------------
class RecruitmentAdSearch:
    """Encapsulates search logic for recruitment advertisements in South Africa."""

    def __init__(self, search_config: Dict[str, Any]) -> None:
        """Initialize the search handler.
        
        Args:
            search_config: Search configuration dictionary
        """
        self.search_name = search_config['id']
        self.days_back = search_config['days_back']
        self.storage = RecruitmentStorage(self.search_name)

        # Optionally, define any domains to exclude.
        self.excluded_domains = {}

        # Optionally, define academic suffixes to filter out.
        self.academic_suffixes = {}

        # Define recruitment-related search terms to capture job adverts.
        self.recruitment_terms = [
            '"recruitment advert"',
            '"job vacancy"',
            '"hiring now"',
            '"employment opportunity"',
            '"career opportunity"',
            '"job advertisement"',
            '"recruitment drive"'
        ]

    def is_valid_recruitment_site(self, url: str) -> bool:
        """Filter out non-relevant domains if necessary.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
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
        """Construct a targeted search query for recruitment adverts.
        
        Args:
            start_date: Start date for search period
            end_date: End date for search period
            
        Returns:
            str: Constructed search query
        """
        recruitment_part = f"({' OR '.join(self.recruitment_terms)})"
        base_query = f'{recruitment_part} AND "South Africa"'
        final_query = f"{base_query} after:{start_date} before:{end_date}"
        logger.debug(f"Constructed query: {final_query}")
        return final_query

    @lru_cache(maxsize=100)
    def fetch_ads_with_retry(self, query: str, max_results: int = 200, max_retries: int = 3) -> List[str]:
        """Fetch articles with retry logic for transient failures.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            List[str]: List of valid URLs
        """
        articles = []
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Try with tld parameter
                try:
                    results = search(query, tld="com", lang="en", num=10, start=0, stop=max_results, pause=2)
                except TypeError:
                    # Fall back to version without tld parameter
                    results = search(query, lang="en", num_results=100, sleep_interval=5)

                # Validate and filter results
                for url in results:
                    if self.is_valid_recruitment_site(url):
                        articles.append(url)
                        logger.debug(f"Found valid URL: {url}")
                
                # Log results per search term
                for term in self.recruitment_terms:
                    term_results = [url for url in articles if term.lower() in url.lower()]
                    logger.info(f"Found {len(term_results)} results for term: {term}")
                
                return articles
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count} failed: {str(e)}")
                if retry_count < max_retries:
                    sleep(2 ** retry_count)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch articles after {max_retries} attempts: {str(e)}")
                    return []

    def get_date_range(self) -> tuple[str, str]:
        """Get the date range for the search.
        
        Returns:
            tuple[str, str]: Start and end dates in YYYY-MM-DD format
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    def get_recent_ads(self) -> List[str]:
        """Retrieve recent recruitment advertisements based on days_back in the configuration.
        
        Returns:
            List[str]: List of valid recruitment advertisement URLs
        """
        start_date, end_date = self.get_date_range()
        query = self.construct_query(start_date=start_date, end_date=end_date)
        logger.info(f"Searching for recruitment ads with query: {query}")
        
        ads = self.fetch_ads_with_retry(query)
        logger.info(f"Retrieved {len(ads)} recruitment ads in the past {self.days_back} day(s).")
        
        return ads

    def save_results(self, urls: List[str]) -> None:
        """Save the search results using the storage handler.
        
        Args:
            urls: List of URLs to save
        """
        if not urls:
            logger.info("No recruitment ads found for this configuration.")
            return

        for ad in urls:
            logger.info(f"Found ad: {ad}")

        # Save to both storage types
        self.storage.save_to_neo4j(urls)
        self.storage.save_to_csv(urls)

# ------------------------------
# Main Entry Point
# ------------------------------
def main() -> None:
    """Execute searches and save recruitment advertisements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, help="Number of days to search back.", default=7)
    args = parser.parse_args()

    try:
        config = load_config("search_config.json")
        
        # Override days_back from command-line if provided
        for search_config in config.get('run_configs', []):
            search_config['days_back'] = args.days_back

            searcher = RecruitmentAdSearch(search_config)
            ads = searcher.get_recent_ads()
            searcher.save_results(ads)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
