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
from typing import List
from urllib.parse import urlparse
import tldextract
from googlesearch import search  # Ensure you are using the correct googlesearch package.
import json
import os

from libraries.neo4j_lib import execute_neo4j_query


# ------------------------------
# Configuration Loading & Logger Setup
# ------------------------------
def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.
    """
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, config_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {str(e)}", e.doc, e.pos)


config = load_config("search_config.json")

logging.basicConfig(
    filename="recruitment_ad_search.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Recruitment Ad Search service started (refactored, sync version).")


# ------------------------------
# RecruitmentAdSearch Class Definition
# ------------------------------
class RecruitmentAdSearch:
    """Encapsulates search logic for recruitment advertisements in South Africa."""

    def __init__(self, search_config: dict) -> None:
        self.search_name = search_config['id']
        self.days_back = search_config['days_back']

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
        """
        Filter out non-relevant domains if necessary.
        Currently, this function only applies basic filtering.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if any(excluded in domain for excluded in self.excluded_domains):
                return False

            if any(domain.endswith(suffix) for suffix in self.academic_suffixes):
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking domain validity for {url}: {e}")
            return False

    def construct_query(self, start_date: str, end_date: str) -> str:
        """
        Construct a targeted search query for recruitment adverts.
        """
        recruitment_part = f"({' OR '.join(self.recruitment_terms)})"
        # Build the base query with recruitment terms and a location filter.
        base_query = f'{recruitment_part} AND "South Africa"'

        # Optionally, add date filters if supported by Google Search.
        final_query = f"{base_query} after:{start_date} before:{end_date}"
        logger.debug(f"Constructed query: {final_query}")
        return final_query

    def fetch_ads(self, query: str, max_results: int = 200) -> List[str]:
        """
        Fetch articles synchronously using google search.
        """
        articles = []
        try:
            # Try with tld parameter
            try:
                results = search(query, tld="com", lang="en", num=10, start=0, stop=max_results, pause=2)
            except TypeError:
                # Fall back to version without tld parameter
                results = search(query, lang="en", num_results=100, sleep_interval=5)

            for url in results:
                if self.is_valid_recruitment_site(url):
                    articles.append(url)
            return articles
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []

    def get_recent_ads(self) -> List[str]:
        """
        Retrieve recent recruitment advertisements based on days_back in the configuration.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        query = self.construct_query(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        logger.info(f"Searching for recruitment ads with query: {query}")
        return self.fetch_ads(query)

    def save_to_neo4j(self, urls: List[str]) -> None:
        """
        Save filtered recruitment advertisements to a Neo4j database.
        """
        query = """
            MERGE (url:Url {url: $url, source: 'recruitment_ad_search'})
            WITH url
            MERGE (domain:Domain { name: $domain_name })
            MERGE (url)-[:HAS_DOMAIN]->(domain)
        """
        for url in urls:
            extracted = tldextract.extract(url)
            domain_name = extracted.domain
            parameters = {"domain_name": domain_name, "url": url}
            execute_neo4j_query(query, parameters)
            logger.info(f"Saved URL to Neo4j: {url}")

    def save_to_csv(self, urls: List[str]) -> None:
        """
        Save filtered recruitment advertisements to a CSV file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "recruitment_output"
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{output_dir}/saved_urls_{self.search_name}_{timestamp}.csv"

        with open(file_name, mode="w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["url", "domain_name", "source"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for url in urls:
                extracted = tldextract.extract(url)
                domain_name = extracted.domain
                writer.writerow({
                    "url": url,
                    "domain_name": domain_name,
                    "source": "recruitment_ad_search"
                })
                logger.info(f"Saved URL to CSV: {url}")

        logger.info(f"CSV file '{file_name}' created successfully.")


# ------------------------------
# Main Entry Point (Synchronous)
# ------------------------------
def main() -> None:
    """Execute searches and save recruitment advertisements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, help="Number of days to search back.", default=7)
    args = parser.parse_args()

    # Override days_back from command-line if provided.
    for search_config in config.get('run_configs', []):
        search_config['days_back'] = args.days_back

        searcher = RecruitmentAdSearch(search_config)
        ads = searcher.get_recent_ads()

        logger.info(f"Retrieved {len(ads)} recruitment ads in the past {search_config['days_back']} day(s).")
        if not ads:
            logger.info("No recruitment ads found for this configuration.")
            continue

        for ad in ads:
            logger.info(f"Found ad: {ad}")

        # Uncomment the next line if you wish to store into Neo4j:
        # searcher.save_to_neo4j(ads)

        searcher.save_to_csv(ads)


if __name__ == "__main__":
    main()
