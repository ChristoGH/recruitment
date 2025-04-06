"""Storage operations for recruitment advertisements."""
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
import tldextract
from logging_config import setup_logging
from libraries.neo4j_lib import execute_neo4j_query

logger = setup_logging("storage")

class RecruitmentStorage:
    """Handles storage operations for recruitment advertisements."""

    def __init__(self, search_name: str) -> None:
        """Initialize the storage handler.
        
        Args:
            search_name: Identifier for the search configuration
        """
        self.search_name = search_name
        self.output_dir = "recruitment_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def save_to_neo4j(self, urls: List[str]) -> None:
        """Save filtered recruitment advertisements to a Neo4j database.
        
        Args:
            urls: List of URLs to save
        """
        query = """
            MERGE (url:Url {url: $url, source: 'recruitment_ad_search'})
            WITH url
            MERGE (domain:Domain { name: $domain_name })
            MERGE (url)-[:HAS_DOMAIN]->(domain)
        """
        for url in urls:
            try:
                extracted = tldextract.extract(url)
                domain_name = extracted.domain
                parameters = {"domain_name": domain_name, "url": url}
                execute_neo4j_query(query, parameters)
                logger.info(f"Saved URL to Neo4j: {url}")
            except Exception as e:
                logger.error(f"Failed to save URL to Neo4j: {url}. Error: {str(e)}")

    def save_to_csv(self, urls: List[str]) -> None:
        """Save filtered recruitment advertisements to a CSV file.
        
        Args:
            urls: List of URLs to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.output_dir}/saved_urls_{self.search_name}_{timestamp}.csv"

        try:
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
        except Exception as e:
            logger.error(f"Failed to save URLs to CSV: {str(e)}") 