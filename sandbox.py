#!/usr/bin/env python3
"""
Recruitment Researcher

This script processes URLs to extract job recruitment data using LLMs and stores the results in a database.
It uses web_crawler_lib for content extraction.
"""

import argparse
import json
import logging
import logging.handlers
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tldextract
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Local imports
from get_urls_from_csvs import get_unique_urls_from_csvs
from prompts import COMPLEX_PROMPTS, LIST_PROMPTS, NON_LIST_PROMPTS
from recruitment_db_lib import DatabaseError, RecruitmentDatabase
from recruitment_models import (AgencyResponse, AttributesResponse, BenefitsResponse,
                                CompanyResponse, ConfirmResponse, ContactPersonResponse,
                                EmailResponse, JobAdvertResponse, JobResponse,
                                LocationResponse, PhoneNumberResponse, LinkResponse, SkillsResponse)
from response_processor_functions import PromptResponseProcessor
from web_crawler_lib import crawl_website_sync, WebCrawlerResult

# Load environment variables
load_dotenv()

# Configure Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "recruitment_research.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_llm_engine(api_key: Optional[str] = None) -> Tuple[OpenAI, ChatMemoryBuffer]:
    """
    Set up the LLM and memory for chat interactions.

    Args:
        api_key: Optional API key for the LLM service

    Returns:
        Tuple containing the LLM instance and memory buffer
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Initialize LLM with conservative settings
    llm = OpenAI(temperature=0, model="o3-mini", request_timeout=120.0)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    return llm, memory


def get_validated_response(prompt_key: str, prompt_text: str, model_class: Any, chat_engine) -> Optional[Any]:
    """
    Sends a prompt to the chat engine and returns a validated Pydantic model instance with exponential backoff.

    Args:
        prompt_key: The key identifying the prompt and model
        prompt_text: The prompt text to send
        model_class: The Pydantic model to validate the response
        chat_engine: The chat engine instance for processing the prompt

    Returns:
        An instance of the Pydantic model or None if validation fails
    """
    max_retries = 5
    base_delay = 5  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = chat_engine.chat(prompt_text)
            logger.info(f"Prompt '{prompt_key}' processed successfully.")

            response_text = response.response
            # Try to clean up the response if it contains markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            response_data = model_class.model_validate_json(response_text)
            return response_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for '{prompt_key}': {e}")
            # Retry with the next attempt rather than returning None immediately
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Failed to parse response for '{prompt_key}': {e}")
                if attempt == max_retries - 1:
                    return None

    logger.error(f"Max retries reached for '{prompt_key}'. Skipping...")
    return None


def verify_recruitment(url: str, chat_engine) -> Tuple[Dict[str, Any], List[str]]:
    """
    Verifies if a URL contains a recruitment advertisement.

    Args:
        url: The URL to verify
        chat_engine: The chat engine instance

    Returns:
        A tuple containing the result dictionary and a list of evidence
    """
    db = RecruitmentDatabase()
    result = {"recruitment_flag": -2}  # Default to -2 for unexpected cases
    incidents = []

    try:
        # Send prompt to chat engine
        prompt_key = "recruitment_prompt"
        prompt_text = LIST_PROMPTS[prompt_key]

        response_data = get_validated_response(
            prompt_key, prompt_text, ConfirmResponse, chat_engine
        )
        if response_data is None:
            logger.warning(f"Received no response or invalid response for URL: {url}")
            return result, incidents

        response_answer = response_data.answer.lower()
        if response_answer == "no":
            logger.info(f"No recruitment evidence detected for URL: {url}")
            db.update_field(url, "recruitment_flag", 0)
            result["recruitment_flag"] = 0
            return result, incidents

        if response_answer == "yes":
            logger.info(f"Recruitment advert detected for URL: {url}")
            db.update_field(url, "recruitment_flag", 1)
            result["recruitment_flag"] = 1
            incidents = response_data.evidence or []
            return result, incidents

        # Handle unexpected response cases
        logger.warning(f"Unexpected response answer for URL: {url} - {response_data.answer}")
        return result, incidents

    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return result, incidents


def get_model_class_for_prompt(prompt_key: str) -> Optional[Any]:
    """
    Get the appropriate Pydantic model class for a prompt key.

    Args:
        prompt_key: The prompt key

    Returns:
        The corresponding Pydantic model class or None if not found
    """
    # Mapping from prompt keys to model classes
    model_map = {
        "recruitment_prompt": ConfirmResponse,
        "company_prompt": CompanyResponse,
        "agency_prompt": AgencyResponse,
        "job_prompt": JobResponse,
        "skills_prompt": SkillsResponse,
        "attributes_prompt": AttributesResponse,
        "contact_prompt": ContactPersonResponse,
        "benefits_prompt": BenefitsResponse,
        "phone_prompt": PhoneNumberResponse,
        "email_prompt": EmailResponse,
        "link_prompt": LinkResponse,
        "location_prompt": LocationResponse,
        "jobadvert_prompt": JobAdvertResponse
    }

    return model_map.get(prompt_key)


def process_url(url: str, db: RecruitmentDatabase, processor: PromptResponseProcessor, llm: OpenAI,
                memory: ChatMemoryBuffer, process_all_prompts: bool = True) -> bool:
    """
    Process a single URL to extract recruitment data.

    Args:
        url: The URL to process
        db: Database instance
        processor: Response processor instance
        llm: LLM instance
        memory: Chat memory buffer
        process_all_prompts: Whether to process all prompts or just basic ones

    Returns:
        True if processing succeeded, False otherwise
    """
    logger.info(f"Processing URL: {url}")

    # Extract content using web_crawler_lib
    crawl_result = crawl_website_sync(
        url=url,
        excluded_tags=['form', 'header'],
        verbose=True
    )

    if not crawl_result.success:
        logger.warning(f"Failed to extract content from URL: {url}")
        # Record the URL in the database as inaccessible
        result = {
            "url": url,
            "domain_name": tldextract.extract(url).domain,
            "source": "crawler",
            "content": "",  # No content because extraction failed
            "recruitment_flag": -1,  # Use a default status indicating not processed
            "accessible": 0,
            "error_message": crawl_result.error_message
        }
        db.insert_url(result)
        return False

    # Extract domain
    extracted = tldextract.extract(url)
    domain_name = extracted.domain

    # Use the markdown content from the crawler
    text = crawl_result.markdown
    if not text:
        logger.warning(f"Empty content extracted from URL: {url}")
        result = {
            "url": url,
            "domain_name": domain_name,
            "source": "crawler",
            "content": "",
            "recruitment_flag": -1,
            "accessible": 1,  # URL was accessible but extraction yielded no content
            "error_message": "Empty content extracted"
        }
        db.insert_url(result)
        return False

    # Build the record
    result = {
        "url": url,
        "domain_name": domain_name,
        "source": "crawler",
        "content": text,
        "recruitment_flag": -1,  # default value until verified
        "accessible": 1
    }

    try:
        # Create document index and chat engine
        documents = [liDocument(text=text)]
        index = VectorStoreIndex.from_documents(documents)
        memory.reset()
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            llm=llm,
            memory=memory,
            system_prompt=(
                "You are a career recruitment analyst with deep insight into the skills and job market. "
                "Your express goal is to investigate online adverts and extract pertinent factual detail."
            )
        )

        # Verify if this is a recruitment advert
        recruitment_type, evidence = verify_recruitment(url, chat_engine)
        result.update(recruitment_type)

        # Insert URL into database
        db.insert_url(result)
        url_id = db.get_url_id(url)

        if url_id is None:
            logger.error(f"Failed to retrieve URL ID for {url}")
            return False

        # Process recruitment confirmation
        prompt_key = "recruitment_prompt"
        prompt_text = LIST_PROMPTS[prompt_key]
        response_data = get_validated_response(prompt_key, prompt_text, ConfirmResponse, chat_engine)
        if response_data:
            processor.process_response(url_id, prompt_key, response_data)

        # If not a recruitment advert and not processing all prompts, stop here
        if result.get("recruitment_flag") == 0 and not process_all_prompts:
            logger.info(f"URL {url} is not a recruitment advert. Skipping detailed extraction.")
            return True

        # Process other prompt types

        # Process NON_LIST_PROMPTS
        for prompt_key, prompt_text in NON_LIST_PROMPTS.items():
            model_class = get_model_class_for_prompt(prompt_key)
            if not model_class:
                logger.warning(f"No model class found for prompt key: {prompt_key}")
                continue

            response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
            if response_data:
                processor.process_response(url_id, prompt_key, response_data)

        # Process LIST_PROMPTS
        for prompt_key, prompt_text in LIST_PROMPTS.items():
            # Skip recruitment_prompt as it's already processed
            if prompt_key == "recruitment_prompt":
                continue

            model_class = get_model_class_for_prompt(prompt_key)
            if not model_class:
                logger.warning(f"No model class found for prompt key: {prompt_key}")
                continue

            response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
            if response_data:
                processor.process_response(url_id, prompt_key, response_data)

        # Process COMPLEX_PROMPTS
        for prompt_key, prompt_text in COMPLEX_PROMPTS.items():
            model_class = get_model_class_for_prompt(prompt_key)
            if not model_class:
                logger.warning(f"No model class found for prompt key: {prompt_key}")
                continue

            response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
            if response_data:
                processor.process_response(url_id, prompt_key, response_data)

        logger.info(f"Successfully processed URL: {url}")
        return True

    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return False


def main():
    """Main function to run the recruitment URL processing."""
    parser = argparse.ArgumentParser(description="Process URLs to extract recruitment data")
    parser.add_argument("--csv-dir", type=str, default="recruitment_output",
                        help="Directory containing CSV files with URLs")
    parser.add_argument("--url-column", type=str, default="url",
                        help="Column name containing URLs in the CSV files")
    parser.add_argument("--min-rows", type=int, default=4,
                        help="Minimum number of rows a CSV file must have to be processed")
    parser.add_argument("--max-urls", type=int, default=1000,
                        help="Maximum number of URLs to process")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Index of the first URL to process")
    parser.add_argument("--single-url", type=str, default=None,
                        help="Process a single URL instead of reading from CSV files")
    parser.add_argument("--process-all", action="store_true",
                        help="Process all prompts even for non-recruitment URLs")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the LLM service (optional)")

    args = parser.parse_args()

    # Initialize database and processor
    db = RecruitmentDatabase()
    processor = PromptResponseProcessor(db)

    # Set up LLM and memory
    llm, memory = setup_llm_engine(args.api_key)

    # Process a single URL if specified
    if args.single_url:
        success = process_url(args.single_url, db, processor, llm, memory, args.process_all)
        if success:
            print(f"Successfully processed URL: {args.single_url}")
        else:
            print(f"Failed to process URL: {args.single_url}")
        return

    # Get URLs to process
    urls_from_files = get_unique_urls_from_csvs(
        args.csv_dir, args.url_column, args.min_rows, args.max_urls
    )

    # Filter out URLs already in the database
    if pd.DataFrame(db.search_urls(limit=1000000)).empty:
        urls = urls_from_files
    else:
        urls_from_db = pd.DataFrame(db.search_urls(limit=1000000))['url'].tolist()
        urls = list(set(urls_from_files) - set(urls_from_db))

    # Limit the number of URLs to process
    if args.start_index >= len(urls):
        print(f"Start index {args.start_index} is out of range. Only {len(urls)} URLs available.")
        return

    urls_to_process = urls[args.start_index:args.start_index + args.max_urls]
    print(f"Processing {len(urls_to_process)} URLs starting from index {args.start_index}")

    # Process each URL
    for i, url in enumerate(urls_to_process):
        print(f"Processing URL {i + 1}/{len(urls_to_process)}: {url}")
        try:
            success = process_url(url, db, processor, llm, memory, args.process_all)
            memory.reset()
            if success:
                logger.info(f"Successfully processed URL: {url}")
            else:
                logger.warning(f"Failed to process URL: {url}")
        except Exception as e:
            logger.error(f"Unexpected error processing URL {url}: {e}")

        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)


if __name__ == "__main__":
    main()