#!/usr/bin/env python3
"""
Recruitment Researcher

This script processes URLs to extract job recruitment data using LLMs and stores the results in a database.
It uses web_crawler_lib for content extraction and supports transaction-based processing.
"""

import argparse
import json
import logging
import logging.handlers
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

import pandas as pd
import tldextract
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from utils import get_model_for_prompt

# Local imports
from recruitment_models import transform_skills_response
from get_urls_from_csvs import get_unique_urls_from_csvs
from prompts import COMPLEX_PROMPTS, LIST_PROMPTS, NON_LIST_PROMPTS
from recruitment_db_lib import DatabaseError, RecruitmentDatabase
from recruitment_models import (AgencyResponse, AttributesResponse, BenefitsResponse,
                                CompanyResponse, ConfirmResponse, ContactPersonResponse,
                                EmailResponse, JobAdvertResponse, JobResponse,
                                LocationResponse, CompanyPhoneNumberResponse, LinkResponse,
                                SkillExperienceResponse, DutiesResponse,
                                QualificationsResponse, AdvertResponse)
from response_processor_functions import PromptResponseProcessor, ResponseProcessingError
from batch_processor import process_all_prompt_responses, direct_insert_skills
from web_crawler_lib import crawl_website_sync, WebCrawlerResult

# Load environment variables
load_dotenv()

from logging_config import setup_logging

# Create module-specific logger
logger = setup_logging("recruitment_researcher")


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
    llm = OpenAI(temperature=0, model="gpt-4o-mini", request_timeout=120.0)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    return llm, memory


def clean_response_text(response_text: str) -> str:
    """
    Clean LLM response text by removing markdown code blocks and other formatting.

    Args:
        response_text: The raw text response from the LLM

    Returns:
        Cleaned text ready for JSON parsing
    """
    # Try to clean up the response if it contains markdown code blocks
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        return response_text.split("```")[1].split("```")[0].strip()
    else:
        return response_text.strip()


def process_url_skills(url_id: int, db, chat_engine) -> None:
    """
    Process skills specifically for a URL ID.
    This can be used to backfill skills for URLs that were already processed.

    Args:
        url_id: The URL ID to process skills for
        db: Database instance
        chat_engine: Chat engine for processing
    """
    # Check if this URL already has skills
    query = "SELECT COUNT(*) FROM skills WHERE url_id = ?"
    with db._execute_query(query, (url_id,)) as cursor:
        count = cursor.fetchone()[0]

    if count > 0:
        logger.info(f"URL ID {url_id} already has {count} skills. Skipping.")
        return

    # Get URL record
    url_record = db.get_url_by_id(url_id)
    if not url_record or not url_record.content:
        logger.warning(f"No content found for URL ID {url_id}")
        return

    # Get skills using prompt
    from prompts import COMPLEX_PROMPTS
    prompt_key = "skills_prompt"
    prompt_text = COMPLEX_PROMPTS[prompt_key]

    # Process response
    from recruitment_models import SkillExperienceResponse
    skills_data = get_validated_response(prompt_key, prompt_text, SkillExperienceResponse, chat_engine)

    if skills_data:
        # Transform to expected format
        from recruitment_models import transform_skills_response
        processed_skills = transform_skills_response(skills_data)

        # Direct insert skills
        if processed_skills and processed_skills.get('skills'):
            direct_insert_skills(db, url_id, processed_skills['skills'])
            logger.info(f"Successfully processed skills for URL ID {url_id}")
            return True

    logger.warning(f"No skills could be extracted for URL ID {url_id}")
    return False


def get_validated_response(prompt_key: str, prompt_text: str, model_class: Any, chat_engine) -> Optional[Any]:
    """
    Sends a prompt to the chat engine and returns a validated Pydantic model instance with exponential backoff.
    """
    max_retries = 5
    base_delay = 5  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = chat_engine.chat(prompt_text)

            # Log only that we received a response, not that it was processed successfully yet
            logger.info(f"Received response for '{prompt_key}' (attempt {attempt + 1}/{max_retries})")

            response_text = clean_response_text(response.response)

            # Special handling for the skills prompt which returns tuples
            if prompt_key == "skills_prompt":
                logger.info(f"Skills response_text for skills_prompt yields '{response_text}'.")
                try:
                    # If the response contains Python-style tuples, convert them to lists for JSON parsing
                    if "skills" in response_text and "(" in response_text and ")" in response_text:
                        # First try direct replacement
                        modified_text = response_text.replace("(", "[").replace(")", "]")
                        logger.info(f"Skills modified_text yields '{modified_text}'.")
                        try:
                            # Try to parse the JSON with replaced brackets
                            response_json = json.loads(modified_text)

                            # Check for skills array
                            if 'skills' in response_json and isinstance(response_json['skills'], list):
                                # Process the skills
                                processed_skills = []

                                for item in response_json['skills']:
                                    # Now item should be a list (previously a tuple)
                                    if isinstance(item, list):
                                        if len(item) >= 2:
                                            skill = item[0]
                                            # Convert "not_listed" to None
                                            experience = None if item[1] == "not_listed" else item[1]
                                            processed_skills.append({"skill": skill, "experience": experience})
                                        else:
                                            processed_skills.append({"skill": item[0], "experience": None})
                                    elif isinstance(item, str):
                                        processed_skills.append({"skill": item, "experience": None})

                                # Create proper response object
                                response_data = {"skills": processed_skills}
                                logger.info(f"Skills prompt yields '{response_data}' after transformation.")

                                # Try to validate with model
                                try:
                                    validated_data = model_class.model_validate(response_data)
                                    logger.info(f"Skills prompt '{prompt_key}' processed successfully.")
                                    logger.info(f"Skills prompt yields '{validated_data}' after transformation.")
                                    return validated_data
                                except Exception as e:
                                    logger.error(f"Skills validation error after transformation: {e}")
                                    # Fall back to returning a dictionary directly
                                    return response_data
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse modified JSON for skills prompt, attempt {attempt + 1}")

                    # If we reach here, more manual parsing may be needed
                    logger.info(f"Attempting manual parsing for skills response")

                except Exception as e:
                    logger.error(f"Error processing skills response: {e}")

            # Standard handling for other prompts
            try:
                response_data = model_class.model_validate_json(response_text)
                logger.info(f"Prompt '{prompt_key}' processed successfully.")
                return response_data
            except Exception as e:
                logger.error(f"Validation error for '{prompt_key}': {e}")
                # Continue to next attempt

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for '{prompt_key}' (attempt {attempt + 1}/{max_retries}): {e}")
            logger.debug(f"Raw response: {response.response[:500]}...")
            # Retry with the next attempt
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Unexpected error for '{prompt_key}': {e}")
                # Continue to next attempt

    logger.error(f"Max retries reached for '{prompt_key}'. Skipping...")
    return None

# Modification to batch_processor.py
# Replace the process_skills function with this improved version:


def verify_recruitment(url: str, chat_engine) -> Tuple[Dict[str, Any], List[str]]:
    """
    Verifies if a URL contains a recruitment advertisement.

    Args:
        url: The URL to verify
        chat_engine: The chat engine instance

    Returns:
        A tuple containing the result dictionary and a list of evidence
    """
    result = {"recruitment_flag": -2}  # Default to -2 for unexpected cases
    incidents = []

    try:
        # Send prompt to chat engine
        prompt_key = "recruitment_prompt"
        prompt_text = LIST_PROMPTS[prompt_key]

        response_data = get_validated_response(
            prompt_key, prompt_text, AdvertResponse, chat_engine
        )
        if response_data is None:
            logger.warning(f"Received no response or invalid response for URL: {url}")
            return result, incidents

        response_answer = response_data.answer.lower()
        if response_answer == "no":
            logger.info(f"No recruitment evidence detected for URL: {url}")
            result["recruitment_flag"] = 0
            return result, incidents

        if response_answer == "yes":
            logger.info(f"Recruitment advert detected for URL: {url}")
            result["recruitment_flag"] = 1
            incidents = response_data.evidence or []
            return result, incidents

        # Handle unexpected response cases
        logger.warning(f"Unexpected response answer for URL: {url} - {response_data.answer}")
        return result, incidents

    except Exception as e:
        logger.error(f"Error verifying recruitment for URL {url}: {e}")
        return result, incidents


def collect_prompt_responses(chat_engine) -> Dict[str, Any]:
    """
    Collect responses for all prompts using the chat engine.

    Args:
        chat_engine: The chat engine for processing prompts

    Returns:
        Dictionary mapping prompt types to their validated responses
    """
    prompt_responses = {}

    # Process recruitment prompt
    recruitment_key = "recruitment_prompt"
    recruitment_text = LIST_PROMPTS[recruitment_key]
    recruitment_data = get_validated_response(
        recruitment_key, recruitment_text, AdvertResponse, chat_engine
    )
    if recruitment_data:
        prompt_responses[recruitment_key] = recruitment_data

    # Process NON_LIST_PROMPTS
    for prompt_key, prompt_text in NON_LIST_PROMPTS.items():
        model_class = get_model_for_prompt(prompt_key)
        if not model_class:
            logger.warning(f"No model class found for prompt key: {prompt_key}")
            continue

        response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
        if response_data:
            prompt_responses[prompt_key] = response_data

    # Process LIST_PROMPTS (excluding recruitment prompt which was already processed)
    for prompt_key, prompt_text in {k: v for k, v in LIST_PROMPTS.items() if k != "recruitment_prompt"}.items():
        model_class = get_model_for_prompt(prompt_key)
        if not model_class:
            logger.warning(f"No model class found for prompt key: {prompt_key}")
            continue

        response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
        if response_data:
            # Handle skills response specially to transform to tuple format
            if prompt_key == "skills_prompt":
                # Convert to dict if it's a model object
                if hasattr(response_data, 'model_dump'):
                    data_dict = response_data.model_dump()
                else:
                    data_dict = response_data
                prompt_responses[prompt_key] = transform_skills_response(data_dict)
            else:
                prompt_responses[prompt_key] = response_data

    # Process COMPLEX_PROMPTS
    for prompt_key, prompt_text in COMPLEX_PROMPTS.items():
        model_class = get_model_for_prompt(prompt_key)
        if not model_class:
            logger.warning(f"No model class found for prompt key: {prompt_key}")
            continue

        response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
        if response_data:
            prompt_responses[prompt_key] = response_data

    return prompt_responses


def validate_and_normalize_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and normalize a URL for crawling.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, normalized_url, error_message)
    """
    # Clean up the URL by trimming whitespace
    url = url.strip()

    # Check if URL has a valid protocol
    valid_protocols = ('http://', 'https://', 'file://', 'raw:')
    has_protocol = any(url.startswith(protocol) for protocol in valid_protocols)

    # If no protocol, try to add https:// (most common)
    if not has_protocol:
        # Check if it might be a search query or invalid URL
        if url.startswith('/') or ' ' in url or url.startswith('('):
            return False, url, f"Invalid URL format: {url}"

        # Try adding https:// to URLs that might be missing the protocol
        url = f"https://{url}"
        logger.info(f"Added https:// protocol to URL: {url}")

    # Additional validation for common issues
    if ' ' in url:
        return False, url, f"URL contains spaces: {url}"

    try:
        # Parse the URL to check if it's valid
        parsed = tldextract.extract(url)
        if not parsed.domain:
            return False, url, f"URL has no valid domain: {url}"

        # Special handling for known job board domains
        known_job_boards = {
            'simplyhired': ['search'],  # Add more job boards and their search paths as needed
            'indeed': ['viewjob', 'jobs'],
            'linkedin': ['jobs', 'job'],
            'glassdoor': ['job-listing', 'jobs'],
            'careerjet': ['job']
        }

        # Check if this is a known job board
        if parsed.domain in known_job_boards:
            # These are valid search/job URLs, don't filter them
            return True, url, None

        # For other domains, check for common search query patterns that shouldn't be crawled
        if any(q in url for q in ['?q=', '/search?', 'query=', 'find=']):
            return False, url, f"URL appears to be a search query: {url}"

        return True, url, None
    except Exception as e:
        return False, url, f"URL validation error: {str(e)}"


def process_url(url: str, db: RecruitmentDatabase, processor: PromptResponseProcessor,
                llm: OpenAI, memory: ChatMemoryBuffer, process_all_prompts: bool = True,
                use_transaction: bool = True) -> bool:
    """
    Process a single URL to extract recruitment data using transaction support.

    Args:
        url: The URL to process
        db: Database instance
        processor: Response processor instance
        llm: LLM instance
        memory: Chat memory buffer
        process_all_prompts: Whether to process all prompts or just basic ones
        use_transaction: Whether to use transaction support

    Returns:
        True if processing succeeded, False otherwise
    """
    logger.info(f"Processing URL: {url}")

    # Validate and normalize the URL
    is_valid, normalized_url, error_message = validate_and_normalize_url(url)

    if not is_valid:
        logger.warning(f"Invalid URL: {url}. {error_message}")
        # Record the URL in the database as inaccessible
        result = {
            "url": url,
            "domain_name": "",  # No valid domain
            "source": "crawler",
            "content": "",  # No content due to invalid URL
            "recruitment_flag": -1,  # Use a default status indicating not processed
            "accessible": 0,
            "error_message": error_message
        }
        try:
            db.insert_url(result)
        except DatabaseError as e:
            logger.error(f"Database error when inserting invalid URL {url}: {e}")

        return False

    # Use the normalized URL for crawling
    url = normalized_url
    logger.info(f"Normalized URL: {url}")

    # Extract content using web_crawler_lib
    try:
        crawl_result = crawl_website_sync(
            url=url,
            excluded_tags=['form', 'header'],
            verbose=True
        )
    except Exception as e:
        logger.error(f"Crawler exception for URL {url}: {e}", exc_info=True)
        # Record the URL in the database as inaccessible
        result = {
            "url": url,
            "domain_name": tldextract.extract(url).domain,
            "source": "crawler",
            "content": "",  # No content because extraction failed
            "recruitment_flag": -1,  # Use a default status indicating not processed
            "accessible": 0,
            "error_message": f"Crawler exception: {str(e)}"
        }
        try:
            db.insert_url(result)
        except DatabaseError as e:
            logger.error(f"Database error when inserting URL with crawler exception {url}: {e}")

        return False

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
        try:
            db.insert_url(result)
            return False
        except DatabaseError as e:
            logger.error(f"Database error when inserting inaccessible URL {url}: {e}")
            return False

    # Extract domain
    extracted = tldextract.extract(url)
    domain_name = extracted.domain

    # Use the markdown content from the crawler
    # text = crawl_result.markdown[:5000]
    text = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
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
        try:
            db.insert_url(result)
            return False
        except DatabaseError as e:
            logger.error(f"Database error when inserting URL with empty content {url}: {e}")
            return False

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
        recruitment_flag = recruitment_type.get("recruitment_flag", -1)

        # Build the record
        result = {
            "url": url,
            "domain_name": domain_name,
            "source": "crawler",
            "content": text,
            "recruitment_flag": recruitment_flag,
            "accessible": 1
        }

        # Insert URL into database
        try:
            db.insert_url(result)
            url_id = db.get_url_id(url)

            if url_id is None:
                logger.error(f"Failed to retrieve URL ID for {url}")
                return False

            # Insert links if available
            if crawl_result.links:
                db.insert_url_links(url_id, crawl_result.links)
        except DatabaseError as e:
            logger.error(f"Database error when inserting URL {url}: {e}")
            return False

        # If not a recruitment advert and not processing all prompts, stop here
        if recruitment_flag == 0 and not process_all_prompts:
            logger.info(f"URL {url} is not a recruitment advert. Skipping detailed extraction.")
            return True

        # Collect responses for all prompts
        prompt_responses = collect_prompt_responses(chat_engine)

        # Process all responses using batch processor
        try:
            results = process_all_prompt_responses(
                db,
                url_id,
                prompt_responses,
                use_transaction=use_transaction
            )

            if results["success"]:
                logger.info(f"Successfully processed {results['processed']} responses for URL: {url}")
            else:
                logger.warning(f"Some errors occurred while processing URL: {url}")
                for error in results["errors"]:
                    logger.error(f"Error in {error['prompt_type']}: {error['error']}")

                # Only mark as failed if all responses failed
                if results["processed"] == 0 and results["failed"] > 0:
                    return False
        except Exception as e:
            logger.error(f"Error in batch processing for URL {url}: {e}", exc_info=True)
            return False

        logger.info(f"Successfully processed URL: {url}")
        return True

    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
        return False


def filter_valid_urls(urls: List[str]) -> List[str]:
    """
    Filter out invalid URLs from a list.

    Args:
        urls: List of URLs to validate

    Returns:
        List of valid URLs
    """
    valid_urls = []
    invalid_count = 0

    for url in urls:
        is_valid, normalized_url, error = validate_and_normalize_url(url)
        if is_valid:
            valid_urls.append(normalized_url)
        else:
            invalid_count += 1
            logger.warning(f"Filtered out invalid URL: {url}. Reason: {error}")

    if invalid_count > 0:
        logger.info(f"Filtered out {invalid_count} invalid URLs from the list.")

    return valid_urls


def backfill_skills(args, db, processor, llm, memory):
    """
    Backfill skills for URLs that already exist in the database but are missing skills.
    """
    # Get all URLs with recruitment_flag = 1 (confirmed recruitment)
    query = """
        SELECT u.id, u.url FROM urls u
        LEFT JOIN (
            SELECT url_id, COUNT(*) as skill_count 
            FROM skills 
            GROUP BY url_id
        ) s ON u.id = s.url_id
        WHERE u.recruitment_flag = 1 
        AND (s.skill_count IS NULL OR s.skill_count = 0)
        ORDER BY u.id
    """

    with db._execute_query(query) as cursor:
        urls_missing_skills = cursor.fetchall()

    if not urls_missing_skills:
        logger.info("No URLs missing skills found")
        return

    logger.info(f"Found {len(urls_missing_skills)} URLs missing skills")

    for url_id, url in urls_missing_skills:
        logger.info(f"Processing skills for URL ID {url_id}: {url}")
        try:
            # Get content
            url_record = db.get_url_by_id(url_id)
            if not url_record or not url_record.content:
                logger.warning(f"No content found for URL ID {url_id}")
                continue

            # Process skills
            from web_crawler_lib import WebCrawlerResult
            from llama_index.core import Document as liDocument
            from llama_index.core import VectorStoreIndex

            documents = [liDocument(text=url_record.content)]
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

            success = process_url_skills(url_id, db, chat_engine)

            if success:
                logger.info(f"Successfully backfilled skills for URL ID {url_id}")
            else:
                logger.warning(f"Failed to backfill skills for URL ID {url_id}")

        except Exception as e:
            logger.error(f"Error backfilling skills for URL ID {url_id}: {e}")

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
    parser.add_argument("--no-transaction", action="store_true",
                        help="Disable transaction support (not recommended)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip URL validation (not recommended)")
    parser.add_argument("--backfill-skills", action="store_true",
                        help="Backfill skills for URLs that are missing them")

    args = parser.parse_args()

    # Initialize database and processor
    db = RecruitmentDatabase()
    processor = PromptResponseProcessor(db)
    # Run diagnostics to check for issues with skills table
    # db.check_skills_table_schema()
    # # Test direct skill insertion with a dummy URL ID (use an ID you know exists)
    # known_url_id = 1  # Change this to an actual URL ID from your database
    # db.test_skill_insertion(known_url_id)
    # Set up LLM and memory
    llm, memory = setup_llm_engine(args.api_key)

    if args.backfill_skills:
        print("Starting skills backfill process...")
        backfill_skills(args, db, processor, llm, memory)
        print("Skills backfill completed.")
        return
    # Process a single URL if specified
    if args.single_url:
        # Validate the URL if validation is enabled
        if not args.skip_validation:
            is_valid, normalized_url, error = validate_and_normalize_url(args.single_url)
            if not is_valid:
                print(f"Invalid URL: {args.single_url}. Error: {error}")
                return
            args.single_url = normalized_url

        success = process_url(args.single_url, db, processor, llm, memory, args.process_all)
        if success:
            print(f"Successfully processed URL: {args.single_url}")
        else:
            print(f"Failed to process URL: {args.single_url}")
        return

    # Get URLs to process
    try:
        urls_from_files = get_unique_urls_from_csvs(
            args.csv_dir, args.url_column, args.min_rows, args.max_urls
        )
        print(f"Found {len(urls_from_files)} unique URLs in CSV files")

        # Filter out invalid URLs if validation is enabled
        if not args.skip_validation:
            urls_from_files = filter_valid_urls(urls_from_files)
            print(f"After validation: {len(urls_from_files)} valid URLs")
    except Exception as e:
        logger.error(f"Error getting URLs from CSV files: {e}", exc_info=True)
        print(f"Error getting URLs from CSV files: {e}")
        return

    # Filter out URLs already in the database
    try:
        urls_in_db = pd.DataFrame(db.search_urls(limit=1000000))
        if urls_in_db.empty:
            urls = urls_from_files
            print("No existing URLs found in database")
        else:
            urls_from_db = urls_in_db['url'].tolist()
            print(f"Found {len(urls_from_db)} URLs already in database")
            urls = list(set(urls_from_files) - set(urls_from_db))
            print(f"After removing existing URLs: {len(urls)} URLs to process")
    except Exception as e:
        logger.error(f"Error retrieving URLs from database: {e}", exc_info=True)
        print(f"Error retrieving URLs from database: {e}")
        # Default to processing all URLs from files
        urls = urls_from_files

    # Sort URLs for more deterministic processing
    urls.sort()

    # Limit the number of URLs to process
    if not urls:
        print("No URLs to process. Exiting.")
        return

    if args.start_index >= len(urls):
        print(f"Start index {args.start_index} is out of range. Only {len(urls)} URLs available.")
        return

    urls_to_process = urls[args.start_index:args.start_index + args.max_urls]
    print(f"Processing {len(urls_to_process)} URLs starting from index {args.start_index}")

    # Process each URL
    success_count = 0
    failure_count = 0

    for i, url in enumerate(urls_to_process):
        print(f"Processing URL {i + 1}/{len(urls_to_process)}: {url}")
        try:
            success = process_url(url, db, processor, llm, memory, args.process_all)
            if success:
                logger.info(f"Successfully processed URL: {url}")
                print(f"✓ Success: {url}")
                success_count += 1
            else:
                logger.warning(f"Failed to process URL: {url}")
                print(f"✗ Failed: {url}")
                failure_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
            print(f"✗ Error: {url} - {str(e)}")
            failure_count += 1

        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)

    print(f"\nProcessing completed. {success_count} URLs succeeded, {failure_count} failed.")
    logger.info(f"Batch processing completed. {success_count} URLs succeeded, {failure_count} failed.")


if __name__ == "__main__":
    main()