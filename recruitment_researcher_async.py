#!/usr/bin/env python3
"""
Async Recruitment Researcher

This script processes URLs to extract job recruitment data using LLMs and stores the results in a database.
It uses web_crawler_lib for content extraction and supports concurrent processing of multiple URLs.
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import aiosqlite
import pandas as pd
import tldextract
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Local imports
from recruitment_models import transform_skills_response
from get_urls_from_csvs import get_unique_urls_from_csvs
from prompts import COMPLEX_PROMPTS, LIST_PROMPTS, NON_LIST_PROMPTS
from recruitment_db import DatabaseError
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
logger = setup_logging("recruitment_researcher_async")

class AsyncRecruitmentDatabase:
    """Async wrapper for database operations."""
    
    def __init__(self, db_path: str = "databases/recruitment.db"):
        self.db_path = db_path

    async def __aenter__(self):
        self.db = await aiosqlite.connect(self.db_path)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db.close()

    async def url_exists(self, url: str) -> bool:
        """Check if a URL already exists in the database."""
        query = "SELECT COUNT(*) FROM urls WHERE url = ?"
        async with self.db.execute(query, (url,)) as cursor:
            result = await cursor.fetchone()
            return result[0] > 0

    async def update_url(self, url: str, url_data: dict) -> int:
        """Update an existing URL in the database."""
        query = """
        UPDATE urls 
        SET domain_name = ?, source = ?, content = ?, 
            recruitment_flag = ?, accessible = ?, error_message = ?
        WHERE url = ?
        """
        params = (
            url_data["domain_name"],
            url_data["source"],
            url_data.get("content", ""),
            url_data.get("recruitment_flag", -1),
            url_data.get("accessible", 1),
            url_data.get("error_message", None),
            url
        )
        await self.db.execute(query, params)
        await self.db.commit()
        return await self.get_url_id(url)

    async def insert_url(self, url_data: dict) -> int:
        """Insert or update a URL in the database."""
        url = url_data["url"]
        try:
            # Check if URL exists
            if await self.url_exists(url):
                # Update existing URL
                return await self.update_url(url, url_data)
            else:
                # Insert new URL
                query = """
                INSERT INTO urls (url, domain_name, source, content, recruitment_flag, accessible, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    url,
                    url_data["domain_name"],
                    url_data["source"],
                    url_data.get("content", ""),
                    url_data.get("recruitment_flag", -1),
                    url_data.get("accessible", 1),
                    url_data.get("error_message", None)
                )
                async with self.db.execute(query, params) as cursor:
                    await self.db.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Database error for URL {url}: {e}")
            raise

    async def get_url_id(self, url: str) -> Optional[int]:
        """Get URL ID asynchronously."""
        query = "SELECT id FROM urls WHERE url = ?"
        async with self.db.execute(query, (url,)) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else None

async def setup_llm_engine(api_key: Optional[str] = None) -> Tuple[OpenAI, ChatMemoryBuffer]:
    """
    Set up the LLM and memory for chat interactions asynchronously.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Initialize LLM with conservative settings
    llm = OpenAI(temperature=0, model="gpt-4-turbo-preview", request_timeout=120.0)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    return llm, memory

def clean_response_text(response_text: str) -> str:
    """Clean LLM response text by removing markdown code blocks and other formatting."""
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        return response_text.split("```")[1].split("```")[0].strip()
    else:
        return response_text.strip()

async def process_url_skills(url_id: int, db, chat_engine) -> bool:
    """Process skills specifically for a URL ID asynchronously."""
    # Check if URL already has skills
    async with db.db.execute("SELECT COUNT(*) FROM skills WHERE url_id = ?", (url_id,)) as cursor:
        result = await cursor.fetchone()
        count = result[0] if result else 0

    if count > 0:
        logger.info(f"URL ID {url_id} already has {count} skills. Skipping.")
        return False

    # Get URL record
    async with db.db.execute("SELECT content FROM urls WHERE id = ?", (url_id,)) as cursor:
        result = await cursor.fetchone()
        if not result or not result[0]:
            logger.warning(f"No content found for URL ID {url_id}")
            return False
        content = result[0]

    # Process skills using the chat engine
    prompt_key = "skills_prompt"
    prompt_text = COMPLEX_PROMPTS[prompt_key]
    
    # Use ThreadPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor() as executor:
        skills_data = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: get_validated_response(prompt_key, prompt_text, SkillExperienceResponse, chat_engine)
        )

    if skills_data:
        processed_skills = transform_skills_response(skills_data)
        if processed_skills and processed_skills.get('skills'):
            # Insert skills
            await insert_skills(db, url_id, processed_skills['skills'])
            logger.info(f"Successfully processed skills for URL ID {url_id}")
            return True

    logger.warning(f"No skills could be extracted for URL ID {url_id}")
    return False

async def insert_skills(db, url_id: int, skills: List[Dict[str, str]]) -> None:
    """Insert skills into the database asynchronously."""
    query = "INSERT INTO skills (url_id, skill, experience) VALUES (?, ?, ?)"
    for skill_data in skills:
        params = (url_id, skill_data['skill'], skill_data.get('experience'))
        await db.db.execute(query, params)
    await db.db.commit()

async def process_url(url: str, db: AsyncRecruitmentDatabase, processor: PromptResponseProcessor,
                     llm: OpenAI, memory: ChatMemoryBuffer) -> bool:
    """Process a single URL asynchronously."""
    logger.info(f"Processing URL: {url}")

    # Validate and normalize URL
    is_valid, normalized_url, error_message = validate_and_normalize_url(url)
    if not is_valid:
        logger.warning(f"Invalid URL: {url}. {error_message}")
        await record_invalid_url(db, url, error_message)
        return False

    url = normalized_url
    logger.info(f"Normalized URL: {url}")

    try:
        # Use ThreadPoolExecutor for CPU-bound tasks like web crawling
        with ThreadPoolExecutor() as executor:
            crawl_result = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: crawl_website_sync(url=url, excluded_tags=['form', 'header'], verbose=True)
            )

        if not crawl_result.success:
            logger.warning(f"Failed to extract content from URL: {url}")
            await record_failed_url(db, url, crawl_result.error_message)
            return False

        # Extract domain
        extracted = tldextract.extract(url)
        domain_name = extracted.domain

        # Process content
        text = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
        if not text:
            logger.warning(f"Empty content extracted from URL: {url}")
            await record_empty_content_url(db, url, domain_name)
            return False

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

        # Process URL with chat engine
        with ThreadPoolExecutor() as executor:
            recruitment_type, evidence = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: verify_recruitment(url, chat_engine)
            )

        recruitment_flag = recruitment_type.get("recruitment_flag", -1)

        # Record URL in database
        url_data = {
            "url": url,
            "domain_name": domain_name,
            "source": "crawler",
            "content": text,
            "recruitment_flag": recruitment_flag,
            "accessible": 1
        }
        
        url_id = await db.insert_url(url_data)
        
        if crawl_result.links:
            await insert_links(db, url_id, crawl_result.links)

        if recruitment_flag == 1:
            # Process recruitment data
            await process_recruitment_data(url_id, db, chat_engine, processor)

        logger.info(f"Successfully processed URL: {url}")
        return True

    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
        return False

async def process_recruitment_data(url_id: int, db: AsyncRecruitmentDatabase, 
                                 chat_engine, processor: PromptResponseProcessor) -> None:
    """Process recruitment data for a URL asynchronously."""
    # Collect responses for all prompts using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        prompt_responses = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: collect_prompt_responses(chat_engine)
        )

    # Process responses
    await process_responses(url_id, db, prompt_responses, processor)

async def process_responses(url_id: int, db: AsyncRecruitmentDatabase, 
                          prompt_responses: dict, processor: PromptResponseProcessor) -> None:
    """Process all prompt responses asynchronously."""
    for prompt_key, response in prompt_responses.items():
        try:
            await processor.process_response_async(prompt_key, response, url_id, db)
        except Exception as e:
            logger.error(f"Error processing {prompt_key} for URL ID {url_id}: {e}")

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

async def record_invalid_url(db: AsyncRecruitmentDatabase, url: str, error_message: str) -> None:
    """Record an invalid URL in the database."""
    url_data = {
        "url": url,
        "domain_name": "",  # No valid domain
        "source": "crawler",
        "content": "",  # No content due to invalid URL
        "recruitment_flag": -1,  # Use a default status indicating not processed
        "accessible": 0,
        "error_message": error_message
    }
    await db.insert_url(url_data)

async def record_failed_url(db: AsyncRecruitmentDatabase, url: str, error_message: str) -> None:
    """Record a failed URL in the database."""
    extracted = tldextract.extract(url)
    url_data = {
        "url": url,
        "domain_name": extracted.domain,
        "source": "crawler",
        "content": "",  # No content because extraction failed
        "recruitment_flag": -1,
        "accessible": 0,
        "error_message": error_message
    }
    await db.insert_url(url_data)

async def record_empty_content_url(db: AsyncRecruitmentDatabase, url: str, domain_name: str) -> None:
    """Record a URL with empty content in the database."""
    url_data = {
        "url": url,
        "domain_name": domain_name,
        "source": "crawler",
        "content": "",
        "recruitment_flag": -1,
        "accessible": 1,  # URL was accessible but extraction yielded no content
        "error_message": "Empty content extracted"
    }
    await db.insert_url(url_data)

async def insert_links(db: AsyncRecruitmentDatabase, url_id: int, links: List[str]) -> None:
    """Insert links into the database asynchronously."""
    query = "INSERT INTO url_links (url_id, link) VALUES (?, ?)"
    for link in links:
        await db.db.execute(query, (url_id, link))
    await db.db.commit()

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

def get_validated_response(prompt_key: str, prompt_text: str, model_class: Any, chat_engine) -> Optional[Any]:
    """
    Sends a prompt to the chat engine and returns a validated Pydantic model instance with exponential backoff.
    """
    max_retries = 5
    base_delay = 5  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = chat_engine.chat(prompt_text)
            logger.info(f"Received response for '{prompt_key}' (attempt {attempt + 1}/{max_retries})")
            response_text = clean_response_text(response.response)

            # Special handling for the skills prompt which returns tuples
            if prompt_key == "skills_prompt":
                return handle_skills_response(response_text, model_class)

            # Standard handling for other prompts
            try:
                response_data = model_class.model_validate_json(response_text)
                logger.info(f"Prompt '{prompt_key}' processed successfully.")
                return response_data
            except Exception as e:
                logger.error(f"Validation error for '{prompt_key}': {e}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for '{prompt_key}' (attempt {attempt + 1}/{max_retries}): {e}")
            logger.debug(f"Raw response: {response.response[:500]}...")
        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Unexpected error for '{prompt_key}': {e}")

    logger.error(f"Max retries reached for '{prompt_key}'. Skipping...")
    return None

def handle_skills_response(response_text: str, model_class: Any) -> Optional[Any]:
    """Handle the special case of skills response parsing."""
    logger.info(f"Skills response_text for skills_prompt yields '{response_text}'.")
    try:
        if "skills" in response_text and "(" in response_text and ")" in response_text:
            modified_text = response_text.replace("(", "[").replace(")", "]")
            logger.info(f"Skills modified_text yields '{modified_text}'.")
            try:
                response_json = json.loads(modified_text)
                if 'skills' in response_json and isinstance(response_json['skills'], list):
                    processed_skills = []
                    for item in response_json['skills']:
                        if isinstance(item, list):
                            if len(item) >= 2:
                                skill = item[0]
                                experience = None if item[1] == "not_listed" else item[1]
                                processed_skills.append({"skill": skill, "experience": experience})
                            else:
                                processed_skills.append({"skill": item[0], "experience": None})
                        elif isinstance(item, str):
                            processed_skills.append({"skill": item, "experience": None})

                    response_data = {"skills": processed_skills}
                    logger.info(f"Skills prompt yields '{response_data}' after transformation.")

                    try:
                        validated_data = model_class.model_validate(response_data)
                        logger.info(f"Skills prompt yields '{validated_data}' after transformation.")
                        return validated_data
                    except Exception as e:
                        logger.error(f"Skills validation error after transformation: {e}")

            except json.JSONDecodeError:
                logger.warning("Failed to parse modified JSON for skills prompt")

    except Exception as e:
        logger.error(f"Error processing skills response: {e}")

    return None

def collect_prompt_responses(chat_engine) -> Dict[str, Any]:
    """
    Collect responses for all prompts using the chat engine.
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

    # Process LIST_PROMPTS (excluding recruitment prompt)
    for prompt_key, prompt_text in {k: v for k, v in LIST_PROMPTS.items() if k != "recruitment_prompt"}.items():
        model_class = get_model_for_prompt(prompt_key)
        if not model_class:
            logger.warning(f"No model class found for prompt key: {prompt_key}")
            continue

        response_data = get_validated_response(prompt_key, prompt_text, model_class, chat_engine)
        if response_data:
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

async def main():
    """Main async function to run the recruitment URL processing."""
    parser = argparse.ArgumentParser(description="Process URLs to extract recruitment data asynchronously")
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
    parser.add_argument("--api-key", type=str, default=None,
                      help="API key for the LLM service (optional)")
    parser.add_argument("--max-concurrent", type=int, default=5,
                      help="Maximum number of URLs to process concurrently")
    parser.add_argument("--force-update", action="store_true",
                      help="Force update of URLs even if they exist in the database")
    parser.add_argument("--skip-validation", action="store_true",
                      help="Skip URL validation (not recommended)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting URL processing with parameters:")
    logger.info(f"  CSV Directory: {args.csv_dir}")
    logger.info(f"  URL Column: {args.url_column}")
    logger.info(f"  Max URLs: {args.max_urls}")
    logger.info(f"  Start Index: {args.start_index}")
    logger.info(f"  Max Concurrent: {args.max_concurrent}")
    logger.info(f"  Force Update: {args.force_update}")
    logger.info(f"  Skip Validation: {args.skip_validation}")

    # Initialize async components
    logger.info("Initializing LLM engine and database connection...")
    llm, memory = await setup_llm_engine(args.api_key)
    
    async with AsyncRecruitmentDatabase() as db:
        processor = PromptResponseProcessor(db)
        logger.info("Database and processor initialized successfully")

        if args.single_url:
            logger.info(f"Processing single URL: {args.single_url}")
            # Validate the URL if validation is enabled
            if not args.skip_validation:
                is_valid, normalized_url, error = validate_and_normalize_url(args.single_url)
                if not is_valid:
                    logger.error(f"Invalid URL: {args.single_url}. Error: {error}")
                    print(f"Invalid URL: {args.single_url}. Error: {error}")
                    return
                args.single_url = normalized_url
                logger.info(f"URL normalized to: {normalized_url}")

            success = await process_url(args.single_url, db, processor, llm, memory)
            logger.info(f"Single URL processing {'succeeded' if success else 'failed'}")
            print(f"{'Successfully' if success else 'Failed to'} processed URL: {args.single_url}")
            return

        # Get URLs to process
        try:
            logger.info(f"Reading URLs from CSV files in {args.csv_dir}...")
            urls_from_files = get_unique_urls_from_csvs(
                args.csv_dir, args.url_column, args.min_rows, args.max_urls
            )
            logger.info(f"Found {len(urls_from_files)} unique URLs in CSV files")
            print(f"Found {len(urls_from_files)} unique URLs in CSV files")

            # Filter out invalid URLs if validation is enabled
            if not args.skip_validation:
                logger.info("Starting URL validation...")
                valid_urls = []
                invalid_count = 0
                for i, url in enumerate(urls_from_files, 1):
                    if i % 100 == 0:  # Log progress every 100 URLs
                        logger.info(f"Validated {i}/{len(urls_from_files)} URLs...")
                    
                    is_valid, normalized_url, error = validate_and_normalize_url(url)
                    if is_valid:
                        valid_urls.append(normalized_url)
                        logger.debug(f"Valid URL: {url} -> {normalized_url}")
                    else:
                        invalid_count += 1
                        logger.warning(f"Invalid URL ({invalid_count}): {url}. Reason: {error}")
                
                if invalid_count > 0:
                    logger.info(f"Filtered out {invalid_count} invalid URLs from the list")
                urls_from_files = valid_urls
                logger.info(f"After validation: {len(urls_from_files)} valid URLs")
                print(f"After validation: {len(urls_from_files)} valid URLs")
            
            # Apply start index and max URLs limit
            start = args.start_index
            end = min(start + args.max_urls, len(urls_from_files))
            urls_slice = urls_from_files[start:end]
            logger.info(f"Selected URLs from index {start} to {end} (total: {len(urls_slice)})")
            
            # Filter out URLs already in the database unless force update is enabled
            if not args.force_update:
                logger.info("Checking for existing URLs in database...")
                urls_to_process = []
                existing_count = 0
                for i, url in enumerate(urls_slice, 1):
                    if i % 100 == 0:  # Log progress every 100 URLs
                        logger.info(f"Checked {i}/{len(urls_slice)} URLs in database...")
                    
                    if not await db.url_exists(url):
                        urls_to_process.append(url)
                        logger.debug(f"New URL to process: {url}")
                    else:
                        existing_count += 1
                        logger.debug(f"Skipping existing URL: {url}")
                        print(f"Skipping existing URL: {url}")
                
                logger.info(f"Found {existing_count} existing URLs in database")
            else:
                logger.info("Force update enabled - processing all URLs regardless of existence")
                urls_to_process = urls_slice

            logger.info(f"Final count: {len(urls_to_process)} URLs to process")
            print(f"Processing {len(urls_to_process)} URLs")

            if not urls_to_process:
                logger.info("No URLs to process. Exiting.")
                print("No URLs to process. Exiting.")
                return

            # Process URLs concurrently with rate limiting
            logger.info(f"Starting concurrent processing with max {args.max_concurrent} simultaneous requests")
            semaphore = asyncio.Semaphore(args.max_concurrent)
            tasks = []
            
            async def process_with_semaphore(url, index):
                async with semaphore:
                    try:
                        logger.info(f"Starting processing of URL {index + 1}/{len(urls_to_process)}: {url}")
                        result = await process_url(url, db, processor, llm, memory)
                        logger.info(f"Completed URL {index + 1}/{len(urls_to_process)}: {url} - {'Success' if result else 'Failed'}")
                        return result
                    except Exception as e:
                        logger.error(f"Error processing URL {index + 1}/{len(urls_to_process)} - {url}: {e}", exc_info=True)
                        return False

            for i, url in enumerate(urls_to_process):
                task = asyncio.create_task(process_with_semaphore(url, i))
                tasks.append(task)

            # Wait for all tasks to complete
            logger.info("Waiting for all URL processing tasks to complete...")
            results = await asyncio.gather(*tasks)
            
            # Count successes and failures
            success_count = sum(1 for r in results if r is True)
            failure_count = len(results) - success_count
            
            logger.info(f"Processing completed. {success_count} URLs succeeded, {failure_count} failed")
            print(f"\nProcessing completed. {success_count} URLs succeeded, {failure_count} failed")

        except Exception as e:
            logger.error(f"Error in main processing loop: {e}", exc_info=True)
            print(f"Error in main processing loop: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 