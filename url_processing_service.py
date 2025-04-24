#!/usr/bin/env python3
"""
URL Processing Service

This FastAPI service consumes URLs from a RabbitMQ queue, processes them using LLMs,
and stores the results in a database. It's based on the new_recruitment_researcher.py script.
"""

import os
import json
import logging
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pika
import asyncio
from dotenv import load_dotenv
import tldextract
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from contextlib import asynccontextmanager

# Local imports
from recruitment_models import transform_skills_response
from prompts import COMPLEX_PROMPTS, LIST_PROMPTS, NON_LIST_PROMPTS
from recruitment.recruitment_db import DatabaseError, RecruitmentDatabase
from recruitment_models import (AgencyResponse, AttributesResponse, BenefitsResponse,
                                CompanyResponse, ConfirmResponse, ContactPersonResponse,
                                EmailResponse, JobAdvertResponse, JobResponse,
                                LocationResponse, CompanyPhoneNumberResponse, LinkResponse,
                                SkillExperienceResponse, DutiesResponse,
                                QualificationsResponse, AdvertResponse)
from response_processor_functions import PromptResponseProcessor, ResponseProcessingError
from batch_processor import process_all_prompt_responses, direct_insert_skills
from web_crawler_lib import crawl_website_sync, WebCrawlerResult, crawl_website_sync_v2
from logging_config import setup_logging
from libraries.config_validator import ConfigValidator
from utils import get_model_for_prompt

# Load environment variables
load_dotenv()

# Create module-specific logger
logger = setup_logging("url_processing_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup: Start background tasks
    background_task = asyncio.create_task(process_urls_from_queue())
    
    yield  # This is where the application runs
    
    # Shutdown: Cancel background tasks
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="URL Processing Service",
    description="Service for processing recruitment URLs and storing results in a database",
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

# Pydantic models for API
class URLProcessingRequest(BaseModel):
    url: str
    search_id: Optional[str] = None
    process_all_prompts: bool = True
    use_transaction: bool = True

class URLProcessingResponse(BaseModel):
    url: str
    success: bool
    recruitment_flag: int
    error_message: Optional[str] = None
    timestamp: str

class URLProcessingStatus(BaseModel):
    url: str
    status: str
    recruitment_flag: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str

# Initialize database
db = RecruitmentDatabase()

# Initialize processor
processor = PromptResponseProcessor(db)

# Set up LLM and memory
def setup_llm_engine(api_key: Optional[str] = None) -> Tuple[OpenAI, ChatMemoryBuffer]:
    """Set up the LLM and memory for chat interactions."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Initialize LLM with conservative settings
    llm = OpenAI(temperature=0, model="gpt-4o-mini", request_timeout=120.0)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    return llm, memory

# Get LLM and memory
llm, memory = setup_llm_engine()

# RabbitMQ connection
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

# URL validation
def validate_and_normalize_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """Validate and normalize a URL for crawling."""
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

# Verify recruitment
def verify_recruitment(url: str, chat_engine) -> Tuple[Dict[str, Any], List[str]]:
    """Verifies if a URL contains a recruitment advertisement."""
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

# Clean response text
def clean_response_text(response_text: str) -> str:
    """Clean LLM response text by removing markdown code blocks and other formatting."""
    # Handle empty responses
    if not response_text or response_text.strip() == "Empty Response":
        # Return appropriate empty JSON structure based on common response formats
        return "{}"
    
    # Try to clean up the response if it contains markdown code blocks
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        return response_text.split("```")[1].split("```")[0].strip()
    else:
        return response_text.strip()

# Get validated response
def get_validated_response(prompt_key: str, prompt_text: str, model_class: Any, chat_engine) -> Optional[Any]:
    """Sends a prompt to the chat engine and returns a validated Pydantic model instance with exponential backoff."""
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

                    # If response is empty or has a null skills value, return an empty skills response
                    if not response_text or response_text == "{}" or '"skills": null' in response_text:
                        default_response = {"skills": []}
                        logger.info(f"Returning default empty skills response for '{prompt_key}'")
                        return model_class.model_validate(default_response)

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
                
                # Provide appropriate default responses based on prompt type
                if response_text == "{}":
                    if prompt_key == "attributes_prompt":
                        return model_class.model_validate({"attributes": []})
                    elif prompt_key == "duties_prompt":
                        return model_class.model_validate({"duties": []})
                    elif prompt_key == "qualifications_prompt":
                        return model_class.model_validate({"qualifications": []})
                    elif prompt_key == "benefits_prompt":
                        return model_class.model_validate({"benefits": []})
                    elif prompt_key == "location_prompt":
                        return model_class.model_validate({"country": None, "province": None, "city": None, "street_address": None})
                    elif prompt_key == "jobadvert_prompt" or prompt_key == "job_advert_prompt":
                        return model_class.model_validate({
                            "description": None, 
                            "salary": None, 
                            "duration": None, 
                            "start_date": None, 
                            "end_date": None, 
                            "posted_date": None, 
                            "application_deadline": None
                        })
                    elif prompt_key == "recruitment_prompt":
                        # Default to "no" when the model returns an empty response
                        logger.info(f"Empty response for recruitment_prompt, defaulting to 'no'")
                        return model_class.model_validate({"answer": "no", "evidence": None})
                # Continue to next attempt if we can't create a default response

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
    
    # When all retries fail, return appropriate empty responses
    if prompt_key == "attributes_prompt":
        return model_class.model_validate({"attributes": []})
    elif prompt_key == "duties_prompt":
        return model_class.model_validate({"duties": []})
    elif prompt_key == "qualifications_prompt":
        return model_class.model_validate({"qualifications": []})
    elif prompt_key == "benefits_prompt":
        return model_class.model_validate({"benefits": []})
    elif prompt_key == "skills_prompt":
        return model_class.model_validate({"skills": []})
    elif prompt_key == "location_prompt":
        return model_class.model_validate({"country": None, "province": None, "city": None, "street_address": None})
    elif prompt_key == "jobadvert_prompt" or prompt_key == "job_advert_prompt":
        return model_class.model_validate({
            "description": None, 
            "salary": None, 
            "duration": None, 
            "start_date": None, 
            "end_date": None, 
            "posted_date": None, 
            "application_deadline": None
        })
    
    return None

# Collect prompt responses
def collect_prompt_responses(chat_engine) -> Dict[str, Any]:
    """Collect responses for all prompts using the chat engine."""
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

# Process URL
def process_url(url: str, process_all_prompts: bool = True, use_transaction: bool = True) -> Tuple[bool, int, Optional[str]]:
    """Process a single URL to extract recruitment data using the new database schema."""
    logger.info(f"Processing URL: {url}")

    # Validate and normalize the URL
    is_valid, normalized_url, error_message = validate_and_normalize_url(url)

    if not is_valid:
        logger.warning(f"Invalid URL: {url}. {error_message}")
        return False, -1, error_message

    # Use the normalized URL for crawling
    url = normalized_url
    logger.info(f"Normalized URL: {url}")

    # Extract domain
    extracted = tldextract.extract(url)
    domain_name = extracted.domain

    # Check if URL already exists in the database
    try:
        # Insert URL into database with pending status
        url_id = db.insert_url(url, domain_name, "crawler")
        
        # Update URL processing status to pending
        db.update_url_processing_status(url_id, "pending")
        
        # Extract content using web_crawler_lib
        try:
            crawl_result = crawl_website_sync(
                url=url,
                excluded_tags=['form', 'header'],
                verbose=True
            )
        except Exception as e:
            logger.error(f"Crawler exception for URL {url}: {e}", exc_info=True)
            # Update URL processing status to failed
            db.update_url_processing_status(url_id, "failed", error_count=1)
            return False, -1, f"Crawler exception: {str(e)}"

        if not crawl_result.success:
            logger.warning(f"Failed to extract content from URL: {url}")
            # Update URL processing status to failed
            db.update_url_processing_status(url_id, "failed", error_count=1)
            return False, -1, crawl_result.error_message

        # Use the markdown content from the crawler
        text = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
        if not text:
            logger.warning(f"Empty content extracted from URL: {url}")
            # Update URL processing status to failed
            db.update_url_processing_status(url_id, "failed", error_count=1)
            return False, -1, "Empty content extracted"

        # Update URL processing status to processing
        db.update_url_processing_status(url_id, "processing")

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
            recruitment_result, evidence = verify_recruitment(url, chat_engine)
            recruitment_flag = recruitment_result["recruitment_flag"]

            # If not a recruitment advert and not processing all prompts, stop here
            if recruitment_flag == 0 and not process_all_prompts:
                logger.info(f"URL {url} is not a recruitment advert. Skipping detailed extraction.")
                # Update URL processing status to completed
                db.update_url_processing_status(url_id, "completed")
                return True, recruitment_flag, None

            # Collect responses for all prompts
            prompt_responses = collect_prompt_responses(chat_engine)

            # Process the responses and store in the database
            try:
                # Create a job record
                job_data = prompt_responses.get("job_prompt", {})
                job_id = db.insert_job(
                    title=job_data.get("title", "Unknown Job Title"),
                    description=job_data.get("description"),
                    salary_min=job_data.get("salary_min"),
                    salary_max=job_data.get("salary_max"),
                    salary_currency=job_data.get("salary_currency"),
                    status="active"
                )
                
                # Create the advert and link it to the URL
                advert_data = prompt_responses.get("jobadvert_prompt", {})
                advert_id = db.insert_advert(
                    job_id=job_id,
                    posted_date=advert_data.get("posted_date"),
                    application_deadline=advert_data.get("application_deadline"),
                    is_remote=advert_data.get("is_remote", False),
                    is_hybrid=advert_data.get("is_hybrid", False)
                )
                
                # Link job to URL
                db.link_job_url(job_id, url_id)
                
                # Process skills
                if "skills_prompt" in prompt_responses:
                    skills_data = prompt_responses["skills_prompt"]
                    if hasattr(skills_data, "skills"):
                        skills = skills_data.skills
                    else:
                        skills = skills_data.get("skills", [])
                        
                    for skill_data in skills:
                        skill_name = skill_data.get("skill")
                        experience = skill_data.get("experience")
                        if skill_name:
                            skill_id = db.insert_skill(skill_name)
                            db.link_job_skill(job_id, skill_id, experience)
                
                # Process qualifications
                if "qualifications_prompt" in prompt_responses:
                    quals_data = prompt_responses["qualifications_prompt"]
                    if hasattr(quals_data, "qualifications"):
                        qualifications = quals_data.qualifications
                    else:
                        qualifications = quals_data.get("qualifications", [])
                        
                    for qual_name in qualifications:
                        if qual_name:
                            qual_id = db.insert_qualification(qual_name)
                            db.link_job_qualification(job_id, qual_id)
                
                # Process attributes
                if "attributes_prompt" in prompt_responses:
                    attrs_data = prompt_responses["attributes_prompt"]
                    if hasattr(attrs_data, "attributes"):
                        attributes = attrs_data.attributes
                    else:
                        attributes = attrs_data.get("attributes", [])
                        
                    for attr_name in attributes:
                        if attr_name:
                            attr_id = db.insert_attribute(attr_name)
                            db.link_job_attribute(job_id, attr_id)
                
                # Process duties
                if "duties_prompt" in prompt_responses:
                    duties_data = prompt_responses["duties_prompt"]
                    if hasattr(duties_data, "duties"):
                        duties = duties_data.duties
                    else:
                        duties = duties_data.get("duties", [])
                        
                    for duty_desc in duties:
                        if duty_desc:
                            duty_id = db.insert_duty(duty_desc)
                            db.link_job_duty(job_id, duty_id)
                
                # Process benefits
                if "benefits_prompt" in prompt_responses:
                    benefits_data = prompt_responses["benefits_prompt"]
                    if hasattr(benefits_data, "benefits"):
                        benefits = benefits_data.benefits
                    else:
                        benefits = benefits_data.get("benefits", [])
                        
                    for benefit_desc in benefits:
                        if benefit_desc:
                            benefit_id = db.insert_benefit(benefit_desc)
                            # Note: There's no direct link between jobs and benefits in the new schema
                
                # Process company
                if "company_prompt" in prompt_responses:
                    company_data = prompt_responses["company_prompt"]
                    company_name = company_data.get("name")
                    if company_name:
                        company_id = db.insert_company(company_name)
                        db.link_job_company(job_id, company_id)
                
                # Process agency
                if "agency_prompt" in prompt_responses:
                    agency_data = prompt_responses["agency_prompt"]
                    agency_name = agency_data.get("name")
                    if agency_name:
                        agency_id = db.insert_agency(agency_name)
                        db.link_job_agency(job_id, agency_id)
                
                # Process location
                if "location_prompt" in prompt_responses:
                    location_data = prompt_responses["location_prompt"]
                    country = location_data.get("country")
                    province = location_data.get("province")
                    city = location_data.get("city")
                    
                    if country or province or city:
                        location_id = db.insert_location(country, province, city)
                        db.link_job_location(job_id, location_id)
                
                # Process industry
                if "industry_prompt" in prompt_responses:
                    industry_data = prompt_responses["industry_prompt"]
                    industry_name = industry_data.get("industry")
                    if industry_name:
                        industry_id = db.insert_industry(industry_name)
                        db.link_job_industry(job_id, industry_id)
                
                # Update URL processing status to completed
                db.update_url_processing_status(url_id, "completed")
                
                logger.info(f"Successfully processed URL: {url}")
                return True, recruitment_flag, None
                
            except Exception as e:
                logger.error(f"Error processing prompt responses for URL {url}: {e}", exc_info=True)
                # Update URL processing status to failed
                db.update_url_processing_status(url_id, "failed", error_count=1)
                return False, recruitment_flag, f"Error processing prompt responses: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
            # Update URL processing status to failed
            db.update_url_processing_status(url_id, "failed", error_count=1)
            return False, -1, f"Unexpected error: {str(e)}"

    except DatabaseError as e:
        logger.error(f"Database error when processing URL {url}: {e}")
        return False, -1, f"Database error: {str(e)}"

# Background task to process URLs from RabbitMQ queue
async def process_urls_from_queue():
    """Process URLs from RabbitMQ queue."""
    while True:
        try:
            # Get RabbitMQ connection
            connection = get_rabbitmq_connection()
            channel = connection.channel()
            
            # Declare queue
            queue_name = "recruitment_urls"
            channel.queue_declare(queue=queue_name, durable=True)
            
            # Set prefetch count to limit the number of unacknowledged messages
            channel.basic_qos(prefetch_count=1)
            
            # Define callback function
            def callback(ch, method, properties, body):
                try:
                    # Parse message
                    message = json.loads(body)
                    url = message.get("url")
                    search_id = message.get("search_id")
                    
                    if not url:
                        logger.error("Received message without URL")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                    
                    logger.info(f"Processing URL from queue: {url}")
                    
                    # Check if URL already exists in the database
                    with db._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM urls WHERE url = ?", (url,))
                        result = cursor.fetchone()
                        
                        if result:
                            url_id = result[0]
                            # Update URL processing status to processing
                            db.update_url_processing_status(url_id, "processing")
                        else:
                            # Insert URL into database with processing status
                            url_id = db.insert_url(url, tldextract.extract(url).domain, "crawler")
                            db.update_url_processing_status(url_id, "processing")
                    
                    # Process URL
                    success, recruitment_flag, error_message = process_url(url)
                    
                    # Update processing status in database
                    status = "completed" if success else "failed"
                    db.update_url_processing_status(url_id, status, error_count=1 if not success else 0)
                    
                    # Acknowledge message
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Reject message and requeue
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
            # Start consuming
            channel.basic_consume(queue=queue_name, on_message_callback=callback)
            
            logger.info("Started consuming from RabbitMQ queue")
            
            # Keep consuming messages until connection is lost
            try:
                channel.start_consuming()
            except (pika.exceptions.ConnectionClosedByBroker,
                   pika.exceptions.AMQPChannelError,
                   pika.exceptions.AMQPConnectionError) as e:
                logger.error(f"RabbitMQ connection lost: {e}")
                # Let the outer loop handle reconnection
                continue
            
        except Exception as e:
            logger.error(f"Error in RabbitMQ consumer: {e}")
            # Wait before retrying
            await asyncio.sleep(5)
            continue

# API endpoints
@app.post("/process", response_model=URLProcessingResponse)
async def process_url_endpoint(request: URLProcessingRequest, background_tasks: BackgroundTasks):
    """Process a single URL."""
    try:
        # Check if URL already exists in the database
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM urls WHERE url = ?", (request.url,))
            result = cursor.fetchone()
            
            if result:
                url_id = result[0]
                # Update URL processing status to processing
                db.update_url_processing_status(url_id, "processing")
            else:
                # Insert URL into database with processing status
                url_id = db.insert_url(request.url, tldextract.extract(request.url).domain, "crawler")
                db.update_url_processing_status(url_id, "processing")
        
        # Process URL
        success, recruitment_flag, error_message = process_url(
            request.url, 
            request.process_all_prompts, 
            request.use_transaction
        )
        
        # Update processing status in database
        status = "completed" if success else "failed"
        db.update_url_processing_status(url_id, status, error_count=1 if not success else 0)
        
        return URLProcessingResponse(
            url=request.url,
            success=success,
            recruitment_flag=recruitment_flag,
            error_message=error_message,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{url:path}", response_model=URLProcessingStatus)
async def get_url_status(url: str):
    """Get the status of a URL processing."""
    try:
        # First check if URL exists in the database
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM urls WHERE url = ?", (url,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"URL {url} not found in database")
            
            url_id = result[0]
            
            # Get the processing status
            status_data = db.get_url_processing_status(url_id)
            
            if not status_data:
                # If no status record exists, create one with pending status
                db.update_url_processing_status(url_id, "pending")
                status_data = {
                    "status": "pending",
                    "recruitment_flag": None,
                    "error_message": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            return URLProcessingStatus(
                url=url,
                status=status_data["status"],
                recruitment_flag=status_data.get("recruitment_flag"),
                error_message=status_data.get("error_message"),
                timestamp=status_data.get("timestamp", datetime.now().isoformat())
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting URL status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    uvicorn.run(app, host="0.0.0.0", port=8002) 