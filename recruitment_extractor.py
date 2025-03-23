#!/usr/bin/env python3
"""
Recruitment Information Extractor

A minimum viable script that demonstrates the recruitment URL processing workflow
without writing to the database. It takes a single URL, crawls it, verifies if it's
a recruitment post, and extracts job-related information.
https://lynnfitho.com/work-from-home-support-specialist-job-vacancy-in-south-africa-for-sabio/

"""

import argparse
import json
import logging
import os
import sys
import ssl
from typing import Dict, Any, Optional, Tuple, List

import tldextract
import requests
from dotenv import load_dotenv
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()


# Mock web crawler function to avoid dependencies
# In a real implementation, you'd use the actual web_crawler_lib
def mock_crawl_website_sync(url, excluded_tags=None, verbose=False):
    """Simplified mock crawler for demonstration purposes"""
    import ssl
    import certifi
    import requests
    import re

    class WebCrawlerResult:
        def __init__(self, success, markdown="", links=None, error_message=None):
            self.success = success
            self.markdown = markdown
            self.links = links or []
            self.error_message = error_message

    try:
        # Use requests library which handles SSL certificates better
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        html = response.text

        # Very basic HTML to plain text conversion (for demonstration)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract links (simplified)
        links = []
        link_pattern = re.compile(r'href=[\'"]?([^\'" >]+)')
        for match in link_pattern.finditer(html):
            link = match.group(1)
            if link.startswith('http'):
                links.append(link)

        return WebCrawlerResult(success=True, markdown=text, links=links)

    except requests.exceptions.RequestException as e:
        return WebCrawlerResult(success=False, error_message=f"Request Error: {str(e)}")
    except Exception as e:
        return WebCrawlerResult(success=False, error_message=f"Error: {str(e)}")


# Prompts dictionary containing the prompts we'll use
from prompts import LIST_PROMPTS, NON_LIST_PROMPTS, COMPLEX_PROMPTS


def clean_response_text(response_text: str) -> str:
    """
    Clean LLM response text by removing markdown code blocks and other formatting.
    """
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        return response_text.split("```")[1].split("```")[0].strip()
    else:
        return response_text.strip()


def setup_llm_engine(api_key: Optional[str] = None):
    """Set up the LLM and memory for chat interactions."""
    # Use the API key from args or from environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OpenAI API key is required. Please provide it using the --api-key flag or set the OPENAI_API_KEY environment variable.")

    # Initialize LLM with conservative settings
    llm = OpenAI(temperature=0, model="gpt-4o-mini", request_timeout=120.0)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    return llm, memory


def validate_and_normalize_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """Validate and normalize a URL for crawling."""
    # Clean up the URL by trimming whitespace
    url = url.strip()

    # Check if URL has a valid protocol
    valid_protocols = ('http://', 'https://')
    has_protocol = any(url.startswith(protocol) for protocol in valid_protocols)

    # If no protocol, try to add https:// (most common)
    if not has_protocol:
        if ' ' in url or url.startswith('('):
            return False, url, f"Invalid URL format: {url}"
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

        return True, url, None
    except Exception as e:
        return False, url, f"URL validation error: {str(e)}"


def verify_recruitment(url: str, chat_engine) -> Tuple[bool, List[str]]:
    """Verifies if a URL contains a recruitment advertisement."""
    try:
        # Send prompt to chat engine
        prompt_key = "recruitment_prompt"
        prompt_text = LIST_PROMPTS[prompt_key]

        response = chat_engine.chat(prompt_text)
        response_text = clean_response_text(response.response)

        try:
            data = json.loads(response_text)
            answer = data.get("answer", "").lower()
            evidence = data.get("evidence", [])

            if answer == "yes":
                logger.info(f"Recruitment advert detected for URL: {url}")
                return True, evidence
            else:
                logger.info(f"No recruitment evidence detected for URL: {url}")
                return False, []

        except json.JSONDecodeError:
            logger.error(f"JSON decoding failed for recruitment verification: {response_text[:100]}...")
            return False, []

    except Exception as e:
        logger.error(f"Error verifying recruitment for URL {url}: {e}")
        return False, []


def process_prompt(prompt_key: str, prompt_text: str, chat_engine) -> Dict[str, Any]:
    """Process a single prompt with the chat engine."""
    try:
        response = chat_engine.chat(prompt_text)
        response_text = clean_response_text(response.response)

        try:
            # Special handling for skills prompt
            if prompt_key == "skills_prompt":
                # Pre-process the response to handle tuple syntax in JSON
                response_text = response_text.replace("(", "[").replace(")", "]")

            data = json.loads(response_text)

            # Transform skills data if needed
            if prompt_key == "skills_prompt" and "skills" in data:
                skills_data = data["skills"]
                if isinstance(skills_data, list):
                    transformed_skills = []

                    for item in skills_data:
                        # Handle array format (converted from tuple)
                        if isinstance(item, list) and len(item) == 2:
                            transformed_skills.append({"skill": item[0], "experience": item[1]})
                        # Handle string format
                        elif isinstance(item, str):
                            transformed_skills.append({"skill": item, "experience": None})
                        # Dictionary format is already compatible
                        elif isinstance(item, dict) and "skill" in item:
                            transformed_skills.append(item)

                    # Replace the skills with the transformed version
                    data["skills"] = transformed_skills

            logger.info(f"Successfully processed prompt: {prompt_key}")
            return data

        except json.JSONDecodeError:
            logger.error(f"JSON decoding failed for {prompt_key}: {response_text[:100]}...")
            return {}

    except Exception as e:
        logger.error(f"Error processing prompt {prompt_key}: {e}")
        return {}


def collect_job_data(chat_engine) -> Dict[str, Any]:
    """Collect job data from all relevant prompts."""
    job_data = {}

    # Process recruitment prompt first
    recruitment_key = "recruitment_prompt"
    recruitment_text = LIST_PROMPTS[recruitment_key]
    recruitment_data = process_prompt(recruitment_key, recruitment_text, chat_engine)

    if recruitment_data:
        job_data[recruitment_key] = recruitment_data

    # Process non-list prompts
    for prompt_key, prompt_text in NON_LIST_PROMPTS.items():
        prompt_data = process_prompt(prompt_key, prompt_text, chat_engine)
        if prompt_data:
            job_data[prompt_key] = prompt_data

    # Process list prompts (excluding recruitment which was already processed)
    list_prompts_to_process = {k: v for k, v in LIST_PROMPTS.items() if k != "recruitment_prompt"}
    for prompt_key, prompt_text in list_prompts_to_process.items():
        prompt_data = process_prompt(prompt_key, prompt_text, chat_engine)
        if prompt_data:
            job_data[prompt_key] = prompt_data

    # Process complex prompts
    for prompt_key, prompt_text in COMPLEX_PROMPTS.items():
        prompt_data = process_prompt(prompt_key, prompt_text, chat_engine)
        if prompt_data:
            job_data[prompt_key] = prompt_data

    return job_data


def extract_job_data_from_responses(prompt_responses: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and combine job-related data from multiple prompt responses."""
    job_data = {}

    # Helper function to get data from responses
    def get_response_data(prompt_key):
        if prompt_key in prompt_responses:
            return prompt_responses[prompt_key]
        return {}

    # Extract job title
    job_response = get_response_data("job_prompt")
    job_data["job_title"] = job_response.get("title")

    # Extract company name
    company_response = get_response_data("company_prompt")
    job_data["company"] = company_response.get("company")

    # Extract location
    location_response = get_response_data("location_prompt")
    if location_response:
        job_data["location"] = {
            "country": location_response.get("country"),
            "province": location_response.get("province"),
            "city": location_response.get("city"),
            "street_address": location_response.get("street_address")
        }

    # Extract job advert details
    jobadvert_response = get_response_data("jobadvert_prompt")
    for key in ["description", "salary", "duration", "start_date",
                "end_date", "posted_date", "application_deadline"]:
        if key in jobadvert_response:
            job_data[key] = jobadvert_response[key]

    # Extract skills
    skills_response = get_response_data("skills_prompt")
    if "skills" in skills_response:
        skills_data = skills_response["skills"]
        if skills_data and isinstance(skills_data, list):
            processed_skills = []
            for skill_item in skills_data:
                if isinstance(skill_item, dict) and "skill" in skill_item:
                    processed_skills.append({
                        "skill": skill_item["skill"],
                        "experience": skill_item.get("experience")
                    })
                elif isinstance(skill_item, (list, tuple)) and len(skill_item) == 2:
                    processed_skills.append({
                        "skill": skill_item[0],
                        "experience": skill_item[1]
                    })
                elif isinstance(skill_item, str):
                    processed_skills.append({
                        "skill": skill_item,
                        "experience": None
                    })
            job_data["skills"] = processed_skills

    # Extract list types (benefits, duties, qualifications)
    for data_type in ["benefits", "duties", "qualifications"]:
        prompt_key = f"{data_type}_prompt"
        response = get_response_data(prompt_key)
        if data_type in response and isinstance(response[data_type], list):
            job_data[data_type] = response[data_type]

    # Extract contacts
    contacts_response = get_response_data("contacts_prompt")
    if "contacts" in contacts_response and isinstance(contacts_response["contacts"], list):
        job_data["contacts"] = contacts_response["contacts"]

    # Extract attributes
    attributes_response = get_response_data("attributes_prompt")
    if "attributes" in attributes_response and isinstance(attributes_response["attributes"], list):
        job_data["attributes"] = attributes_response["attributes"]

    # Extract agency
    agency_response = get_response_data("agency_prompt")
    job_data["agency"] = agency_response.get("agency")

    # Extract contact details
    email_response = get_response_data("email_prompt")
    job_data["email"] = email_response.get("email")

    phone_response = get_response_data("company_phone_number_prompt")
    job_data["phone"] = phone_response.get("number")

    link_response = get_response_data("link_prompt")
    job_data["link"] = link_response.get("link")

    return job_data


def process_url(url: str) -> Tuple[bool, Dict[str, Any]]:
    """Process a URL and extract recruitment information."""
    # Validate and normalize the URL
    is_valid, normalized_url, error_message = validate_and_normalize_url(url)

    if not is_valid:
        logger.warning(f"Invalid URL: {url}. {error_message}")
        return False, {"error": error_message}

    # Use the normalized URL
    url = normalized_url
    logger.info(f"Processing URL: {url}")

    # Extract content using web crawler
    try:
        crawl_result = mock_crawl_website_sync(url, excluded_tags=['form', 'header'], verbose=True)
    except Exception as e:
        logger.error(f"Crawler exception for URL {url}: {e}")
        return False, {"error": f"Crawler exception: {str(e)}"}

    if not crawl_result.success:
        logger.warning(f"Failed to extract content from URL: {url}")
        return False, {"error": crawl_result.error_message}

    # Extract domain
    extracted = tldextract.extract(url)
    domain_name = extracted.domain

    # Use the markdown content from the crawler
    text = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
    if not text:
        logger.warning(f"Empty content extracted from URL: {url}")
        return False, {"error": "Empty content extracted"}

    try:
        # Set up LLM and create chat engine
        llm, memory = setup_llm_engine()

        # Create document index and chat engine
        documents = [liDocument(text=text)]
        index = VectorStoreIndex.from_documents(documents)
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
        is_recruitment, evidence = verify_recruitment(url, chat_engine)

        # If not a recruitment advert, return basic info
        if not is_recruitment:
            return True, {
                "url": url,
                "domain": domain_name,
                "is_recruitment": False,
                "message": "This URL does not contain a recruitment advertisement."
            }

        # Collect job data from all prompts
        prompt_responses = collect_job_data(chat_engine)

        # Extract and format job data
        job_data = extract_job_data_from_responses(prompt_responses)

        # Add metadata
        result = {
            "url": url,
            "domain": domain_name,
            "is_recruitment": True,
            "evidence": evidence,
            "job_data": job_data,
            "raw_responses": prompt_responses
        }

        logger.info(f"Successfully processed URL: {url}")
        return True, result

    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}")
        return False, {"error": str(e)}


def main():
    """Main function to run the extraction."""
    parser = argparse.ArgumentParser(description="Extract recruitment information from a URL")
    parser.add_argument("url", help="URL to process")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--output", type=str, default=None, help="Output file path (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process the URL
    success, result = process_url(args.url)

    if success:
        # Print or save the result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Failed to process URL: {args.url}")
        print(f"Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()