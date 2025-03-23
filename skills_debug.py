#!/usr/bin/env python3
"""
Skills Fix Runner

This script integrates all the fixes for the skills processing issue
and provides a simple way to verify that skills are being properly
extracted from LLM responses and inserted into the database.
"""

import argparse
import json
import logging
import os
import sys
from pprint import pprint

import tldextract
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Import the web crawler
from web_crawler_lib import crawl_website_sync

# Import the prompts
from prompts import COMPLEX_PROMPTS

# Import the database
from recruitment_db_lib import RecruitmentDatabase

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("skills_fix_runner.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def clean_response_text(response_text):
    """Clean LLM response text by removing markdown code blocks."""
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        return response_text.split("```")[1].split("```")[0].strip()
    else:
        return response_text.strip()


def parse_skills_response(response_text):
    """
    Parse and transform the skills response from the LLM.
    Focus on handling the tuple format from the skills prompt.
    """
    logger.debug("RAW RESPONSE TEXT:")
    logger.debug(response_text)

    # First, clean the response text
    clean_text = clean_response_text(response_text)
    logger.debug("CLEANED RESPONSE TEXT:")
    logger.debug(clean_text)

    # Handle tuple syntax in LLM response by converting to JSON-compatible format
    if "(" in clean_text and ")" in clean_text:
        logger.debug("Response contains tuple syntax, converting to JSON-compatible format")
        json_compatible = clean_text.replace("(", "[").replace(")", "]")
        logger.debug("JSON-COMPATIBLE TEXT:")
        logger.debug(json_compatible)

        try:
            data = json.loads(json_compatible)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after tuple conversion: {e}")
            return None
    else:
        # Try to parse as regular JSON
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            logger.debug(f"Response text: {clean_text}")
            return None

    # Extract skills data
    skills_data = data.get("skills", [])
    logger.debug("EXTRACTED SKILLS DATA:")
    logger.debug(skills_data)

    if not skills_data:
        logger.warning("No skills found in the response")
        return {"skills": []}

    # Process skills data to handle different formats
    processed_skills = []

    for item in skills_data:
        logger.debug(f"Processing item: {item} (type: {type(item).__name__})")

        # Handle list format (converted from tuple)
        if isinstance(item, list):
            if len(item) >= 2:
                skill, experience = item[0], item[1]
                # Handle "not_listed" as None
                if experience == "not_listed":
                    experience = None
                processed_skills.append((skill.strip(), experience))
                logger.debug(f"Processed list: ({skill}, {experience})")
            else:
                skill = item[0]
                processed_skills.append((skill.strip(), None))
                logger.debug(f"Processed list as single value: ({skill}, None)")

        # Handle dictionary format
        elif isinstance(item, dict) and "skill" in item:
            skill = item["skill"]
            experience = item.get("experience")
            # Handle "not_listed" as None
            if experience == "not_listed":
                experience = None
            processed_skills.append((skill.strip(), experience))
            logger.debug(f"Processed dict: ({skill}, {experience})")

        # Handle string format
        elif isinstance(item, str):
            if item.strip():
                processed_skills.append((item.strip(), None))
                logger.debug(f"Processed string: ({item}, None)")

        else:
            logger.warning(f"Unrecognized format: {item}")

    logger.debug("FINAL PROCESSED SKILLS:")
    logger.debug(processed_skills)

    return {"skills": processed_skills}


def insert_skills_to_db(db, url_id, skills_data):
    """Insert processed skills data into the database."""
    if not skills_data or "skills" not in skills_data:
        logger.warning("No valid skills data to insert")
        return False

    skills_list = skills_data["skills"]
    if not skills_list:
        logger.warning("Empty skills list")
        return False

    success = True
    count = 0

    for skill, experience in skills_list:
        try:
            db.insert_skill(url_id, skill, experience)
            logger.info(f"Inserted skill: '{skill}', experience: {experience}")
            count += 1
        except Exception as e:
            logger.error(f"Failed to insert skill '{skill}': {e}")
            success = False

    logger.info(f"Inserted {count} skills into database")
    return success


def setup_llm_engine(api_key=None):
    """Set up the LLM engine."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key is required.")

    llm = OpenAI(temperature=0, model="gpt-4o-mini", request_timeout=120.0)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    return llm, memory


def process_url(url, api_key=None, db_path=None):
    """
    Process a URL to extract skills and insert them into the database.

    Args:
        url: The URL to process
        api_key: Optional OpenAI API key
        db_path: Optional database path

    Returns:
        Dictionary with processing results
    """
    # Validate URL
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    logger.info(f"Processing URL: {url}")

    # Create database connection
    try:
        db = RecruitmentDatabase(db_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return {"success": False, "error": f"Database connection error: {e}"}

    # Extract domain
    extracted = tldextract.extract(url)
    domain_name = f"{extracted.domain}.{extracted.suffix}"
    logger.info(f"Domain: {domain_name}")

    # Check if URL exists in database
    url_id = db.get_url_id(url)

    if not url_id:
        logger.info(f"URL not found in database, creating new entry")
        try:
            db.insert_url({
                "url": url,
                "domain_name": domain_name,
                "source": "skills_fix_runner",
                "recruitment_flag": 1  # Assume it's a recruitment post for testing
            })
            url_id = db.get_url_id(url)
            if not url_id:
                raise ValueError("Failed to get URL ID after insertion")
        except Exception as e:
            logger.error(f"Failed to insert URL: {e}")
            return {"success": False, "error": f"Failed to insert URL: {e}"}

    logger.info(f"Using URL ID: {url_id}")

    # Extract content using web crawler
    try:
        logger.info("Starting web crawler")
        crawl_result = crawl_website_sync(
            url=url,
            excluded_tags=['form', 'header'],
            word_count_threshold=10,
            verbose=True
        )
    except Exception as e:
        logger.error(f"Web crawler exception: {e}")
        return {"success": False, "error": f"Web crawler error: {e}"}

    if not crawl_result.success:
        logger.error(f"Web crawler failed: {crawl_result.error_message}")
        return {"success": False, "error": crawl_result.error_message}

    # Process content with LLM
    try:
        # Get the content
        text = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
        if not text:
            logger.error("Empty content extracted")
            return {"success": False, "error": "Empty content extracted"}

        # Set up LLM
        llm, memory = setup_llm_engine(api_key)

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

        # Get the skills prompt
        skills_prompt = COMPLEX_PROMPTS.get("skills_prompt")
        if not skills_prompt:
            logger.error("Skills prompt not found in COMPLEX_PROMPTS")
            return {"success": False, "error": "Skills prompt not found"}

        logger.info("Sending skills prompt to LLM")
        response = chat_engine.chat(skills_prompt)

        # Process the response
        logger.info("Processing LLM response")
        skills_data = parse_skills_response(response.response)

        if not skills_data:
            logger.error("Failed to parse skills response")
            return {"success": False, "error": "Failed to parse skills response"}

        # Insert the skills into the database
        logger.info("Inserting skills into database")
        insert_success = insert_skills_to_db(db, url_id, skills_data)

        # Verify the inserted skills
        try:
            query = "SELECT skill, experience FROM skills WHERE url_id = ?"
            with db._execute_query(query, (url_id,)) as cursor:
                inserted_skills = cursor.fetchall()
                logger.info(f"Found {len(inserted_skills)} skills in database:")
                for skill, experience in inserted_skills:
                    logger.info(f"  * '{skill}', experience: {experience}")
        except Exception as e:
            logger.error(f"Error verifying inserted skills: {e}")

        return {
            "success": insert_success,
            "url_id": url_id,
            "skills_count": len(skills_data.get("skills", [])),
            "skills_data": skills_data,
            "raw_response": response.response
        }

    except Exception as e:
        logger.error(f"Error processing content: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Extract skills from a URL and insert them into the database")
    parser.add_argument("url", help="URL to process")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--db-path", help="Path to the database")
    parser.add_argument("--output", help="Output file for results (JSON)")

    args = parser.parse_args()

    # Process the URL
    result = process_url(args.url, args.api_key, args.db_path)

    if result["success"]:
        logger.info("Successfully processed URL and inserted skills")

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            print("\nPROCESSING RESULTS:")
            print(f"URL ID: {result['url_id']}")
            print(f"Skills count: {result['skills_count']}")
            print("\nEXTRACTED SKILLS:")
            for skill, experience in result.get("skills_data", {}).get("skills", []):
                print(f"  * '{skill}', experience: {experience}")

        return 0
    else:
        logger.error(f"Failed to process URL: {result.get('error', 'Unknown error')}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())