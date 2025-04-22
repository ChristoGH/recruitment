import streamlit as st
import pandas as pd
import tldextract
from dotenv import load_dotenv
from logging_config import setup_logging
from libraries.config_validator import ConfigValidator
from utils import get_model_for_prompt
import typing
from typing import Any, Dict, List, Optional, Union, Tuple
from web_crawler_lib import crawl_website_sync, WebCrawlerResult, crawl_website_sync_v2
from prompts import COMPLEX_PROMPTS, LIST_PROMPTS, NON_LIST_PROMPTS
from llama_index.core import Document as liDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from recruitment_models import transform_skills_response
from recruitment_models import (AgencyResponse, AttributesResponse, BenefitsResponse,
                                CompanyResponse, ConfirmResponse, ContactPersonResponse,
                                EmailResponse, JobAdvertResponse, JobResponse,
                                LocationResponse, CompanyPhoneNumberResponse, LinkResponse,
                                SkillExperienceResponse, DutiesResponse,
                                QualificationsResponse, AdvertResponse)
import json
import random
import time
import os
from datetime import datetime

logger = setup_logging("streamlit")

# Load environment variables
load_dotenv()

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

def validate_and_normalize_url(url: str) -> typing.Tuple[bool, str, Optional[str]]:
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

def init_session_state() -> None:
    """Initialize session state variables."""


    # Initialize LLM with conservative settings
    if 'llm' not in st.session_state:
        st.session_state['llm'] = OpenAI(temperature=0, model="gpt-4o-mini", request_timeout=120.0)
    
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    if 'input_url' not in st.session_state:
        st.session_state['input_url'] = ""
    
    if 'previous_input_url' not in st.session_state:
        st.session_state['previous_input_url'] = ""
        
    if 'text' not in st.session_state:
        st.session_state['text'] = "## no scraped content to show..."

    st.session_state['prompt_values'] = {**COMPLEX_PROMPTS, **LIST_PROMPTS, **NON_LIST_PROMPTS}.values()
    st.session_state['prompt_names'] = {**COMPLEX_PROMPTS, **LIST_PROMPTS, **NON_LIST_PROMPTS}.keys()
    st.session_state['prompts'] = {**COMPLEX_PROMPTS, **LIST_PROMPTS, **NON_LIST_PROMPTS}
    
    if 'selected_prompts' not in st.session_state:
        st.session_state['selected_prompts'] = []


def get_url_from_session_state() -> str:
    """Get the URL from session state."""
    return st.session_state['input_url']

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

def write_responses_to_file(url: str, prompt_responses: Dict[str, Any]) -> str:
    """Write prompt responses to a JSON file in the current directory."""
    try:
        # Create a filename based on the URL and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a safe filename from the URL
        safe_url = "".join(c if c.isalnum() else "_" for c in url)
        filename = f"{safe_url}_{timestamp}.json"
        
        # Prepare the data structure
        data = {
            "url": url,
            "timestamp": timestamp,
            "responses": {}
        }
        
        # Convert Pydantic models to dictionaries
        for prompt_name, response in prompt_responses.items():
            if hasattr(response, 'model_dump'):
                data["responses"][prompt_name] = response.model_dump()
            else:
                data["responses"][prompt_name] = response
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully wrote responses to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error writing responses to file: {e}")
        raise

def main()-> None:
    # Initialize session state if it doesn't exist
    init_session_state()
    
    # Create the text input with a key that automatically updates the session state
    input_url = st.text_input(label="Enter URL", value=st.session_state['input_url'], key='input_url')
    
    # Check if the input URL has changed
    url_changed = input_url != st.session_state['previous_input_url']
    
    # Update the previous input URL
    st.session_state['previous_input_url'] = input_url
    
    # Display validation results if we have an input
    if input_url:
        is_valid, normalized_url, error_message = validate_and_normalize_url(input_url)
        
        # Create a container for all the information
        info_container = st.container()
        with info_container:
            st.write(f"Original URL: {input_url}")
            st.write(f"Is a valid URL: {is_valid}")
            st.write(f"Normalized URL: {normalized_url}")
            if error_message:
                st.write(f"Error message: {error_message}")
    
    # Run code only when the URL has changed
    if url_changed and input_url:
        st.write("URL has changed! Running specific code...")
        # Add your specific code here that should run only when the URL changes
        # For example, you might want to automatically scrape the content
        try:
            crawl_result = crawl_website_sync(
                url=input_url,
                excluded_tags=['form', 'header'],
                verbose=True
            )
            if not crawl_result.success:
                logger.warning(f"Failed to extract content from URL: {input_url}")
                st.error(f"Failed to extract content: {crawl_result.error_message}")
            else:    
                st.session_state['text'] = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
                st.write("Content automatically scraped!")
                
        except Exception as e:
            logger.error(f"Crawler exception for URL {input_url}: {e}", exc_info=True)
            st.error(f"Crawler exception: {str(e)}")
            
    # Add a multi-select box for prompt names in the sidebar
    st.sidebar.markdown("### Select Prompts")
    prompt_options = list(st.session_state['prompt_names'])
    selected_prompts = st.sidebar.multiselect(
        "Choose prompts to use",
        options=prompt_options,
        default=[],
        key="prompt_selector"
    )
    
    # Update the selected prompts in session state
    st.session_state['selected_prompts'] = selected_prompts
    
    # Display selected prompts
    if selected_prompts:
        st.sidebar.markdown("**Selected prompts:**")
        for prompt in selected_prompts:
            st.sidebar.markdown(f"- {prompt}")
    
    if st.sidebar.button(label="Initialize LLM", key="init_llm_button"):
        # Add your scraping logic here
        st.write("Initialize...")
        try:
            # Create document index and chat engine
            if ('text' in st.session_state) and ('llm' in st.session_state):
                documents = [liDocument(text=st.session_state['text'])]
                index = VectorStoreIndex.from_documents(documents)
                st.session_state['memory'].reset()
                st.session_state['chat_engine'] = index.as_chat_engine(
                    chat_mode="context",
                    llm=st.session_state['llm'],
                    memory=st.session_state['memory'],
                    system_prompt=(
                        "You are a career recruitment analyst with deep insight into the skills and job market. "
                        "Your express goal is to investigate online adverts and extract pertinent factual detail."
                    )
                )
            st.success("LLM initialized!", icon=":material/check:")
            
        except Exception as e:
            logger.error(f"Error creating chat engine: {e}", exc_info=True)
            st.error(f"Error creating chat engine: {str(e)}")
            
    
    if st.sidebar.button(label="scrape content", key="scrape_button"):
        # Add your scraping logic here
        st.write("Scraping content...")
        
        # Extract content using web_crawler_lib
        if input_url:
            st.write(f"Processing URL: {input_url}")
            try:
                crawl_result = crawl_website_sync(
                    url=input_url,
                    excluded_tags=['form', 'header'],
                    verbose=True
                )
                if not crawl_result.success:
                    logger.warning(f"Failed to extract content from URL: {input_url}")
                    st.error(f"Failed to extract content: {crawl_result.error_message}")
                else:    
                    st.session_state['text'] = crawl_result.markdown[:min(5000, len(crawl_result.markdown))]
                    st.markdown(st.session_state['text'])
                
            except Exception as e:
                logger.error(f"Crawler exception for URL {input_url}: {e}", exc_info=True)
                st.error(f"Crawler exception: {str(e)}")

    with st.expander("Prompts:"):
        for prompt_name, prompt in zip(st.session_state['prompt_names'], st.session_state['prompt_values']):
            st.markdown(f"#### {prompt_name}")
            st.markdown(prompt)
            st.markdown("----")
            
    with st.expander("Selected prompts:"):
        if st.session_state['selected_prompts']:
            for prompt_name in st.session_state['selected_prompts']:
                st.write(st.session_state['prompts'])
                st.markdown(f"#### Selected prompt: {prompt_name}")
                st.markdown(st.session_state['prompts'][prompt_name])
                st.markdown("----")


    with st.expander("Page content:"):
        st.markdown(st.session_state['text'])
        
    if st.sidebar.button(label="Process prompts", key="process_prompts_button") and (st.session_state['input_url'] != "") and ('chat_engine' in st.session_state):
        st.write("Processing prompts...")
        recruitment_type, evidence = verify_recruitment(st.session_state['input_url'], st.session_state['chat_engine'])
        recruitment_flag = recruitment_type.get("recruitment_flag", -1)
        st.write(f"Recruitment flag: {recruitment_flag}")
        if recruitment_flag == 1:
            for prompt_name in st.session_state['selected_prompts']:
                if (prompt_name in st.session_state['selected_prompts']) and (recruitment_flag == 1):
                    st.write(f"Processing prompt: {prompt_name}")
                    st.write(st.session_state['prompts'][prompt_name])
                    prompt_responses = collect_prompt_responses(st.session_state['chat_engine'])
                    
                    # Write responses to file
                    try:
                        filename = write_responses_to_file(st.session_state['input_url'], prompt_responses)
                        st.success(f"Responses written to: {filename}")
                    except Exception as e:
                        st.error(f"Failed to write responses to file: {str(e)}")
                    
                    st.write("----")
            for prompt_name, prompt_response in prompt_responses.items():
                st.write(f"Prompt: {prompt_name}")
                st.write(prompt_response)
                st.write("----")


if __name__ == "__main__":
    main()
    