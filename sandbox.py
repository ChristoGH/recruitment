import pandas as pd
from recruitment_db_lib import DatabaseError, RecruitmentDatabase
from get_urls_from_csvs import get_unique_urls_from_csvs
import tldextract
import logging
import random
from recruitment_models import ConfirmResponse,CompanyResponse,AgencyResponse,JobResponse,SkillsResponse
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except ImportError:
    webdriver = None
import time
import requests
from llama_index.core import Document as liDocument, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from readability import Document
from bs4 import BeautifulSoup
from newspaper import Article
from prompts import LIST_PROMPTS,NON_LIST_PROMPTS, COMPLEX_PROMPTS
import json
from typing import List, Optional, Any
llm = OpenAI(temperature=0, model="o3-mini", request_timeout=120.0)
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
def is_url_accessible(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/90.0.4430.85 Safari/537.36")
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        # You may want to wait for certain elements if needed
        status = driver.title != ""  # simple check: did the page load?
        driver.quit()
        return status
    except Exception as e:
        logger.error(f"Exception for URL {url}: {e}")
        return False

def extract_with_newspaper(url: str) -> str:
    """Try to extract the article text using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) >= MIN_TEXT_LENGTH:
            logger.info("Article extracted successfully using newspaper3k.")
            return text
        else:
            logger.warning("Newspaper3k extraction returned insufficient text.")
    except Exception as e:
        logger.error(f"Error using newspaper3k for {url}: {e}")
    return ""


def extract_with_readability(url: str) -> str:
    """Fetch the page with requests and extract text using readability-lxml."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.85 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.error(f"Requests returned status code {response.status_code} for {url}")
            return ""
        doc = Document(response.text)
        # The summary() method returns the main content as HTML.
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        if len(text) >= MIN_TEXT_LENGTH:
            logger.info("Article extracted successfully using readability-lxml.")
            return text
        else:
            logger.warning("Readability extraction returned insufficient text.")
    except Exception as e:
        logger.error(f"Error using readability for {url}: {e}")
    return ""


def extract_with_selenium(url: str) -> str:
    """Render the page with Selenium and extract the main content."""
    if webdriver is None:
        logger.error("Selenium is not installed; cannot use Selenium fallback.")
        return ""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.85 Safari/537.36"
        )
        # Adjust or specify the path to chromedriver if needed.
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        # Allow time for dynamic content to load
        time.sleep(3)
        html = driver.page_source
        driver.quit()
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        if len(text) >= MIN_TEXT_LENGTH:
            logger.info("Article extracted successfully using Selenium with readability.")
            return text
        else:
            logger.warning("Selenium extraction returned insufficient text.")
    except Exception as e:
        logger.error(f"Error using Selenium for {url}: {e}")
    return ""


def extract_main_text(url: str) -> str:
    """
    Extracts the main text from an article URL using multiple fallback methods.

    1. Try newspaper3k.
    2. Fall back to requests + readability-lxml.
    3. Finally, if needed, use Selenium to render JavaScript.
    """
    logger.info(f"Attempting to extract article text from {url}")

    # First attempt: newspaper3k
    text = extract_with_newspaper(url)
    if text:
        return text

    # Second attempt: requests + readability-lxml
    text = extract_with_readability(url)
    if text:
        return text

    # Third attempt: Selenium (if available)
    text = extract_with_selenium(url)
    return text


def get_validated_response(prompt_key: str, prompt_text: str, model_class: Any, chat_engine) -> Optional[Any]:
    """
    Sends a prompt to the chat engine and returns a validated Pydantic model instance with exponential backoff.

    Args:
        prompt_key (str): The key identifying the prompt and model.
        prompt_text (str): The prompt text to send.
        model_class (BaseModel): The Pydantic model to validate the response.
        chat_engine: The chat engine instance for processing the prompt.

    Returns:
        Optional[Any]: An instance of the Pydantic model or None if validation fails.
    """
    max_retries = 5
    base_delay = 5  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = chat_engine.chat(prompt_text)
            logger.info(f"Prompt '{prompt_key}' processed successfully.")

            response_text = response.response
            response_data = model_class.model_validate_json(response_text)

            return response_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for '{prompt_key}': {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for '{prompt_key}': {e}")

        except Exception as e:
            if "429 Too Many Requests" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to parse response for '{prompt_key}': {e}")
                return None

    logger.error(f"Max retries reached for '{prompt_key}'. Skipping...")
    return None


def verify_recruitment(url: str, chat_engine):
    """
    Verifies incidents based on the URL and updates the database.
    Optionally uploads to Neo4j and Google Sheets based on user input.

    Args:
        url (str): The URL of the analyzed article.
        chat_engine: The chat engine instance used for processing.

    Returns:
        tuple: A dictionary containing the result and a list of incidents.
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
            db.update_field(url, "recruitment_flag", 1)  # Optionally log this change
            result["recruitment_flag"] = 1
            incidents = response_data.evidence or []
            return result, incidents

        # Handle unexpected response cases
        logger.warning(f"Unexpected response answer for URL: {url} - {response_data.answer}")
        return result, incidents

    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return result, incidents
# Configure Logging
logging.basicConfig(
    filename="sandbox_recruitment_research.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
MIN_TEXT_LENGTH = 100  # adjust as necessary
db = RecruitmentDatabase()
urls_from_files = get_unique_urls_from_csvs('recruitment_output', 'url', 4, 1000)
if pd.DataFrame(db.search_urls(limit=1000000)).empty:
    urls = urls_from_files
else:
    urls_from_db = pd.DataFrame(db.search_urls(limit=1000000))['url'].tolist()
    urls = list(set(urls_from_files) - set(urls_from_db))

url = urls[105]

# Check accessibility before processing
if not is_url_accessible(url):
    logger.warning(f"URL not accessible: {url}")
    # Record the URL in the database as inaccessible.
    # Set accessible to 0 and leave content empty.
    result = {
        "url": url,
        "domain_name": tldextract.extract(url).domain,
        "source": "google_search",
        "content": "",  # No content because URL is inaccessible
        "recruitment_flag": -1,  # Use a default status (or -2) indicating not processed
        "accessible": 0
    }
    db.insert_url(result)
    continue

logger.info(f"Processing URL: {url}")
extracted = tldextract.extract(url)
domain_name = extracted.domain

# Extract Main Text
text = extract_main_text(url)
from response_processor_functions import PromptResponseProcessor
processor = PromptResponseProcessor(db)
# Build the record dictionary. Note that 'accessible' is 1 if text was successfully extracted.
result = {
    "url": url,
    "domain_name": domain_name,
    "source": "google_search",
    "content": text,
    "recruitment_flag": -1,  # default value until verified
    "accessible": 1  # accessible if we reached this point
}
if text:
    try:
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
        recruitment_type, evidence = verify_recruitment(url, chat_engine)
        result.update(recruitment_type)

        db.insert_url(result)
        url_id = db.get_url_id(url)
        prompt_key="recruitment_prompt"
        prompt_text=LIST_PROMPTS[prompt_key]
        response = chat_engine.chat(prompt_text)
        logger.info(f"Prompt '{prompt_key}' processed successfully.")

        response_text = response.response
        response_data = ConfirmResponse.model_validate_json(response_text)
        processor.process_response(url_id, prompt_key, response_data)

        prompt_key="company_prompt"
        prompt_text=NON_LIST_PROMPTS[prompt_key]
        response = chat_engine.chat(prompt_text)
        logger.info(f"Prompt '{prompt_key}' processed successfully.")
        response_text = response.response
        response_data = CompanyResponse.model_validate_json(response_text)
        processor.process_response(url_id, prompt_key, response_data)

        prompt_key="agency_prompt"
        prompt_text=NON_LIST_PROMPTS[prompt_key]
        response = chat_engine.chat(prompt_text)
        logger.info(f"Prompt '{prompt_key}' processed successfully.")
        response_text = response.response
        response_data = AgencyResponse.model_validate_json(response_text)
        processor.process_response(url_id, prompt_key, response_data)

        prompt_key="job_prompt"
        prompt_text=NON_LIST_PROMPTS[prompt_key]
        response = chat_engine.chat(prompt_text)
        logger.info(f"Prompt '{prompt_key}' processed successfully.")
        response_text = response.response
        response_data = JobResponse.model_validate_json(response_text)
        processor.process_response(url_id, prompt_key, response_data)

        prompt_key="skills_prompt"
        prompt_text=LIST_PROMPTS[prompt_key]
        response = chat_engine.chat(prompt_text)
        logger.info(f"Prompt '{prompt_key}' processed successfully.")
        response_text = response.response
        response_data = SkillsResponse.model_validate_json(response_text)
        processor.process_response(url_id, prompt_key, response_data)


    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")

