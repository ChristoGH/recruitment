# batch_processor.py

import logging
from typing import Dict, Any, List, Optional
import json
from recruitment_db_lib import DatabaseError

logger = logging.getLogger(__name__)


def process_all_prompt_responses(db, url_id: int, prompt_responses: Dict[str, Any],
                                 use_transaction: bool = True) -> Dict[str, Any]:
    """
    Process all prompt responses for a URL, optionally using a transaction.

    Args:
        db: RecruitmentDatabase instance
        url_id: The URL ID
        prompt_responses: Dictionary mapping prompt_type to response (model instance or dict)
        use_transaction: Whether to use a transaction (default: True)

    Returns:
        Dict containing status information about the processing
    """
    results = {
        "success": True,
        "processed": 0,
        "failed": 0,
        "errors": []
    }

    # Process with transaction if supported and requested
    if use_transaction and hasattr(db, "process_prompt_responses_in_transaction"):
        try:
            # Convert responses to dicts if they are model instances
            parsed_responses = {}
            for prompt_type, response in prompt_responses.items():
                try:
                    if hasattr(response, "model_dump"):
                        # It's a Pydantic model
                        parsed_responses[prompt_type] = response.model_dump()
                    elif isinstance(response, dict):
                        # It's already a dict
                        parsed_responses[prompt_type] = response
                    elif isinstance(response, str):
                        # It's a JSON string
                        parsed_responses[prompt_type] = json.loads(response)
                    else:
                        # Unknown format, skip
                        logger.warning(f"Skipping unknown response format for {prompt_type}")
                        results["failed"] += 1
                        results["errors"].append({
                            "prompt_type": prompt_type,
                            "error": "Unknown response format"
                        })
                        continue
                except Exception as e:
                    logger.error(f"Failed to parse response for {prompt_type}: {e}")
                    results["failed"] += 1
                    results["errors"].append({
                        "prompt_type": prompt_type,
                        "error": f"Parse error: {e}"
                    })

            if parsed_responses:
                db.process_prompt_responses_in_transaction(url_id, parsed_responses)
                results["processed"] = len(parsed_responses)
        except Exception as e:
            logger.error(f"Transaction processing failed: {e}", exc_info=True)
            results["success"] = False
            results["errors"].append({
                "prompt_type": "transaction",
                "error": str(e)
            })
    else:
        # Process responses individually using the standard methods
        for prompt_type, response in prompt_responses.items():
            try:
                # Handle different response types
                if prompt_type == "recruitment_prompt":
                    process_recruitment(db, url_id, response)
                elif prompt_type == "company_prompt":
                    process_company(db, url_id, response)
                elif prompt_type == "agency_prompt":
                    process_agency(db, url_id, response)
                elif prompt_type == "job_prompt":
                    process_job(db, url_id, response)
                elif prompt_type == "company_phone_number_prompt":
                    process_company_phone_number(db, url_id, response)
                elif prompt_type == "email_prompt":
                    process_email(db, url_id, response)
                elif prompt_type == "link_prompt":
                    process_link(db, url_id, response)
                elif prompt_type == "benefits_prompt":
                    process_benefits(db, url_id, response)
                elif prompt_type == "skills_prompt":
                    process_skills(db, url_id, response)
                elif prompt_type == "attributes_prompt":
                    process_attributes(db, url_id, response)
                elif prompt_type == "contacts_prompt":
                    process_contacts(db, url_id, response)
                elif prompt_type == "location_prompt":
                    process_location(db, url_id, response)
                elif prompt_type == "jobadvert_prompt":
                    process_job_advert(db, url_id, response)
                elif prompt_type == "qualifications_prompt":
                    process_qualifications(db, url_id, response)
                elif prompt_type == "duties_prompt":
                    process_duties(db, url_id, response)
                else:
                    logger.warning(f"Unknown prompt type: {prompt_type}")
                    continue

                results["processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {prompt_type}: {e}")
                results["failed"] += 1
                results["errors"].append({
                    "prompt_type": prompt_type,
                    "error": str(e)
                })

    return results


def extract_job_data_from_responses(prompt_responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and combine job-related data from multiple prompt responses.

    Args:
        prompt_responses: Dictionary of prompt responses

    Returns:
        Combined job data dictionary
    """
    job_data = {}

    # Helper function to parse responses if they're strings
    def parse_if_string(resp):
        if isinstance(resp, str):
            try:
                return json.loads(resp)
            except json.JSONDecodeError:
                return {}
        elif hasattr(resp, "model_dump"):
            return resp.model_dump()
        return resp if isinstance(resp, dict) else {}

    # Extract job title
    if "job_prompt" in prompt_responses:
        job_response = parse_if_string(prompt_responses["job_prompt"])
        job_data["job_title"] = job_response.get("title")

    # Extract company name
    if "company_prompt" in prompt_responses:
        company_response = parse_if_string(prompt_responses["company_prompt"])
        job_data["company"] = company_response.get("company")

    # Extract location
    if "location_prompt" in prompt_responses:
        location_response = parse_if_string(prompt_responses["location_prompt"])
        job_data["location"] = {
            "country": location_response.get("country"),
            "province": location_response.get("province"),
            "city": location_response.get("city"),
            "street_address": location_response.get("street_address")
        }

    # Extract job advert details
    if "jobadvert_prompt" in prompt_responses:
        jobadvert_response = parse_if_string(prompt_responses["jobadvert_prompt"])
        for key in ["description", "salary", "duration", "start_date",
                    "end_date", "posted_date", "application_deadline"]:
            if key in jobadvert_response:
                job_data[key] = jobadvert_response[key]

    # Extract skills, benefits, duties, qualifications if available
    for data_type in ["skills", "benefits", "duties", "qualifications"]:
        prompt_key = f"{data_type}_prompt"
        if prompt_key in prompt_responses:
            response = parse_if_string(prompt_responses[prompt_key])
            if data_type in response and isinstance(response[data_type], list):
                job_data[data_type] = response[data_type]

    return job_data


# Individual processing functions for non-transaction mode

def process_recruitment(db, url_id: int, response: Any) -> None:
    """Process recruitment flag and evidence."""
    data = _ensure_dict(response)
    answer = data.get("answer")
    if answer == "yes":
        # Update recruitment flag to 1 (is recruitment)
        db.update_field_by_id(url_id, "recruitment_flag", 1)

        # Process evidence if available
        evidence = data.get("evidence")
        if evidence and isinstance(evidence, list):
            db.insert_recruitment_evidence_list(url_id, evidence)
    elif answer == "no":
        # Update recruitment flag to 0 (not recruitment)
        db.update_field_by_id(url_id, "recruitment_flag", 0)


def process_company(db, url_id: int, response: Any) -> None:
    """Process company name."""
    data = _ensure_dict(response)
    company = data.get("company")
    if company:
        db.insert_company(url_id, company)


def process_agency(db, url_id: int, response: Any) -> None:
    """Process agency name."""
    data = _ensure_dict(response)
    agency = data.get("agency")
    if agency:
        db.insert_agency(url_id, agency)


def process_job(db, url_id: int, response: Any) -> None:
    """Process job title."""
    data = _ensure_dict(response)
    title = data.get("title")
    if title:
        db.insert_job_title(url_id, title)


def process_company_phone_number(db, url_id: int, response: Any) -> None:
    """Process phone number."""
    data = _ensure_dict(response)
    number = data.get("number")
    if number:
        db.insert_company_phone_number(url_id, number)


def process_email(db, url_id: int, response: Any) -> None:
    """Process email address."""
    data = _ensure_dict(response)
    email = data.get("email")
    if email:
        db.insert_email(url_id, email)


def process_link(db, url_id: int, response: Any) -> None:
    """Process contact URL link."""
    data = _ensure_dict(response)
    link = data.get("link")
    if link:
        db.insert_link(url_id, link)


def process_benefits(db, url_id: int, response: Any) -> None:
    """Process benefits list."""
    data = _ensure_dict(response)
    benefits = data.get("benefits")
    if benefits and isinstance(benefits, list):
        db.insert_benefits_list(url_id, benefits)


def process_duties(db, url_id: int, response: Any) -> None:
    """Process duties list."""
    data = _ensure_dict(response)
    duties = data.get("duties")
    if duties and isinstance(duties, list):
        db.insert_duties_list(url_id, duties)


def process_qualifications(db, url_id: int, response: Any) -> None:
    """Process qualifications list."""
    data = _ensure_dict(response)
    qualifications = data.get("qualifications")
    if qualifications and isinstance(qualifications, list):
        db.insert_qualifications_list(url_id, qualifications)


def process_skills(db, url_id: int, response: Any) -> None:
    """Process skills list."""
    data = _ensure_dict(response)
    skills = data.get("skills")
    if skills and isinstance(skills, list):
        db.insert_skills_list(url_id, skills)


def process_attributes(db, url_id: int, response: Any) -> None:
    """Process attributes list."""
    data = _ensure_dict(response)
    attributes = data.get("attributes")
    if attributes and isinstance(attributes, list):
        db.insert_attributes_list(url_id, attributes)


def process_contacts(db, url_id: int, response: Any) -> None:
    """Process contact persons list."""
    data = _ensure_dict(response)
    contacts = data.get("contacts")
    if contacts and isinstance(contacts, list):
        db.insert_contact_persons_list(url_id, contacts)


def process_location(db, url_id: int, response: Any) -> None:
    """Process location information."""
    data = _ensure_dict(response)
    country = data.get("country")
    province = data.get("province")
    city = data.get("city")
    street_address = data.get("street_address")

    if any([country, province, city, street_address]):
        db.insert_location(
            url_id=url_id,
            country=country,
            province=province,
            city=city,
            street_address=street_address
        )


def process_job_advert(db, url_id: int, response: Any) -> None:
    """Process job advertisement details."""
    data = _ensure_dict(response)
    db.insert_job_advert_details(
        url_id=url_id,
        description=data.get("description"),
        salary=data.get("salary"),
        duration=data.get("duration"),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
        posted_date=data.get("posted_date"),
        application_deadline=data.get("application_deadline")
    )


def _ensure_dict(response: Any) -> Dict[str, Any]:
    """
    Ensure response is a dictionary.

    Args:
        response: Response object, string, or dictionary

    Returns:
        Dictionary containing response data
    """
    if hasattr(response, "model_dump"):
        # It's a Pydantic model
        return response.model_dump()
    elif isinstance(response, dict):
        # It's already a dict
        return response
    elif isinstance(response, str):
        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Return empty dict if parsing fails
            return {}
    else:
        # Unknown type, return empty dict
        return {}