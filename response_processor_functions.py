import json
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel


class PromptResponseProcessor:
    """
    Process responses from LLM prompts and insert data into the database.
    """

    def __init__(self, db):
        """
        Initialize the processor with a database connection.

        Args:
            db: RecruitmentDatabase instance
        """
        self.db = db

    def process_response(self, url_id: int, prompt_type: str, response: Union[str, BaseModel]) -> None:
        """
        Process a response from an LLM based on the prompt type.

        Args:
            url_id: The URL ID in the database
            prompt_type: The type of prompt that generated the response
            response: The JSON response string or Pydantic model from the LLM
        """
        try:
            # Convert response to dict if it's a Pydantic model
            if isinstance(response, BaseModel):
                data = response.model_dump()
            else:
                # Otherwise assume it's a JSON string
                data = json.loads(response)

            # Process based on prompt type
            if prompt_type == "recruitment_prompt":
                self._process_recruitment(url_id, data)
            elif prompt_type == "company_prompt":
                self._process_company(url_id, data)
            elif prompt_type == "agency_prompt":
                self._process_agency(url_id, data)
            elif prompt_type == "job_prompt":
                self._process_job(url_id, data)
            elif prompt_type == "skill_prompt":
                self._process_single_skill(url_id, data)
            elif prompt_type == "phone_prompt":
                self._process_phone(url_id, data)
            elif prompt_type == "email_prompt":
                self._process_email(url_id, data)
            elif prompt_type == "link_prompt":
                self._process_link(url_id, data)
            elif prompt_type == "benefits_prompt":
                self._process_benefits(url_id, data)
            elif prompt_type == "skills_prompt":
                self._process_skills(url_id, data)
            elif prompt_type == "attributes_prompt":
                self._process_attributes(url_id, data)
            elif prompt_type == "contact_prompt":
                self._process_contacts(url_id, data)
            elif prompt_type == "location_prompt":
                self._process_location(url_id, data)
            elif prompt_type == "jobadvert_prompt":
                self._process_job_advert(url_id, data)
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response}")
        except KeyError as e:
            print(f"Missing key in response data: {e}")
        except Exception as e:
            print(f"Error processing response: {e}")


    def _process_recruitment(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process recruitment flag and evidence."""
        answer = data.get("answer")
        if answer == "yes":
            # Update recruitment flag to 1 (is recruitment)
            self.db.update_field_by_id(url_id, "recruitment_flag", 1)

            # Process evidence if available
            evidence = data.get("evidence")
            if evidence and isinstance(evidence, list):
                self.db.insert_recruitment_evidence_list(url_id, evidence)
        elif answer == "no":
            # Update recruitment flag to 0 (not recruitment)
            self.db.update_field_by_id(url_id, "recruitment_flag", 0)

    def _process_company(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process company name."""
        company = data.get("company")
        if company:
            self.db.insert_company(url_id, company)

    def _process_agency(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process agency name."""
        agency = data.get("agency")
        if agency:
            self.db.insert_agency(url_id, agency)

    def _process_job(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process job title."""
        title = data.get("title")
        if title:
            self.db.insert_job_title(url_id, title)

    def _process_single_skill(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process single skill."""
        skill = data.get("skill")
        if skill:
            self.db.insert_skill(url_id, skill)

    def _process_phone(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process phone number."""
        number = data.get("number")
        if number:
            self.db.insert_phone_number(url_id, number)

    def _process_email(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process email address."""
        email = data.get("email")
        if email:
            self.db.insert_email(url_id, email)

    def _process_link(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process contact URL link."""
        link = data.get("link")
        if link:
            self.db.insert_link(url_id, link)

    def _process_benefits(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process benefits list."""
        benefits = data.get("benefit")
        if benefits and isinstance(benefits, list):
            self.db.insert_benefits_list(url_id, benefits)

    def _process_skills(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process skills list."""
        skills = data.get("skill")
        if skills and isinstance(skills, list):
            self.db.insert_skills_list(url_id, skills)

    def _process_attributes(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process attributes list."""
        attributes = data.get("attribute")
        if attributes and isinstance(attributes, list):
            self.db.insert_attributes_list(url_id, attributes)

    def _process_contacts(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process contact persons list."""
        contacts = data.get("contact")
        if contacts and isinstance(contacts, list):
            self.db.insert_contact_persons_list(url_id, contacts)

    def _process_location(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process location information."""
        country = data.get("country")
        province = data.get("province")
        city = data.get("city")
        street_address = data.get("street_address")

        if any([country, province, city, street_address]):
            self.db.insert_location(
                url_id=url_id,
                country=country,
                province=province,
                city=city,
                street_address=street_address
            )

    def _process_job_advert(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process job advertisement details."""
        self.db.insert_job_advert_details(
            url_id=url_id,
            description=data.get("description"),
            salary=data.get("salary"),
            duration=data.get("duration"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            posted_date=data.get("posted_date"),
            application_deadline=data.get("application_deadline")
        )


# Usage example
def process_all_prompt_responses(db, url_id: int, prompt_responses: Dict[str, str]) -> None:
    """
    Process all prompt responses for a URL.

    Args:
        db: RecruitmentDatabase instance
        url_id: The URL ID
        prompt_responses: Dictionary mapping prompt_type to JSON response string
    """
    processor = PromptResponseProcessor(db)

    for prompt_type, response in prompt_responses.items():
        processor.process_response(url_id, prompt_type, response)