import json
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, ValidationError

# Import the model mapping functionality
from utils import get_model_for_prompt

# Set up logging
logger = logging.getLogger(__name__)


class ResponseProcessingError(Exception):
    """Exception raised for errors during response processing."""
    pass


class ValidationFailedError(ResponseProcessingError):
    """Exception raised when validation fails for a response."""
    pass


class JSONDecodeError(ResponseProcessingError):
    """Exception raised when JSON decoding fails for a response."""
    pass


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

    def process_response(self, url_id: int, prompt_type: str, response: Union[str, Dict, BaseModel]) -> None:
        """
        Process a response from an LLM based on the prompt type.

        Args:
            url_id: The URL ID in the database
            prompt_type: The type of prompt that generated the response
            response: The JSON response string, dict, or Pydantic model from the LLM

        Raises:
            ValidationFailedError: If response validation fails
            JSONDecodeError: If JSON parsing fails
            ResponseProcessingError: For other processing errors
            DatabaseError: If a database operation fails
        """
        try:
            # Parse the response into a dictionary
            data = self._parse_response(response)

            # Validate the data
            data = self._validate_data(prompt_type, data)

            # Process the validated data
            self._process_validated_data(url_id, prompt_type, data)

        except (ValidationFailedError, JSONDecodeError) as e:
            # These are expected errors we want to log but might not halt execution
            logger.error(f"{e.__class__.__name__} for {prompt_type}: {e}")
            raise
        except Exception as e:
            # Unexpected errors - log and re-raise
            logger.error(f"Unexpected error processing {prompt_type}: {e}", exc_info=True)
            raise ResponseProcessingError(f"Failed to process {prompt_type}: {e}") from e

    def _parse_response(self, response: Union[str, Dict, BaseModel]) -> Dict:
        """
        Parse the response into a dictionary.

        Args:
            response: The response to parse

        Returns:
            Dict: The parsed response data

        Raises:
            JSONDecodeError: If JSON parsing fails
        """
        # Handle case when response is already a Pydantic model
        if isinstance(response, BaseModel):
            return response.model_dump()
        # Handle case when response is a dict
        elif isinstance(response, dict):
            return response
        else:
            # Otherwise assume it's a JSON string
            try:
                return json.loads(response)
            except json.decoder.JSONDecodeError as e:
                msg = f"Failed to parse JSON response: {response[:100]}..." if len(response) > 100 else response
                logger.error(msg)
                logger.error(f"JSON error: {e}")
                raise JSONDecodeError(f"Invalid JSON format: {e}") from e

    def _validate_data(self, prompt_type: str, data: Dict) -> Dict:
        """
        Validate the data against the appropriate model.

        Args:
            prompt_type: The type of prompt
            data: The data to validate

        Returns:
            Dict: The validated data

        Raises:
            ValidationFailedError: If validation fails
        """
        try:
            model_class = get_model_for_prompt(prompt_type)
            validated_data = model_class(**data)
            return validated_data.model_dump()
        except KeyError:
            logger.warning(f"No model class found for prompt key: {prompt_type}")
            return data  # Return unvalidated data if no model found
        except ValidationError as e:
            msg = f"Validation error for {prompt_type}: {e}"
            logger.error(msg)
            raise ValidationFailedError(msg) from e

    def _process_validated_data(self, url_id: int, prompt_type: str, data: Dict) -> None:
        """
        Process the validated data based on the prompt type.

        Args:
            url_id: The URL ID
            prompt_type: The type of prompt
            data: The validated data

        Raises:
            ResponseProcessingError: If processing fails
        """
        # Map prompt types to their processor methods
        processors = {
            "recruitment_prompt": self._process_recruitment,
            "company_prompt": self._process_company,
            "agency_prompt": self._process_agency,
            "job_prompt": self._process_job,
            "company_phone_number_prompt": self._process_company_phone_number,
            "email_prompt": self._process_email,
            "link_prompt": self._process_link,
            "benefits_prompt": self._process_benefits,
            "skills_prompt": self._process_skills,
            "attributes_prompt": self._process_attributes,
            "contacts_prompt": self._process_contacts,
            "location_prompt": self._process_location,
            "jobadvert_prompt": self._process_job_advert,
            "qualifications_prompt": self._process_qualifications,
            "duties_prompt": self._process_duties,
        }

        # Get the processor method for this prompt type
        processor = processors.get(prompt_type)

        if processor:
            try:
                # Call the processor method
                processor(url_id, data)
            except Exception as e:
                msg = f"Error in processor for {prompt_type}: {e}"
                logger.error(msg, exc_info=True)
                raise ResponseProcessingError(msg) from e
        else:
            logger.warning(f"Unknown prompt type: {prompt_type}")

    # Example of a multi-table database operation using transaction
    def process_job_posting(self, url_id: int, data: Dict[str, Any]) -> None:
        """
        Process a complete job posting with transaction support.

        This handles multiple related database operations as an atomic unit.

        Args:
            url_id: The URL ID
            data: Combined data from multiple prompts
        """
        try:
            # Extract data from multiple prompt responses
            job_title = data.get("job_title")
            company = data.get("company")
            description = data.get("description")
            salary = data.get("salary")
            location = data.get("location", {})
            skills = data.get("skills", [])
            benefits = data.get("benefits", [])

            # Use the new transaction-based method
            if job_title:
                self.db.insert_job_with_details(
                    url_id=url_id,
                    job_title=job_title,
                    company_name=company,
                    description=description,
                    salary=salary
                )

                # Add location if provided
                if any(location.values()):
                    self.db.insert_location(
                        url_id=url_id,
                        country=location.get("country"),
                        province=location.get("province"),
                        city=location.get("city"),
                        street_address=location.get("street_address")
                    )

                # Add skills and benefits
                if skills:
                    self.db.insert_skills_list(url_id, skills)

                if benefits:
                    self.db.insert_benefits_list(url_id, benefits)

        except Exception as e:
            logger.error(f"Error processing job posting: {e}", exc_info=True)
            raise ResponseProcessingError(f"Failed to process job posting: {e}") from e

    # The rest of the individual processor methods remain the same
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

    def _process_company_phone_number(self, url_id: int, data: Dict[str, Any]) -> None:
        """Process phone number."""
        number = data.get("number")
        if number:
            self.db.insert_company_phone_number(url_id, number)

    # Additional processor methods would follow...