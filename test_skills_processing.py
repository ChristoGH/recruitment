# test_skills_processing.py

import unittest
import json
from unittest.mock import MagicMock, patch

# Import the modules to test
from batch_processor import process_skills, extract_job_data_from_responses


class TestSkillsProcessing(unittest.TestCase):
    """Test cases for the refactored skills processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.db_mock = MagicMock()
        self.url_id = 123

    def test_process_skills_tuples(self):
        """Test processing skills in tuple format."""
        # Test data in tuple format
        skills_response = {
            "skills": [
                ("Python", "3+ years"),
                ("SQL", "2 years"),
                ("Docker", "1 year")
            ]
        }

        # Call the function
        process_skills(self.db_mock, self.url_id, skills_response)

        # Verify the database was called with the correct parameters
        self.db_mock.insert_skills_list.assert_called_once()
        args = self.db_mock.insert_skills_list.call_args[0]
        self.assertEqual(args[0], self.url_id)

        # Verify the skills data was processed correctly
        processed_skills = args[1]
        self.assertEqual(len(processed_skills), 3)
        self.assertEqual(processed_skills[0], ("Python", "3+ years"))
        self.assertEqual(processed_skills[1], ("SQL", "2 years"))
        self.assertEqual(processed_skills[2], ("Docker", "1 year"))

    def test_process_skills_dicts(self):
        """Test processing skills in dictionary format."""
        # Test data in dictionary format
        skills_response = {
            "skills": [
                {"skill": "Python", "experience": "3+ years"},
                {"skill": "SQL", "experience": "2 years"},
                {"skill": "Docker", "experience": "1 year"}
            ]
        }

        # Call the function
        process_skills(self.db_mock, self.url_id, skills_response)

        # Verify the skills data was processed correctly
        processed_skills = self.db_mock.insert_skills_list.call_args[0][1]
        self.assertEqual(len(processed_skills), 3)
        self.assertEqual(processed_skills[0], ("Python", "3+ years"))
        self.assertEqual(processed_skills[1], ("SQL", "2 years"))
        self.assertEqual(processed_skills[2], ("Docker", "1 year"))

    def test_process_skills_strings(self):
        """Test processing skills in string format (backward compatibility)."""
        # Test data in string format
        skills_response = {
            "skills": ["Python", "SQL", "Docker"]
        }

        # Call the function
        process_skills(self.db_mock, self.url_id, skills_response)

        # Verify the skills data was processed correctly
        processed_skills = self.db_mock.insert_skills_list.call_args[0][1]
        self.assertEqual(len(processed_skills), 3)
        self.assertEqual(processed_skills[0], ("Python", None))
        self.assertEqual(processed_skills[1], ("SQL", None))
        self.assertEqual(processed_skills[2], ("Docker", None))

    def test_extract_job_data(self):
        """Test extracting job data including skills with experience."""
        # Create test prompt responses
        prompt_responses = {
            "job_prompt": {"title": "Software Engineer"},
            "company_prompt": {"company": "Tech Corp"},
            "skills_prompt": {
                "skills": [
                    ("Python", "3+ years"),
                    ("SQL", "2 years"),
                    {"skill": "Docker", "experience": "1 year"}
                ]
            },
            "benefits_prompt": {"benefits": ["Remote work", "Health insurance"]}
        }

        # Call the function
        job_data = extract_job_data_from_responses(prompt_responses)

        # Verify job data
        self.assertEqual(job_data["job_title"], "Software Engineer")
        self.assertEqual(job_data["company"], "Tech Corp")

        # Verify skills data
        skills = job_data["skills"]
        self.assertEqual(len(skills), 3)

        # Check that skills were transformed into dictionary format
        self.assertEqual(skills[0]["skill"], "Python")
        self.assertEqual(skills[0]["experience"], "3+ years")
        self.assertEqual(skills[1]["skill"], "SQL")
        self.assertEqual(skills[1]["experience"], "2 years")
        self.assertEqual(skills[2]["skill"], "Docker")
        self.assertEqual(skills[2]["experience"], "1 year")

        # Verify other data
        self.assertEqual(job_data["benefits"], ["Remote work", "Health insurance"])


if __name__ == "__main__":
    unittest.main()