# Enhanced models.py with Pydantic V2 validation

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, EmailStr, model_validator
import re
from datetime import datetime


class AdvertResponse(BaseModel):
    answer: Literal["yes", "no"]  # Restrict to only valid values
    evidence: Optional[List[str]] = None

    @model_validator(mode='after')
    def validate_evidence_provided(self):
        if self.answer == 'yes' and (not self.evidence or len(self.evidence) == 0):
            raise ValueError("Evidence must be provided when answer is 'yes'")
        return self


class ConfirmResponse(BaseModel):
    answer: Literal["yes", "no"]
    evidence: Optional[List[str]] = None


class JobResponse(BaseModel):
    title: Optional[str] = Field(None, min_length=2, max_length=200)


class LocationResponse(BaseModel):
    country: Optional[str] = Field(None, min_length=2, max_length=100)
    province: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    street_address: Optional[str] = Field(None, max_length=500)


class ContactPersonResponse(BaseModel):
    contacts: Optional[List[str]] = None

    @field_validator('contacts')
    @classmethod
    def validate_contacts(cls, v):
        if v:
            for contact in v:
                if len(contact.strip().split()) < 2:
                    raise ValueError(f"Contact '{contact}' should include first and last name")
        return v


class SkillsResponse(BaseModel):
    skills: Optional[List[str]] = None

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v):
        if v:
            return [skill.strip() for skill in v if skill.strip()]
        return v


class AttributesResponse(BaseModel):
    attributes: Optional[List[str]] = None

    @field_validator('attributes')
    @classmethod
    def validate_attributes(cls, v):
        if v:
            return [attr.strip() for attr in v if attr.strip()]
        return v


class AgencyResponse(BaseModel):
    agency: Optional[str] = Field(None, min_length=2, max_length=200)


class CompanyResponse(BaseModel):
    company: Optional[str] = Field(None, min_length=2, max_length=200)


class BenefitsResponse(BaseModel):
    benefits: Optional[List[str]] = None

    @field_validator('benefits')
    @classmethod
    def validate_benefits(cls, v):
        if v:
            return [benefit.strip() for benefit in v if benefit.strip()]
        return v


class DutiesResponse(BaseModel):
    duties: Optional[List[str]] = None

    @field_validator('duties')
    @classmethod
    def validate_duties(cls, v):
        if v:
            return [duty.strip() for duty in v if duty.strip()]
        return v


class QualificationsResponse(BaseModel):
    qualifications: Optional[List[str]] = None

    @field_validator('qualifications')
    @classmethod
    def validate_qualifications(cls, v):
        if v:
            return [qual.strip() for qual in v if qual.strip()]
        return v


class LinkResponse(BaseModel):
    link: Optional[str] = None

    @field_validator('link')
    @classmethod
    def validate_link(cls, v):
        if v:
            # Simple URL validation
            url_pattern = re.compile(
                r'^(?:http|ftp)s?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
                r'localhost|'  # localhost
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)

            if not url_pattern.match(v):
                raise ValueError('Invalid URL format')
        return v


class EmailResponse(BaseModel):
    email: Optional[EmailStr] = None  # Using EmailStr for email validation


class CompanyPhoneNumberResponse(BaseModel):
    number: Optional[str] = None

    @field_validator('number')
    @classmethod
    def validate_phone(cls, v):
        if v:
            # Remove common separators for validation
            clean_number = re.sub(r'[\s\-\(\)\.]', '', v)

            # Check if it's all digits after cleaning
            if not clean_number.isdigit():
                raise ValueError('Phone number should contain only digits, spaces, and common separators')

            # Check reasonable length
            if len(clean_number) < 7 or len(clean_number) > 15:
                raise ValueError('Phone number length is invalid')
        return v


class JobAdvertResponse(BaseModel):
    description: Optional[str] = None
    salary: Optional[str] = None
    duration: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    posted_date: Optional[str] = None
    application_deadline: Optional[str] = None

    @field_validator('start_date', 'end_date', 'posted_date', 'application_deadline')
    @classmethod
    def validate_dates(cls, v):
        if v:
            try:
                # Check if date follows the required format
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in format YYYY-MM-DD')
        return v

    @model_validator(mode='after')
    def validate_date_ranges(self):
        if self.start_date and self.end_date:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            if end < start:
                raise ValueError('End date cannot be before start date')
        return self