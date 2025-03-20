# models.py

from typing import List, Optional
from pydantic import BaseModel, Field


class AdvertResponse(BaseModel):
    answer: str  # "yes" or "no"
    reasons: Optional[List[str]] = None


class ConfirmResponse(BaseModel):
    answer: str  # "yes" or "no"
    evidence: Optional[List[str]] = None


class JobResponse(BaseModel):
    title: Optional[str] = None


class LocationResponse(BaseModel):
    country: Optional[str] = None
    province: Optional[str] = None
    city: Optional[str] = None
    street_address: Optional[str] = None


class ContactPersonResponse(BaseModel):
    contacts: Optional[List[str]] = None  # Changed from 'name' to 'contact' to match processor


class SkillsResponse(BaseModel):
    skills: Optional[List[str]] = None  # Changed from 'skills' to 'skill' to match processor


class AttributesResponse(BaseModel):
    attributes: Optional[List[str]] = None  # Ensure this is a list


class AgencyResponse(BaseModel):
    agency: Optional[str] = None


class CompanyResponse(BaseModel):
    company: Optional[str] = None


class BenefitsResponse(BaseModel):
    benefits: Optional[List[str]] = None  # Ensure this is a list

class DutiesResponse(BaseModel):
    duties: Optional[List[str]] = None  # Ensure this is a list

class QualificationsResponse(BaseModel):
    qualifications: Optional[List[str]] = None  # Ensure this is a list

class LinkResponse(BaseModel):
    link: Optional[str] = None


class EmailResponse(BaseModel):
    email: Optional[str] = None


class PhoneNumberResponse(BaseModel):
    number: Optional[str] = None


class JobAdvertResponse(BaseModel):
    description: Optional[str] = None
    salary: Optional[str] = None
    duration: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    posted_date: Optional[str] = None
    application_deadline: Optional[str] = None