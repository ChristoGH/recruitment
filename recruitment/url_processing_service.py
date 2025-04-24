"""
Module for processing job URLs and extracting job details.
"""
import re
from typing import Dict, List
import requests
from bs4 import BeautifulSoup

def process_url(url: str, db) -> None:
    """Process a URL and extract job details."""
    # Get the page content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL: {url}")

    content = response.text

    # Extract job details
    job_details = extract_job_details(content)
    job_id = db.insert_job(
        job_details["title"],
        job_details["description"],
        job_details["posted_date"],
        job_details["type"],
        job_details["location"],
        1  # URL ID
    )

    # Extract and insert skills
    skills = extract_skills(content)
    for skill in skills:
        skill_id = db.insert_skill(skill)
        db.link_job_skill(job_id, skill_id)

    # Extract and insert qualifications
    qualifications = extract_qualifications(content)
    for qualification in qualifications:
        qualification_id = db.insert_qualification(qualification)
        db.link_job_qualification(job_id, qualification_id)

    # Extract and insert attributes
    attributes = extract_attributes(content)
    for attribute in attributes:
        attribute_id = db.insert_attribute(attribute)
        db.link_job_attribute(job_id, attribute_id)

    # Extract and insert duties
    duties = extract_duties(content)
    for duty in duties:
        duty_id = db.insert_duty(duty)
        db.link_job_duty(job_id, duty_id)

    # Extract and insert benefits
    benefits = extract_benefits(content)
    for benefit in benefits:
        benefit_id = db.insert_benefit(benefit)
        db.link_job_benefit(job_id, benefit_id)

    # Extract and insert company details
    company_details = extract_company_details(content)
    company_id = db.insert_company(company_details["name"], company_details["website"])

    # Extract and insert agency details
    agency_details = extract_agency_details(content)
    agency_id = db.insert_agency(agency_details["name"], agency_details["website"])

    # Extract and insert location details
    location_details = extract_location_details(content)
    location_id = db.insert_location(
        location_details["city"],
        location_details["province"],
        location_details["country"]
    )
    db.link_job_location(job_id, location_id)

    # Extract and insert industries
    industries = extract_industries(content)
    for industry in industries:
        industry_id = db.insert_industry(industry)
        db.link_job_industry(job_id, industry_id)

def extract_job_details(content: str) -> Dict[str, str]:
    """Extract job details from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract title
    title = soup.find('h1').text.strip()
    
    # Extract description
    description = soup.find('p').text.strip()
    
    # Extract other details
    posted_date = "2024-03-01"  # Mock value
    job_type = "Full-time"  # Mock value
    location = "Remote"  # Mock value
    
    return {
        "title": title,
        "description": description,
        "posted_date": posted_date,
        "type": job_type,
        "location": location
    }

def extract_skills(content: str) -> List[str]:
    """Extract skills from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    skills = []
    
    # Find skills section and extract skills
    skills_section = soup.find('h2', string=re.compile(r'Required Skills', re.I))
    if skills_section:
        skills_list = skills_section.find_next('ul')
        if skills_list:
            skills = [skill.text.strip() for skill in skills_list.find_all('li')]
    
    return skills

def extract_qualifications(content: str) -> List[str]:
    """Extract qualifications from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    qualifications = []
    
    # Find qualifications section and extract qualifications
    qualifications_section = soup.find('h2', string=re.compile(r'Required Qualifications', re.I))
    if qualifications_section:
        qualifications_list = qualifications_section.find_next('ul')
        if qualifications_list:
            qualifications = [qual.text.strip() for qual in qualifications_list.find_all('li')]
    
    return qualifications

def extract_attributes(content: str) -> List[str]:
    """Extract attributes from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    attributes = []
    
    # Find attributes section and extract attributes
    attributes_section = soup.find('h2', string=re.compile(r'Required Attributes', re.I))
    if attributes_section:
        attributes_list = attributes_section.find_next('ul')
        if attributes_list:
            attributes = [attr.text.strip() for attr in attributes_list.find_all('li')]
    
    return attributes

def extract_duties(content: str) -> List[str]:
    """Extract duties from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    duties = []
    
    # Find duties section and extract duties
    duties_section = soup.find('h2', string=re.compile(r'Job Duties', re.I))
    if duties_section:
        duties_list = duties_section.find_next('ul')
        if duties_list:
            duties = [duty.text.strip() for duty in duties_list.find_all('li')]
    
    return duties

def extract_benefits(content: str) -> List[str]:
    """Extract benefits from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    benefits = []
    
    # Find benefits section and extract benefits
    benefits_section = soup.find('h2', string=re.compile(r'Benefits', re.I))
    if benefits_section:
        benefits_list = benefits_section.find_next('ul')
        if benefits_list:
            benefits = [benefit.text.strip() for benefit in benefits_list.find_all('li')]
    
    return benefits

def extract_company_details(content: str) -> Dict[str, str]:
    """Extract company details from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find company section
    company_section = soup.find('h2', string=re.compile(r'About the Company', re.I))
    if company_section:
        company_div = company_section.find_next('div')
        if company_div:
            name = company_div.find('h3').text.strip()
            website = company_div.find('a')['href']
            return {"name": name, "website": website}
    
    return {"name": "Unknown", "website": ""}

def extract_agency_details(content: str) -> Dict[str, str]:
    """Extract agency details from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find agency section
    agency_section = soup.find('h2', string=re.compile(r'Posted by', re.I))
    if agency_section:
        agency_div = agency_section.find_next('div')
        if agency_div:
            name = agency_div.find('h3').text.strip()
            website = agency_div.find('a')['href']
            return {"name": name, "website": website}
    
    return {"name": "Unknown", "website": ""}

def extract_location_details(content: str) -> Dict[str, str]:
    """Extract location details from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find location section
    location_section = soup.find('h2', string=re.compile(r'Location', re.I))
    if location_section:
        location_div = location_section.find_next('div')
        if location_div:
            location_text = location_div.find('p').text.strip()
            parts = location_text.split(',')
            if len(parts) == 3:
                return {
                    "city": parts[0].strip(),
                    "province": parts[1].strip(),
                    "country": parts[2].strip()
                }
    
    return {"city": "", "province": "", "country": ""}

def extract_industries(content: str) -> List[str]:
    """Extract industries from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    industries = []
    
    # Find industries section and extract industries
    industries_section = soup.find('h2', string=re.compile(r'Industry|Sector', re.I))
    if industries_section:
        industries_list = industries_section.find_next('ul')
        if industries_list:
            industries = [industry.text.strip() for industry in industries_list.find_all('li')]
        else:
            # If no list is found, try to find a direct text mention
            industry_text = industries_section.find_next('p')
            if industry_text:
                # Split by common separators and clean up
                industries = [i.strip() for i in re.split(r'[,;/]', industry_text.text.strip())]
                industries = [i for i in industries if i]  # Remove empty strings
    
    return industries 