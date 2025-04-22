
# Recruitment Database Schema

## Overview
This document outlines the new database schema for the recruitment system. The schema is designed to be more normalized and maintainable than the previous version.

## Database Migration Notes
- The migration process moves data from `recruitment.db` to `recruitment_new.db`
- The old database uses singular table names (e.g., `company`, `agency`)
- The new database uses plural table names (e.g., `companies`, `agencies`)
- The `hiring_company` table from the old database is not used and has been removed
- Companies are distinguished from agencies based on their role:
  - Companies: Organizations hiring for positions
  - Agencies: Recruitment firms that post jobs on behalf of companies

## Naming Conventions
- Table names are plural, lowercase, and use underscores
- Primary keys are named `id`
- Foreign keys follow the pattern `table_name_id`
- Relationship tables use the pattern `table1_table2`
- Timestamps use `created_at` and `updated_at`

## Core Tables

### urls
Primary entry point for all crawled URLs.
```sql
CREATE TABLE urls (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    domain_name TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### jobs
Core job information.
```sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    salary_min DECIMAL,
    salary_max DECIMAL,
    salary_currency TEXT,
    status TEXT CHECK(status IN ('active', 'inactive', 'filled', 'expired', 'draft', 'published')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### adverts
Job advertisement details.
```sql
CREATE TABLE adverts (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    posted_date DATE,
    application_deadline DATE,
    is_remote BOOLEAN DEFAULT FALSE,
    is_hybrid BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
```

### companies
Organizations hiring for positions.
```sql
CREATE TABLE companies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### agencies
Recruitment firms that post jobs on behalf of companies.
```sql
CREATE TABLE agencies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### skills
Job skills.
```sql
CREATE TABLE skills (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### qualifications
Required qualifications.
```sql
CREATE TABLE qualifications (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### attributes
Job attributes.
```sql
CREATE TABLE attributes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### duties
Job duties.
```sql
CREATE TABLE duties (
    id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### locations
Geographic locations.
```sql
CREATE TABLE locations (
    id INTEGER PRIMARY KEY,
    country TEXT NOT NULL,
    province TEXT,
    city TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### phones
Phone numbers.
```sql
CREATE TABLE phones (
    id INTEGER PRIMARY KEY,
    number TEXT NOT NULL,
    type TEXT CHECK(type IN ('mobile', 'landline', 'fax')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### emails
Email addresses.
```sql
CREATE TABLE emails (
    id INTEGER PRIMARY KEY,
    address TEXT NOT NULL,
    type TEXT CHECK(type IN ('primary', 'secondary', 'work')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### benefits
Job benefits.
```sql
CREATE TABLE benefits (
    id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### recruitment_evidence
Recruitment evidence.
```sql
CREATE TABLE recruitment_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_id INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE
);
```

## Relationship Tables

### job_urls
Links jobs to URLs.
```sql
CREATE TABLE job_urls (
    job_id INTEGER NOT NULL,
    url_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, url_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (url_id) REFERENCES urls(id) ON DELETE CASCADE
);
```

### job_skills
Links jobs to skills with experience levels.
```sql
CREATE TABLE job_skills (
    job_id INTEGER NOT NULL,
    skill_id INTEGER NOT NULL,
    experience TEXT CHECK(experience IN ('entry', 'intermediate', 'senior', 'expert')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, skill_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
);
```

### job_qualifications
Links jobs to qualifications.
```sql
CREATE TABLE job_qualifications (
    job_id INTEGER NOT NULL,
    qualification_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, qualification_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (qualification_id) REFERENCES qualifications(id) ON DELETE CASCADE
);
```

### job_attributes
Links jobs to attributes.
```sql
CREATE TABLE job_attributes (
    job_id INTEGER NOT NULL,
    attribute_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, attribute_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (attribute_id) REFERENCES attributes(id) ON DELETE CASCADE
);
```

### job_companies
Links jobs to companies.
```sql
CREATE TABLE job_companies (
    job_id INTEGER NOT NULL,
    company_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, company_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
);
```

### job_agencies
Links jobs to agencies.
```sql
CREATE TABLE job_agencies (
    job_id INTEGER NOT NULL,
    agency_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, agency_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (agency_id) REFERENCES agencies(id) ON DELETE CASCADE
);
```

### job_duties
Links jobs to duties.
```sql
CREATE TABLE job_duties (
    job_id INTEGER NOT NULL,
    duty_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, duty_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (duty_id) REFERENCES duties(id) ON DELETE CASCADE
);
```

### job_locations
Links jobs to locations.
```sql
CREATE TABLE job_locations (
    job_id INTEGER NOT NULL,
    location_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (job_id, location_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
);
```

### company_phones
Links companies to phone numbers.
```sql
CREATE TABLE company_phones (
    company_id INTEGER NOT NULL,
    phone_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_id, phone_id),
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE,
    FOREIGN KEY (phone_id) REFERENCES phones(id) ON DELETE CASCADE
);
```

### agency_phones
Links agencies to phone numbers.
```sql
CREATE TABLE agency_phones (
    agency_id INTEGER NOT NULL,
    phone_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agency_id, phone_id),
    FOREIGN KEY (agency_id) REFERENCES agencies(id) ON DELETE CASCADE,
    FOREIGN KEY (phone_id) REFERENCES phones(id) ON DELETE CASCADE
);
```

### company_emails
Links companies to email addresses.
```sql
CREATE TABLE company_emails (
    company_id INTEGER NOT NULL,
    email_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_id, email_id),
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE,
    FOREIGN KEY (email_id) REFERENCES emails(id) ON DELETE CASCADE
);
```

### agency_emails
Links agencies to email addresses.
```sql
CREATE TABLE agency_emails (
    agency_id INTEGER NOT NULL,
    email_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agency_id, email_id),
    FOREIGN KEY (agency_id) REFERENCES agencies(id) ON DELETE CASCADE,
    FOREIGN KEY (email_id) REFERENCES emails(id) ON DELETE CASCADE
);
```

### url_processing_status
Tracks URL processing status.
```sql
CREATE TABLE url_processing_status (
    url_id INTEGER PRIMARY KEY,
    status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    last_processed_at TIMESTAMP,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (url_id) REFERENCES urls(id) ON DELETE CASCADE
);
```

## Migration Strategy
1. Create new database with the schema above
2. Create migration scripts to:
   - Map existing data to new schema
   - Preserve relationships
   - Handle data transformation
3. Validate data integrity
4. Switch application to new database

## Next Steps
1. Create migration scripts
2. Update application code
3. Test data integrity
4. Deploy changes

# Microservices Architecture

## Overview

The recruitment system has been refactored into a microservices architecture using FastAPI and RabbitMQ. This architecture replaces the previous CSV-based workflow with a more scalable, event-driven approach.

## Components

### 1. URL Discovery Service

The URL Discovery Service (`url_discovery_service.py`) is responsible for:
- Searching for recruitment URLs using Google with specific terms and date ranges
- Validating URLs to ensure they are recruitment-related
- Publishing valid URLs to a RabbitMQ queue for processing
- Providing API endpoints for initiating searches and checking results

### 2. URL Processing Service

The URL Processing Service (`url_processing_service.py`) is responsible for:
- Consuming URLs from the RabbitMQ queue
- Crawling websites to extract content
- Using LLMs to verify recruitment advertisements and extract structured data
- Storing results in the Neo4j database
- Providing API endpoints for processing individual URLs and checking status

### 3. RabbitMQ

RabbitMQ serves as the message broker between services:
- The "recruitment_urls" queue holds URLs waiting to be processed
- Messages are acknowledged only after successful processing
- Failed messages are requeued for retry

### 4. Neo4j Database

The Neo4j database stores all processed data:
- URLs and their metadata
- Extracted recruitment information
- Relationships between entities

## Setup and Configuration

### Environment Variables

Create a `.env` file with the following variables:
```
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration (for local Neo4j Desktop)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# RabbitMQ Configuration (optional, defaults shown)
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
```

### Docker Compose

The system is containerized using Docker Compose:
```bash
docker-compose up --build
```

This starts:
- URL Discovery Service on port 8000
- URL Processing Service on port 8001
- RabbitMQ on ports 5672 (AMQP) and 15672 (management UI)

## API Usage

### URL Discovery Service

#### Create a Search
```bash
curl -X POST "http://localhost:8000/search" \
-H "Content-Type: application/json" \
-d '{
  "id": "batch1",
  "days_back": 7,
  "excluded_domains": [],
  "academic_suffixes": [],
  "recruitment_terms": [
    "recruitment advert",
    "job vacancy",
    "hiring now",
    "employment opportunity",
    "career opportunity",
    "job advertisement",
    "recruitment drive"
  ]
}'
```

#### Check Search Status
```bash
curl "http://localhost:8000/search/status/test_search"
```

#### Get Found URLs
```bash
curl "http://localhost:8000/search/urls/test_search"
```

### URL Processing Service

#### Process a Single URL
```bash
curl -X POST "http://localhost:8001/process" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/job-posting",
    "process_all_prompts": true,
    "use_transaction": true
  }'
```

#### Check Processing Status
```bash
curl "http://localhost:8001/status/https://example.com/job-posting"
```

## Local Development

For local development without Docker:

1. Start RabbitMQ:
   ```bash
   docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
   ```

2. Run the services directly:
   ```bash
   # Terminal 1
   python url_discovery_service.py
   
   # Terminal 2
   python url_processing_service.py
   ```

3. If you need to run both services and avoid port conflicts:
   ```bash
   # Terminal 1 - URL Discovery Service runs on port 8000 by default
   python url_discovery_service.py
   
   # Terminal 2 - URL Processing Service can run on port 8002 or another port
   python -m url_processing_service
   ```

4. To restart the URL processing service after code changes:
   ```bash
   # Stop the current service with Ctrl+C
   # Then restart it with
   cd /path/to/project/root && python -m url_processing_service
   ```

## Handling Common Issues

### Empty Response Issues

The URL processing service now handles empty responses from the LLM by providing appropriate default responses:
- For list-based prompts (attributes, duties, etc.), it returns empty lists
- For complex prompts (location, job advert), it returns structured objects with null fields
- This ensures processing continues even if some prompts fail

### Port Conflicts

If you encounter port conflicts:
1. Check which services are running on which ports:
   ```bash
   lsof -i :8000
   lsof -i :8001
   lsof -i :8002
   ```
2. Modify the port in the service file if needed:
   ```python
   # At the bottom of url_processing_service.py
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8002)  # Change port as needed
   ```

### Model Mapping Issues

If you encounter a "No model class found for prompt key" warning:
1. Check that all prompt keys in the prompts.py file are properly mapped in the get_model_for_prompt function
2. Add any missing mappings to the model_map dictionary

## Troubleshooting

### Connection Issues with Neo4j

If the URL Processing Service can't connect to your local Neo4j instance:
1. Verify Neo4j Desktop is running and the database is started
2. Check the connection details in your `.env` file
3. Ensure the URL Processing Service has network access to your local machine
4. Try using `host.docker.internal` instead of `localhost` in the NEO4J_URI if needed

### RabbitMQ Issues

If you encounter RabbitMQ connection issues:
1. Check the RabbitMQ management UI at http://localhost:15672 (login with guest/guest)
2. Verify the queue "recruitment_urls" exists
3. Check the logs for connection errors

cat logs/url_processing_service.log | grep "Inserted"
