# Recruitment System

A distributed system for discovering and processing recruitment advertisements using FastAPI, RabbitMQ, and SQLite.

## System Architecture

The system consists of two main services:

1. **URL Discovery Service** (`url_discovery_service.py`)
   - Searches for recruitment URLs using Google Search
   - Filters and validates URLs
   - Publishes valid URLs to RabbitMQ queue
   - Runs on port 8000

2. **URL Processing Service** (`url_processing_service.py`)
   - Consumes URLs from RabbitMQ queue
   - Crawls and extracts content from URLs
   - Uses LLMs to analyze and extract structured data
   - Stores results in SQLite database
   - Runs on port 8001

### Database Schema

The system uses a SQLite database (`recruitment.db`) with the following main tables:

- `urls`: Stores crawled URLs and their processing status
- `jobs`: Core job information
- `adverts`: Job advertisement details
- `companies`: Hiring organizations
- `agencies`: Recruitment firms
- `skills`: Required skills with experience levels
- `qualifications`: Required qualifications
- `attributes`: Job attributes
- `duties`: Job responsibilities
- `locations`: Geographic locations
- `benefits`: Job benefits
- `industries`: Industry categories

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Neo4j database (optional)

## Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_api_key
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PWD=password
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recruitment.git
cd recruitment
```

2. Build and start the services:
```bash
docker-compose up --build
```

## Usage

### Automatic Discovery Process

The system includes an automated discovery process that:
- Runs every 60 minutes via a cron job
- Searches for new recruitment URLs
- Automatically publishes found URLs to RabbitMQ
- Triggers immediate processing of discovered URLs

You can monitor this process through:
- RabbitMQ management interface (http://localhost:15672)
- Service logs: `docker-compose logs url_discovery`
- Database updates in `databases/recruitment.db`

### URL Discovery Service (Port 8000)

1. Create a new search for recruitment URLs:
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

2. Check search status:
```bash
curl "http://localhost:8000/search/status/test_search"
```

3. Get URLs found by a search:
```bash
curl "http://localhost:8000/search/urls/test_search"
```

### URL Processing Service (Port 8001)

1. Process a single URL:
```bash
curl -X POST "http://localhost:8001/process" \
-H "Content-Type: application/json" \
-d '{
  "url": "https://example.com/job-posting",
  "process_all_prompts": true,
  "use_transaction": true
}'
```

2. Check URL processing status:
```bash
curl "http://localhost:8001/status/https://example.com/job-posting"
```

### Monitoring

1. View service logs:
```bash
docker-compose logs url_discovery
docker-compose logs url_processing
```

2. Check service health:
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

3. Access RabbitMQ management interface:
   - URL: `http://localhost:15672`
   - Default credentials: guest/guest

4. View database contents:
```bash
sqlite3 databases/recruitment.db
```

## Testing

Run the test suite:

```bash
pytest tests/
```

The test suite includes:
- Unit tests for URL discovery service
- Unit tests for URL processing service
- Unit tests for database operations
- Integration tests for the complete pipeline

## Development

### Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and run tests:
```bash
pytest tests/
```

3. Update documentation if needed

4. Create a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep functions small and focused

## Troubleshooting

1. Check service logs:
```bash
docker-compose logs url_discovery
docker-compose logs url_processing
```

2. Verify RabbitMQ connection:
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

3. Check database:
```bash
sqlite3 databases/recruitment.db
```

4. Common issues:
   - Ensure all environment variables are set correctly
   - Verify RabbitMQ is running and accessible
   - Check database permissions and path
   - Monitor OpenAI API rate limits

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
