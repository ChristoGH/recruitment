
---

# Trafficking News Article Crawler



This Python script is designed to perform targeted searches for news articles related to trafficking incidents, 
particularly within a specified time range. 
By constructing advanced search queries using predefined keywords and filters, it fetches relevant URLs via Google search, 
logs the activity, and then exports the results either to a CSV file or a Neo4j database.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Script Workflow](#script-workflow)
- [Logging](#logging)
- [Output](#output)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Developer](#developer)

## Overview

This module searches for news articles that mention trafficking-related terms (e.g., "human trafficking", "child trafficking", etc.) 
along with evidence keywords (e.g., "arrest", "rescue", "investigation") within the last _n_ days. 
It excludes specific domains if needed and limits the search to South Africa by default. 
After gathering the articles, it logs the results and exports the URLs along with their domain names to a CSV file. 
There is also an option to save the data to a Neo4j graph database.

## Features

- **Targeted Search:** Combines trafficking-related and evidence terms with date filters to create precise Google search queries.
- **Domain Filtering:** Excludes unwanted domains from the search results.
- **Configurable Time Range:** Searches articles from a user-defined number of days in the past.
- **Flexible Output Options:** Save results to a CSV file and optionally to a Neo4j database.
- **Robust Error Handling & Logging:** Uses logging to track execution details and potential errors.
- **Configuration Driven:** Loads search parameters from an external JSON configuration file.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**
   I use uv to manage dependencies in a virtual environment. Then, install the required packages:
   ```bash
   pip install argparse tldextract googlesearch-python google neo4j
   ```
   > **Note:** Make sure that you are using the correct `googlesearch-python` package as required by the script. You may need 
   > to verify which version works best with your environment.

3. **Set Up Custom Libraries:**
   This script relies on a custom module `libraries.neo4j_lib` for Neo4j interactions. Ensure that this module is in the correct path or update the import paths as necessary.

## Configuration

The script expects a JSON configuration file (`search_config.json`) in the current working directory. This file should define one or more search configurations under the key `run_configs`. A sample configuration might look like:

```json
{
  "run_configs": [
    {
      "id": "trafficking_search",
      "days_back": 7,
      "excluded_domains": ["example.com", "anotherdomain.com"]
    }
  ]
}
```

- **id:** Identifier for the search configuration.
- **days_back:** Number of days to look back from the current date (this can also be overridden by the command-line argument).
- **excluded_domains:** List of domains to exclude from the search results.

## Usage

Run the script from the command line with an optional argument to specify how many days back to search:

```bash
python web_crawler.py --days_back 7
```

If the `--days_back` argument is not provided, it defaults to 7 days.

## Script Workflow

1. **Configuration Loading:**  
   The script begins by loading search parameters from the `search_config.json` file. If the file is missing or contains invalid JSON, it raises an error.

2. **Logging Setup:**  
   Logging is configured to record messages to `web_crawler.log`, providing insights into the script’s activity.

3. **Search Query Construction:**  
   The `TraffickingNewsSearch` class constructs a Google search query combining:
   - **Trafficking Terms:** e.g., `"human trafficking"`, `"child trafficking"`.
   - **Evidence Terms:** e.g., `"arrest"`, `"rescue"`, `"investigation"`.
   - **News/Article Keywords:** Ensuring that the results are related to news content.
   - **Geographic Filter:** Limits search to "South Africa".
   - **Date Filters:** Uses `after:` and `before:` operators to restrict the search within a specified date range.
   - **Excluded Domains:** Appends `-site:` filters to remove undesired sources.

4. **Article Fetching:**  
   It utilizes the `googlesearch` module to fetch URLs. Each URL is then filtered to ensure it does not belong to any excluded domains.

5. **Data Storage Options:**
   - **CSV File:** The script saves valid URLs and their domain names to a timestamped CSV file in an `output` directory.
   - **Neo4j Integration:** Optionally (if uncommented), it saves the URL information to a Neo4j database by executing a Cypher query.

## Logging

- The script logs key events such as:
  - Startup and configuration loading.
  - Construction of search queries.
  - Number of articles found.
  - Saving of each URL (both to CSV and, if enabled, to Neo4j).
- Errors encountered during execution are logged with an error level for easier debugging.

## Output

- **CSV File:**  
  Output files are saved in the `output` directory with filenames in the format:
  ```
  saved_urls_<search_id>_<timestamp>.csv
  ```
  Each CSV file contains columns for `url`, `domain_name`, and `source`.

- **Neo4j Database:**  
  If enabled, each article URL is saved as a node in a Neo4j graph with relationships that map URLs to their respective domains.

## Dependencies

- **Standard Libraries:** `csv`, `argparse`, `logging`, `datetime`, `json`, `os`, `urllib.parse`
- **Third-Party Libraries:** 
  - `tldextract`: For extracting domain names.
  - `googlesearch`: For performing Google searches.
  - Custom: `libraries.neo4j_lib` for interfacing with Neo4j.

Ensure these dependencies are installed before running the script.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [Issues](#) or open a pull request.


## Developer

This script was developed by **Christo Strydom**. For further details or inquiries, please refer to the repository's contact information.

