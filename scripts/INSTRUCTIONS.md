# Scripts Directory Instructions

This directory contains utility scripts for managing and testing the memegen service.

## test_scraping.py

Tests the scraping and embedding functionality for meme templates.

### Purpose
- Scrapes content from source URLs specified in template configs
- Generates embeddings for template search functionality
- Verifies embedding persistence in ChromaDB
- Helps debug template processing pipeline

### Usage
```bash
python test_scraping.py <path_to_template_config.yml>
```

### Example
```bash
python test_scraping.py ../templates/drake.yml
```

### Features
- Supports both YAML and JSON template configs
- Force rescraping with `force_rescrape` flag
- Saves scraped content to a new file with `_scraped` suffix
- Verifies embeddings in both ChromaDB and filesystem
- Detailed logging for debugging

### Output
- Creates embeddings in the configured ChromaDB collection
- Saves scraped content to a new YAML file
- Logs detailed information about the scraping and embedding process

### Requirements
- ChromaDB setup and running
- OpenAI API key configured (for embeddings)
- Template config file with a valid 'source' URL

## Other Scripts

### check_db.ipynb
Jupyter notebook for inspecting and debugging the ChromaDB database.

### process_missing_embeddings.py
Processes templates that are missing embeddings in the database.

### search_templates.py
CLI tool for testing template search functionality.

### check_deployment.py
Verifies deployment status and health of the service.

### simulate_load.py
Simulates load on the service for performance testing. 