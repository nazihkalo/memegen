import asyncio
import json
import yaml
from pathlib import Path
import sys
import logging
import argparse
from app.embeddings import scrape_webpage, template_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_all_templates(force_rescrape: bool = False):
    """Process all templates in the templates directory."""
    templates_dir = Path(__file__).parent.parent / "templates"
    if not templates_dir.exists():
        logger.error(f"Templates directory not found: {templates_dir}")
        return
        
    # Find all config files
    config_files = []
    for template_dir in templates_dir.iterdir():
        if template_dir.is_dir():
            config_path = template_dir / "config.yml"
            if config_path.exists():
                config_files.append(config_path)
                
    logger.info(f"Found {len(config_files)} template configs")
    
    # Process each template
    for config_path in config_files:
        logger.info(f"\nProcessing template: {config_path.parent.name}")
        await test_scrape_and_embed(str(config_path), force_rescrape)

async def test_scrape_and_embed(template_path: str, force_rescrape: bool = False):
    """Test scraping and embedding functionality for a template."""
    template_path = Path(template_path)
    if not template_path.exists():
        logger.error(f"Template path does not exist: {template_path}")
        return
        
    try:
        # Try reading as YAML first, then JSON if that fails
        with open(template_path) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError:
                # Reset file pointer and try JSON
                f.seek(0)
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"File is neither valid YAML nor JSON: {template_path}")
                    return
            
        if 'source' not in config or not config['source']:
            logger.error(f"No source URL found in template config: {template_path}")
            return
            
        # Only scrape if forced or no scraped content exists
        if force_rescrape or 'scraped_content' not in config:
            logger.info(f"Scraping source URL: {config['source']}")
            result = await scrape_webpage(config['source'])
            
            if result:
                content = result["text"]
                aside_content = result.get("aside_content", "")
                added_at = result.get("added_at")
                
                logger.info("Successfully scraped content:")
                print("\n" + "="*80)
                print("MAIN CONTENT:")
                print(content[:500] + "..." if len(content) > 500 else content)
                
                if aside_content:
                    print("\nASIDE CONTENT:")
                    print(aside_content[:500] + "..." if len(aside_content) > 500 else aside_content)
                    
                if added_at:
                    print(f"\nAdded at: {added_at}")
                print("="*80 + "\n")
                
                # Update config with scraped content and metadata
                config['scraped_content'] = content
                if aside_content:
                    config['aside_content'] = aside_content
                if added_at:
                    config['added_at'] = added_at
                
                # Save back to the original config file
                with open(template_path, 'w') as f:
                    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
                logger.info(f"Updated config file: {template_path}")
            else:
                logger.error("Failed to scrape content")
                return
        else:
            logger.info("Using existing scraped content")
            
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        await template_embeddings.initialize()
        
        # Check embeddings directory before adding
        embeddings_dir = template_embeddings.embeddings_dir
        logger.info(f"Embeddings directory: {embeddings_dir}")
        if embeddings_dir.exists():
            logger.info("Contents before adding template:")
            for f in embeddings_dir.glob('*'):
                logger.info(f"  - {f.name}")
        
        # Generate embedding for this template
        logger.info("Generating embedding for template...")
        template_text = template_embeddings._create_template_text(config)
        
        # Add template to collection
        template_id = config.get('template_id', template_path.stem)
        await template_embeddings.add_template(template_id, config, template_text)
        
        # Verify embedding persistence
        logger.info("Verifying embedding persistence...")
        
        # Check ChromaDB collection
        result = template_embeddings.collection.get(
            ids=[template_id],
            include=['embeddings', 'documents', 'metadatas']
        )
        
        if result and result['ids']:
            logger.info(f"Successfully verified embedding in ChromaDB for template: {template_id}")
            logger.info(f"Document text length: {len(result['documents'][0])}")
            logger.info(f"Embedding dimensions: {len(result['embeddings'][0])}")
            logger.info(f"Metadata: {result['metadatas'][0]}")
        else:
            logger.error(f"Failed to verify embedding in ChromaDB for template: {template_id}")
            
        # Check filesystem
        logger.info("\nChecking filesystem persistence...")
        if embeddings_dir.exists():
            logger.info("Contents after adding template:")
            for f in embeddings_dir.glob('*'):
                logger.info(f"  - {f.name} ({f.stat().st_size} bytes)")
        else:
            logger.error(f"Embeddings directory not found at: {embeddings_dir}")
            
    except Exception as e:
        logger.error(f"Error processing template: {str(e)}")
        raise  # Re-raise to see full traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test template scraping and embedding")
    parser.add_argument("template_path", nargs="?", help="Path to template config file")
    parser.add_argument("--force-rescrape", action="store_true", help="Force rescraping of source URLs")
    parser.add_argument("--all", action="store_true", help="Process all templates in templates directory")
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(process_all_templates(args.force_rescrape))
    elif args.template_path:
        asyncio.run(test_scrape_and_embed(args.template_path, args.force_rescrape))
    else:
        parser.print_help()
        sys.exit(1) 