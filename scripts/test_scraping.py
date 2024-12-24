import asyncio
import json
import yaml
from pathlib import Path
import sys
import logging
from app.embeddings import scrape_webpage, template_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            content = await scrape_webpage(config['source'])
            
            if content:
                logger.info("Successfully scraped content:")
                print("\n" + "="*80)
                print(content[:500] + "..." if len(content) > 500 else content)
                print("="*80 + "\n")
                
                # Update config with scraped content
                config['scraped_content'] = content
                
                # Save to a new file with _scraped suffix
                new_path = template_path.with_name(template_path.stem + "_scraped.yml")
                with open(new_path, 'w') as f:
                    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
                logger.info(f"Saved updated config to: {new_path}")
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
    if len(sys.argv) != 2:
        print("Usage: python test_scraping.py <path_to_template_config.yml>")
        sys.exit(1)
        
    asyncio.run(test_scrape_and_embed(sys.argv[1])) 