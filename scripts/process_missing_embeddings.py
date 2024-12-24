import asyncio
import argparse
import logging
from app.embeddings import template_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_missing_embeddings(force_rescrape: bool = False, batch_size: int = None):
    """Process templates that don't have embeddings."""
    try:
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        await template_embeddings.initialize()
        
        # Get missing templates
        missing_configs = await template_embeddings.get_missing_templates()
        total_missing = len(missing_configs)
        
        if not total_missing:
            logger.info("No missing templates found")
            return
            
        logger.info(f"Found {total_missing} templates without embeddings")
        
        if batch_size:
            logger.info(f"Processing in batches of {batch_size}")
            for i in range(0, total_missing, batch_size):
                batch = missing_configs[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} templates)")
                
                success_count = 0
                for config in batch:
                    try:
                        # Scrape and update config if needed
                        config = await template_embeddings._scrape_and_update_config(config, force_rescrape)
                        
                        template_text = template_embeddings._create_template_text(config)
                        template_id = config['template_id']
                        
                        if await template_embeddings.add_template(template_id, config, template_text):
                            success_count += 1
                            logger.info(f"Successfully processed template {template_id} ({success_count}/{len(batch)})")
                            
                    except Exception as e:
                        logger.error(f"Error processing template {config.get('template_id')}: {e}")
                        
                logger.info(f"Batch {i//batch_size + 1} complete: {success_count}/{len(batch)} successful")
        else:
            # Process all at once
            success_count = await template_embeddings.process_missing_templates(force_rescrape)
            
        logger.info("Finished processing missing templates")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process templates without embeddings")
    parser.add_argument("--force-rescrape", action="store_true", help="Force rescraping of source URLs")
    parser.add_argument("--batch-size", type=int, help="Process templates in batches of this size")
    args = parser.parse_args()
    
    asyncio.run(process_missing_embeddings(args.force_rescrape, args.batch_size)) 