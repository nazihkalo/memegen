import asyncio
import argparse
import logging
from pathlib import Path
import yaml
from app.embeddings import template_embeddings, date_to_timestamp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_updated_metadata():
    """Update metadata for existing embeddings with latest config values."""
    try:
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        await template_embeddings.initialize()
        
        # Get all templates from ChromaDB
        existing_templates = template_embeddings.list_all_templates()
        total_templates = len(existing_templates)
        
        if not total_templates:
            logger.info("No existing templates found in ChromaDB")
            return
            
        logger.info(f"Found {total_templates} templates in ChromaDB")
        
        # Get templates directory
        templates_dir = Path(__file__).parent.parent / "templates"
        if not templates_dir.exists():
            logger.error(f"Templates directory not found: {templates_dir}")
            return
            
        success_count = 0
        for template_id in existing_templates:
            try:
                # Read current config
                config_path = templates_dir / template_id / "config.yml"
                if not config_path.exists():
                    logger.warning(f"Config file not found for template {template_id}")
                    continue
                    
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                # Get added_at date and convert to timestamp
                added_at = config.get('added_at', '')
                added_at_timestamp = date_to_timestamp(added_at) if added_at else 0  # Use 0 instead of None
                
                if added_at and not added_at_timestamp:
                    logger.warning(f"Failed to convert date {added_at} to timestamp for template {template_id}")
                
                # Create updated metadata
                metadata = {
                    'name': config.get('name', ''),
                    'template_id': template_id,
                    'text_zones': str(config.get('text_zones', 2)),
                    'added_at': added_at or '',  # Use empty string instead of None
                    'added_at_ts': added_at_timestamp,  # Will be 0 if no date
                    'config_json': yaml.dump(config)
                }
                
                # Get existing document and embedding
                result = template_embeddings.collection.get(
                    ids=[template_id],
                    include=['embeddings', 'documents']
                )
                
                if not result['ids']:
                    logger.warning(f"No embedding found for template {template_id}")
                    continue
                
                # Update with existing embedding and document but new metadata
                template_embeddings.collection.update(
                    ids=[template_id],
                    embeddings=[result['embeddings'][0]],
                    documents=[result['documents'][0]],
                    metadatas=[metadata]
                )
                
                success_count += 1
                logger.info(f"Successfully updated metadata for template {template_id} ({success_count}/{total_templates})")
                
            except Exception as e:
                logger.error(f"Error updating metadata for template {template_id}: {e}")
        
        logger.info(f"Finished updating metadata: {success_count}/{total_templates} successful")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

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
    parser = argparse.ArgumentParser(description="Process templates without embeddings or update metadata")
    parser.add_argument("--force-rescrape", action="store_true", help="Force rescraping of source URLs")
    parser.add_argument("--batch-size", type=int, help="Process templates in batches of this size")
    parser.add_argument("--update-metadata", action="store_true", help="Update metadata for existing embeddings")
    args = parser.parse_args()
    
    if args.update_metadata:
        asyncio.run(process_updated_metadata())
    else:
        asyncio.run(process_missing_embeddings(args.force_rescrape, args.batch_size)) 