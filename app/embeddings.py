import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

import chromadb
from chromadb.config import Settings
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TemplateEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Create embeddings directory if it doesn't exist
        embeddings_dir = Path(__file__).parent / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using embeddings directory: {embeddings_dir}")
        
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(embeddings_dir),
            anonymized_telemetry=False
        ))
        self.collection = self.chroma_client.get_or_create_collection(name="template_embeddings")
        
        # Check if collection is empty and initialize if needed
        if not self._has_embeddings():
            logger.info("No embeddings found, initializing...")
            self.update_embeddings()
            
    def _has_embeddings(self) -> bool:
        """Check if embeddings exist in the collection."""
        try:
            result = self.collection.get()
            has_embeddings = bool(result and result.get("ids"))
            logger.info(f"Embeddings check: {'found' if has_embeddings else 'not found'}")
            if has_embeddings:
                logger.debug(f"Found templates: {result.get('ids', [])}")
            return has_embeddings
        except Exception as e:
            logger.error(f"Error checking embeddings: {e}")
            return False
            
    def list_all_templates(self) -> List[str]:
        """List all available template IDs."""
        try:
            result = self.collection.get()
            templates = result.get("ids", [])
            logger.info(f"Found {len(templates)} templates in collection")
            logger.debug(f"Available templates: {templates}")
            return templates
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
            
    def _load_template_configs(self) -> List[Dict[str, Any]]:
        """Load all template configs from the templates directory."""
        templates_dir = Path(__file__).parent.parent / "templates"
        logger.info(f"Loading templates from: {templates_dir}")
        configs = []
        
        for template_dir in templates_dir.iterdir():
            if template_dir.is_dir():
                config_path = template_dir / "config.yml"
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                            config['template_id'] = template_dir.name
                            # Count text zones from config
                            if 'text' in config:
                                config['text_zones'] = len(config['text'])
                            else:
                                config['text_zones'] = 2  # Default fallback
                            configs.append(config)
                            logger.debug(f"Loaded template config: {template_dir.name}")
                    except Exception as e:
                        logger.error(f"Error loading template {template_dir.name}: {e}")
        
        logger.info(f"Loaded {len(configs)} template configs")
        return configs
    
    def _create_template_text(self, config: Dict[str, Any]) -> str:
        """Create searchable text from template config."""
        parts = []
        
        if 'name' in config:
            parts.append(f"Name: {config['name']}")
            
        if 'keywords' in config:
            keywords = config['keywords']
            if keywords and isinstance(keywords, list):
                # Filter out None values and convert all items to strings
                valid_keywords = [str(k) for k in keywords if k is not None]
                if valid_keywords:
                    parts.append(f"Keywords: {', '.join(valid_keywords)}")
                
        if 'source' in config:
            parts.append(f"Source: {config['source']}")
            
        if 'example' in config:
            examples = config['example']
            if isinstance(examples, list):
                # Filter out None values from examples too
                valid_examples = [str(e) for e in examples if e is not None]
                if valid_examples:
                    parts.append(f"Example: {' '.join(valid_examples)}")
            elif examples is not None:
                parts.append(f"Example: {examples}")
                
        return " ".join(parts)
    
    def update_embeddings(self):
        """Update the embeddings database with current templates."""
        logger.info("Starting embeddings update")
        configs = self._load_template_configs()
        
        # Clear existing embeddings if any exist
        existing = self.collection.get()
        if existing and existing["ids"]:
            logger.info(f"Clearing {len(existing['ids'])} existing embeddings")
            self.collection.delete(ids=existing["ids"])
        
        # Create new embeddings
        for config in configs:
            try:
                template_text = self._create_template_text(config)
                template_id = config['template_id']
                logger.debug(f"Processing template: {template_id}")
                
                # Store metadata with serialized config
                metadata = {
                    'name': config.get('name', ''),
                    'template_id': template_id,
                    'text_zones': str(config['text_zones']),  # Convert to string
                    'config_json': json.dumps(config)  # Serialize config to JSON string
                }
                
                # Generate embedding
                self.collection.add(
                    documents=[template_text],
                    ids=[template_id],
                    metadatas=[metadata]
                )
                logger.debug(f"Added embedding for template: {template_id}")
            except Exception as e:
                logger.error(f"Error adding template {config.get('template_id')}: {e}")
        
        logger.info("Finished updating embeddings")
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template info directly by ID without semantic search."""
        logger.info(f"Looking up template: {template_id}")
        try:
            result = self.collection.get(
                ids=[template_id],
                include=['metadatas']
            )
            
            if not result["ids"]:
                logger.warning(f"Template not found: {template_id}")
                return None
                
            # Return the full template info from metadata
            metadata = result["metadatas"][0]
            config = json.loads(metadata.get('config_json', '{}'))
            text_zones = int(metadata.get('text_zones', 2))
            
            logger.debug(f"Found template {template_id} with {text_zones} text zones")
            return {
                'template_id': template_id,
                'metadata': {
                    'name': metadata.get('name', ''),
                    'template_id': metadata.get('template_id', ''),
                    'text_zones': text_zones
                },
                'config': config,
                'text_zones': text_zones
            }
        except Exception as e:
            logger.error(f"Error getting template {template_id}: {e}")
            return None
    
    async def search_templates(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for templates similar to the query."""
        logger.info(f"Searching templates with query: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        templates = [
            {
                'template_id': id,
                'metadata': {
                    'name': metadata.get('name', ''),
                    'template_id': metadata.get('template_id', ''),
                    'text_zones': int(metadata.get('text_zones', 2))
                },
                'config': json.loads(metadata.get('config_json', '{}')),
                'text_zones': int(metadata.get('text_zones', 2)),
                'distance': distance
            }
            for id, metadata, distance in zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
        
        logger.info(f"Found {len(templates)} matching templates")
        return templates

# Global instance
template_embeddings = TemplateEmbeddings() 