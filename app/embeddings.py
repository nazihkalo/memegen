import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import httpx
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

async def scrape_webpage(url: str) -> Optional[str]:
    """Scrape text content from a webpage."""
    try:
        # Configure client with redirect following and longer timeout
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
                
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {str(e)}")
        return None

def truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncate text to approximately fit within token limits.
    Using rough estimate of 4 chars per token, 6000 chars ≈ 1500 tokens,
    leaving plenty of room under the 8192 token limit."""
    if len(text) <= max_chars:
        return text
    
    # Try to truncate at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:  # Only use period if it's not too far back
        return truncated[:last_period + 1]
    return truncated

def get_embedding(text: str, model="text-embedding-3-large") -> Optional[List[float]]:
    """Get embedding from OpenAI API."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        # Truncate text to fit within token limits
        truncated_text = truncate_text(text)
        if len(truncated_text) < len(text):
            logger.info(f"Truncated text from {len(text)} to {len(truncated_text)} characters")
            
        response = client.embeddings.create(
            model=model,
            input=truncated_text,
            dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

class TemplateEmbeddings:
    def __init__(self):
        # Create embeddings directory with absolute path
        self.embeddings_dir = Path(__file__).parent.resolve() / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using embeddings directory: {self.embeddings_dir}")
        
        # Configure ChromaDB with explicit settings
        settings = Settings(
            persist_directory=str(self.embeddings_dir),
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.embeddings_dir),
            settings=settings
        )
        
        # Create or get collection with explicit settings
        self.collection = self.chroma_client.get_or_create_collection(
            name="template_embeddings",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for OpenAI embeddings
        )
        
        # Note: initialization happens in initialize() method
        self._initialized = False
            
    def _has_embeddings(self) -> bool:
        """Check if embeddings exist in the collection."""
        try:
            result = self.collection.get()
            has_embeddings = bool(result and result.get("ids"))
            logger.info(f"Embeddings check: {'found' if has_embeddings else 'not found'}")
            if has_embeddings:
                logger.debug(f"Found templates: {result.get('ids', [])}")
                logger.debug(f"Embeddings directory contents: {list(self.embeddings_dir.glob('*'))}")
            return has_embeddings
        except Exception as e:
            logger.error(f"Error checking embeddings: {e}")
            return False
            
    async def initialize(self):
        """Initialize embeddings if needed."""
        if self._initialized:
            return
            
        if not self._has_embeddings():
            logger.info("No embeddings found, initializing...")
            await self.update_embeddings()
            
        self._initialized = True
            
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

    async def _scrape_and_update_config(self, config: Dict[str, Any], force_rescrape: bool = False) -> Dict[str, Any]:
        """Scrape webpage content and update config if needed."""
        if 'source' not in config or not config['source']:
            return config
            
        # Skip if we already have scraped content and not forcing rescrape
        if not force_rescrape and 'scraped_content' in config and config['scraped_content']:
            return config
            
        content = await scrape_webpage(config['source'])
        if content:
            config['scraped_content'] = content
            
            # Save updated config back to file if template_id is available
            if 'template_id' in config:
                templates_dir = Path(__file__).parent.parent / "templates"
                config_path = templates_dir / config['template_id'] / "config.yml"
                if config_path.exists():
                    with open(config_path, 'w') as f:
                        yaml.safe_dump(config, f)
                    
        return config
            
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
                valid_keywords = [str(k) for k in keywords if k is not None]
                if valid_keywords:
                    parts.append(f"Keywords: {', '.join(valid_keywords)}")
                
        if 'source' in config:
            parts.append(f"Source: {config['source']}")
            
        if 'example' in config:
            examples = config['example']
            if isinstance(examples, list):
                valid_examples = [str(e) for e in examples if e is not None]
                if valid_examples:
                    parts.append(f"Example: {' '.join(valid_examples)}")
            elif examples is not None:
                parts.append(f"Example: {examples}")
                
        if 'scraped_content' in config and config['scraped_content']:
            parts.append(f"Description: {config['scraped_content']}")
                
        return " ".join(parts)
        
    async def add_template(self, template_id: str, config: Dict[str, Any], template_text: str) -> bool:
        """Add a single template to the collection."""
        try:
            # Get embedding from OpenAI
            embedding = get_embedding(template_text)
            if not embedding:
                logger.error(f"Failed to get embedding for template: {template_id}")
                return False
            
            # Store metadata with serialized config
            metadata = {
                'name': config.get('name', ''),
                'template_id': template_id,
                'text_zones': str(config.get('text_zones', 2)),
                'config_json': json.dumps(config)
            }
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[template_text],
                ids=[template_id],
                metadatas=[metadata]
            )
            
            # Force persistence
            if hasattr(self.collection, 'persist'):
                self.collection.persist()
            elif hasattr(self.chroma_client, 'persist'):
                self.chroma_client.persist()
                
            logger.info(f"Added and persisted embedding for template: {template_id}")
            logger.debug(f"Embeddings directory contents: {list(self.embeddings_dir.glob('*'))}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template {template_id}: {e}")
            return False
            
    async def update_embeddings(self, force_rescrape: bool = False):
        """Update the embeddings database with current templates."""
        logger.info("Starting embeddings update")
        configs = self._load_template_configs()
        
        # Clear existing embeddings if any exist
        existing = self.collection.get()
        if existing and existing["ids"]:
            logger.info(f"Clearing {len(existing['ids'])} existing embeddings")
            self.collection.delete(ids=existing["ids"])
            
        # Create new embeddings
        success_count = 0
        for config in configs:
            try:
                # Scrape and update config if needed
                config = await self._scrape_and_update_config(config, force_rescrape)
                
                template_text = self._create_template_text(config)
                template_id = config['template_id']
                
                if await self.add_template(template_id, config, template_text):
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error adding template {config.get('template_id')}: {e}")
                
        # Force final persistence
        if hasattr(self.collection, 'persist'):
            self.collection.persist()
        elif hasattr(self.chroma_client, 'persist'):
            self.chroma_client.persist()
            
        logger.info(f"Finished updating embeddings. Successfully added {success_count}/{len(configs)} templates")
        logger.debug(f"Final embeddings directory contents: {list(self.embeddings_dir.glob('*'))}")
    
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
        
        # Get embedding for query
        query_embedding = get_embedding(query)
        if not query_embedding:
            logger.error("Failed to get embedding for query")
            return []
            
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
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

    async def get_missing_templates(self) -> List[Dict[str, Any]]:
        """Get template configs that don't have embeddings in ChromaDB."""
        # Get all template configs
        all_configs = self._load_template_configs()
        all_template_ids = {config['template_id'] for config in all_configs}
        
        # Get existing template IDs from ChromaDB
        existing_ids = set(self.list_all_templates())
        
        # Find missing template IDs
        missing_ids = all_template_ids - existing_ids
        logger.info(f"Found {len(missing_ids)} templates without embeddings")
        
        # Return configs for missing templates
        missing_configs = [
            config for config in all_configs 
            if config['template_id'] in missing_ids
        ]
        return missing_configs
        
    async def process_missing_templates(self, force_rescrape: bool = False) -> int:
        """Process only templates that don't have embeddings yet.
        Returns the number of successfully processed templates."""
        logger.info("Checking for templates without embeddings...")
        missing_configs = await self.get_missing_templates()
        
        if not missing_configs:
            logger.info("No missing templates found")
            return 0
            
        logger.info(f"Processing {len(missing_configs)} missing templates...")
        success_count = 0
        
        for config in missing_configs:
            try:
                # Scrape and update config if needed
                config = await self._scrape_and_update_config(config, force_rescrape)
                
                template_text = self._create_template_text(config)
                template_id = config['template_id']
                
                if await self.add_template(template_id, config, template_text):
                    success_count += 1
                    logger.info(f"Successfully processed template {template_id} ({success_count}/{len(missing_configs)})")
                    
            except Exception as e:
                logger.error(f"Error processing template {config.get('template_id')}: {e}")
                
        logger.info(f"Finished processing missing templates. Successfully added {success_count}/{len(missing_configs)}")
        return success_count

# Global instance
template_embeddings = TemplateEmbeddings() 