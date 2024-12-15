import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
import yaml
from sentence_transformers import SentenceTransformer

class TemplateEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Create embeddings directory if it doesn't exist
        embeddings_dir = Path(__file__).parent / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(embeddings_dir),
            anonymized_telemetry=False
        ))
        self.collection = self.chroma_client.get_or_create_collection(name="template_embeddings")
        
    def _load_template_configs(self) -> List[Dict[str, Any]]:
        """Load all template configs from the templates directory."""
        templates_dir = Path(__file__).parent.parent / "templates"
        configs = []
        
        for template_dir in templates_dir.iterdir():
            if template_dir.is_dir():
                config_path = template_dir / "config.yml"
                if config_path.exists():
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                        config['template_id'] = template_dir.name
                        configs.append(config)
        
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
        configs = self._load_template_configs()
        
        # Clear existing embeddings if any exist
        existing = self.collection.get()
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        
        # Create new embeddings
        for config in configs:
            template_text = self._create_template_text(config)
            template_id = config['template_id']
            
            # Generate embedding
            self.collection.add(
                documents=[template_text],
                ids=[template_id],
                metadatas=[{
                    'name': config.get('name', ''),
                    'template_id': template_id
                }]
            )
    
    async def search_templates(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for templates similar to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [
            {
                'template_id': id,
                'metadata': metadata,
                'distance': distance
            }
            for id, metadata, distance in zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]

# Global instance
template_embeddings = TemplateEmbeddings() 