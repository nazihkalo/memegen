import requests
import json
from urllib.parse import quote
import argparse
from typing import Optional, List, Dict, Any
import logging
from IPython.display import Image, display, HTML
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_templates_notebook(
    query: str, 
    n_results: Optional[int] = 5, 
    base_url: str = "http://localhost:5001",
    display_images: bool = True
) -> List[Dict[str, Any]]:
    """Search for meme templates and optionally display results in a Jupyter notebook.
    
    Args:
        query: Search query text
        n_results: Number of results to return
        base_url: Base URL of the API
        display_images: Whether to display images in the notebook
        
    Returns:
        List of template results with metadata
    """
    # URL encode the query
    encoded_query = quote(query)
    
    # Construct URL with parameters
    url = f"{base_url}/search?q={encoded_query}"
    if n_results:
        url += f"&n={n_results}"
        
    try:
        # Make the request
        logger.info(f"Searching for: {query}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse results
        results = response.json()["results"]
        logger.info(f"Found {len(results)} results")
        
        if display_images:
            # Display results with images in notebook
            for result in results:
                try:
                    template_id = result['template_id']
                    name = result['metadata'].get('name', 'Unnamed Template')
                    similarity = 1 - result.get('distance', 0)
                    text_zones = result.get('text_zones', 0)
                    
                    # Create HTML for template info
                    info_html = f"""
                    <div style="margin-bottom: 20px">
                        <h3>{name} (ID: {template_id})</h3>
                        <p>Similarity score: {similarity:.3f}</p>
                        <p>Text zones: {text_zones}</p>
                    """
                    
                    # Handle keywords safely
                    keywords = result.get('config', {}).get('keywords', [])
                    if keywords:
                        # Filter out None values and convert to strings
                        valid_keywords = [str(k) for k in keywords if k is not None]
                        if valid_keywords:
                            info_html += f"<p>Keywords: {', '.join(valid_keywords)}</p>"
                    
                    # Handle source URL safely
                    source = result.get('config', {}).get('source')
                    if source:
                        info_html += f"<p>Source: <a href='{source}'>{source}</a></p>"
                    
                    info_html += "</div>"
                    
                    # Display template info
                    display(HTML(info_html))
                    
                    # Display template image
                    try:
                        image_url = f"{base_url}/templates/{template_id}/preview"
                        img_response = requests.get(image_url)
                        img_response.raise_for_status()
                        display(Image(img_response.content))
                    except Exception as e:
                        logger.error(f"Error displaying image for template {template_id}: {e}")
                        display(HTML(f"<p style='color: red'>Error loading image for template {template_id}</p>"))
                    
                    display(HTML("<hr style='margin: 20px 0'>"))
                except Exception as e:
                    logger.error(f"Error displaying result: {e}")
                    continue
        
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request: {e}")
        return []
    except KeyError as e:
        logger.error(f"Error parsing response: {e}")
        logger.debug(f"Response content: {response.text}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []

def search_templates(query: str, n_results: Optional[int] = 5, base_url: str = "http://localhost:5001") -> None:
    """Command-line version of template search."""
    results = search_templates_notebook(query, n_results, base_url, display_images=False)
    
    if results:
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            try:
                name = result['metadata'].get('name', 'Unnamed Template')
                template_id = result['template_id']
                similarity = 1 - result.get('distance', 0)
                text_zones = result.get('text_zones', 0)
                
                print(f"\n{i}. {name} (ID: {template_id})")
                print(f"   Similarity score: {similarity:.3f}")
                print(f"   Text zones: {text_zones}")
                
                # Handle keywords safely
                keywords = result.get('config', {}).get('keywords', [])
                if keywords:
                    valid_keywords = [str(k) for k in keywords if k is not None]
                    if valid_keywords:
                        print(f"   Keywords: {', '.join(valid_keywords)}")
                
                # Handle source URL safely
                source = result.get('config', {}).get('source')
                if source:
                    print(f"   Source: {source}")
            except Exception as e:
                logger.error(f"Error displaying result {i}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for meme templates")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--n", type=int, default=5, help="Number of results to return")
    parser.add_argument("--url", default="http://localhost:5001", help="Base URL of the API")
    args = parser.parse_args()
    
    search_templates(args.query, args.n, args.url) 