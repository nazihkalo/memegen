import asyncio
import logging

from sanic import Sanic, response
from sanic.request import Request
from sanic_ext import openapi

from app import config, helpers, settings, utils
from app.embeddings import template_embeddings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Sanic(name="memegen")
config.init(app)


@app.get("/")
@openapi.exclude(True)
def index(request: Request):
    return response.redirect("/docs")


@app.get("/templates/list")
@openapi.summary("List all available templates")
@openapi.description("Get a list of all template IDs in the system")
async def list_templates(request: Request):
    """List all available templates."""
    try:
        templates = template_embeddings.list_all_templates()
        return response.json({
            "templates": templates,
            "count": len(templates)
        })
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return response.json({"error": str(e)}, status=500)


@app.get("/templates/<template_id:str>/info")
@openapi.summary("Get template information")
@openapi.description("Get detailed information about a specific template")
@openapi.parameter("template_id", str, "Template ID", required=True)
async def get_template_info(request: Request, template_id: str):
    """Get information about a specific template."""
    try:
        logger.info(f"Getting info for template: {template_id}")
        
        # Debug: Check if embeddings are initialized
        logger.debug("Checking embeddings collection state...")
        collection_state = template_embeddings.collection.get()
        logger.debug(f"Collection has {len(collection_state.get('ids', []))} templates")
        
        # Get template info using direct lookup
        template_info = template_embeddings.get_template(template_id)
        
        if not template_info:
            logger.warning(f"Template not found: {template_id}")
            # Debug: List available templates
            available = template_embeddings.collection.get()
            logger.debug(f"Available templates: {available.get('ids', [])}")
            return response.json({"error": f"Template {template_id} not found"}, status=404)
            
        logger.debug(f"Returning template info: {template_info}")
        return response.json(template_info)
        
    except Exception as e:
        logger.error(f"Error getting template info: {e}", exc_info=True)
        return response.json({"error": str(e)}, status=500)


@app.get("/test")
@openapi.exclude(True)
async def test(request: Request):
    if not settings.DEBUG:
        return response.redirect("/")

    urls = await asyncio.to_thread(helpers.get_test_images, request)
    content = utils.html.gallery(urls, columns=False, refresh=20)
    return response.html(content)


@app.get("/favicon.ico")
@openapi.exclude(True)
async def favicon(request: Request):
    return await response.file("app/static/favicon.ico")


@app.get("/robots.txt")
@openapi.exclude(True)
async def robots(request: Request):
    return await response.file("app/static/robots.txt")


@app.get("/search")
@openapi.summary("Search for templates using natural language")
@openapi.description("Search for meme templates using natural language queries")
@openapi.parameter("q", str, "Search query", required=True)
@openapi.parameter("n", int, "Number of results to return", required=False)
async def search_templates(request: Request):
    query = request.args.get("q")
    n_results = int(request.args.get("n", 5))
    
    if not query:
        return response.json({"error": "Missing query parameter 'q'"}, status=400)
        
    logger.info(f"Searching templates with query: {query}")
    results = await template_embeddings.search_templates(query, n_results)
    logger.debug(f"Search returned {len(results)} results")
    return response.json({"results": results})


@app.before_server_start
async def setup_embeddings(app, loop):
    """Initialize embeddings before server starts."""
    logger.info("Checking embeddings initialization")
    try:
        # Force update if no embeddings exist
        if not template_embeddings._has_embeddings():
            logger.info("No embeddings found, initializing...")
            template_embeddings.update_embeddings()
        else:
            logger.info("Embeddings already initialized")
    except Exception as e:
        logger.error(f"Error during embeddings initialization: {e}")
        raise  # Re-raise to prevent server start if initialization fails


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=settings.DEBUG,
        auto_reload=True,
        access_log=False,
        motd=False,
        fast=not settings.DEBUG,
        workers=1  # Use single worker to avoid multiprocessing issues
    )
