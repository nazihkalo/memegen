import asyncio

from sanic import Sanic, response
from sanic.request import Request
from sanic_ext import openapi

from app import config, helpers, settings, utils
from app.embeddings import template_embeddings

app = Sanic(name="memegen")
config.init(app)


@app.get("/")
@openapi.exclude(True)
def index(request: Request):
    return response.redirect("/docs")


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
        
    results = await template_embeddings.search_templates(query, n_results)
    return response.json({"results": results})


@app.before_server_start
async def setup_embeddings(app, loop):
    """Initialize embeddings before server starts."""
    template_embeddings.update_embeddings()


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
