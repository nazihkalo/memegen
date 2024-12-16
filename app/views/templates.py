import asyncio
from pathlib import Path
import logging
import io

from sanic import Blueprint, exceptions, response
from sanic.request import Request
from sanic_ext import openapi

from .. import helpers, utils
from ..models import Template
from .helpers import generate_url
from .schemas import CustomRequest, MemeResponse, MemeTemplateRequest, TemplateResponse

logger = logging.getLogger(__name__)
blueprint = Blueprint("Templates", url_prefix="/templates")


@blueprint.get("/")
@openapi.summary("List all templates")
@openapi.parameter(
    "animated",
    bool,
    "query",
    description="Limit results to templates supporting animation",
)
@openapi.parameter(
    "filter", str, "query", description="Part of the name, keyword, or example to match"
)
@openapi.response(
    200,
    {"application/json": list[TemplateResponse]},
    "Successfully returned a list of all templates",
)
async def index(request: Request):
    query = request.args.get("filter", "").lower()
    animated = utils.urls.flag(request, "animated")
    data = await asyncio.to_thread(
        helpers.get_valid_templates, request, query, animated
    )
    return response.json(data)


@blueprint.get("/<id:slug>")
@openapi.summary("View a specific template")
@openapi.parameter("id", str, "path", description="ID of a meme template")
@openapi.response(
    200,
    {"application/json": TemplateResponse},
    "Successfully returned a specific template",
)
@openapi.response(404, str, description="Template not found")
async def detail(request, id):
    template: Template = Template.objects.get_or_none(id)
    if template:
        return response.json(template.jsonify(request))
    raise exceptions.NotFound(f"Template not found: {id}")


@blueprint.post("/<id:slug>")
@openapi.summary("Create a meme from a template")
@openapi.parameter("id", str, "path", description="ID of a meme template")
@openapi.body({"application/json": MemeTemplateRequest})
@openapi.response(
    201,
    {"image/png": bytes},
    "Successfully created a meme image",
)
async def build(request, id):
    template: Template = Template.objects.get_or_none(id)
    if not template:
        raise exceptions.NotFound(f"Template not found: {id}")
        
    # Get text lines from request
    data = request.json or {}
    text_lines = data.get("text", [])
    
    logger.info(f"Generating meme for template {id} with text lines: {text_lines}")
    logger.debug(f"Template details - id: {template.id}, name: {template.name}, text zones: {len(template.text)}")
    logger.debug(f"Request data: {data}")
    
    try:
        # Generate the meme image using the image utils
        image = await asyncio.to_thread(
            utils.images.render_image,
            template,
            data.get("style", "default"),
            text_lines,
            (0, 0),  # Use default size
            data.get("font", ""),
            watermark=""
        )
        
        if not image:
            logger.error("Failed to generate image")
            raise exceptions.ServerError("Failed to generate image")
            
        logger.info("Successfully generated meme image")
        
        # Convert PIL Image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        
        return response.raw(
            image_bytes.getvalue(),
            content_type="image/png",
            status=201
        )
            
    except Exception as e:
        logger.error(f"Failed to generate meme: {str(e)}", exc_info=True)
        raise exceptions.ServerError(f"Failed to generate meme: {str(e)}")


@blueprint.post("/custom")
@openapi.summary("Create a meme from any image")
@openapi.body({"application/json": CustomRequest})
@openapi.response(
    201,
    {"application/json": MemeResponse},
    "Successfully created a meme from a custom image",
)
async def custom(request: Request):
    return await generate_url(request)
