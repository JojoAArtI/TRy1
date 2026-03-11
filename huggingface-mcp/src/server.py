"""
HuggingFace MCP Server
Gives Claude the ability to search models, datasets, pull model cards,
compare benchmarks, and explore the HuggingFace Hub directly.
"""

import json
import httpx
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("huggingface_mcp")

HF_API_BASE = "https://huggingface.co/api"
HF_HUB_BASE = "https://huggingface.co"


# ── Shared HTTP client ────────────────────────────────────────────────────────

async def hf_get(path: str, params: dict = None) -> dict:
    """Make a GET request to the HuggingFace API."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API_BASE}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()


def _handle_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 404:
            return "Error: Resource not found on HuggingFace Hub."
        if e.response.status_code == 429:
            return "Error: Rate limit hit. Try again shortly."
        return f"Error: HuggingFace API returned status {e.response.status_code}."
    return f"Error: {type(e).__name__}: {e}"


# ── Input models ──────────────────────────────────────────────────────────────

class SearchModelsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Search query, e.g. 'bert sentiment classification'", min_length=1)
    task: Optional[str] = Field(default=None, description="Filter by task, e.g. 'text-classification', 'text-generation'")
    library: Optional[str] = Field(default=None, description="Filter by library, e.g. 'transformers', 'diffusers'")
    limit: int = Field(default=10, description="Number of results to return", ge=1, le=50)
    sort: str = Field(default="downloads", description="Sort by: 'downloads', 'likes', 'lastModified'")


class ModelCardInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_id: str = Field(..., description="Full model ID, e.g. 'bert-base-uncased' or 'google/flan-t5-base'", min_length=1)


class SearchDatasetsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Search query for datasets", min_length=1)
    task: Optional[str] = Field(default=None, description="Filter by task category")
    limit: int = Field(default=10, ge=1, le=50)


class CompareModelsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_ids: List[str] = Field(..., description="List of model IDs to compare, e.g. ['bert-base-uncased', 'roberta-base']", min_items=2, max_items=5)


class TrendingInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    resource_type: str = Field(default="models", description="What to fetch trending items for: 'models' or 'datasets'")
    limit: int = Field(default=10, ge=1, le=30)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="hf_search_models",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def hf_search_models(params: SearchModelsInput) -> str:
    """Search HuggingFace Hub for models by query, task, or library.

    Returns a ranked list of models with download counts, likes, tags,
    and direct Hub URLs. Useful for finding the best model for a task.

    Args:
        params: SearchModelsInput with query, optional task/library filter, limit, sort

    Returns:
        str: JSON list of matching models with metadata
    """
    try:
        api_params = {
            "search": params.query,
            "limit": params.limit,
            "sort": params.sort,
            "direction": -1,
            "full": True,
        }
        if params.task:
            api_params["pipeline_tag"] = params.task
        if params.library:
            api_params["library"] = params.library

        data = await hf_get("/models", api_params)

        results = [
            {
                "model_id": m.get("modelId", m.get("id")),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "task": m.get("pipeline_tag"),
                "tags": m.get("tags", [])[:8],
                "last_modified": m.get("lastModified"),
                "url": f"{HF_HUB_BASE}/{m.get('modelId', m.get('id'))}",
            }
            for m in data
        ]
        return json.dumps({"query": params.query, "count": len(results), "models": results}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hf_get_model_card",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def hf_get_model_card(params: ModelCardInput) -> str:
    """Fetch detailed metadata and model card info for a specific HuggingFace model.

    Returns architecture details, training data, evaluation results,
    intended uses, limitations, and license information.

    Args:
        params: ModelCardInput with model_id (e.g. 'google/flan-t5-base')

    Returns:
        str: JSON with model metadata and card summary
    """
    try:
        data = await hf_get(f"/models/{params.model_id}")

        card = {
            "model_id": data.get("modelId", data.get("id")),
            "author": data.get("author"),
            "task": data.get("pipeline_tag"),
            "downloads_last_month": data.get("downloads"),
            "likes": data.get("likes"),
            "tags": data.get("tags", []),
            "license": next((t for t in data.get("tags", []) if t.startswith("license:")), None),
            "languages": [t.replace("language:", "") for t in data.get("tags", []) if t.startswith("language:")],
            "datasets_used": [t.replace("dataset:", "") for t in data.get("tags", []) if t.startswith("dataset:")],
            "library": data.get("library_name"),
            "last_modified": data.get("lastModified"),
            "private": data.get("private", False),
            "siblings": [f["rfilename"] for f in data.get("siblings", [])],
            "url": f"{HF_HUB_BASE}/{data.get('modelId', data.get('id'))}",
            "card_data": data.get("cardData", {}),
        }
        return json.dumps(card, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hf_search_datasets",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def hf_search_datasets(params: SearchDatasetsInput) -> str:
    """Search HuggingFace Hub for datasets by query or task.

    Returns dataset names, download stats, task categories, and Hub URLs.

    Args:
        params: SearchDatasetsInput with query, optional task, limit

    Returns:
        str: JSON list of matching datasets
    """
    try:
        api_params = {"search": params.query, "limit": params.limit, "sort": "downloads", "direction": -1}
        if params.task:
            api_params["task_categories"] = params.task

        data = await hf_get("/datasets", api_params)

        results = [
            {
                "dataset_id": d.get("id"),
                "downloads": d.get("downloads", 0),
                "likes": d.get("likes", 0),
                "tags": d.get("tags", [])[:6],
                "task_categories": d.get("taskCategories", []),
                "url": f"{HF_HUB_BASE}/datasets/{d.get('id')}",
            }
            for d in data
        ]
        return json.dumps({"query": params.query, "count": len(results), "datasets": results}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hf_compare_models",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def hf_compare_models(params: CompareModelsInput) -> str:
    """Compare multiple HuggingFace models side-by-side.

    Fetches metadata for each model and returns a structured comparison
    of downloads, likes, tasks, tags, license, and library.

    Args:
        params: CompareModelsInput with list of 2–5 model IDs

    Returns:
        str: JSON comparison table of the requested models
    """
    results = []
    for model_id in params.model_ids:
        try:
            data = await hf_get(f"/models/{model_id}")
            results.append({
                "model_id": model_id,
                "task": data.get("pipeline_tag"),
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "library": data.get("library_name"),
                "license": next((t for t in data.get("tags", []) if t.startswith("license:")), "unknown"),
                "last_modified": data.get("lastModified"),
                "url": f"{HF_HUB_BASE}/{model_id}",
            })
        except Exception as e:
            results.append({"model_id": model_id, "error": _handle_error(e)})

    return json.dumps({"comparison": results}, indent=2)


@mcp.tool(
    name="hf_trending",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def hf_trending(params: TrendingInput) -> str:
    """Get currently trending models or datasets on HuggingFace Hub.

    Args:
        params: TrendingInput with resource_type ('models' or 'datasets') and limit

    Returns:
        str: JSON list of trending items with stats
    """
    try:
        path = "/models" if params.resource_type == "models" else "/datasets"
        data = await hf_get(path, {"sort": "trendingScore", "direction": -1, "limit": params.limit, "full": True})

        key = "model_id" if params.resource_type == "models" else "dataset_id"
        results = [
            {
                key: item.get("modelId", item.get("id")),
                "trending_score": item.get("trendingScore"),
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "task": item.get("pipeline_tag"),
                "url": f"{HF_HUB_BASE}/{'/datasets/' if params.resource_type == 'datasets' else ''}{item.get('modelId', item.get('id'))}",
            }
            for item in data
        ]
        return json.dumps({"trending": params.resource_type, "items": results}, indent=2)
    except Exception as e:
        return _handle_error(e)


if __name__ == "__main__":
    mcp.run()
