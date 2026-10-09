"""FastAPI service exposing the load generator.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, IPvAnyAddress

from core import (
    GeneratorSettings,
    LoadAlreadyRunningError,
    LoadConfig,
    LoadGenerator,
    PromptSampler,
)

app = FastAPI(title="LLM Load Generator", version="0.1.0")

CONFIG_PATH = Path(
    os.environ.get("LOAD_GENERATOR_CONFIG", Path(__file__).parent / "config.example.yaml")
)

load_generator = LoadGenerator(
    sampler=PromptSampler.from_recipe(CONFIG_PATH),
    settings=GeneratorSettings.from_file(CONFIG_PATH),
)


class LoadRequest(BaseModel):
    """Request body to trigger load against a target server."""

    ip_address: IPvAnyAddress = Field(..., description="Target server IP address")
    port: int = Field(8000, gt=0, le=65535, description="Target server port")
    model: str = Field(
        "default", description="Model name to send in the request payload"
    )
    samples_per_second: float = Field(
        ..., gt=0, description="Number of samples (requests) to send per second"
    )
    duration_seconds: float = Field(
        ..., gt=0, description="How long the load generation run should last, in seconds"
    )


class LoadResponse(BaseModel):
    ip_address: str
    port: int
    model: str
    samples_per_second: float
    duration_seconds: float
    status: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
async def list_runs() -> list[dict]:
    """List all load runs (most recent last) with their statistics."""
    return [
        {
            "ip_address": run.config.ip_address,
            "port": run.config.port,
            "model": run.config.model,
            "samples_per_second": run.config.samples_per_second,
            "duration_seconds": run.config.duration_seconds,
            "running": run.finished_at is None,
            **run.stats,
        }
        for run in load_generator.active_runs
    ]


@app.post("/load", response_model=LoadResponse, status_code=202)
async def trigger_load(request: LoadRequest) -> LoadResponse:
    """Trigger load generation against the given IP address."""
    config = LoadConfig(
        ip_address=str(request.ip_address),
        samples_per_second=request.samples_per_second,
        duration_seconds=request.duration_seconds,
        port=request.port,
        model=request.model,
    )
    try:
        await load_generator.start_load(config)
    except LoadAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LoadResponse(
        ip_address=config.ip_address,
        port=config.port,
        model=config.model,
        samples_per_second=config.samples_per_second,
        duration_seconds=config.duration_seconds,
        status="accepted",
    )
