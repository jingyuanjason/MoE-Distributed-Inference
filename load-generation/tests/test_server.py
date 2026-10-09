"""Tests for the load generator FastAPI service."""

import pytest
from httpx import ASGITransport, AsyncClient

from server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_trigger_load_accepts_valid_request(client: AsyncClient):
    resp = await client.post(
        "/load",
        json={
            "ip_address": "127.0.0.1",
            "samples_per_second": 10,
            "duration_seconds": 30,
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["ip_address"] == "127.0.0.1"
    assert body["samples_per_second"] == 10.0
    assert body["duration_seconds"] == 30.0
    assert body["status"] == "accepted"


@pytest.mark.asyncio
async def test_trigger_load_rejects_invalid_ip(client: AsyncClient):
    resp = await client.post(
        "/load",
        json={
            "ip_address": "not-an-ip",
            "samples_per_second": 10,
            "duration_seconds": 30,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_trigger_load_rejects_non_positive_rate(client: AsyncClient):
    resp = await client.post(
        "/load",
        json={
            "ip_address": "127.0.0.1",
            "samples_per_second": 0,
            "duration_seconds": 30,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_trigger_load_rejects_non_positive_duration(client: AsyncClient):
    resp = await client.post(
        "/load",
        json={
            "ip_address": "127.0.0.1",
            "samples_per_second": 10,
            "duration_seconds": 0,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_trigger_load_rejects_missing_fields(client: AsyncClient):
    resp = await client.post("/load", json={})
    assert resp.status_code == 422
