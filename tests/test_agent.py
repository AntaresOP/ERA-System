import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert response.json()["system"] == "ERA System"


def test_search_endpoint():
    payload = {
        "query": "What is long term semantic memory in LLM agents?",
        "use_cache": False
    }
    response = client.post("/api/v1/engine/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert data["query"] == payload["query"]