"""API tests.

Exercise the FastAPI endpoints against a live Neo4j + pgvector. Requires
``NEO4J_PASSWORD`` to be set (see .env.example). Clears the graph before each
test so results are deterministic regardless of prior runs.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from kg_rag.api import app

NEO4J_PASSWORD = "kg_rag_local_dev_2026"


@pytest.fixture(scope="module")
def client():
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
    # Clear the graph so each module run starts from a blank slate.
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", NEO4J_PASSWORD)
    )
    try:
        try:
            driver.verify_connectivity()
        except ServiceUnavailable:
            pytest.skip("requires a reachable Neo4j service on localhost:7687")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()

    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_populates_graph(client):
    r = client.post("/ingest")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == 8
    assert body["chunks"] == 16
    assert body["nodes_written"] == 61
    assert body["edges_written"] == 18


def test_ask_supplier(client):
    # Ingest first so the graph has data.
    client.post("/ingest")
    r = client.post("/ask", json={"question": "Who supplies NVIDIA's foundry?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "graph"
    assert body["refused"] is False
    assert "Taiwan Semiconductor" in body["answer"]
    assert body["citations"]


def test_ask_competitor(client):
    client.post("/ingest")
    r = client.post("/ask", json={"question": "Who does NVIDIA compete with?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "graph"
    assert body["refused"] is False
    assert "Advanced Micro Devices" in body["answer"]
    assert "Intel Corporation" in body["answer"]


def test_ask_out_of_scope_refused(client):
    client.post("/ingest")
    r = client.post("/ask", json={"question": "What is the weather in Paris?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "out_of_scope"
    assert body["refused"] is True


def test_ask_blank_question_rejected(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400
