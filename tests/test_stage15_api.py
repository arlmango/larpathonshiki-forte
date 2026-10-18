"""Synthetic-only checks for the standalone, fail-closed research adapter."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tools.stage15_api import ResearchResponse, create_research_app
from research_fixtures import Probe

KEY="fixture-only"

def client(provider=None):
    return TestClient(create_research_app(provider, api_key=KEY), headers={"X-API-Key":KEY})


@pytest.mark.parametrize("logit,candidates", [(1.2, [3.5]), (0.0, [3.5]), (-0.1, [])])
def test_fixed_gate_and_unvalidated_semantics(logit, candidates):
    provider={"synthetic_01":{"clip_logit":logit,"candidate_time_sec":3.5}}.get
    response=client(provider).post("/research/stage15/predict",json={"sample_id":"synthetic_01"})
    assert response.status_code == 200
    assert response.json() == {
        "research_only": True, "production_enabled": False, "status": "not_validated", "human_review_required": True,
        "uncalibrated_clip_logit": logit, "candidate_times_sec": candidates,
    }


@pytest.mark.parametrize("output", [
    None, {}, {"clip_logit": 1},
    {"clip_logit": float("nan"), "candidate_time_sec": 1},
    {"clip_logit": 1, "candidate_time_sec": float("inf")},
    {"clip_logit": 1, "candidate_time_sec": -1},
    {"clip_logit": "1", "candidate_time_sec": 1},
    {"clip_logit": True, "candidate_time_sec": 1},
    {"clip_logit": 1, "candidate_time_sec": 1, "verdict": "violation"},
])
def test_malformed_provider_fails_closed(output):
    response = client({"synthetic_01":output}.get).post(
        "/research/stage15/predict", json={"sample_id": "synthetic_01"}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "research prediction unavailable"}


def test_provider_exception_is_private_and_missing_provider_unavailable():
    broken=Probe("exception")

    for provider, detail in [(broken, "research prediction unavailable"), (None, "research provider unavailable")]:
        response = client(provider).post(
            "/research/stage15/predict", json={"sample_id": "synthetic_01"}
        )
        assert response.status_code == 503
        assert response.json() == {"detail": detail}


@pytest.mark.parametrize("payload", [
    {"sample_id": "../../private"}, {"sample_id": "a" * 65}, {"sample_id": ""},
    {"sample_id": 123}, {"sample_id": "synthetic_01", "embeddings": [1, 2]},
    {"sample_id": "synthetic_01", "path": "/private"},
])
def test_request_rejects_paths_payloads_and_never_echoes(payload):
    unexpected=Probe("exception")

    response = client(unexpected).post("/research/stage15/predict", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid research request"}


def test_response_forbids_decision_fields_and_app_has_no_production_routes():
    for field in ["verdict", "type", "norm"]:
        with pytest.raises(ValidationError):
            ResearchResponse(uncalibrated_clip_logit=1.0, candidate_times_sec=[1.0], **{field: "injected"})
    paths = {route.path for route in create_research_app(api_key=KEY).routes}
    assert paths == {"/", "/research/health", "/research/stage15/predict", "/research/stage15/analyze"}
