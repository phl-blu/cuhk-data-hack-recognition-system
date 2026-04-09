"""
Unit tests for api.py — health endpoint lifecycle, CORS headers.
Requirements: 1.2, 1.3, 1.4, 3.1
"""
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers to build a patched version of api.py
# ---------------------------------------------------------------------------

def _make_fake_feature_extractor():
    """Return a minimal FeatureExtractor stub."""
    fe = MagicMock()
    fe.extract_features.return_value = [0.0] * 1280
    return fe


def _make_fake_joblib(scaler, svm):
    """Return a joblib stub whose load() returns scaler then svm."""
    jl = MagicMock()
    jl.load.side_effect = [scaler, svm]
    return jl


def _build_app(*, models_should_load: bool = True, cors_origins_env: str = ""):
    """
    Import api.py in a controlled way so we can test startup behaviour.

    Because api.py uses a lifespan context manager, we use TestClient as a
    context manager to trigger startup/shutdown.
    """
    # Remove any previously cached module so each test gets a fresh import
    for key in list(sys.modules.keys()):
        if key in ("api",):
            del sys.modules[key]

    fake_scaler = MagicMock()
    fake_scaler.transform.return_value = [[0.0] * 1280]
    fake_svm = MagicMock()
    fake_svm.predict_proba.return_value = [[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    patches = {
        "CORS_ORIGINS": cors_origins_env,
    }

    if models_should_load:
        fe_instance = _make_fake_feature_extractor()
        jl = _make_fake_joblib(fake_scaler, fake_svm)
    else:
        # Simulate a missing model file
        jl = MagicMock()
        jl.load.side_effect = FileNotFoundError("svm_model.pkl not found")
        fe_instance = _make_fake_feature_extractor()

    with (
        patch.dict("os.environ", patches),
        patch("feature_extractor.FeatureExtractor", return_value=fe_instance),
        patch("joblib.load", side_effect=jl.load),
    ):
        import api as api_module
        return api_module


# ---------------------------------------------------------------------------
# Task 4.1 — Health endpoint lifecycle
# ---------------------------------------------------------------------------

class TestHealthEndpointLifecycle:
    """Requirements 1.3, 1.4 — /health returns 503 before load, 200 after."""

    def test_health_returns_200_after_models_loaded(self):
        """After successful startup models_loaded is True → 200 {"status":"ok"}."""
        api_module = _build_app(models_should_load=True)
        with TestClient(api_module.app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_returns_503_before_models_loaded(self):
        """
        Requirement 1.4 — if models_loaded is False the endpoint returns 503.
        We patch the flag directly after import to simulate the pre-load state.
        """
        api_module = _build_app(models_should_load=True)
        # Manually reset the flag to simulate the window before startup completes
        api_module.models_loaded = False
        client = TestClient(api_module.app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"status": "loading"}

    def test_missing_model_file_causes_sys_exit(self):
        """
        Requirement 1.2 — a missing model file must cause sys.exit(1).
        We run the lifespan startup directly and assert sys.exit is called
        with a non-zero code.
        """
        for key in list(sys.modules.keys()):
            if key == "api":
                del sys.modules[key]

        fe_instance = _make_fake_feature_extractor()
        exit_calls = []

        def fake_exit(code=0):
            exit_calls.append(code)
            raise SystemExit(code)

        with (
            patch("feature_extractor.FeatureExtractor", return_value=fe_instance),
            patch("joblib.load", side_effect=FileNotFoundError("svm_model.pkl not found")),
            patch("sys.exit", side_effect=fake_exit),
        ):
            import api as api_module
            import asyncio

            async def run_lifespan():
                async with api_module.lifespan(api_module.app):
                    pass

            with pytest.raises(SystemExit):
                asyncio.run(run_lifespan())

        assert exit_calls, "sys.exit was never called"
        assert exit_calls[0] != 0, f"Expected non-zero exit code, got {exit_calls[0]}"





# ---------------------------------------------------------------------------
# Task 4.1 — CORS headers on HTTP responses
# ---------------------------------------------------------------------------

class TestCORSHeaders:
    """Requirement 3.1 — CORS headers must be present on HTTP responses."""

    def test_cors_headers_present_with_wildcard_default(self):
        """When CORS_ORIGINS is unset, allow-origin should be '*'."""
        api_module = _build_app(models_should_load=True, cors_origins_env="")
        with TestClient(api_module.app, raise_server_exceptions=False) as client:
            resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in resp.headers

    def test_cors_headers_present_with_explicit_origin(self):
        """When CORS_ORIGINS is set, the header should reflect the configured origin."""
        api_module = _build_app(
            models_should_load=True,
            cors_origins_env="http://localhost:3000",
        )
        with TestClient(api_module.app, raise_server_exceptions=False) as client:
            resp = client.get(
                "/health",
                headers={"Origin": "http://localhost:3000"},
            )
        assert "access-control-allow-origin" in resp.headers
        assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_cors_preflight_returns_200(self):
        """OPTIONS preflight should succeed (CORS middleware handles it)."""
        api_module = _build_app(models_should_load=True, cors_origins_env="")
        with TestClient(api_module.app, raise_server_exceptions=False) as client:
            resp = client.options(
                "/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.status_code == 200
