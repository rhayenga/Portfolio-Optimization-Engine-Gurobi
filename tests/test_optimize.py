import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import OptimizeRequest
from app.optimizer import optimize_mean_variance

client = TestClient(app)


def _two_asset_request(**kwargs) -> OptimizeRequest:
    base = dict(
        tickers=["AAA", "BBB"],
        expected_returns=[0.10, 0.05],
        covariance=[[0.04, 0.0], [0.0, 0.01]],
        risk_aversion=1.0,
        max_weight=1.0,
        long_only=True,
    )
    base.update(kwargs)
    return OptimizeRequest(**base)


def _four_asset_payload(**kwargs) -> dict:
    base = {
        "tickers": ["AAA", "BBB", "CCC", "DDD"],
        "expected_returns": [0.12, 0.10, 0.08, 0.06],
        "covariance": [
            [0.04, 0.01, 0.01, 0.00],
            [0.01, 0.03, 0.01, 0.00],
            [0.01, 0.01, 0.025, 0.00],
            [0.00, 0.00, 0.00, 0.02],
        ],
        "risk_aversion": 1.0,
        "max_weight": 1.0,
        "long_only": True,
        "sectors": ["TECH", "TECH", "FIN", "FIN"],
        "sector_limits": {"TECH": 0.5, "FIN": 0.6},
        "max_assets": 2,
        "min_weight": 0.05,
    }
    base.update(kwargs)
    return base


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_optimize_weights_sum_to_one():
    req = _two_asset_request()
    out = optimize_mean_variance(req)
    assert abs(sum(out.weights) - 1.0) < 1e-6
    assert out.status == "optimal"
    assert out.holdings_count >= 1
    assert out.portfolio_volatility == pytest.approx(math.sqrt(out.portfolio_variance))


def test_optimize_api():
    payload = {
        "tickers": ["AAA", "BBB"],
        "expected_returns": [0.12, 0.04],
        "covariance": [[0.04, 0.0], [0.0, 0.01]],
        "risk_aversion": 2.0,
        "max_weight": 1.0,
        "long_only": True,
    }
    r = client.post("/optimize", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(sum(body["weights"]) - 1.0) < 1e-6
    assert body["solver"] == "gurobi"
    assert "holdings_count" in body


def test_rejects_mismatched_lengths():
    r = client.post(
        "/optimize",
        json={
            "tickers": ["AAA", "BBB"],
            "expected_returns": [0.1],
            "covariance": [[0.04, 0.0], [0.0, 0.01]],
        },
    )
    assert r.status_code == 422


def test_max_assets_and_sector_caps():
    r = client.post("/optimize", json=_four_asset_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["holdings_count"] <= 2
    assert abs(sum(body["weights"]) - 1.0) < 1e-6
    # TECH = AAA+BBB, FIN = CCC+DDD
    tech = body["weights"][0] + body["weights"][1]
    fin = body["weights"][2] + body["weights"][3]
    assert tech <= 0.5 + 1e-6
    assert fin <= 0.6 + 1e-6


def test_min_return_constraint():
    req = _two_asset_request(min_return=0.09, risk_aversion=10.0)
    out = optimize_mean_variance(req)
    assert out.expected_return >= 0.09 - 1e-6


def test_turnover_limits_rebalance():
    # Unconstrained prefers AAA; with tiny turnover from 100% BBB, stay near BBB
    free = optimize_mean_variance(
        OptimizeRequest(
            tickers=["AAA", "BBB"],
            expected_returns=[0.20, 0.05],
            covariance=[[0.04, 0.0], [0.0, 0.01]],
            risk_aversion=0.5,
            long_only=True,
        )
    )
    assert free.weights[0] > free.weights[1]

    limited = optimize_mean_variance(
        OptimizeRequest(
            tickers=["AAA", "BBB"],
            expected_returns=[0.20, 0.05],
            covariance=[[0.04, 0.0], [0.0, 0.01]],
            risk_aversion=0.5,
            long_only=True,
            current_weights=[0.0, 1.0],
            max_turnover=0.2,
        )
    )
    turnover = abs(limited.weights[0] - 0.0) + abs(limited.weights[1] - 1.0)
    assert turnover <= 0.2 + 1e-5
    assert limited.weights[1] > limited.weights[0]


def test_infeasible_constraints_return_400():
    r = client.post(
        "/optimize",
        json={
            "tickers": ["AAA", "BBB"],
            "expected_returns": [0.05, 0.04],
            "covariance": [[0.04, 0.0], [0.0, 0.01]],
            "min_return": 0.50,
            "long_only": True,
        },
    )
    assert r.status_code == 400
    assert "infeasible" in r.json()["detail"].lower()
