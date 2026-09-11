from fastapi.testclient import TestClient

from app.main import app
from app.models import EfficientFrontierRequest
from app.optimizer import compute_efficient_frontier

client = TestClient(app)


def test_efficient_frontier_volatility_rises_with_return():
    req = EfficientFrontierRequest(
        tickers=["AAA", "BBB", "CCC"],
        expected_returns=[0.05, 0.10, 0.15],
        covariance=[
            [0.02, 0.005, 0.0],
            [0.005, 0.03, 0.005],
            [0.0, 0.005, 0.05],
        ],
        n_points=8,
        long_only=True,
        max_weight=1.0,
    )
    out = compute_efficient_frontier(req)
    assert out.n_points >= 3
    # Higher target return should not decrease volatility on the frontier
    vols = [p.portfolio_volatility for p in out.points]
    rets = [p.expected_return for p in out.points]
    assert rets == sorted(rets)
    assert vols[-1] >= vols[0] - 1e-8
    for p in out.points:
        assert abs(sum(p.weights) - 1.0) < 1e-5


def test_frontier_api():
    r = client.post(
        "/frontier",
        json={
            "tickers": ["AAA", "BBB"],
            "expected_returns": [0.05, 0.12],
            "covariance": [[0.02, 0.0], [0.0, 0.05]],
            "n_points": 5,
            "long_only": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["solver"] == "gurobi"
    assert len(body["points"]) >= 2
