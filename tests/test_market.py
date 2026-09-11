import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.market_data import estimate_mu_sigma

client = TestClient(app)


def _fake_closes() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    dates = pd.bdate_range("2024-01-01", periods=n)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n)))
    b = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=n)))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=dates)


def test_estimate_mu_sigma_shapes():
    tickers, mu, cov, intensity = estimate_mu_sigma(_fake_closes())
    assert tickers == ["AAA", "BBB"]
    assert len(mu) == 2
    assert len(cov) == 2 and len(cov[0]) == 2
    assert abs(cov[0][1] - cov[1][0]) < 1e-12
    assert 0.0 <= intensity <= 1.0


def test_ledoit_wolf_reduces_off_diagonals_toward_identity():
    from app.market_data import ledoit_wolf_shrinkage

    sample = np.array([[0.04, 0.03], [0.03, 0.09]], dtype=float)
    shrunk, intensity = ledoit_wolf_shrinkage(sample, n_obs=50)
    assert 0.0 < intensity <= 1.0
    assert abs(shrunk[0, 1]) <= abs(sample[0, 1]) + 1e-12


def test_from_market_endpoint(monkeypatch):
    def fake_fetch(tickers, lookback_days):
        assert tickers == ["AAA", "BBB"]
        return _fake_closes()

    monkeypatch.setattr("app.main.fetch_adjusted_closes", fake_fetch)
    r = client.post(
        "/optimize/from-market",
        json={
            "tickers": ["AAA", "BBB"],
            "lookback_days": 60,
            "risk_aversion": 2.0,
            "long_only": True,
            "max_weight": 1.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["solver"] == "gurobi"
    assert abs(sum(body["weights"]) - 1.0) < 1e-6
    assert body["lookback_days"] == 60
    assert len(body["expected_returns"]) == 2
    assert len(body["covariance"]) == 2


def test_from_market_rejects_one_ticker():
    r = client.post(
        "/optimize/from-market",
        json={"tickers": ["AAA"], "lookback_days": 60},
    )
    assert r.status_code == 422
