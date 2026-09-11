import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_backtest_from_market(monkeypatch):
    rng = np.random.default_rng(1)
    n = 400
    dates = pd.bdate_range("2023-01-01", periods=n)
    prices = {
        "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=n))),
        "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=n))),
    }
    closes = pd.DataFrame(prices, index=dates)

    monkeypatch.setattr("app.backtest.fetch_adjusted_closes", lambda tickers, days: closes)

    r = client.post(
        "/backtest/from-market",
        json={
            "tickers": ["AAA", "BBB"],
            "train_days": 252,
            "test_days": 63,
            "risk_aversion": 2.0,
            "long_only": True,
            "max_weight": 1.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(sum(body["weights"]) - 1.0) < 1e-6
    assert "sharpe_ratio" in body["optimized"]
    assert "sharpe_ratio" in body["equal_weight"]
    assert body["test_days"] > 0
