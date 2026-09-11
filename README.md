# Portfolio Optimization Engine (Gurobi)

Mean-variance portfolio optimizer exposed as a **FastAPI** service, solved with **Gurobi 13**, packaged with **Docker**.

**Project page:** [rhayenga.github.io/Portfolio-Optimization-Engine-Gurobi](https://rhayenga.github.io/Portfolio-Optimization-Engine-Gurobi/)  
**Repo:** [github.com/rhayenga/Portfolio-Optimization-Engine-Gurobi](https://github.com/rhayenga/Portfolio-Optimization-Engine-Gurobi)

## Stack

- Python 3.10+ / FastAPI + Uvicorn
- Gurobi (`gurobipy`) for the QP / MIQP solve
- yfinance for historical prices → estimated μ / Σ
- Docker (package & ship; academic license stays local)

## Quick Start & Setup (local — recommended with academic license)

Your Free Academic license is host-locked to this Mac (`~/gurobi.lic`):

```bash
git clone https://github.com/rhayenga/Portfolio-Optimization-Engine-Gurobi.git
cd Portfolio-Optimization-Engine-Gurobi
/opt/anaconda3/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --reload --reload-exclude '.venv' --host 127.0.0.1 --port 8000
```

Docs: `http://127.0.0.1:8000/docs` (once the server above is running)

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/optimize` | Optimize with your own μ and Σ |
| POST | `/optimize/from-market` | Download prices, estimate μ/Σ, then optimize |
| POST | `/frontier` | Efficient frontier (min variance at target returns) |
| POST | `/frontier/from-market` | Estimate μ/Σ from prices, then sweep the frontier |
| POST | `/backtest/from-market` | Train on history, optimize, evaluate OOS vs equal-weight |

### From-market example (real tickers)

```bash
curl -s http://127.0.0.1:8000/optimize/from-market \
  -H 'Content-Type: application/json' \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "JPM"],
    "lookback_days": 252,
    "risk_aversion": 2.0,
    "max_weight": 0.4,
    "long_only": true,
    "max_assets": 3,
    "min_weight": 0.05,
    "sectors": ["TECH", "TECH", "TECH", "FIN"],
    "sector_limits": {"TECH": 0.7, "FIN": 0.4}
  }'
```

### Manual μ / Σ example

```bash
curl -s http://127.0.0.1:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{
    "tickers": ["AAA", "BBB"],
    "expected_returns": [0.10, 0.05],
    "covariance": [[0.04, 0.0], [0.0, 0.01]],
    "risk_aversion": 1.0,
    "max_weight": 1.0,
    "long_only": true
  }'
```

Objective: maximize `μ'w − (λ/2) w'Σw` with budget, optional `min_return`, sector caps, and `max_assets` (Gurobi binaries).

## Tests

```bash
source .venv/bin/activate
pytest -q
```

GitHub Actions runs the same suite on push/PR (`.github/workflows/ci.yml`).

## CLI

```bash
source .venv/bin/activate
python -m app.cli optimize --tickers AAPL,MSFT,GOOGL,JPM --risk-aversion 2 --max-weight 0.4
python -m app.cli frontier --tickers AAPL,MSFT,GOOGL --n-points 10
python -m app.cli backtest --tickers AAPL,MSFT,GOOGL,JPM --train-days 252 --test-days 63
```

Market endpoints shrink Σ with **Ledoit–Wolf** by default (`shrink_covariance: true`).

## Docker

Academic named-user licenses **do not unlock Gurobi inside Docker**. Use the local venv on this Mac for solves. The image still packages the FastAPI + data layer; for containerized solves you need [WLS](https://www.gurobi.com/features/web-license-service/) env vars (see `docker-compose.yml`).

```bash
docker compose build
docker compose up
```

## License

MIT (see `LICENSE`). Gurobi itself requires a separate Gurobi license.
