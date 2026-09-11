"""Mean-variance portfolio optimization with Gurobi."""

from __future__ import annotations

import math
from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from app.models import OptimizeRequest, OptimizeResponse


def optimize_mean_variance(req: OptimizeRequest) -> OptimizeResponse:
    """
    Solve: maximize μ'w − (λ/2) w'Σw
    s.t.   1'w = 1
           weight bounds / long-only
           optional μ'w >= min_return
           optional sector caps
           optional cardinality (max_assets) via binary indicators
    """
    n = len(req.tickers)
    mu = np.asarray(req.expected_returns, dtype=float)
    sigma = np.asarray(req.covariance, dtype=float)
    lam = float(req.risk_aversion)

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance matrix must be symmetric")

    model = gp.Model("mean_variance")
    model.Params.OutputFlag = 0

    use_cardinality = req.max_assets is not None
    lo = 0.0 if req.long_only else -req.max_weight
    # When using binaries, keep continuous vars free in [0, max_weight] and link via z
    hi = req.max_weight
    w = model.addVars(n, lb=lo, ub=hi, name="w")

    model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0, name="budget")

    if req.min_return is not None:
        model.addConstr(
            gp.quicksum(float(mu[i]) * w[i] for i in range(n)) >= float(req.min_return),
            name="min_return",
        )

    if req.sectors is not None and req.sector_limits is not None:
        by_sector: dict[str, list[int]] = defaultdict(list)
        for i, sector in enumerate(req.sectors):
            by_sector[sector].append(i)
        for sector, limit in req.sector_limits.items():
            idxs = by_sector.get(sector, [])
            if not idxs:
                continue
            model.addConstr(
                gp.quicksum(w[i] for i in idxs) <= float(limit),
                name=f"sector_{sector}",
            )

    if use_cardinality:
        z = model.addVars(n, vtype=GRB.BINARY, name="z")
        model.addConstr(gp.quicksum(z[i] for i in range(n)) <= int(req.max_assets), name="cardinality")
        for i in range(n):
            model.addConstr(w[i] <= hi * z[i], name=f"link_ub_{i}")
            if req.min_weight > 0:
                model.addConstr(w[i] >= float(req.min_weight) * z[i], name=f"link_lb_{i}")

    # Quadratic objective: μ'w − (λ/2) w'Σw
    obj = gp.QuadExpr()
    for i in range(n):
        obj += float(mu[i]) * w[i]
    for i in range(n):
        for j in range(n):
            c = 0.5 * lam * float(sigma[i, j])
            if c != 0.0:
                obj -= c * w[i] * w[j]

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("constraints are infeasible (no portfolio satisfies them)")
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution (status={model.Status})")

    weights = np.array([w[i].X for i in range(n)], dtype=float)
    weights[np.abs(weights) < 1e-8] = 0.0
    if req.long_only:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            weights = weights / s

    port_ret = float(mu @ weights)
    port_var = float(weights @ sigma @ weights)
    port_vol = math.sqrt(max(port_var, 0.0))
    holdings = int(np.count_nonzero(weights > 1e-8))

    return OptimizeResponse(
        tickers=list(req.tickers),
        weights=[round(float(x), 8) for x in weights],
        expected_return=round(port_ret, 8),
        portfolio_variance=round(port_var, 10),
        portfolio_volatility=round(port_vol, 8),
        holdings_count=holdings,
        status="optimal",
        solver="gurobi",
    )
