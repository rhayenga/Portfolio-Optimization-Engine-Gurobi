"""Mean-variance portfolio optimization with Gurobi."""

from __future__ import annotations

import math
from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from app.models import (
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    FrontierPoint,
    OptimizeRequest,
    OptimizeResponse,
)


def _add_common_constraints(
    model: gp.Model,
    w: gp.tupledict,
    req: OptimizeRequest,
    n: int,
    hi: float,
) -> None:
    model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0, name="budget")

    if req.min_return is not None:
        mu = req.expected_returns
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

    if req.max_assets is not None:
        z = model.addVars(n, vtype=GRB.BINARY, name="z")
        model.addConstr(gp.quicksum(z[i] for i in range(n)) <= int(req.max_assets), name="cardinality")
        for i in range(n):
            model.addConstr(w[i] <= hi * z[i], name=f"link_ub_{i}")
            if req.min_weight > 0:
                model.addConstr(w[i] >= float(req.min_weight) * z[i], name=f"link_lb_{i}")

    if req.max_turnover is not None and req.current_weights is not None:
        w0 = req.current_weights
        t = model.addVars(n, lb=0.0, name="turnover")
        for i in range(n):
            model.addConstr(t[i] >= w[i] - float(w0[i]), name=f"turn_pos_{i}")
            model.addConstr(t[i] >= float(w0[i]) - w[i], name=f"turn_neg_{i}")
        model.addConstr(
            gp.quicksum(t[i] for i in range(n)) <= float(req.max_turnover),
            name="max_turnover",
        )


def _extract_weights(w: gp.tupledict, n: int, long_only: bool) -> np.ndarray:
    weights = np.array([w[i].X for i in range(n)], dtype=float)
    weights[np.abs(weights) < 1e-8] = 0.0
    if long_only:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            weights = weights / s
    return weights


def _response_from_weights(
    tickers: list[str],
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> OptimizeResponse:
    port_ret = float(mu @ weights)
    port_var = float(weights @ sigma @ weights)
    port_vol = math.sqrt(max(port_var, 0.0))
    holdings = int(np.count_nonzero(weights > 1e-8))
    return OptimizeResponse(
        tickers=list(tickers),
        weights=[round(float(x), 8) for x in weights],
        expected_return=round(port_ret, 8),
        portfolio_variance=round(port_var, 10),
        portfolio_volatility=round(port_vol, 8),
        holdings_count=holdings,
        status="optimal",
        solver="gurobi",
    )


def optimize_mean_variance(req: OptimizeRequest) -> OptimizeResponse:
    """
    Solve: maximize μ'w − (λ/2) w'Σw
    s.t.   1'w = 1 and optional portfolio constraints.
    """
    n = len(req.tickers)
    mu = np.asarray(req.expected_returns, dtype=float)
    sigma = np.asarray(req.covariance, dtype=float)
    lam = float(req.risk_aversion)

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance matrix must be symmetric")

    model = gp.Model("mean_variance")
    model.Params.OutputFlag = 0

    lo = 0.0 if req.long_only else -req.max_weight
    hi = req.max_weight
    w = model.addVars(n, lb=lo, ub=hi, name="w")
    _add_common_constraints(model, w, req, n, hi)

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

    weights = _extract_weights(w, n, req.long_only)
    return _response_from_weights(list(req.tickers), weights, mu, sigma)


def minimize_variance_at_return(req: OptimizeRequest, target_return: float) -> OptimizeResponse | None:
    """
    Classic frontier point: minimize (1/2) w'Σw
    s.t. μ'w = target_return, 1'w = 1, plus shared constraints.
    Returns None if this target is infeasible.
    """
    n = len(req.tickers)
    mu = np.asarray(req.expected_returns, dtype=float)
    sigma = np.asarray(req.covariance, dtype=float)

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance matrix must be symmetric")

    # Build a request without min_return so the target equality is the return constraint
    constr_req = req.model_copy(update={"min_return": None})

    model = gp.Model("min_var_at_return")
    model.Params.OutputFlag = 0

    lo = 0.0 if constr_req.long_only else -constr_req.max_weight
    hi = constr_req.max_weight
    w = model.addVars(n, lb=lo, ub=hi, name="w")
    _add_common_constraints(model, w, constr_req, n, hi)
    model.addConstr(
        gp.quicksum(float(mu[i]) * w[i] for i in range(n)) == float(target_return),
        name="target_return",
    )

    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            c = 0.5 * float(sigma[i, j])
            if c != 0.0:
                obj += c * w[i] * w[j]
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        return None
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution (status={model.Status})")

    weights = _extract_weights(w, n, constr_req.long_only)
    return _response_from_weights(list(req.tickers), weights, mu, sigma)


def compute_efficient_frontier(req: EfficientFrontierRequest) -> EfficientFrontierResponse:
    """Sweep target returns between asset μ min/max and solve min-variance at each."""
    mu = np.asarray(req.expected_returns, dtype=float)
    lo_r, hi_r = float(mu.min()), float(mu.max())
    if hi_r - lo_r < 1e-12:
        raise ValueError("expected returns are identical; frontier is a single point")

    # Stay slightly inside the raw min/max so long-only problems stay feasible more often
    span = hi_r - lo_r
    pad = 0.02 * span
    targets = np.linspace(lo_r + pad, hi_r - pad, req.n_points)

    opt_base = OptimizeRequest(
        tickers=req.tickers,
        expected_returns=req.expected_returns,
        covariance=req.covariance,
        risk_aversion=1.0,
        max_weight=req.max_weight,
        min_weight=req.min_weight,
        long_only=req.long_only,
        min_return=None,
        max_assets=req.max_assets,
        sectors=req.sectors,
        sector_limits=req.sector_limits,
    )

    points: list[FrontierPoint] = []
    for t in targets:
        sol = minimize_variance_at_return(opt_base, float(t))
        if sol is None:
            continue
        points.append(
            FrontierPoint(
                target_return=round(float(t), 8),
                expected_return=sol.expected_return,
                portfolio_volatility=sol.portfolio_volatility,
                portfolio_variance=sol.portfolio_variance,
                weights=sol.weights,
                holdings_count=sol.holdings_count,
            )
        )

    if not points:
        raise ValueError("no feasible frontier points for the given constraints")

    # Keep the efficient (upper) branch: from global min-variance upward in return
    gmv_idx = min(range(len(points)), key=lambda i: points[i].portfolio_volatility)
    points = points[gmv_idx:]

    return EfficientFrontierResponse(
        tickers=list(req.tickers),
        n_points=len(points),
        points=points,
        solver="gurobi",
    )
