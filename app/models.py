from pydantic import BaseModel, Field, field_validator, model_validator


class PortfolioConstraints(BaseModel):
    """Shared portfolio constraints for raw and market-data optimize calls."""

    risk_aversion: float = Field(
        1.0,
        gt=0,
        description="Risk aversion λ in max μ'w − (λ/2) w'Σw",
    )
    max_weight: float = Field(
        1.0,
        gt=0,
        le=1.0,
        description="Per-asset weight upper bound",
    )
    min_weight: float = Field(
        0.0,
        ge=0,
        le=1.0,
        description="Minimum weight if an asset is selected (used with max_assets)",
    )
    long_only: bool = Field(True, description="If true, weights >= 0")
    min_return: float | None = Field(
        None,
        description="Optional floor on portfolio expected return μ'w",
    )
    max_assets: int | None = Field(
        None,
        ge=1,
        description="Optional max number of holdings (Gurobi binary cardinality)",
    )
    sectors: list[str] | None = Field(
        None,
        description="Optional sector label per ticker (same order as tickers)",
    )
    sector_limits: dict[str, float] | None = Field(
        None,
        description='Optional max total weight per sector, e.g. {"TECH": 0.4}',
    )

    @field_validator("sectors")
    @classmethod
    def normalize_sectors(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [s.strip().upper() for s in v]

    @field_validator("sector_limits")
    @classmethod
    def normalize_sector_limits(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return None
        out: dict[str, float] = {}
        for k, lim in v.items():
            key = k.strip().upper()
            if not (0.0 < lim <= 1.0):
                raise ValueError(f"sector_limits[{key}] must be in (0, 1]")
            out[key] = lim
        return out

    @model_validator(mode="after")
    def check_constraint_fields(self) -> "PortfolioConstraints":
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot exceed max_weight")
        if self.max_assets is not None and not self.long_only:
            raise ValueError("max_assets currently requires long_only=true")
        if self.sector_limits is not None and self.sectors is None:
            raise ValueError("sector_limits requires sectors")
        return self


class OptimizeRequest(PortfolioConstraints):
    """Mean-variance portfolio optimization inputs."""

    tickers: list[str] = Field(..., min_length=2, description="Asset tickers")
    expected_returns: list[float] = Field(
        ...,
        min_length=2,
        description="Expected return for each ticker (same order)",
    )
    covariance: list[list[float]] = Field(
        ...,
        min_length=2,
        description="Covariance matrix (n x n), annualized",
    )

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in v if t.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("tickers must be unique")
        return cleaned

    @field_validator("expected_returns")
    @classmethod
    def check_returns_len(cls, v: list[float], info) -> list[float]:
        tickers = info.data.get("tickers")
        if tickers is not None and len(v) != len(tickers):
            raise ValueError("expected_returns length must match tickers")
        return v

    @field_validator("covariance")
    @classmethod
    def check_covariance(cls, v: list[list[float]], info) -> list[list[float]]:
        tickers = info.data.get("tickers")
        n = len(tickers) if tickers is not None else len(v)
        if len(v) != n:
            raise ValueError("covariance must be n x n matching tickers")
        for row in v:
            if len(row) != n:
                raise ValueError("covariance must be square")
        return v

    @model_validator(mode="after")
    def check_aligned_fields(self) -> "OptimizeRequest":
        if self.sectors is not None and len(self.sectors) != len(self.tickers):
            raise ValueError("sectors length must match tickers")
        return self


class MarketOptimizeRequest(PortfolioConstraints):
    """Optimize from live/historical market prices (estimates μ and Σ)."""

    tickers: list[str] = Field(..., min_length=2, description="Asset tickers")
    lookback_days: int = Field(
        252,
        ge=30,
        le=2520,
        description="Trading days of history used to estimate μ and Σ",
    )

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in v if t.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("tickers must be unique")
        return cleaned

    @model_validator(mode="after")
    def check_aligned_fields(self) -> "MarketOptimizeRequest":
        if self.sectors is not None and len(self.sectors) != len(self.tickers):
            raise ValueError("sectors length must match tickers")
        return self


class OptimizeResponse(BaseModel):
    tickers: list[str]
    weights: list[float]
    expected_return: float
    portfolio_variance: float
    portfolio_volatility: float
    holdings_count: int
    status: str
    solver: str = "gurobi"


class MarketOptimizeResponse(OptimizeResponse):
    lookback_days: int
    expected_returns: list[float]
    covariance: list[list[float]]
