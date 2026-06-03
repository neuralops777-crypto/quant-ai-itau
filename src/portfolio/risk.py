"""Métricas de risco e performance.

Funções puras (sem estado) para cálculo de:
    * Volatilidade anualizada
    * VaR e CVaR históricos
    * Sharpe e Sortino anualizados
    * Max Drawdown
    * Alpha de Jensen e Beta

Todas as funções esperam retornos diários como ``pd.Series``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass de saída
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskMetrics:
    """Container imutável de métricas de risco."""

    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    var_5pct: float
    cvar_5pct: float


# ---------------------------------------------------------------------------
# Funções de base
# ---------------------------------------------------------------------------

def annualize_return(daily_ret: pd.Series, periods: int = 252) -> float:
    """Retorno composto anualizado."""
    r = daily_ret.dropna()
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0)


def annualize_vol(daily_ret: pd.Series, periods: int = 252) -> float:
    """Volatilidade anualizada (desvio padrão)."""
    r = daily_ret.dropna()
    return float(r.std(ddof=0) * np.sqrt(periods)) if not r.empty else 0.0


def sharpe_ratio(
    daily_ret: pd.Series, rf_annual: float = 0.0, periods: int = 252
) -> float:
    """Sharpe ratio anualizado.

    Args:
        daily_ret: Retornos diários.
        rf_annual: Taxa livre de risco anual.
        periods: Dias úteis por ano.

    Returns:
        Sharpe anualizado.
    """
    r = daily_ret.dropna()
    if r.empty:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = r - rf_daily
    denom = excess.std(ddof=0)
    return float(excess.mean() / denom * np.sqrt(periods)) if denom > 0 else 0.0


def sortino_ratio(
    daily_ret: pd.Series, rf_annual: float = 0.0, periods: int = 252
) -> float:
    """Sortino ratio anualizado (usa apenas retornos negativos no denominador)."""
    r = daily_ret.dropna()
    if r.empty:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = r - rf_daily
    downside_std = excess[excess < 0].std(ddof=0)
    return float(excess.mean() / downside_std * np.sqrt(periods)) if downside_std > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """Drawdown máximo (pico a vale) da curva de equity."""
    e = equity.dropna()
    if e.empty:
        return 0.0
    dd = e / e.cummax() - 1.0
    return float(dd.min())


def var_cvar(
    daily_ret: pd.Series, alpha: float = 0.05
) -> Tuple[float, float]:
    """VaR e CVaR históricos.

    Args:
        daily_ret: Retornos diários.
        alpha: Nível de significância (ex.: 0.05 → VaR 95 %).

    Returns:
        Tupla ``(VaR, CVaR)``.
    """
    r = daily_ret.dropna()
    if r.empty:
        return 0.0, 0.0
    var = float(np.quantile(r, alpha))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var
    return var, cvar


def alpha_beta(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    rf_annual: float = 0.0,
    periods: int = 252,
) -> Tuple[float, float]:
    """Alpha de Jensen e Beta vs benchmark.

    Args:
        port_ret: Retornos diários do portfólio.
        bench_ret: Retornos diários do benchmark.
        rf_annual: Taxa livre de risco anual.
        periods: Dias úteis por ano.

    Returns:
        Tupla ``(alpha_anual, beta)``.
    """
    idx = port_ret.index.intersection(bench_ret.index)
    p = port_ret.loc[idx].dropna()
    b = bench_ret.loc[idx].dropna()
    common = p.index.intersection(b.index)
    p, b = p.loc[common], b.loc[common]

    if len(p) < 10:
        return 0.0, 0.0

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    ep, eb = p - rf_daily, b - rf_daily

    beta_val = float(np.cov(ep, eb, ddof=0)[0, 1] / max(np.var(eb), 1e-12))
    alpha_daily = float(ep.mean() - beta_val * eb.mean())
    alpha_annual = float((1.0 + alpha_daily) ** periods - 1.0)
    return alpha_annual, beta_val


# ---------------------------------------------------------------------------
# Sumário agregado
# ---------------------------------------------------------------------------

def risk_summary(
    daily_ret: pd.Series,
    equity: pd.Series,
    rf_annual: float,
    bench_ret: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Calcula todas as métricas de uma vez.

    Args:
        daily_ret: Retornos diários do portfólio.
        equity: Curva de equity (valor absoluto).
        rf_annual: Taxa livre de risco anual.
        bench_ret: Retornos do benchmark (opcional, para Alpha/Beta).

    Returns:
        Dicionário de métricas.
    """
    var, cvar = var_cvar(daily_ret)
    out: Dict[str, float] = {
        "ann_return": annualize_return(daily_ret),
        "ann_vol": annualize_vol(daily_ret),
        "sharpe": sharpe_ratio(daily_ret, rf_annual=rf_annual),
        "sortino": sortino_ratio(daily_ret, rf_annual=rf_annual),
        "max_drawdown": max_drawdown(equity),
        "VaR_5%": var,
        "CVaR_5%": cvar,
    }
    if bench_ret is not None:
        a, b = alpha_beta(daily_ret, bench_ret, rf_annual=rf_annual)
        out["alpha"] = a
        out["beta"] = b
    return out
