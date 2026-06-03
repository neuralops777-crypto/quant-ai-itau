"""Métricas de performance do backtest.

Calcula CAGR, Sharpe, Sortino, Max Drawdown, Alpha e Beta
a partir do DataFrame retornado pelo engine de backtest.

Fluxo:
    ``compute_metrics(bt, bench_ret, rf_annual)`` → ``Dict[str, float]``
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.portfolio.risk import (
    alpha_beta,
    annualize_return,
    annualize_vol,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    var_cvar,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Args:
        equity: Série de valores absolutos do portfólio.

    Returns:
        CAGR anualizado.
    """
    e = equity.dropna()
    if len(e) < 2:
        return 0.0
    years = (e.index[-1] - e.index[0]).days / 365.25
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def compute_metrics(
    bt: pd.DataFrame,
    bench_ret: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
) -> Dict[str, float]:
    """Compila todas as métricas de performance do backtest.

    Args:
        bt: DataFrame retornado por :func:`~src.backtest.engine.run_backtest`.
            Deve conter colunas ``equity``, ``ret``, ``turnover``, ``cost``.
        bench_ret: Retornos diários do benchmark (opcional).
        rf_annual: Taxa livre de risco anual.

    Returns:
        Dicionário de métricas.
    """
    port_ret = bt["ret"].astype(float)
    equity_curve = bt["equity"].astype(float)

    var, cvar = var_cvar(port_ret)

    out: Dict[str, float] = {
        "CAGR": cagr(equity_curve),
        "TotalReturn": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
        "AnnReturn": annualize_return(port_ret),
        "AnnVol": annualize_vol(port_ret),
        "Sharpe": sharpe_ratio(port_ret, rf_annual=rf_annual),
        "Sortino": sortino_ratio(port_ret, rf_annual=rf_annual),
        "MaxDrawdown": max_drawdown(equity_curve),
        "VaR_5%": var,
        "CVaR_5%": cvar,
        "AvgTurnover": float(bt["turnover"].mean()) if "turnover" in bt.columns else 0.0,
        "TotalCosts": float(bt["cost"].sum()) if "cost" in bt.columns else 0.0,
    }

    if bench_ret is not None:
        a, b = alpha_beta(port_ret, bench_ret, rf_annual=rf_annual)
        out["Alpha"] = a
        out["Beta"] = b
    else:
        out["Alpha"] = 0.0
        out["Beta"] = 0.0

    logger.info("Métricas calculadas", extra=out)
    return out
