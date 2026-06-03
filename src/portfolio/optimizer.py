"""Otimização de portfólio via PyPortfolioOpt.

Estratégias suportadas:
    * ``max_sharpe``  — Máximo Sharpe Ratio.
    * ``min_vol``     — Mínima Volatilidade.
    * ``markowitz``   — Máxima Utilidade Quadrática.

Diferenciais em relação à implementação básica:
    * **Ledoit-Wolf shrinkage** na estimativa de covariância — muito mais
      estável com poucos ativos (< 30) do que ``sample_cov``.
    * **Regularização L2** (``gamma`` configurável) — penaliza concentração
      excessiva de pesos, resultando em carteiras mais diversificadas.
    * Expected returns injetáveis — permite usar predições do XGBoost
      em vez da média histórica simples.

Fluxo:
    ``optimize_weights(prices_wide, exp_returns, cfg)`` → ``Dict[str, float]``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, objective_functions, risk_models

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceção
# ---------------------------------------------------------------------------

class OptimizationError(RuntimeError):
    """Levantada quando a otimização falha."""


# ---------------------------------------------------------------------------
# Value object de configuração
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizerConfig:
    """Parâmetros imutáveis do otimizador."""

    method: str               # max_sharpe | min_vol | markowitz
    risk_free_rate_annual: float
    min_weight: float
    max_weight: float
    l2_reg: float = 0.1       # gamma da regularização L2


# ---------------------------------------------------------------------------
# Otimização
# ---------------------------------------------------------------------------

def optimize_weights(
    prices_wide: pd.DataFrame,
    exp_returns: Optional[pd.Series],
    cfg: OptimizerConfig,
) -> Dict[str, float]:
    """Calcula pesos ótimos para o portfólio.

    Args:
        prices_wide: Preços com colunas = tickers e índice = datas.
        exp_returns: Retornos esperados por ticker (do XGBoost).
            Se ``None``, usa média histórica anualizada.
        cfg: Configuração do otimizador (:class:`OptimizerConfig`).

    Returns:
        Dicionário ``{ticker: peso}`` com pesos limpos (somam 1.0,
        pesos < 1e-4 zerados).

    Raises:
        OptimizationError: Se a otimização falhar ou houver < 2 ativos.
    """
    prices = prices_wide.dropna(how="all").ffill().dropna(axis=1, how="any")

    if prices.shape[1] < 2:
        raise OptimizationError("Otimização requer pelo menos 2 ativos sem NaN.")

    # Expected returns: XGBoost ou média histórica
    if exp_returns is not None:
        mu = exp_returns.reindex(prices.columns).astype(float)
        # Fallback para média histórica em tickers sem predição
        hist_mu = expected_returns.mean_historical_return(prices)
        mu = mu.fillna(hist_mu)
    else:
        mu = expected_returns.mean_historical_return(prices)

    # Covariância com Ledoit-Wolf shrinkage (mais estável que sample_cov)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=(cfg.min_weight, cfg.max_weight))

    # Regularização L2 para evitar concentração extrema
    if cfg.l2_reg > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=cfg.l2_reg)

    try:
        if cfg.method == "min_vol":
            ef.min_volatility()
        elif cfg.method == "max_sharpe":
            ef.max_sharpe(risk_free_rate=cfg.risk_free_rate_annual)
        elif cfg.method == "markowitz":
            ef.max_quadratic_utility()
        else:
            raise OptimizationError(f"Método desconhecido: {cfg.method}")

        weights = ef.clean_weights()

        perf = ef.portfolio_performance(
            risk_free_rate=cfg.risk_free_rate_annual, verbose=False
        )
        logger.info(
            "Otimização concluída",
            extra={
                "method": cfg.method,
                "exp_return": f"{perf[0]:.2%}",
                "volatility": f"{perf[1]:.2%}",
                "sharpe": f"{perf[2]:.2f}",
                "n_ativos_ativos": sum(1 for v in weights.values() if v > 1e-4),
            },
        )
        return dict(weights)

    except Exception as exc:
        raise OptimizationError(f"Otimização falhou ({cfg.method}): {exc}") from exc
