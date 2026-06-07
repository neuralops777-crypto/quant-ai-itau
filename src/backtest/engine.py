"""Engine de backtest com rebalanceamento periódico e custos de transação.

Design:
    * Aceita ``weights_fn`` como callable — permite reotimizar a cada
      rebalanceamento com dados apenas do passado, evitando look-ahead bias.
    * Custo aplicado **antes** de atualizar os pesos e **antes** de calcular
      o retorno do dia — garante ordem correta de operações.
    * Slippage modelado separadamente do custo de corretagem.
    * Suporta qualquer frequência de rebalanceamento via ``pandas.resample``.

Fluxo:
    ``run_backtest(prices, weights_fn, cfg)`` → DataFrame com colunas
    ``equity, ret, turnover, cost``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Value object de configuração
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """Parâmetros imutáveis do backtest."""

    rebalance: str            # ex.: 'M' (mensal), 'Q' (trimestral)
    initial_capital: float
    transaction_cost_bps: float   # custo de corretagem em bps
    slippage_bps: float = 5.0     # slippage em bps


# Tipo do callable de pesos
WeightsFn = Callable[[pd.DataFrame], Dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FREQ_ALIASES = {
    "M": "ME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
    "BM": "BME",
}
def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """Retorna conjunto de datas de rebalanceamento."""
    freq = _FREQ_ALIASES.get(freq.upper(), freq)  # normaliza alias
    s = pd.Series(1, index=index)
    return set(s.resample(freq).last().dropna().index.to_pydatetime().tolist())


def _normalize_weights(w: Dict[str, float]) -> pd.Series:
    """Clipa negativos e normaliza para soma = 1."""
    s = pd.Series(w).clip(lower=0.0)
    total = s.sum()
    return s / total if total > 0 else s


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    weights_fn: WeightsFn,
    cfg: BacktestConfig,
    warmup_days: int = 63,
) -> pd.DataFrame:
    """Executa o backtest com rebalanceamento e custos.

    Args:
        prices: DataFrame largo (datas × tickers) de preços ajustados.
        weights_fn: Callable que recebe janela histórica de preços e
            retorna ``Dict[ticker, peso]``.  É chamado apenas em datas
            de rebalanceamento, **usando somente dados passados**.
        cfg: Configuração do backtest (:class:`BacktestConfig`).
        warmup_days: Mínimo de dias antes do primeiro rebalanceamento
            para garantir indicadores técnicos válidos.

    Returns:
        DataFrame com índice de datas e colunas:
        ``equity, ret, turnover, cost``.
    """
    px = prices.copy().sort_index().ffill().dropna(how="all")
    tickers = list(px.columns)

    rebal_set = _rebalance_dates(px.index, cfg.rebalance)
    total_cost_factor = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0

    equity = cfg.initial_capital
    # Inicia igualmente ponderado (ou zero — será ajustado no 1º rebal)
    current_w = pd.Series(0.0, index=tickers)

    records = []
    ret_series = px.pct_change().fillna(0.0)

    for i, dt in enumerate(px.index):
        dt_py = dt.to_pydatetime()
        turnover = 0.0
        cost = 0.0

        # Rebalanceamento: usa APENAS dados até dt (exclusive) para evitar look-ahead
        if dt_py in rebal_set and i >= warmup_days:
            window = px.iloc[:i]   # histórico até o dia anterior inclusive
            try:
                new_weights = weights_fn(window)
                new_w = _normalize_weights(new_weights).reindex(tickers).fillna(0.0)
                turnover = float((new_w - current_w).abs().sum())

                # Custo ANTES de atualizar pesos e ANTES do retorno do dia
                cost = equity * turnover * total_cost_factor
                equity -= cost
                current_w = new_w
            except Exception as exc:
                logger.warning(
                    "Falha no rebalanceamento; mantendo pesos anteriores",
                    extra={"date": str(dt), "error": str(exc)},
                )

        # Retorno do dia com pesos correntes
        day_ret = float((current_w * ret_series.loc[dt].reindex(tickers).fillna(0.0)).sum())
        equity *= (1.0 + day_ret)

        records.append(
            {
                "date": dt,
                "equity": equity,
                "ret": day_ret,
                "turnover": turnover,
                "cost": cost,
            }
        )

    out = pd.DataFrame(records).set_index("date")
    logger.info(
        "Backtest concluído",
        extra={
            "dias": len(out),
            "equity_final": round(equity, 2),
            "custo_total": round(out["cost"].sum(), 2),
        },
    )
    return out
