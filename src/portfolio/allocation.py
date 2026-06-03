"""Alocação de capital e análise de exposição setorial.

Converte pesos contínuos em quantidades discretas de ações e
agrega exposição por setor para análise de concentração.

Fluxo:
    ``Allocator.summary(weights, prices, capital, sector_map)``
    → ``pd.DataFrame`` com colunas ticker, setor, peso, quantidade, valor (R$).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

class Allocator:
    """Converte pesos contínuos em alocação discreta com análise setorial."""

    def __init__(self, sector_map: Optional[Dict[str, str]] = None) -> None:
        """Inicializa o alocador.

        Args:
            sector_map: Mapeamento ``{ticker: setor}``.
                Se ``None``, todos os tickers recebem ``'Outros'``.
        """
        self.sector_map = sector_map or {}

    # ------------------------------------------------------------------
    def discrete_allocation(
        self,
        weights: Dict[str, float],
        latest_prices: pd.Series,
        total_capital: float,
    ) -> Dict[str, int]:
        """Calcula quantidades inteiras de ações por ticker.

        Usa alocação greedy (floor de shares) — o saldo residual
        fica em caixa.

        Args:
            weights: Pesos ``{ticker: w}`` (somam ≤ 1.0).
            latest_prices: Últimos preços disponíveis por ticker.
            total_capital: Capital total em R$.

        Returns:
            Dicionário ``{ticker: quantidade_inteira}``.
        """
        alloc: Dict[str, int] = {}
        for ticker, w in weights.items():
            price = float(latest_prices.get(ticker, 0.0))
            if price <= 0 or w <= 0:
                alloc[ticker] = 0
                continue
            alloc[ticker] = int((w * total_capital) // price)
        return alloc

    # ------------------------------------------------------------------
    def sector_exposure(self, weights: Dict[str, float]) -> pd.DataFrame:
        """Agrega exposição por setor.

        Args:
            weights: Pesos por ticker.

        Returns:
            DataFrame com colunas ``sector`` e ``exposure`` (soma dos pesos).
        """
        rows: List[Dict] = []
        for ticker, w in weights.items():
            rows.append(
                {"sector": self.sector_map.get(ticker, "Outros"), "ticker": ticker, "weight": float(w)}
            )
        df = pd.DataFrame(rows)
        return (
            df.groupby("sector", as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "exposure"})
            .sort_values("exposure", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def weights_to_table(self, weights: Dict[str, float]) -> pd.DataFrame:
        """Converte pesos em DataFrame ordenado.

        Args:
            weights: Pesos por ticker.

        Returns:
            DataFrame com colunas ``ticker``, ``sector``, ``weight``.
        """
        rows = [
            {
                "ticker": t,
                "sector": self.sector_map.get(t, "Outros"),
                "weight": round(float(w), 4),
            }
            for t, w in weights.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def summary(
        self,
        weights: Dict[str, float],
        latest_prices: pd.Series,
        total_capital: float,
    ) -> pd.DataFrame:
        """Tabela consolidada: peso, setor, quantidade e valor financeiro.

        Args:
            weights: Pesos por ticker.
            latest_prices: Últimos preços disponíveis.
            total_capital: Capital total em R$.

        Returns:
            DataFrame com colunas:
            ``ticker, sector, weight, price, quantity, value_brl``.
        """
        qty = self.discrete_allocation(weights, latest_prices, total_capital)
        rows = []
        for ticker, w in weights.items():
            price = float(latest_prices.get(ticker, 0.0))
            q = qty.get(ticker, 0)
            rows.append(
                {
                    "ticker": ticker,
                    "sector": self.sector_map.get(ticker, "Outros"),
                    "weight": round(w, 4),
                    "price_brl": round(price, 2),
                    "quantity": q,
                    "value_brl": round(q * price, 2),
                }
            )
        df = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

        allocated = df["value_brl"].sum()
        cash = total_capital - allocated
        logger.info(
            "Alocação calculada",
            extra={
                "capital": total_capital,
                "alocado": round(allocated, 2),
                "caixa_residual": round(cash, 2),
            },
        )
        return df
