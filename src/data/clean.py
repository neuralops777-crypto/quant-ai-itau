"""Limpeza e padronização de dados de mercado.

Estratégia de limpeza:
    1. Achatar colunas MultiIndex do Yahoo Finance.
    2. Forward-fill + drop de colunas com > 20 % de NaN.
    3. **Winsorização** nos log-retornos (percentis 1–99) — mais robusta
       do que remoção por Z-score ou zeragem de outliers.
    4. Reconstrução dos preços ajustados a partir dos retornos limpos.
    5. Reindexação para dias úteis contínuos (bdate_range).
    6. Limpeza e forward-fill do macro.

Fluxo:
    ``run_cleaning(raw_dir, processed_dir)``
    → ``data/processed/prices_clean.parquet``
    → ``data/processed/macro_clean.parquet``
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceção
# ---------------------------------------------------------------------------

class DataCleaningError(RuntimeError):
    """Levantada quando a etapa de limpeza falha."""


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte MultiIndex de colunas do Yahoo Finance em nomes planos.

    Exemplo: ``('Adj Close', 'PETR4.SA')`` → ``'Adj Close__PETR4.SA'``.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "__".join(str(lvl) for lvl in col).strip("_")
            for col in df.columns.to_list()
        ]
    return df


def _winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Aplica winsorização simétrica aos percentis informados.

    Args:
        series: Série numérica de entrada.
        lower_q: Percentil inferior (ex.: 0.01 → 1 %).
        upper_q: Percentil superior (ex.: 0.99 → 99 %).

    Returns:
        Série com valores clampados entre os percentis.
    """
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexar para dias úteis contínuos, forward-filling gaps.

    Args:
        df: DataFrame com DatetimeIndex.

    Returns:
        DataFrame reindexado e forward-filled.
    """
    bidx = pd.bdate_range(df.index.min(), df.index.max())
    return df.reindex(bidx).ffill()


# ---------------------------------------------------------------------------
# Limpeza de preços
# ---------------------------------------------------------------------------

def clean_prices(raw_prices_path: Path) -> pd.DataFrame:
    """Limpa os preços brutos do Yahoo Finance.

    Passos:
        * Flatten de colunas MultiIndex.
        * Seleciona colunas ``Adj Close`` e ``Volume``.
        * Remove colunas com > 20 % de NaN, forward-fill nas demais.
        * Winsorização dos log-retornos e reconstrução dos preços.
        * Reindexação para dias úteis contínuos.

    Args:
        raw_prices_path: Caminho para ``data/raw/prices.parquet``.

    Returns:
        DataFrame limpo com colunas ``Adj Close__<ticker>`` e
        ``Volume__<ticker>``.

    Raises:
        DataCleaningError: Se o parquet estiver vazio.
    """
    df = pd.read_parquet(raw_prices_path)
    if df.empty:
        raise DataCleaningError("Parquet de preços brutos está vazio.")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = _flatten_yahoo_columns(df)

    adj_cols = [c for c in df.columns if c.startswith("Adj Close__")]
    vol_cols = [c for c in df.columns if c.startswith("Volume__")]

    if not adj_cols:
        raise DataCleaningError("Nenhuma coluna 'Adj Close' encontrada após flatten.")

    df = df[adj_cols + vol_cols].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove colunas com muitos NaN antes do forward-fill
    min_valid = int(len(df) * 0.80)
    df = df.dropna(axis=1, thresh=min_valid)

    df = df.ffill().bfill()

    # Atualiza lista após possível drop de colunas
    adj_cols = [c for c in df.columns if c.startswith("Adj Close__")]

    # Winsorização nos log-retornos → reconstrução dos preços
    adj = df[adj_cols].copy()
    logret = np.log(adj).diff()
    logret_clean = logret.apply(_winsorize, axis=0)

    # Reconstrução: acumulação a partir do primeiro preço observado
    first_valid = adj.apply(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan)
    adj_clean = np.exp(logret_clean.cumsum()).mul(first_valid, axis=1)
    df[adj_cols] = adj_clean

    df = _normalize_index(df)

    logger.info(
        "Preços limpos",
        extra={"shape": str(df.shape), "tickers": len(adj_cols)},
    )
    return df


# ---------------------------------------------------------------------------
# Limpeza de macro
# ---------------------------------------------------------------------------

def clean_macro(raw_macro_path: Path) -> pd.DataFrame:
    """Limpa as séries macroeconômicas do BCB.

    Args:
        raw_macro_path: Caminho para ``data/raw/macro.parquet``.

    Returns:
        DataFrame com índice de dias úteis e séries forward-filled.

    Raises:
        DataCleaningError: Se o parquet estiver vazio.
    """
    macro = pd.read_parquet(raw_macro_path)
    if macro.empty:
        raise DataCleaningError("Parquet de macro bruto está vazio.")

    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    macro = (
        macro.sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    macro = _normalize_index(macro)

    logger.info("Macro limpo", extra={"shape": str(macro.shape)})
    return macro


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_cleaning(raw_dir: Path, processed_dir: Path) -> Tuple[Path, Path]:
    """Executa o pipeline completo de limpeza e persiste os resultados.

    Args:
        raw_dir: Diretório com os parquets brutos.
        processed_dir: Diretório de destino dos parquets processados.

    Returns:
        Tupla ``(clean_prices_path, clean_macro_path)``.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    prices = clean_prices(raw_dir / "prices.parquet")
    macro = clean_macro(raw_dir / "macro.parquet")

    prices_path = processed_dir / "prices_clean.parquet"
    macro_path = processed_dir / "macro_clean.parquet"

    prices.to_parquet(prices_path)
    macro.to_parquet(macro_path)

    logger.info(
        "Limpeza concluída",
        extra={"prices": str(prices_path), "macro": str(macro_path)},
    )
    return prices_path, macro_path
