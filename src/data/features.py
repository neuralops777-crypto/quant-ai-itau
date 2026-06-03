"""Engenharia de features técnicas, quantitativas e macro.

Para cada ticker gera um DataFrame tabular com:
    * Indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger Bands.
    * Indicadores quantitativos: volatilidade, Sharpe rolling, drawdown, beta.
    * Macro (BCB): nível + variação percentual em 21 dias.
    * Target: retorno forward em ``horizon_days`` dias úteis.

Fluxo:
    ``run_features(processed_dir, horizon_days)``
    → ``data/processed/features.parquet``
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceção
# ---------------------------------------------------------------------------

class FeatureEngineeringError(RuntimeError):
    """Levantada quando a geração de features falha."""


# ---------------------------------------------------------------------------
# Indicadores técnicos (funções puras, testáveis individualmente)
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (método Wilder).

    Args:
        series: Série de preços de fechamento.
        period: Janela RSI.

    Returns:
        Série RSI em [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (EMA) sem ajuste de inicialização."""
    return series.ewm(span=span, adjust=False).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD linha, sinal e histograma.

    Returns:
        Tupla ``(macd_line, signal_line, histogram)``.
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(
    series: pd.Series, window: int = 20, n_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger.

    Returns:
        Tupla ``(lower, mid, upper)``.
    """
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return mid - n_std * std, mid, mid + n_std * std


def rolling_beta(
    asset_ret: pd.Series, mkt_ret: pd.Series, window: int = 252
) -> pd.Series:
    """Beta rolling contra o mercado.

    Args:
        asset_ret: Retornos diários do ativo.
        mkt_ret: Retornos diários do índice de mercado.
        window: Janela em dias úteis.

    Returns:
        Série de beta rolling.
    """
    cov = asset_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def max_drawdown_series(prices: pd.Series) -> pd.Series:
    """Drawdown contínuo a partir do pico acumulado.

    Returns:
        Série de drawdown em [-1, 0].
    """
    peak = prices.cummax()
    return prices / peak - 1.0


# ---------------------------------------------------------------------------
# Construção da tabela de features por ticker
# ---------------------------------------------------------------------------

def _build_ticker_features(
    px: pd.Series,
    vol: pd.Series,
    mkt_ret: pd.Series,
    horizon_days: int,
    feat_cfg: dict,
) -> pd.DataFrame:
    """Gera todas as features para um único ticker.

    Args:
        px: Série de preços Adj Close.
        vol: Série de volume.
        mkt_ret: Retornos do índice de mercado (para beta).
        horizon_days: Janela do target forward.
        feat_cfg: Sub-dict ``features`` do model.yaml.

    Returns:
        DataFrame com features e coluna ``target_ret_{horizon_days}d``.
    """
    ret1 = px.pct_change()
    lr = np.log(px).diff()

    # Médias móveis
    sma_feats = {f"sma_{w}": px.rolling(w).mean() for w in feat_cfg["sma_windows"]}
    ema_feats = {f"ema_{w}": ema(px, w) for w in feat_cfg["ema_windows"]}

    # Osciladores
    rsi14 = rsi(px, feat_cfg["rsi_window"])
    macd_l, macd_s, macd_h = macd(
        px, feat_cfg["macd_fast"], feat_cfg["macd_slow"], feat_cfg["macd_signal"]
    )
    bb_lo, bb_mid, bb_up = bollinger(px, feat_cfg["bb_window"], feat_cfg["bb_std"])

    # Posição relativa nas bandas de Bollinger
    bb_pct = (px - bb_lo) / (bb_up - bb_lo).replace(0.0, np.nan)

    # Quantitativos
    vol_ann = lr.rolling(feat_cfg["volatility_window"]).std(ddof=0) * np.sqrt(252)
    sharpe_roll = (
        ret1.rolling(63).mean() / ret1.rolling(63).std(ddof=0).replace(0.0, np.nan)
    ) * np.sqrt(252)
    dd = max_drawdown_series(px)
    dd_60 = dd.rolling(60).min()
    beta_252 = rolling_beta(ret1, mkt_ret, 252)

    # Target: retorno forward
    target = px.shift(-horizon_days) / px - 1.0

    df = pd.DataFrame(
        {
            "px": px,
            "ret_1d": ret1,
            "logret_1d": lr,
            "volume": vol,
            **sma_feats,
            **ema_feats,
            "rsi": rsi14,
            "macd": macd_l,
            "macd_signal": macd_s,
            "macd_hist": macd_h,
            "bb_lower": bb_lo,
            "bb_mid": bb_mid,
            "bb_upper": bb_up,
            "bb_pct": bb_pct,
            "vol_ann": vol_ann,
            "sharpe_roll": sharpe_roll,
            "drawdown": dd,
            "maxdd_60": dd_60,
            "beta_252": beta_252,
            f"target_ret_{horizon_days}d": target,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_feature_table(
    prices_clean_path: Path,
    macro_clean_path: Path,
    horizon_days: int,
    feat_cfg: dict,
    market_index_ticker: str = "^BVSP",
) -> pd.DataFrame:
    """Constrói a tabela de features para todos os tickers.

    Args:
        prices_clean_path: Parquet de preços limpos.
        macro_clean_path: Parquet de macro limpo.
        horizon_days: Janela do retorno alvo.
        feat_cfg: Configuração de features (dict do model.yaml).
        market_index_ticker: Ticker do índice de mercado para cálculo de beta.

    Returns:
        DataFrame long com colunas ``date``, ``ticker``, features e target.
    """
    df = pd.read_parquet(prices_clean_path)
    macro = pd.read_parquet(macro_clean_path)

    if df.empty:
        raise FeatureEngineeringError("Preços limpos estão vazios.")

    adj_cols = [c for c in df.columns if c.startswith("Adj Close__")]
    vol_cols = [c for c in df.columns if c.startswith("Volume__")]

    tickers: List[str] = [c.split("__", 1)[1] for c in adj_cols]
    adj = df[adj_cols].copy()
    adj.columns = tickers
    vol_df = df[vol_cols].copy()
    vol_df.columns = [c.split("__", 1)[1] for c in vol_cols]

    # Retorno de mercado (fallback: média ponderada simples)
    if market_index_ticker in adj.columns:
        mkt_ret = adj[market_index_ticker].pct_change()
    else:
        mkt_ret = adj.mean(axis=1).pct_change()

    rows: List[pd.DataFrame] = []
    for t in tickers:
        px = adj[t]
        v = vol_df[t] if t in vol_df.columns else pd.Series(np.nan, index=adj.index)
        feat = _build_ticker_features(px, v, mkt_ret, horizon_days, feat_cfg)
        feat.insert(0, "ticker", t)
        feat.insert(0, "date", adj.index)
        rows.append(feat)

    out = pd.concat(rows, axis=0, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)

    # Merge macro (as-of join por data)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    macro_reset = macro.reset_index().rename(columns={"index": "date"})
    out = (
        out.sort_values("date")
        .merge(macro_reset, on="date", how="left")
        .ffill()
    )

    # Variação percentual das séries macro (21 dias úteis)
    for c in macro.columns:
        out[f"{c}_chg21"] = out.groupby("ticker")[c].transform(
            lambda s: s.pct_change(21)  # noqa: B023
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    target_col = f"target_ret_{horizon_days}d"
    out = out.dropna(subset=[target_col])

    logger.info(
        "Features geradas",
        extra={"rows": len(out), "cols": len(out.columns), "tickers": len(tickers)},
    )
    return out


def run_features(processed_dir: Path, horizon_days: int, feat_cfg: dict) -> Path:
    """Gera e persiste a tabela de features.

    Args:
        processed_dir: Diretório de processados.
        horizon_days: Janela do retorno alvo.
        feat_cfg: Configuração de features.

    Returns:
        Caminho do parquet gerado.
    """
    features = build_feature_table(
        prices_clean_path=processed_dir / "prices_clean.parquet",
        macro_clean_path=processed_dir / "macro_clean.parquet",
        horizon_days=horizon_days,
        feat_cfg=feat_cfg,
    )
    path = processed_dir / "features.parquet"
    features.to_parquet(path)
    logger.info("Features salvas", extra={"path": str(path)})
    return path
