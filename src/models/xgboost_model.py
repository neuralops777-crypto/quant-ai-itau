"""Modelo XGBoost para previsão de retorno futuro.

Treina um ``XGBRegressor`` para prever o retorno acumulado em
``horizon_days`` dias úteis à frente.  As predições da última
observação de cada ticker alimentam o otimizador de portfólio
como *expected returns*.

Fluxo:
    ``train_xgboost(df, ...)`` → :class:`XGBResult`
    ``predict_expected_returns_latest(model, feature_df)`` → ``pd.Series``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DROP_COLS = frozenset({"date", "ticker", "llm_summary"})


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XGBResult:
    """Modelo XGBoost treinado e suas métricas de validação."""

    model: XGBRegressor
    metrics: Dict[str, float]
    feature_names: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    drop = _DROP_COLS | {target_col}
    X = (
        df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = df[target_col].astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_xgboost(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    xgb_params: Dict[str, Any],
) -> XGBResult:
    """Treina XGBRegressor com split temporal e avalia no período mais recente.

    Utiliza ``early_stopping_rounds`` quando existe conjunto de validação
    suficientemente grande (> 50 amostras).

    Args:
        df: Dataset de features (long format).
        target_col: Nome da coluna target.
        test_size: Fração de teste (split temporal).
        random_state: Semente.
        xgb_params: Hiperparâmetros do XGBoost (do YAML).

    Returns:
        :class:`XGBResult` com modelo, métricas e nomes de features.
    """
    X, y = _prepare_xy(df, target_col)
    n = int(len(X) * (1.0 - test_size))
    X_tr, X_te = X.iloc[:n], X.iloc[n:]
    y_tr, y_te = y.iloc[:n], y.iloc[n:]

    model = XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **xgb_params,
    )

    fit_kwargs: Dict[str, Any] = {}
    if len(X_te) > 50:
        fit_kwargs["eval_set"] = [(X_te, y_te)]
        fit_kwargs["verbose"] = False

    model.fit(X_tr, y_tr, **fit_kwargs)

    pred = model.predict(X_te)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
    }

    logger.info("XGBoost treinado", extra={"metrics": metrics})

    return XGBResult(
        model=model,
        metrics=metrics,
        feature_names=tuple(X.columns.tolist()),
    )


# ---------------------------------------------------------------------------
# Inferência — expected returns
# ---------------------------------------------------------------------------

def predict_expected_returns_latest(
    result: XGBResult,
    feature_df: pd.DataFrame,
    asof_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Prediz retorno esperado usando a observação mais recente de cada ticker.

    Args:
        result: :class:`XGBResult` com modelo treinado.
        feature_df: Dataset com colunas ``date`` e ``ticker``.
        asof_date: Corte temporal opcional (inclusive).

    Returns:
        Série indexada por ticker com retorno esperado predito.
    """
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if asof_date is not None:
        df = df[df["date"] <= asof_date]

    latest = df.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)

    drop = _DROP_COLS
    X = (
        latest.drop(columns=[c for c in drop if c in latest.columns], errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Garante mesma ordem de features do treino
    X = X.reindex(columns=list(result.feature_names), fill_value=0.0)

    preds = result.model.predict(X)
    return pd.Series(preds, index=latest["ticker"].values, name="exp_ret")
