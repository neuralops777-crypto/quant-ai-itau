"""Modelos baseline para previsão de retornos.

Treina LinearRegression e RandomForestRegressor como benchmarks
para comparação com o modelo XGBoost principal.

Fluxo:
    ``train_baselines(df, ...)`` → lista de :class:`ModelResult`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelResult:
    """Container imutável para modelo treinado e suas métricas."""

    name: str
    model: Any
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DROP_COLS = frozenset({"date", "ticker", "llm_summary"})


def _prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X e y a partir do dataset de features.

    Args:
        df: Dataset completo (long format).
        target_col: Nome da coluna target.

    Returns:
        Tupla ``(X, y)`` prontos para sklearn.
    """
    drop = _DROP_COLS | {target_col}
    X = (
        df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = df[target_col].astype(float)
    return X, y


def _temporal_split(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split temporal (sem shuffle) para evitar look-ahead bias."""
    n = int(len(X) * (1.0 - test_size))
    return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def train_baselines(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    rf_params: Dict[str, Any],
    lr_params: Dict[str, Any],
) -> List[ModelResult]:
    """Treina modelos baseline e avalia com split temporal.

    Args:
        df: Dataset de features (long format).
        target_col: Nome da coluna target.
        test_size: Fração de teste (ex.: 0.2 → 20 % mais recentes).
        random_state: Semente para reprodutibilidade.
        rf_params: Hiperparâmetros do RandomForestRegressor.
        lr_params: Hiperparâmetros do LinearRegression.

    Returns:
        Lista de :class:`ModelResult` com modelos e métricas.
    """
    X, y = _prepare_xy(df, target_col)
    X_tr, X_te, y_tr, y_te = _temporal_split(X, y, test_size)

    results: List[ModelResult] = []

    # Linear Regression
    lr = LinearRegression(**lr_params)
    lr.fit(X_tr, y_tr)
    pred_lr = lr.predict(X_te)
    results.append(
        ModelResult(
            name="linear_regression",
            model=lr,
            metrics=_eval_metrics(y_te.values, pred_lr),
        )
    )

    # Random Forest
    rf = RandomForestRegressor(
        random_state=random_state, n_jobs=-1, **rf_params
    )
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    results.append(
        ModelResult(
            name="random_forest",
            model=rf,
            metrics=_eval_metrics(y_te.values, pred_rf),
        )
    )

    for r in results:
        logger.info(
            "Baseline treinado",
            extra={"model": r.name, "metrics": r.metrics},
        )

    return results
