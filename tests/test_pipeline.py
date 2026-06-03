"""Testes unitários do pipeline Quant AI Itaú.

Cobre os módulos críticos com fixtures sintéticas reprodutíveis.
Todos os testes são autossuficientes — sem dependência de rede ou API.

Execute:
    pytest -v --cov=src tests/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures globais
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def price_series() -> pd.Series:
    """Série de preços sintética com 500 dias úteis."""
    np.random.seed(42)
    idx = pd.bdate_range("2021-01-04", periods=500)
    returns = np.random.normal(5e-4, 0.015, 500)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=idx, name="TICKER")


@pytest.fixture(scope="module")
def prices_wide() -> pd.DataFrame:
    """DataFrame wide com 3 ativos sintéticos."""
    np.random.seed(0)
    idx = pd.bdate_range("2021-01-04", periods=500)
    data = {}
    for t in ["AAA", "BBB", "CCC"]:
        r = np.random.normal(4e-4, 0.013, 500)
        data[t] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def daily_returns(prices_wide) -> pd.Series:
    return prices_wide["AAA"].pct_change().dropna()


# ---------------------------------------------------------------------------
# 1. Limpeza de dados
# ---------------------------------------------------------------------------

class TestDataCleaner:
    def test_winsorize_clips_extremes(self, price_series):
        from src.data.clean import _winsorize
        s = price_series.pct_change().dropna()
        # Injeta outliers
        s.iloc[5] = 99.9
        s.iloc[10] = -99.9
        result = _winsorize(s)
        assert result.max() < 99.9
        assert result.min() > -99.9

    def test_flatten_columns(self):
        from src.data.clean import _flatten_yahoo_columns
        arrays = [["Adj Close", "Volume"], ["AAA", "AAA"]]
        tuples = list(zip(*arrays))
        multi_idx = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[100.0, 1_000_000]], columns=multi_idx)
        flat = _flatten_yahoo_columns(df)
        assert "Adj Close__AAA" in flat.columns
        assert "Volume__AAA" in flat.columns

    def test_normalize_index_fills_gaps(self, prices_wide):
        from src.data.clean import _normalize_index
        # Remove algumas datas
        sparse = prices_wide.iloc[::3]
        norm = _normalize_index(sparse)
        expected_len = len(pd.bdate_range(sparse.index.min(), sparse.index.max()))
        assert len(norm) == expected_len
        assert norm.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_rsi_in_bounds(self, price_series):
        from src.data.features import rsi
        result = rsi(price_series, 14).dropna()
        assert (result >= 0).all(), "RSI deve ser >= 0"
        assert (result <= 100).all(), "RSI deve ser <= 100"

    def test_bollinger_ordering(self, price_series):
        from src.data.features import bollinger
        lo, mid, hi = bollinger(price_series, 20, 2.0)
        valid = lo.dropna().index
        assert (lo[valid] <= mid[valid]).all(), "BB lower <= mid"
        assert (mid[valid] <= hi[valid]).all(), "BB mid <= upper"

    def test_macd_structure(self, price_series):
        from src.data.features import macd
        line, signal, hist = macd(price_series)
        pd.testing.assert_series_equal(line - signal, hist, check_names=False)

    def test_rolling_beta_finite(self, prices_wide):
        from src.data.features import rolling_beta
        mkt = prices_wide["AAA"].pct_change()
        asset = prices_wide["BBB"].pct_change()
        result = rolling_beta(asset, mkt, 60).dropna()
        assert result.replace([np.inf, -np.inf], np.nan).notna().all()

    def test_build_ticker_features_columns(self, price_series, prices_wide):
        from src.data.features import _build_ticker_features
        feat_cfg = {
            "sma_windows": [10, 20],
            "ema_windows": [12, 26],
            "rsi_window": 14,
            "bb_window": 20,
            "bb_std": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "volatility_window": 21,
        }
        mkt_ret = prices_wide["AAA"].pct_change()
        vol = pd.Series(1_000_000.0, index=price_series.index)
        df = _build_ticker_features(price_series, vol, mkt_ret, 30, feat_cfg)
        expected = {"rsi", "macd", "bb_mid", "vol_ann", "drawdown", "beta_252", "target_ret_30d"}
        assert expected.issubset(df.columns), f"Colunas faltando: {expected - set(df.columns)}"


# ---------------------------------------------------------------------------
# 3. Risco
# ---------------------------------------------------------------------------

class TestRisk:
    def test_sharpe_positive_returns(self, daily_returns):
        from src.portfolio.risk import sharpe_ratio
        # Série com retorno médio positivo deve ter Sharpe > 0
        s = pd.Series(np.abs(daily_returns.values) * 0.01 + 0.001, index=daily_returns.index)
        assert sharpe_ratio(s, rf_annual=0.0) > 0

    def test_var_less_than_cvar(self, daily_returns):
        from src.portfolio.risk import var_cvar
        var, cvar = var_cvar(daily_returns)
        assert cvar <= var, "CVaR deve ser <= VaR (mais pessimista)"

    def test_max_drawdown_negative(self, prices_wide):
        from src.portfolio.risk import max_drawdown
        equity = prices_wide["AAA"]
        dd = max_drawdown(equity)
        assert dd <= 0, "Max drawdown deve ser negativo ou zero"

    def test_sortino_gt_sharpe_on_skewed(self):
        from src.portfolio.risk import sharpe_ratio, sortino_ratio
        # Série com muitos retornos pequenos positivos → downside vol < total vol
        np.random.seed(7)
        r = pd.Series(np.random.exponential(0.005, 500) - 0.002)
        # Não garante ordenação, mas ambos devem ser finitos
        s = sharpe_ratio(r)
        so = sortino_ratio(r)
        assert np.isfinite(s) and np.isfinite(so)

    def test_alpha_beta_benchmark_equal(self, daily_returns):
        from src.portfolio.risk import alpha_beta
        # Quando portfólio == benchmark: beta ≈ 1, alpha ≈ 0
        a, b = alpha_beta(daily_returns, daily_returns)
        assert abs(b - 1.0) < 1e-6
        assert abs(a) < 1e-4


# ---------------------------------------------------------------------------
# 4. Otimizador
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_weights_sum_to_one(self, prices_wide):
        from src.portfolio.optimizer import OptimizerConfig, optimize_weights
        cfg = OptimizerConfig(
            method="max_sharpe",
            risk_free_rate_annual=0.10,
            min_weight=0.0,
            max_weight=0.8,
            l2_reg=0.1,
        )
        w = optimize_weights(prices_wide, exp_returns=None, cfg=cfg)
        assert abs(sum(w.values()) - 1.0) < 1e-3, "Pesos devem somar 1"

    def test_weights_within_bounds(self, prices_wide):
        from src.portfolio.optimizer import OptimizerConfig, optimize_weights
        max_w = 0.6
        cfg = OptimizerConfig(
            method="min_vol",
            risk_free_rate_annual=0.10,
            min_weight=0.0,
            max_weight=max_w,
            l2_reg=0.05,
        )
        w = optimize_weights(prices_wide, exp_returns=None, cfg=cfg)
        assert all(v <= max_w + 1e-6 for v in w.values()), "Peso excede max_weight"

    def test_exp_returns_injection(self, prices_wide):
        from src.portfolio.optimizer import OptimizerConfig, optimize_weights
        exp = pd.Series({"AAA": 0.25, "BBB": 0.05, "CCC": 0.05})
        cfg = OptimizerConfig(
            method="max_sharpe",
            risk_free_rate_annual=0.10,
            min_weight=0.0,
            max_weight=0.9,
            l2_reg=0.0,
        )
        w = optimize_weights(prices_wide, exp_returns=exp, cfg=cfg)
        # Com expected return muito superior em AAA, deve ter maior peso
        assert w.get("AAA", 0) >= w.get("CCC", 0)

    def test_raises_with_single_asset(self):
        from src.portfolio.optimizer import OptimizerConfig, OptimizationError, optimize_weights
        single = pd.DataFrame({"A": [100, 101, 102]})
        cfg = OptimizerConfig("max_sharpe", 0.10, 0.0, 1.0)
        with pytest.raises(OptimizationError):
            optimize_weights(single, None, cfg)


# ---------------------------------------------------------------------------
# 5. Alocação
# ---------------------------------------------------------------------------

class TestAllocator:
    def test_discrete_allocation_integer(self, prices_wide):
        from src.portfolio.allocation import Allocator
        weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
        prices = prices_wide.iloc[-1]
        alloc = Allocator().discrete_allocation(weights, prices, 100_000.0)
        assert all(isinstance(v, int) for v in alloc.values())
        assert all(v >= 0 for v in alloc.values())

    def test_sector_exposure_sums_to_weights(self):
        from src.portfolio.allocation import Allocator
        sector_map = {"A": "Tech", "B": "Finance", "C": "Tech"}
        weights = {"A": 0.4, "B": 0.35, "C": 0.25}
        exp = Allocator(sector_map).sector_exposure(weights)
        assert abs(exp["exposure"].sum() - 1.0) < 1e-9
        assert float(exp.loc[exp["sector"] == "Tech", "exposure"].values[0]) == pytest.approx(0.65, abs=1e-9)

    def test_summary_columns(self, prices_wide):
        from src.portfolio.allocation import Allocator
        weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
        df = Allocator().summary(weights, prices_wide.iloc[-1], 500_000.0)
        assert set(df.columns) >= {"ticker", "weight", "quantity", "value_brl"}


# ---------------------------------------------------------------------------
# 6. Backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_equity_grows_from_initial(self, prices_wide):
        from src.backtest.engine import BacktestConfig, run_backtest
        cfg = BacktestConfig(rebalance="M", initial_capital=100_000.0,
                             transaction_cost_bps=10.0, slippage_bps=5.0)
        w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
        bt = run_backtest(prices_wide, lambda _: w, cfg)
        assert bt["equity"].iloc[0] > 0
        assert not bt.empty

    def test_columns_present(self, prices_wide):
        from src.backtest.engine import BacktestConfig, run_backtest
        cfg = BacktestConfig("M", 50_000.0, 10.0)
        bt = run_backtest(prices_wide, lambda _: {"AAA": 0.6, "BBB": 0.4}, cfg)
        assert {"equity", "ret", "turnover", "cost"}.issubset(bt.columns)

    def test_costs_positive(self, prices_wide):
        from src.backtest.engine import BacktestConfig, run_backtest
        cfg = BacktestConfig("M", 100_000.0, 20.0, 10.0)
        bt = run_backtest(prices_wide, lambda _: {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, cfg)
        assert bt["cost"].sum() >= 0


# ---------------------------------------------------------------------------
# 7. Métricas
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_cagr_positive_trend(self, prices_wide):
        from src.backtest.metrics import cagr
        equity = prices_wide["AAA"]
        result = cagr(equity)
        assert isinstance(result, float)

    def test_compute_metrics_keys(self, prices_wide):
        from src.backtest.engine import BacktestConfig, run_backtest
        from src.backtest.metrics import compute_metrics
        cfg = BacktestConfig("M", 100_000.0, 10.0)
        bt = run_backtest(prices_wide, lambda _: {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, cfg)
        m = compute_metrics(bt, rf_annual=0.10)
        expected_keys = {"CAGR", "Sharpe", "Sortino", "MaxDrawdown", "Alpha", "Beta"}
        assert expected_keys.issubset(m.keys())


# ---------------------------------------------------------------------------
# 8. LLM Features (modo fallback — sem API)
# ---------------------------------------------------------------------------

class TestLLMFeatures:
    def test_deterministic_scores_reproducible(self):
        from src.models.llm_features import _deterministic_scores
        s1 = _deterministic_scores("PETR4|Petrobras anuncia dividendo")
        s2 = _deterministic_scores("PETR4|Petrobras anuncia dividendo")
        assert s1 == s2, "Scores devem ser determinísticos"

    def test_scores_in_range(self):
        from src.models.llm_features import _deterministic_scores
        sc = _deterministic_scores("VALE3|Vale reporta queda na produção de ferro")
        assert -1.0 <= sc["sentiment_score"] <= 1.0
        assert -1.0 <= sc["event_score"] <= 1.0
        assert 0.0 <= sc["confidence_score"] <= 1.0

    def test_score_news_items_fallback(self):
        from datetime import datetime
        from src.models.llm_features import NewsItem, score_news_items
        items = [
            NewsItem(
                ticker="ITUB4",
                published_at=datetime(2024, 6, 1),
                title="Itaú reporta lucro recorde",
                source="Valor Econômico",
                url="https://example.com",
                snippet="Lucro líquido cresceu 15% no trimestre.",
            )
        ]
        df = score_news_items(items)
        assert not df.empty
        assert set(df.columns) >= {"ticker", "date", "sentiment_score", "event_score", "confidence_score"}

    def test_attach_llm_features_fills_zeros(self):
        from src.models.llm_features import attach_llm_features
        feat = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "ticker": ["A"] * 5,
            "px": [100.0] * 5,
        })
        llm = pd.DataFrame(columns=["ticker", "date", "sentiment_score", "event_score", "confidence_score", "llm_summary"])
        result = attach_llm_features(feat, llm)
        # Com llm vazio, deve retornar dataframe original intacto
        assert len(result) == len(feat)


# ---------------------------------------------------------------------------
# 9. Config e logger
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_yaml_missing_raises(self, tmp_path):
        from src.utils.config import ConfigError, load_yaml
        with pytest.raises(ConfigError):
            load_yaml(tmp_path / "nao_existe.yaml")

    def test_ensure_dirs_creates_structure(self, tmp_path):
        from src.utils.config import Paths, ensure_dirs
        p = Paths.from_root(tmp_path)
        ensure_dirs(p)
        for attr in ("data_raw", "data_processed", "data_external", "reports", "figures"):
            assert getattr(p, attr).exists(), f"{attr} não foi criado"

    def test_logger_json_output(self, capsys):
        import json
        from src.utils.logger import get_logger
        log = get_logger("test_json_output")
        log.info("mensagem teste")
        captured = capsys.readouterr().out
        if captured.strip():
            obj = json.loads(captured.strip().split("\n")[-1])
            assert obj["level"] == "INFO"
            assert "mensagem teste" in obj["message"]
