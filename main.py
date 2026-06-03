"""Pipeline principal do Quant AI Itaú.

Orquestra as 8 etapas do pipeline end-to-end:
    1. Coleta  (Yahoo Finance + BCB)
    2. Limpeza (winsorização, calendário, NaN)
    3. Features técnicas + quantitativas + macro
    4. Features LLM (sentimento, evento, confiança)
    5. Modelos baseline (Linear + RandomForest)
    6. XGBoost → expected returns
    7. Otimização de portfólio (Ledoit-Wolf + L2_reg)
    8. Backtest + relatório PDF

Uso:
    python main.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.reports import (
    generate_report,
    plot_allocation,
    plot_drawdown,
    plot_equity,
    plot_sector,
)
from src.data.clean import run_cleaning
from src.data.collect import collect_all
from src.data.features import run_features
from src.models.baseline import train_baselines
from src.models.llm_features import NewsItem, attach_llm_features, score_news_items
from src.models.xgboost_model import predict_expected_returns_latest, train_xgboost
from src.portfolio.allocation import Allocator
from src.portfolio.optimizer import OptimizerConfig, optimize_weights
from src.utils.config import Paths, ensure_dirs, load_yaml
from src.utils.logger import get_logger

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prices_wide(processed_dir: Path) -> pd.DataFrame:
    """Carrega preços limpos e retorna matriz wide (datas × tickers)."""
    df = pd.read_parquet(processed_dir / "prices_clean.parquet")
    adj_cols = [c for c in df.columns if c.startswith("Adj Close__")]
    tickers = [c.split("__", 1)[1] for c in adj_cols]
    wide = df[adj_cols].copy()
    wide.columns = tickers
    return wide.dropna(how="all").ffill()


def _mock_news(tickers: List[str], asof: pd.Timestamp) -> List[NewsItem]:
    """Gera notícias sintéticas para demo sem provedor externo."""
    return [
        NewsItem(
            ticker=t,
            published_at=datetime(asof.year, asof.month, asof.day),
            title=f"Atualização corporativa de {t}",
            source="Public",
            url=f"https://example.com/{t}",
            snippet=f"Análise de mercado para {t} — data de referência {asof.date()}.",
        )
        for t in tickers
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Executa o pipeline completo de ponta a ponta."""
    root = Path(__file__).resolve().parent
    paths = Paths.from_root(root)
    ensure_dirs(paths)

    # Carrega configurações
    model_cfg = load_yaml(paths.root / "configs" / "model.yaml")
    strat_cfg = load_yaml(paths.root / "configs" / "strategy.yaml")

    data_cfg = strat_cfg["data"]
    uni = strat_cfg["universe"]
    macro_sgs = strat_cfg["macro_bcb_sgs"]
    port_cfg = strat_cfg["portfolio"]
    bt_cfg_raw = strat_cfg["backtest"]
    sector_map = strat_cfg.get("sector_map", {})

    horizon = int(model_cfg["general"]["horizon_days"])
    test_size = float(model_cfg["general"]["test_size"])
    random_state = int(model_cfg["general"]["random_state"])
    feat_cfg = model_cfg["features"]
    rf_annual = float(port_cfg["risk_free_rate_annual"])

    # ------------------------------------------------------------------ 1
    logger.info("ETAPA 1/8 — Coleta de dados")
    collect_all(
        raw_dir=paths.data_raw,
        tickers_stocks=uni["stocks_br"],
        tickers_etfs=uni["etfs"],
        tickers_indices=uni["indices"],
        macro_sgs=macro_sgs,
        start=data_cfg["start"],
        end=data_cfg["end"],
        interval=data_cfg["interval"],
    )

    # ------------------------------------------------------------------ 2
    logger.info("ETAPA 2/8 — Limpeza")
    run_cleaning(paths.data_raw, paths.data_processed)

    # ------------------------------------------------------------------ 3
    logger.info("ETAPA 3/8 — Engenharia de features")
    feat_path = run_features(paths.data_processed, horizon_days=horizon, feat_cfg=feat_cfg)
    feat = pd.read_parquet(feat_path)

    # ------------------------------------------------------------------ 4
    logger.info("ETAPA 4/8 — Features LLM")
    asof = pd.to_datetime(feat["date"].max())
    tickers_list = sorted(feat["ticker"].unique().tolist())
    news_items = _mock_news(tickers_list, asof)
    llm_df = score_news_items(news_items, model=model_cfg["llm"]["model"])
    feat = attach_llm_features(feat, llm_df)
    feat.to_parquet(paths.data_processed / "features_enriched.parquet")

    # ------------------------------------------------------------------ 5
    logger.info("ETAPA 5/8 — Modelos baseline")
    target_col = f"target_ret_{horizon}d"
    train_baselines(
        feat,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        rf_params=model_cfg["baseline"]["random_forest"],
        lr_params=model_cfg["baseline"]["linear"],
    )

    # ------------------------------------------------------------------ 6
    logger.info("ETAPA 6/8 — XGBoost + expected returns")
    xgb_res = train_xgboost(
        feat,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        xgb_params=model_cfg["xgboost"],
    )
    exp_ret = predict_expected_returns_latest(xgb_res, feat)
    logger.info("Expected returns calculados", extra={"tickers": exp_ret.to_dict()})

    # ------------------------------------------------------------------ 7
    logger.info("ETAPA 7/8 — Otimização de portfólio")
    prices_wide = _load_prices_wide(paths.data_processed)

    opt_cfg = OptimizerConfig(
        method=port_cfg["optimizer"],
        risk_free_rate_annual=rf_annual,
        min_weight=float(port_cfg["min_weight"]),
        max_weight=float(port_cfg["max_weight"]),
        l2_reg=float(port_cfg.get("l2_reg", 0.1)),
    )
    weights = optimize_weights(prices_wide=prices_wide, exp_returns=exp_ret, cfg=opt_cfg)

    allocator = Allocator(sector_map=sector_map)
    w_table = allocator.weights_to_table(weights)
    alloc_summary = allocator.summary(
        weights=weights,
        latest_prices=prices_wide.iloc[-1],
        total_capital=float(bt_cfg_raw["initial_capital"]),
    )
    sector_df = allocator.sector_exposure(weights)

    # ------------------------------------------------------------------ 8a Backtest
    logger.info("ETAPA 8/8 — Backtest + relatório")

    bt_config = BacktestConfig(
        rebalance=bt_cfg_raw["rebalance"],
        initial_capital=float(bt_cfg_raw["initial_capital"]),
        transaction_cost_bps=float(bt_cfg_raw["transaction_cost_bps"]),
        slippage_bps=float(bt_cfg_raw.get("slippage_bps", 5.0)),
    )

    # weights_fn reotimiza a cada rebalanceamento com dados do passado
    def weights_fn(window: pd.DataFrame) -> dict:
        try:
            return optimize_weights(prices_wide=window, exp_returns=exp_ret, cfg=opt_cfg)
        except Exception:
            return weights  # fallback para pesos fixos

    bt = run_backtest(prices=prices_wide, weights_fn=weights_fn, cfg=bt_config)

    # Benchmark
    bench_px = prices_wide.get("^BVSP") or prices_wide.get("BOVA11.SA")
    bench_ret = bench_px.pct_change().fillna(0.0) if bench_px is not None else None

    metrics = compute_metrics(bt, bench_ret=bench_ret, rf_annual=rf_annual)

    # ------------------------------------------------------------------ 8b Relatório
    equity_fig = plot_equity(bt, paths.figures / "equity.png", bench=bench_px)
    dd_fig = plot_drawdown(bt, paths.figures / "drawdown.png")
    alloc_fig = plot_allocation(weights, paths.figures / "allocation.png")
    sector_fig = plot_sector(sector_df, paths.figures / "sector.png")

    llm_insights = (
        "Insights extraídos por IA Generativa:\n\n"
        + "\n".join(
            f"[{row['ticker']}] Sentimento: {row['sentiment_score']:.2f} | "
            f"Evento: {row['event_score']:.2f} | "
            f"Confiança: {row['confidence_score']:.2f}\n{row['llm_summary']}"
            for _, row in llm_df.iterrows()
        )
        if not llm_df.empty
        else None
    )

    generate_report(
        out_pdf=paths.reports / "final_report.pdf",
        metrics=metrics,
        weights_table=w_table,
        alloc_summary=alloc_summary,
        equity_fig=equity_fig,
        dd_fig=dd_fig,
        alloc_fig=alloc_fig,
        sector_fig=sector_fig,
        llm_insights=llm_insights,
    )

    logger.info(
        "Pipeline concluído",
        extra={"pdf": str(paths.reports / "final_report.pdf"), "metricas": metrics},
    )


if __name__ == "__main__":
    main()
