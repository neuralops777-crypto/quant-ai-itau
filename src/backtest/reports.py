"""Geração de gráficos e relatório PDF executivo.

Gera:
    * ``equity.png``    — Curva de equity vs. benchmark.
    * ``drawdown.png``  — Drawdown contínuo.
    * ``allocation.png`` — Pie chart de alocação por ativo.
    * ``sector.png``    — Bar chart de exposição setorial.
    * ``final_report.pdf`` — Relatório consolidado com todas as seções.

Fluxo:
    ``generate_report(...)`` → ``Path`` do PDF gerado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

from src.utils.logger import get_logger

def _safe(text: str) -> str:
    """Remove caracteres fora do range latin-1 para compatibilidade com fpdf."""
    return (
        text
        .replace("\u2014", "-")   # em dash —
        .replace("\u2013", "-")   # en dash –
        .replace("\u2018", "'")   # aspas esquerdas '
        .replace("\u2019", "'")   # aspas direitas '
        .replace("\u201c", '"')   # aspas duplas esquerdas "
        .replace("\u201d", '"')   # aspas duplas direitas "
        .encode("latin-1", errors="replace").decode("latin-1")
    )

logger = get_logger(__name__)

# Paleta consistente com identidade visual financeira
_BLUE = "#003087"
_RED = "#D52B1E"
_GRAY = "#F5F5F5"


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plot_equity(
    bt: pd.DataFrame,
    out_path: Path,
    bench: Optional[pd.Series] = None,
) -> Path:
    """Curva de equity com benchmark opcional.

    Args:
        bt: DataFrame do backtest (coluna ``equity``).
        out_path: Caminho de saída da imagem.
        bench: Série de preços do benchmark para normalização.

    Returns:
        Caminho do arquivo salvo.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))

    eq = bt["equity"]
    eq_norm = eq / eq.iloc[0] * 100
    ax.plot(eq_norm.index, eq_norm, lw=2, color=_BLUE, label="Portfólio")

    if bench is not None:
        b = bench.reindex(eq.index).ffill().dropna()
        b_norm = b / b.iloc[0] * 100
        ax.plot(b_norm.index, b_norm, lw=1.5, color=_GRAY, linestyle="--",
                label="Benchmark", alpha=0.8)
        ax.fill_between(
            eq_norm.index,
            eq_norm,
            b_norm.reindex(eq_norm.index).ffill(),
            where=(eq_norm > b_norm.reindex(eq_norm.index).ffill()),
            alpha=0.12,
            color=_BLUE,
        )

    ax.set_title("Curva de Equity (base 100)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (base 100)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_drawdown(bt: pd.DataFrame, out_path: Path) -> Path:
    """Gráfico de drawdown.

    Args:
        bt: DataFrame do backtest (coluna ``equity``).
        out_path: Caminho de saída.

    Returns:
        Caminho do arquivo salvo.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    equity = bt["equity"]
    dd = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(dd.index, dd * 100, 0, color=_RED, alpha=0.65)
    ax.set_title("Drawdown (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_allocation(weights: Dict[str, float], out_path: Path) -> Path:
    """Pie chart de alocação por ativo.

    Args:
        weights: Pesos por ticker.
        out_path: Caminho de saída.

    Returns:
        Caminho do arquivo salvo.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = {k: v for k, v in sorted(weights.items(), key=lambda x: -x[1]) if v > 1e-4}

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        items.values(),
        labels=items.keys(),
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Alocação por Ativo", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sector(sector_df: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart de exposição setorial.

    Args:
        sector_df: DataFrame com colunas ``sector`` e ``exposure``.
        out_path: Caminho de saída.

    Returns:
        Caminho do arquivo salvo.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sector_df["sector"], sector_df["exposure"] * 100, color=_BLUE, alpha=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlabel("Exposição (%)")
    ax.set_title("Exposição Setorial", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def generate_report(
    out_pdf: Path,
    metrics: Dict[str, float],
    weights_table: pd.DataFrame,
    alloc_summary: pd.DataFrame,
    equity_fig: Path,
    dd_fig: Path,
    alloc_fig: Path,
    sector_fig: Path,
    llm_insights: Optional[str] = None,
) -> Path:
    """Gera o relatório PDF executivo final.

    Args:
        out_pdf: Caminho de saída do PDF.
        metrics: Métricas de backtest.
        weights_table: DataFrame com pesos por ativo.
        alloc_summary: Tabela de alocação discreta com valores em R$.
        equity_fig: Caminho da figura de equity.
        dd_fig: Caminho da figura de drawdown.
        alloc_fig: Caminho do pie chart.
        sector_fig: Caminho do bar chart setorial.
        llm_insights: Texto de insights LLM (opcional).

    Returns:
        Caminho do PDF gerado.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)

    # -------- Capa --------
    pdf.add_page()
    pdf.set_fill_color(0, 48, 135)  # azul Itaú
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_y(12)
    pdf.cell(0, 10, _safe("Quant AI Itau - Relatorio Final"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe("Plataforma de Investimentos com Machine Learning e IA Generativa"), ln=True, align="C")

    pdf.set_text_color(30, 30, 30)
    pdf.set_y(48)

    # -------- Métricas --------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe("1. Metricas de Performance"), ln=True)
    pdf.set_font("Courier", "", 9)

    fmt_map = {
        "CAGR": "{:.2%}", "TotalReturn": "{:.2%}", "AnnReturn": "{:.2%}",
        "AnnVol": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
        "MaxDrawdown": "{:.2%}", "VaR_5%": "{:.2%}", "CVaR_5%": "{:.2%}",
        "AvgTurnover": "{:.2%}", "TotalCosts": "R$ {:.2f}",
        "Alpha": "{:.2%}", "Beta": "{:.2f}",
    }

    col_w = 80
    for k, v in metrics.items():
        fmt = fmt_map.get(k, "{:.4f}")
        try:
            val_str = fmt.format(v)
        except Exception:
            val_str = str(round(v, 4))
        pdf.cell(col_w, 5, f"  {k}", border="B")
        pdf.cell(0, 5, val_str, border="B", ln=True)

    pdf.ln(4)

    # -------- Pesos --------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe("2. Pesos Otimizados"), ln=True)
    pdf.set_font("Courier", "", 8)

    header = ["Ticker", "Setor", "Peso", "Preco (R$)", "Qtd", "Valor (R$)"]
    col_widths = [30, 40, 20, 28, 20, 30]

    # Cabeçalho
    pdf.set_fill_color(0, 48, 135)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(header, col_widths):
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln()

    pdf.set_text_color(30, 30, 30)
    for _, row in alloc_summary.iterrows():
        for val, w in zip(
            [
                str(row.get("ticker", "")),
                str(row.get("sector", "")),
                f"{row.get('weight', 0):.2%}",
                f"{row.get('price_brl', 0):.2f}",
                str(row.get("quantity", 0)),
                f"{row.get('value_brl', 0):,.2f}",
            ],
            col_widths,
        ):
            pdf.cell(w, 5, val, border=1)
        pdf.ln()

    pdf.ln(4)

    # -------- Gráficos --------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe("3. Graficos de Performance"), ln=True)
    pdf.image(str(equity_fig), w=185)
    pdf.ln(2)
    pdf.image(str(dd_fig), w=185)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe("4. Alocacao e Exposicao Setorial"), ln=True)
    pdf.image(str(alloc_fig), x=15, w=90)
    pdf.image(str(sector_fig), x=110, y=pdf.get_y() - 90, w=90)

    # -------- Insights LLM --------
    if llm_insights:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _safe("5. Insights da IA Generativa"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _safe(llm_insights[:4000]))

    pdf.output(str(out_pdf))
    logger.info("Relatorio PDF gerado", extra={"path": str(out_pdf)})
    return out_pdf
