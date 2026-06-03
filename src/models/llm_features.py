"""Features baseadas em IA Generativa (sentimento, evento, confiança).

Extrai sinais quantitativos de notícias via LLM (OpenAI GPT).

Modos de operação:
    * **Real** (``OPENAI_API_KEY`` presente e ``llm.enabled: true``):
      chama a API e parseia o JSON retornado.
    * **Fallback** (ausência de chave ou falha): gera scores
      determinísticos via SHA-256 — garante reprodutibilidade do
      pipeline mesmo sem acesso à API.

Fluxo:
    ``score_news_items(items)`` → DataFrame com colunas
    ``ticker, date, sentiment_score, event_score, confidence_score, llm_summary``.

    ``attach_llm_features(feature_df, llm_df)`` → enriquece o dataset principal.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceção
# ---------------------------------------------------------------------------

class LLMFeatureError(RuntimeError):
    """Levantada quando a extração via LLM falha irrecuperavelmente."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsItem:
    """Representação imutável de uma notícia financeira."""

    ticker: str
    published_at: datetime
    title: str
    source: str
    url: str
    snippet: Optional[str] = None


@dataclass
class NewsInsight:
    """Resultado estruturado da análise de uma notícia."""

    ticker: str
    date: Any                  # date object
    sentiment_score: float     # [-1, 1]
    event_score: float         # [-1, 1]
    confidence_score: float    # [0, 1]
    llm_summary: str


# ---------------------------------------------------------------------------
# Fallback determinístico
# ---------------------------------------------------------------------------

def _deterministic_scores(text: str) -> Dict[str, float]:
    """Gera scores deterministicos a partir de SHA-256 do texto.

    Garante que o mesmo input sempre produz o mesmo output, permitindo
    reprodutibilidade de backtests sem acesso à API.

    Args:
        text: Texto concatenado ticker + título + snippet.

    Returns:
        Dicionário com ``sentiment_score``, ``event_score`` e
        ``confidence_score``.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    a = int(digest[0:8], 16) / 0xFFFFFFFF
    b = int(digest[8:16], 16) / 0xFFFFFFFF
    c = int(digest[16:24], 16) / 0xFFFFFFFF
    return {
        "sentiment_score": round((a * 2.0) - 1.0, 4),
        "event_score": round((b * 2.0) - 1.0, 4),
        "confidence_score": round(0.5 + 0.5 * c, 4),
    }


# ---------------------------------------------------------------------------
# Extração via LLM
# ---------------------------------------------------------------------------

_SYSTEM_MSG = (
    "Você é um analista quantitativo sênior. "
    "Sua tarefa é extrair sinais quantitativos de notícias financeiras."
)

_USER_TEMPLATE = (
    "Ticker: {ticker}\n"
    "Título: {title}\n"
    "Fonte: {source}\n"
    "Resumo: {snippet}\n\n"
    "Retorne SOMENTE um JSON válido (sem markdown) com as chaves:\n"
    "  sentiment_score: número entre -1 e 1\n"
    "  event_score: número entre -1 e 1 (impacto do evento)\n"
    "  confidence_score: número entre 0 e 1\n"
    "  summary: frase executiva em português (máx. 120 chars)\n"
)


def _parse_llm_response(content: str) -> Dict[str, Any]:
    """Extrai o primeiro bloco JSON de uma resposta LLM.

    Args:
        content: Texto bruto retornado pelo modelo.

    Returns:
        Dicionário parseado.

    Raises:
        LLMFeatureError: Se não houver JSON válido na resposta.
    """
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise LLMFeatureError(f"Resposta LLM sem JSON: {content[:200]}")
    return json.loads(content[start: end + 1])


def score_news_items(
    items: List[NewsItem],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> pd.DataFrame:
    """Pontua notícias com LLM ou fallback determinístico.

    Args:
        items: Lista de :class:`NewsItem`.
        model: Modelo OpenAI a utilizar.
        temperature: Temperatura do LLM (0 = determinístico).

    Returns:
        DataFrame agregado por ``(ticker, date)`` com colunas:
        ``sentiment_score``, ``event_score``, ``confidence_score``,
        ``llm_summary``.
    """
    if not items:
        return pd.DataFrame(
            columns=["ticker", "date", "sentiment_score", "event_score",
                     "confidence_score", "llm_summary"]
        )

    api_key = os.getenv("OPENAI_API_KEY")
    use_llm = bool(api_key)

    llm_client = None
    if use_llm:
        try:
            from langchain_openai import ChatOpenAI  # noqa: WPS433
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433
            llm_client = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
            logger.info("LLM ativo", extra={"model": model})
        except Exception as exc:
            logger.warning("Falha ao inicializar LLM; usando fallback", extra={"error": str(exc)})
            use_llm = False

    records: List[Dict[str, Any]] = []

    for item in items:
        fingerprint = f"{item.ticker}|{item.title}|{item.source}|{item.snippet or ''}|{item.url}"

        if not use_llm:
            sc = _deterministic_scores(fingerprint)
            records.append(
                {
                    "ticker": item.ticker,
                    "date": item.published_at.date(),
                    **sc,
                    "llm_summary": f"[fallback] {(item.snippet or item.title)[:120]}",
                }
            )
            continue

        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: WPS433,F811
            msgs = [
                SystemMessage(content=_SYSTEM_MSG),
                HumanMessage(
                    content=_USER_TEMPLATE.format(
                        ticker=item.ticker,
                        title=item.title,
                        source=item.source,
                        snippet=item.snippet or "",
                    )
                ),
            ]
            resp = llm_client.invoke(msgs)  # type: ignore[union-attr]
            obj = _parse_llm_response(
                resp.content if isinstance(resp.content, str) else str(resp.content)
            )
            records.append(
                {
                    "ticker": item.ticker,
                    "date": item.published_at.date(),
                    "sentiment_score": float(obj["sentiment_score"]),
                    "event_score": float(obj["event_score"]),
                    "confidence_score": float(obj["confidence_score"]),
                    "llm_summary": str(obj.get("summary", ""))[:500],
                }
            )
        except Exception as exc:
            logger.warning(
                "LLM falhou; usando fallback",
                extra={"ticker": item.ticker, "error": str(exc)},
            )
            sc = _deterministic_scores(fingerprint)
            records.append(
                {
                    "ticker": item.ticker,
                    "date": item.published_at.date(),
                    **sc,
                    "llm_summary": "[fallback]",
                }
            )

    df = pd.DataFrame.from_records(records)

    # Agrega por ticker/data (média de scores, concat de summaries)
    agg = (
        df.groupby(["ticker", "date"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            event_score=("event_score", "mean"),
            confidence_score=("confidence_score", "mean"),
            llm_summary=("llm_summary", lambda s: " | ".join(list(s)[:3])),
        )
    )
    logger.info("LLM features geradas", extra={"registros": len(agg)})
    return agg


# ---------------------------------------------------------------------------
# Enriquecimento do dataset principal
# ---------------------------------------------------------------------------

def attach_llm_features(feature_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Une features LLM ao dataset principal por ``(ticker, date)``.

    Valores ausentes são preenchidos com 0 (scores) e '' (summary).

    Args:
        feature_df: Dataset principal com colunas ``date`` e ``ticker``.
        llm_df: DataFrame retornado por :func:`score_news_items`.

    Returns:
        Dataset enriquecido.
    """
    if llm_df.empty:
        return feature_df

    out = feature_df.copy()
    out["_date_only"] = pd.to_datetime(out["date"]).dt.date

    llm = llm_df.copy()
    llm = llm.rename(columns={"date": "_date_only"})

    out = out.merge(llm, on=["ticker", "_date_only"], how="left")
    out = out.drop(columns=["_date_only"])

    for col in ("sentiment_score", "event_score", "confidence_score"):
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    if "llm_summary" in out.columns:
        out["llm_summary"] = out["llm_summary"].fillna("")

    return out
