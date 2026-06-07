"""Coleta de dados de mercado (Yahoo Finance) e macro (BCB SGS).

Fluxo:
    ``collect_all(raw_dir, tickers_stocks, tickers_etfs, tickers_indices,
                  macro_sgs, start, end, interval)``
    → ``data/raw/prices.parquet``
    → ``data/raw/macro.parquet``
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Exceção
# ---------------------------------------------------------------------------

class DataCollectionError(RuntimeError):
    """Levantada quando a coleta de dados falha."""


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def collect_prices(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Baixa preços históricos do Yahoo Finance para uma lista de tickers.

    Args:
        tickers: Lista de tickers (ex.: ["PETR4.SA", "VALE3.SA"]).
        start: Data de início no formato "YYYY-MM-DD".
        end: Data de fim no formato "YYYY-MM-DD".
        interval: Intervalo dos dados ("1d", "1wk", "1mo").
        max_retries: Número máximo de tentativas em caso de falha.
        retry_delay: Segundos de espera entre tentativas.

    Returns:
        DataFrame com MultiIndex de colunas (field, ticker) do yfinance.

    Raises:
        DataCollectionError: Se a coleta falhar após todas as tentativas.
    """
    if not tickers:
        raise DataCollectionError("Lista de tickers está vazia.")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Baixando preços Yahoo Finance",
                extra={"tickers": len(tickers), "start": start, "end": end},
            )
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df.empty:
                raise DataCollectionError(
                    f"yfinance retornou DataFrame vazio para {tickers}."
                )
            logger.info("Preços coletados", extra={"shape": str(df.shape)})
            return df
        except DataCollectionError:
            raise
        except Exception as exc:
            logger.warning(
                f"Tentativa {attempt}/{max_retries} falhou: {exc}"
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise DataCollectionError(
        f"Falha ao coletar preços após {max_retries} tentativas."
    )


# ---------------------------------------------------------------------------
# BCB SGS (Séries macroeconômicas)
# ---------------------------------------------------------------------------

_BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


def _fetch_bcb_series(
    code: int,
    start: str,
    end: str,
    timeout: int = 30,
) -> Optional[pd.Series]:
    """Baixa uma série do Sistema Gerenciador de Séries (SGS) do BCB.

    Args:
        code: Código da série no SGS (ex.: 432 = Selic).
        start: Data de início "YYYY-MM-DD".
        end: Data de fim "YYYY-MM-DD".
        timeout: Timeout HTTP em segundos.

    Returns:
        pd.Series com índice DatetimeIndex, ou None se falhar.
    """
    url = _BCB_SGS_URL.format(code=code)
    params = {
        "formato": "json",
        "dataInicial": (pd.Timestamp(start) if start else pd.Timestamp("2018-01-01")).strftime("%d/%m/%Y"),
        "dataFinal": (pd.Timestamp(end) if end else pd.Timestamp.now()).strftime("%d/%m/%Y"),
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            logger.warning(f"BCB SGS {code}: resposta vazia.")
            return None
        series = pd.DataFrame(data)
        series["data"] = pd.to_datetime(series["data"], format="%d/%m/%Y")
        series["valor"] = pd.to_numeric(series["valor"], errors="coerce")
        series = series.set_index("data")["valor"].sort_index()
        series.name = str(code)
        return series
    except Exception as exc:
        logger.warning(f"BCB SGS {code}: falha na coleta — {exc}")
        return None


def collect_macro(
    sgs_codes: Dict[str, int],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Coleta séries macroeconômicas do BCB SGS.

    Args:
        sgs_codes: Dicionário nome → código SGS.
                   Ex.: {"selic": 432, "ipca": 433, "cambio": 1}.
        start: Data de início "YYYY-MM-DD".
        end: Data de fim "YYYY-MM-DD".

    Returns:
        DataFrame com uma coluna por série e DatetimeIndex.
        Séries que falharem serão preenchidas com NaN.
    """
    frames: Dict[str, pd.Series] = {}
    for name, code in sgs_codes.items():
        logger.info(f"Coletando BCB SGS: {name} (código {code})")
        s = _fetch_bcb_series(code, start, end)
        if s is not None:
            frames[name] = s
        else:
            logger.warning(f"Série BCB '{name}' ({code}) não disponível — NaN.")

    if not frames:
        logger.warning("Nenhuma série BCB coletada; retornando DataFrame vazio.")
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1, keys=frames.keys())
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    macro = macro.sort_index()

    logger.info("Macro coletado", extra={"shape": str(macro.shape), "series": list(frames.keys())})
    return macro


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def collect_all(
    raw_dir: Path,
    tickers_stocks: List[str],
    tickers_etfs: List[str],
    tickers_indices: List[str],
    macro_sgs: Dict[str, int],
    start: str,
    end: str,
    interval: str = "1d",
) -> None:
    """Coleta preços e macro e persiste em ``raw_dir``.

    Args:
        raw_dir: Diretório onde os parquets brutos serão salvos.
        tickers_stocks: Tickers de ações brasileiras (ex.: ["PETR4.SA"]).
        tickers_etfs: Tickers de ETFs (ex.: ["BOVA11.SA"]).
        tickers_indices: Tickers de índices (ex.: ["^BVSP"]).
        macro_sgs: Dict nome → código SGS do BCB.
        start: Data de início "YYYY-MM-DD".
        end: Data de fim "YYYY-MM-DD".
        interval: Intervalo dos dados de preço.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Preços ---
    all_tickers = list(dict.fromkeys(
        tickers_stocks + tickers_etfs + tickers_indices
    ))
    prices = collect_prices(
        tickers=all_tickers,
        start=start,
        end=end,
        interval=interval,
    )
    prices_path = raw_dir / "prices.parquet"
    prices.to_parquet(prices_path)
    logger.info("Preços brutos salvos", extra={"path": str(prices_path)})

    # --- Macro ---
    macro = collect_macro(sgs_codes=macro_sgs, start=start, end=end)
    macro_path = raw_dir / "macro.parquet"
    if not macro.empty:
        macro.to_parquet(macro_path)
        logger.info("Macro bruto salvo", extra={"path": str(macro_path)})
    else:
        # Salva DataFrame vazio para não quebrar etapas seguintes
        pd.DataFrame().to_parquet(macro_path)
        logger.warning("Macro vazio salvo como placeholder.")