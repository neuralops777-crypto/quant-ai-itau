"""Logger estruturado com saída JSON Lines.

Compatível com ELK Stack, Datadog e Cloud Logging.
Cada linha de log é um JSON válido com timestamp UTC,
nível, nome do módulo, mensagem e campos extras.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class LogEvent:
    """Payload imutável de um evento de log estruturado."""

    ts: str
    level: str
    name: str
    message: str
    extra: Dict[str, Any]


class JsonFormatter(logging.Formatter):
    """Formata registros de log como JSON Lines.

    Campos internos do LogRecord são excluídos do bloco `extra`
    para evitar ruído; apenas kwargs passados via `extra=` chegam lá.
    """

    _SKIP: frozenset[str] = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        extra: Dict[str, Any] = {
            k: v for k, v in record.__dict__.items() if k not in self._SKIP
        }
        event = LogEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            name=record.name,
            message=record.getMessage(),
            extra=extra,
        )
        return json.dumps(asdict(event), ensure_ascii=False, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Cria ou recupera um logger estruturado configurado.

    Utiliza o padrão de *idempotência*: se o logger já tiver handlers,
    apenas ajusta o nível e o retorna — evitando handlers duplicados em
    ambientes com imports repetidos (ex.: notebooks).

    Args:
        name: Nome do logger; use ``__name__`` nos módulos.
        level: Nível de logging (default: INFO).

    Returns:
        Logger configurado com JsonFormatter no stdout.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
