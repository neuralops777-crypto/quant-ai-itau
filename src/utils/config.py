"""Gerenciamento centralizado de configurações e caminhos do projeto.

Carrega arquivos YAML de ``configs/`` com cache interno (Singleton),
expõe o dataclass ``Paths`` para navegação no sistema de arquivos e
garante a criação dos diretórios necessários.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceção
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Levantada quando o carregamento ou parsing do YAML falha."""


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Carrega um arquivo YAML em um dicionário.

    Args:
        path: Caminho para o arquivo ``.yaml`` / ``.yml``.

    Returns:
        Dicionário com o conteúdo do arquivo.

    Raises:
        ConfigError: Se o arquivo não existir ou o YAML for inválido.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Arquivo de configuração não encontrado: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config deve ser um mapeamento no topo: {p}")
        return data
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido em {p}: {exc}") from exc


# ---------------------------------------------------------------------------
# Singleton de configuração
# ---------------------------------------------------------------------------

class ConfigManager:
    """Singleton para carregar e cachear configurações YAML."""

    _instance: Optional["ConfigManager"] = None
    _cache: Dict[str, Dict[str, Any]] = {}

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, filename: str, root: Optional[Path] = None) -> Dict[str, Any]:
        """Carrega e cacheia um arquivo YAML de ``configs/``.

        Args:
            filename: Nome do arquivo (ex.: ``'model.yaml'``).
            root: Raiz do projeto; se None, resolve a partir deste arquivo.

        Returns:
            Dicionário com as configurações.
        """
        if filename in self._cache:
            return self._cache[filename]
        base = root or Path(__file__).resolve().parents[2]
        data = load_yaml(base / "configs" / filename)
        self._cache[filename] = data
        return data

    @staticmethod
    def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
        """Obtém variável de ambiente.

        Args:
            key: Nome da variável.
            default: Valor padrão se ausente.

        Returns:
            Valor da variável ou *default*.
        """
        return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Caminhos do projeto derivados de uma raiz comum."""

    root: Path
    data_raw: Path
    data_processed: Path
    data_external: Path
    reports: Path
    figures: Path

    @staticmethod
    def from_root(root: Path) -> "Paths":
        """Constrói a instância a partir da raiz do projeto."""
        return Paths(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            data_external=root / "data" / "external",
            reports=root / "reports",
            figures=root / "reports" / "figures",
        )


def ensure_dirs(paths: Paths) -> None:
    """Cria todos os diretórios do projeto se não existirem.

    Args:
        paths: Instância de :class:`Paths`.
    """
    for p in (
        paths.data_raw,
        paths.data_processed,
        paths.data_external,
        paths.reports,
        paths.figures,
    ):
        p.mkdir(parents=True, exist_ok=True)
    logger.info("Diretórios verificados", extra={"root": str(paths.root)})


# Instância global (uso nos módulos)
config = ConfigManager()
