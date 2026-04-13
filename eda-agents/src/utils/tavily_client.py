"""Utilidad — Cliente Tavily para búsqueda web académica.

Wrapper sobre tavily-python que ejecuta búsquedas web orientadas a
literatura académica y retorna resultados normalizados.

Usada por Agent 01 (Research Lead) como complemento a la búsqueda con Claude.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


def search_tavily(
    query: str,
    api_key: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Ejecuta una búsqueda con Tavily y retorna resultados normalizados.

    Parameters
    ----------
    query : texto de búsqueda.
    api_key : Tavily API key.
    max_results : máximo de resultados (default 5).
    search_depth : "basic" o "advanced".
    include_domains : dominios académicos prioritarios.

    Returns
    -------
    Lista de dicts con {title, url, content, score}.
    """
    if not api_key:
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)

        if include_domains is None:
            include_domains = [
                "scholar.google.com",
                "pubmed.ncbi.nlm.nih.gov",
                "arxiv.org",
                "researchgate.net",
                "sciencedirect.com",
                "springer.com",
                "ieee.org",
                "nature.com",
                "wiley.com",
            ]

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
        )

        results: list[dict[str, Any]] = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
                "score": item.get("score", 0.0),
            })

        return results

    except Exception as e:
        logger.warning("tavily_search_failed", error=str(e))
        return []


def search_literature_tavily(
    equations: list[str],
    api_key: str,
    max_results_per_eq: int = 3,
) -> list[dict[str, Any]]:
    """Busca literatura para cada ecuación PICO usando Tavily.

    Parameters
    ----------
    equations : lista de ecuaciones booleanas PICO.
    api_key : Tavily API key.
    max_results_per_eq : resultados por ecuación.

    Returns
    -------
    Lista unificada y deduplicada de resultados.
    """
    all_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for eq in equations:
        results = search_tavily(
            query=eq,
            api_key=api_key,
            max_results=max_results_per_eq,
        )
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    return all_results
