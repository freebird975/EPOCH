"""Paper-record links derived from stable source identifiers."""

from __future__ import annotations

import re
from urllib.parse import quote


def semantic_scholar_record_api_url(paper: dict | str | int) -> str:
    """Build a request-free Semantic Scholar paper-record URL from a CorpusId.

    The record endpoint returns the canonical paper URL and external identifiers
    when available. It does not imply that the paper's full text is open access.
    """
    if isinstance(paper, dict):
        value = paper.get("corpusid") or paper.get("paper_id")
    else:
        value = paper
    corpus_id = str(value or "").strip()
    if not re.fullmatch(r"\d+", corpus_id):
        return ""
    identifier = quote(f"CorpusId:{corpus_id}", safe="")
    return (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{identifier}?fields=title%2Curl%2CexternalIds"
    )
