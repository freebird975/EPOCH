"""Dependency-free BM25 baseline. Chinese text is indexed with character bigrams."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    result = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group()
        if "\u3400" <= token[0] <= "\u9fff":
            result.extend(token[i:i + 2] for i in range(len(token) - 1))
            if len(token) == 1:
                result.append(token)
        else:
            result.append(token)
    return result


def search(chunks: list[dict], question: str, top_k: int = 4) -> list[dict]:
    query_terms = set(tokenize(question))
    if not query_terms or not chunks:
        return []
    documents = [Counter(tokenize(chunk["text"])) for chunk in chunks]
    lengths = [sum(terms.values()) for terms in documents]
    average_length = sum(lengths) / len(lengths) or 1
    document_frequency = Counter()
    for terms in documents:
        document_frequency.update(terms.keys())

    ranked = []
    for chunk, terms, length in zip(chunks, documents, lengths):
        score = 0.0
        for term in query_terms:
            frequency = terms[term]
            if not frequency:
                continue
            idf = math.log(1 + (len(chunks) - document_frequency[term] + 0.5)
                           / (document_frequency[term] + 0.5))
            score += idf * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / average_length)
            )
        if score > 0:
            # A query matching the section title is usually more precise than
            # one matching generic words in a long body paragraph.
            heading_matches = len(query_terms & set(tokenize(chunk["section"])))
            score *= 1 + 0.25 * heading_matches
            ranked.append({**chunk, "score": round(score, 4)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]
