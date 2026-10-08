"""Fetch and normalize arXiv Atom metadata without downloading full papers."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
DEFAULT_QUERY = 'all:"retrieval augmented generation" AND (cat:cs.IR OR cat:cs.CL)'
API_URL = "https://export.arxiv.org/api/query"


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def parse_atom(data: bytes) -> tuple[list[dict], dict]:
    root = ET.fromstring(data)
    if root.tag != ATOM + "feed":
        raise ValueError("响应不是 arXiv Atom feed")

    papers: list[dict] = []
    seen_ids: set[str] = set()
    for entry in root.findall(ATOM + "entry"):
        raw_url = _clean(entry.findtext(ATOM + "id"))
        parsed_url = urllib.parse.urlparse(raw_url)
        if parsed_url.hostname != "arxiv.org" or not parsed_url.path.startswith("/abs/"):
            continue
        source_url = "https://arxiv.org" + parsed_url.path
        versioned_id = parsed_url.path.rstrip("/").rsplit("/", 1)[-1]
        match = re.fullmatch(r"(.+?)(v\d+)?", versioned_id)
        if not match:
            continue
        paper_id, version = match.group(1), match.group(2) or ""
        if paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        title = _clean(entry.findtext(ATOM + "title"))
        abstract = _clean(entry.findtext(ATOM + "summary"))
        if not title or not abstract:
            continue
        authors = [
            _clean(author.findtext(ATOM + "name"))
            for author in entry.findall(ATOM + "author")
        ]
        categories = [
            item.attrib["term"]
            for item in entry.findall(ATOM + "category")
            if item.attrib.get("term")
        ]
        primary = entry.find(ARXIV + "primary_category")
        pdf_url = next((
            link.attrib.get("href", "")
            for link in entry.findall(ATOM + "link")
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf"
        ), "")
        papers.append({
            "paper_id": paper_id,
            "version": version,
            "title": title,
            "authors": [name for name in authors if name],
            "abstract": abstract,
            "published_at": _clean(entry.findtext(ATOM + "published")),
            "updated_at": _clean(entry.findtext(ATOM + "updated")),
            "year": int((_clean(entry.findtext(ATOM + "published")) or "0000")[:4]),
            "doi": _clean(entry.findtext(ARXIV + "doi")),
            "primary_category": primary.attrib.get("term", "") if primary is not None else "",
            "categories": categories,
            "source_url": source_url,
            "pdf_url": pdf_url,
            "evidence_scope": "abstract",
        })

    metadata = {
        "api": API_URL,
        "feed_title": _clean(root.findtext(ATOM + "title")),
        "feed_updated_at": _clean(root.findtext(ATOM + "updated")),
        "total_results": int(root.findtext(OPENSEARCH + "totalResults") or 0),
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "normalized_count": len(papers),
        "evidence_scope": "abstract_only",
    }
    if not papers:
        raise ValueError("Atom feed 中没有可用论文；请检查查询条件和 API 响应")
    return papers, metadata


def import_atom(raw_path: Path, papers_path: Path, manifest_path: Path) -> dict:
    data = raw_path.read_bytes()
    papers, metadata = parse_atom(data)
    papers_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    papers_path.write_text(
        "".join(json.dumps(paper, ensure_ascii=False) + "\n" for paper in papers),
        encoding="utf-8",
    )
    metadata.update({
        "raw_path": raw_path.as_posix(),
        "papers_path": papers_path.as_posix(),
    })
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def fetch_arxiv(raw_path: Path, query: str = DEFAULT_QUERY, limit: int = 100) -> str:
    if not 1 <= limit <= 200:
        raise ValueError("首版一次只获取 1–200 条，请缩小 limit")
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{API_URL}?{params}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LitAgent/0.1 (personal research prototype)"},
    )
    # The arXiv legacy API asks clients to keep requests at least 3 seconds apart.
    time.sleep(3.1)
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("arXiv 响应超过 5 MB，已拒绝保存")
    parse_atom(data)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(data)
    return url


def load_papers(path: Path) -> list[dict]:
    papers = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        paper = json.loads(line)
        for field in ("paper_id", "source_url", "evidence_scope"):
            if not paper.get(field):
                raise ValueError(f"{path}:{number} 缺少 {field}")
        if not isinstance(paper.get("title"), str) or not isinstance(paper.get("abstract"), str):
            raise ValueError(f"{path}:{number} 缺少标题或摘要文本字段")
        if paper["paper_id"] in seen:
            raise ValueError(f"重复论文 ID: {paper['paper_id']}")
        seen.add(paper["paper_id"])
        papers.append(paper)
    if not papers:
        raise ValueError(f"语料为空: {path}")
    return papers
