"""Conservative claim-to-evidence verification for numbered citations.

Citation numbers are 1-based indexes into the ordered ``contexts`` argument.
Without a semantic judge this module only recognizes an exact normalized claim
inside its cited text as support; all other claims remain insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Iterable, Mapping

VERDICTS = frozenset({"support", "contradiction", "insufficient"})
_CITATION = re.compile(r"\[([0-9\s,，;；\-–]+)\]")
_SPLIT = re.compile(r"(?<!\bal\.)(?<=[.!?。！？])\s+|\n+")
_MAX_CONTEXTS = 1000
_MAX_TEXT_CHARS = 100_000


@dataclass(frozen=True)
class Evidence:
    paper_id: str
    chunk_id: str
    section: str
    text: str
    title: str = ""
    # Page is copied only when the upstream source explicitly provides it.
    page: Any = None


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    citations: tuple[int, ...]
    evidence: tuple[Evidence, ...]
    verdict: str
    reason: str = ""


@dataclass
class VerificationResult:
    claims: list[ClaimVerdict] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    safe: bool = False
    needs_revision: bool = True

    @property
    def verdicts(self) -> list[ClaimVerdict]:
        """Alias convenient for callers that expect a verdict list."""
        return self.claims


def _contexts(items: Iterable[Mapping[str, Any]]) -> list[Evidence]:
    result: list[Evidence] = []
    for i, item in enumerate(items):
        if i >= _MAX_CONTEXTS:
            raise ValueError(f"contexts exceeds maximum of {_MAX_CONTEXTS}")
        if not isinstance(item, Mapping):
            raise TypeError("each context must be a mapping")
        text = item.get("text", "")
        if not isinstance(text, str):
            raise TypeError("context text must be a string")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"context text exceeds maximum of {_MAX_TEXT_CHARS} characters")
        for key in ("paper_id", "chunk_id", "section"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"context {key} must be a non-empty string")
        result.append(Evidence(
            paper_id=item["paper_id"], chunk_id=item["chunk_id"],
            section=item["section"], text=text, title=str(item.get("title", "")),
            **({"page": item["page"]} if "page" in item else {}),
        ))
    return result


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).split())


def citation_numbers(text: str) -> list[int]:
    """Parse numbered lists and bounded numeric ranges in first-appearance order."""
    numbers: list[int] = []
    for group in _CITATION.findall(text):
        for part in re.split(r"\s*[,，;；]\s*", group):
            part = part.strip()
            range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
            if range_match:
                start, end = map(int, range_match.groups())
                expanded = range(start, end + 1) if start <= end and end - start <= 100 else (start, end)
            elif part.isdigit():
                value = int(part)
                expanded = () if len(part) == 4 and 1900 <= value <= 2100 else (value,)
            else:
                continue
            for number in expanded:
                if number not in numbers:
                    numbers.append(number)
    return numbers


def _strip_citations(text: str) -> str:
    return _CITATION.sub(
        lambda match: match.group(0) if re.fullmatch(r"(?:19|20)\d{2}", match.group(1).strip()) else "",
        text,
    )


def _epistemic_caveat(text: str) -> bool:
    lead = text.casefold().lstrip(" ,;:，；：")
    if lead.startswith((
        "however, the blocks do not establish", "the evidence does not establish",
        "the provided blocks do not establish", "the blocks do not establish",
        "therefore, the evidence cannot", "the evidence is insufficient",
        "现有证据不足", "证据不足", "无法根据现有证据", "无法从这些证据",
    )):
        return True
    return bool(re.match(r"^(?:however,\s*)?the evidence does not "
                         r"(?:establish|identify|prove|show|confirm)\b", lead)
                and re.search(r"\b(?:first|earliest|priority|only)\b", lead))


def _judge(judge: Callable[..., Any], claim: str, evidence: list[Evidence]) -> tuple[str, str]:
    try:
        result = judge(claim, evidence)
    except Exception:
        # Never include exception text: model/client errors can contain secrets.
        return "insufficient", "judge_error"
    if isinstance(result, str):
        verdict, reason = result.strip().casefold(), ""
    elif isinstance(result, Mapping):
        verdict = str(result.get("verdict", "")).strip().casefold()
        reason = str(result.get("reason", ""))[:500]
    else:
        return "insufficient", "invalid_judge_result"
    if verdict not in VERDICTS:
        return "insufficient", "invalid_judge_verdict"
    return verdict, reason


def verify_answer(
    answer_text: str,
    contexts: Iterable[Mapping[str, Any]],
    judge: Callable[[str, list[Evidence]], Any] | None = None,
) -> VerificationResult:
    """Verify claims against cited evidence blocks.

    ``judge`` may return ``support``, ``contradiction`` or ``insufficient``, or
    a mapping with ``verdict`` and optional ``reason``. A judge failure is
    converted to ``insufficient`` and does not expose exception details.
    ``safe`` is true only when every extracted claim has valid cited support.
    """
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string")
    if len(answer_text) > _MAX_TEXT_CHARS:
        raise ValueError(f"answer_text exceeds maximum of {_MAX_TEXT_CHARS} characters")
    evidence_blocks = _contexts(contexts)
    claims: list[ClaimVerdict] = []
    invalid: set[int] = set()

    for raw in _SPLIT.split(answer_text.strip()):
        raw = raw.strip()
        if not raw:
            continue
        numbers = tuple(citation_numbers(raw))
        claim = _strip_citations(raw).strip(" \t,;:，；：")
        if not claim:
            continue
        prior_paper_answer = any(
            previous.citations and re.search(r"\bpaper\b|论文", previous.claim.casefold())
            for previous in claims
        )
        priority_caveat = bool(re.search(r"\b(?:first|earliest|priority|only)\b|首次|最早|唯一",
                                          claim.casefold()))
        if not numbers and _epistemic_caveat(claim) and (not priority_caveat or prior_paper_answer):
            continue
        valid = [n for n in numbers if 1 <= n <= len(evidence_blocks)]
        invalid.update(n for n in numbers if n < 1 or n > len(evidence_blocks))
        cited = [evidence_blocks[n - 1] for n in valid]
        if not numbers:
            verdict, reason = "insufficient", "missing_citation"
        elif len(valid) != len(numbers):
            verdict, reason = "insufficient", "invalid_citation_number"
        elif judge is not None:
            verdict, reason = _judge(judge, claim, cited)
        else:
            needle = _normalize(claim)
            exact = bool(needle) and any(needle in _normalize(e.text) for e in cited)
            verdict, reason = ("support", "exact_text_match") if exact else ("insufficient", "no_semantic_judge")
        claims.append(ClaimVerdict(claim, numbers, tuple(cited), verdict, reason))

    safe = bool(claims) and not invalid and all(c.verdict == "support" for c in claims)
    return VerificationResult(
        claims=claims, unsupported_citations=sorted(invalid),
        safe=safe, needs_revision=not safe,
    )
