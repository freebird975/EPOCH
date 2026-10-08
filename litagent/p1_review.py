"""Offline human review export and scoring for P1 JSON reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

LABELS = {"correct", "incorrect", "uncertain"}
SUPPORT_LABELS = {"yes", "no", "uncertain"}


def _refs(value: Any) -> list[dict]:
    """Collect source locators from evidence, preserving report order."""
    found: list[dict] = []
    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if node.get("chunk_id") and node.get("paper_id"):
                ref = {k: node[k] for k in ("paper_id", "chunk_id", "section", "start_line", "end_line", "location_type", "quote") if node.get(k) is not None}
                if ref not in found:
                    found.append(ref)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(value)
    return found


def _row(kind: str, path: str, value: Any, report_id: str) -> dict:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    key = f"{report_id}\0{kind}\0{path}\0{text}"
    rid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return {"review_id": rid, "kind": kind, "path": path, "content": value,
            "source_references": _refs(value), "label": "", "evidence_support": ""}


def report_to_rows(report: Mapping[str, Any], report_type: str = "auto") -> list[dict]:
    """Flatten paper_understanding or related_work report into editable review rows."""
    if report_type not in {"auto", "paper_understanding", "related_work"}:
        raise ValueError("report_type must be auto, paper_understanding, or related_work")
    if report_type == "auto":
        report_type = "paper_understanding" if report.get("task") == "paper_understanding" or "fields" in report else "related_work"
    if report_type == "related_work" and isinstance(report.get("result"), Mapping):
        content = report["result"]
    else:
        content = report
    identity = str(report.get("paper_id", "")) if report_type == "paper_understanding" else str(report.get("topic", report.get("query", "")))
    # Include a stable digest of the report so IDs cannot collide across distinct report files.
    fingerprint = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]
    report_id = f"{report_type}:{identity}:{fingerprint}"
    rows: list[dict] = []
    if report_type == "paper_understanding":
        fields = content.get("fields", {})
        if isinstance(fields, Mapping):
            for field, claims in fields.items():
                if isinstance(claims, list):
                    for i, claim in enumerate(claims):
                        if claim not in (None, "", [], {}):
                            rows.append(_row("paper_field", f"fields.{field}[{i}]", claim, report_id))
    else:
        for i, paper in enumerate(content.get("papers", []) if isinstance(content.get("papers"), list) else []):
            if not isinstance(paper, Mapping):
                continue
            pid = str(paper.get("paper_id", i))
            for field in ("task", "method", "data", "findings"):
                val = paper.get(field)
                if isinstance(val, Mapping):
                    field_content = {"paper_id": pid, "field": field, **val}
                    if val.get("text") and str(val.get("text")).casefold() != "unknown":
                        rows.append(_row("related_paper_field", f"papers.{pid}.{field}", field_content, report_id))
                elif val not in (None, "", "unknown"):
                    rows.append(_row("related_paper_field", f"papers.{pid}.{field}", {"paper_id": pid, "field": field, "value": val}, report_id))
        for i, item in enumerate(content.get("groups", []) if isinstance(content.get("groups"), list) else []):
            if isinstance(item, Mapping) and item:
                rows.append(_row("group", f"groups[{i}]", item, report_id))
        for i, item in enumerate(content.get("comparisons", []) if isinstance(content.get("comparisons"), list) else []):
            if not isinstance(item, Mapping) or not item:
                continue
            # One row for the synthesis and one for each populated paper comparison value.
            synthesis = {k: v for k, v in item.items() if k != "entries"}
            if any(v not in (None, "", [], {}) for v in synthesis.values()):
                rows.append(_row("comparison", f"comparisons[{i}]", synthesis, report_id))
            for j, entry in enumerate(item.get("entries", []) if isinstance(item.get("entries"), list) else []):
                if isinstance(entry, Mapping) and entry:
                    rows.append(_row("comparison_entry", f"comparisons[{i}].entries[{j}]", entry, report_id))
    return rows


def score_rows(rows: list[Mapping[str, Any]]) -> dict:
    """Score only explicitly labeled rows; blank labels never count as correct."""
    total = len(rows)
    labeled = [r for r in rows if r.get("label") in LABELS]
    evidence_labeled = [r for r in rows if r.get("evidence_support") in SUPPORT_LABELS]
    counts = {label: sum(r.get("label") == label for r in labeled) for label in sorted(LABELS)}
    decided = [r for r in labeled if r.get("label") in {"correct", "incorrect"}]
    evidence_decided = [r for r in evidence_labeled if r.get("evidence_support") in {"yes", "no"}]
    supported = sum(r.get("evidence_support") == "yes" for r in evidence_labeled)
    return {"total_rows": total, "labeled_rows": len(labeled),
            "label_coverage": len(labeled) / total if total else 0.0,
            "labels": counts, "decided_rows": len(decided),
            "correctness_rate_among_decided": counts["correct"] / len(decided) if decided else None,
            "evidence_labeled_rows": len(evidence_labeled),
            "evidence_label_coverage": len(evidence_labeled) / total if total else 0.0,
            "evidence_decided_rows": len(evidence_decided),
            "evidence_support_rate_among_decided": supported / len(evidence_decided) if evidence_decided else None,
            "evidence_support": {"yes": supported,
                                 "no": sum(r.get("evidence_support") == "no" for r in evidence_labeled),
                                 "uncertain": sum(r.get("evidence_support") == "uncertain" for r in evidence_labeled)}}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="export a JSON report as editable JSONL review rows")
    exp.add_argument("report", type=Path)
    exp.add_argument("output", type=Path)
    exp.add_argument("--type", choices=("auto", "paper_understanding", "related_work"), default="auto")
    score = sub.add_parser("score", help="score a reviewed JSONL file")
    score.add_argument("reviewed", type=Path)
    score.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        rows = report_to_rows(json.loads(args.report.read_text(encoding="utf-8")), args.type)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"Exported {len(rows)} rows to {args.output}")
        return 0
    rows = _read_jsonl(args.reviewed)
    for row in rows:
        if row.get("label") not in (None, "", *LABELS):
            raise ValueError(f"invalid label for {row.get('review_id')}: {row.get('label')}")
        if row.get("evidence_support") not in (None, "", *SUPPORT_LABELS):
            raise ValueError(f"invalid evidence_support for {row.get('review_id')}: {row.get('evidence_support')}")
    result = score_rows(rows)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
