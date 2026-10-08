"""LitSearch 全文 RAG 的本地网页界面（零第三方依赖，仅用标准库）。

启动方式（项目根目录）：

    .venv\\Scripts\\python webui\\server.py
    .venv\\Scripts\\python webui\\server.py --port 8080

打开终端里打印的 http://127.0.0.1:8000 即可使用。
问答（ask）与 Agentic 检索需要 .env 中的 DEEPSEEK_API_KEY；英文 classic 检索完全离线。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 流水线默认路径是相对于项目根目录的

from litagent.runtime import BudgetExceeded, RunRuntime, RuntimePolicy, use_runtime  # noqa: E402
from litsearch_fulltext import (  # noqa: E402
    CHILD_DATA, CHILD_INDEX, DENSE_CHILD_INDEX, PARENT_DATA,
    retrieve_parents, run_fulltext_ask,
)

PAGE = Path(__file__).with_name("index.html")

# 数据路径，默认与 litsearch_fulltext.py 一致，可用命令行参数切换到其他一致数据集
PATHS = {
    "parents": PARENT_DATA,
    "chunks": CHILD_DATA,
    "index": CHILD_INDEX,
    "dense_index": DENSE_CHILD_INDEX,
}

# 索引、向量模型与重排器在一次加载后驻留内存，供后续请求复用
RESOURCES: dict = {}
# 检索/生成计算串行执行，避免并发重复加载模型或打满 API 预算
LOCK = threading.Lock()
STATE = {"warm_status": "idle", "warm_error": None, "started_at": time.time()}

MAX_QUESTION_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024


def api_key_configured() -> bool:
    """只判断是否配置，绝不读取或外发密钥本身。"""
    if os.getenv("DEEPSEEK_API_KEY") or os.getenv("RAG_API_KEY"):
        return True
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY=") and line.split("=", 1)[1].strip():
                return True
    return False


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status_payload() -> dict:
    corpus_manifest = _read_json(PATHS["parents"].with_name("corpus_fulltext_manifest.json"))
    chunk_manifest = _read_json(PATHS["chunks"].with_suffix(".manifest.json"))
    return {
        "corpus": {
            "parents": (corpus_manifest.get("counts") or {}).get("with_full_text"),
            "chunks": chunk_manifest.get("chunks"),
            "chunk_size": chunk_manifest.get("chunk_size_chars"),
            "source": corpus_manifest.get("source"),
        },
        "indexes": {
            "bm25": PATHS["index"].is_file(),
            "dense": PATHS["dense_index"].is_file(),
        },
        "models": {
            "embedding_loaded": "embedding_model" in RESOURCES,
            "reranker_loaded": "reranker_model" in RESOURCES,
        },
        "api_configured": api_key_configured(),
        "warm_status": STATE["warm_status"],
        "warm_error": STATE["warm_error"],
        "uptime_seconds": round(time.time() - STATE["started_at"], 1),
    }


def warm_up() -> None:
    """后台预热：加载索引、向量模型与重排器，让首次查询不等模型加载。"""
    with LOCK:
        STATE["warm_status"] = "warming"
        try:
            retriever = "hybrid" if PATHS["dense_index"].is_file() else "bm25"
            runtime = RunRuntime(RuntimePolicy(deadline_seconds=120, failure_policy="strict"))
            with use_runtime(runtime):
                retrieve_parents(
                    "retrieval augmented generation", top_k=1, candidate_k=20,
                    retriever=retriever, pipeline="classic", rerank=True, rerank_k=5,
                    trace={}, resources=RESOURCES,
                    parent_path=PATHS["parents"], child_path=PATHS["chunks"],
                    index_path=PATHS["index"], dense_index_path=PATHS["dense_index"],
                )
            STATE["warm_status"] = "ready"
        except Exception as exc:  # 预热失败不阻断服务，状态页会显示原因
            STATE["warm_status"] = "failed"
            STATE["warm_error"] = f"{type(exc).__name__}: {exc}"[:300]


def _clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def parse_options(payload: dict) -> dict:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"问题过长，请控制在 {MAX_QUESTION_CHARS} 字以内")
    pipeline = payload.get("pipeline", "agentic")
    retriever = payload.get("retriever", "hybrid")
    if pipeline not in {"agentic", "classic"}:
        raise ValueError("pipeline 必须是 agentic 或 classic")
    if retriever not in {"hybrid", "bm25", "dense"}:
        raise ValueError("retriever 必须是 hybrid、bm25 或 dense")
    if pipeline == "agentic" and retriever != "hybrid":
        raise ValueError("agentic 流程要求 retriever=hybrid（每轮执行 BM25+Dense RRF）")
    failure_policy = "degrade" if payload.get("allow_degrade") else "strict"
    return {
        "question": question,
        "pipeline": pipeline,
        "retriever": retriever,
        "rerank": bool(payload.get("rerank", True)) and pipeline == "agentic",
        "top_k": _clamp_int(payload.get("top_k"), 5, 1, 10),
        "candidate_k": _clamp_int(payload.get("candidate_k"), 100, 20, 200),
        "rerank_k": _clamp_int(payload.get("rerank_k"), 20, 5, 50),
        "policy": RuntimePolicy(deadline_seconds=240, api_timeout_seconds=60,
                                max_api_calls=8, max_completion_tokens=12000,
                                max_retrieval_rounds=2, failure_policy=failure_policy),
    }


def trim_parent(parent: dict) -> dict:
    """去掉整篇全文等大字段，保留结果卡片需要的展示信息。"""
    return {
        "paper_id": parent.get("paper_id"),
        "title": parent.get("title", ""),
        "abstract": parent.get("abstract", ""),
        "source_url": parent.get("source_url", ""),
        "best_chunk_rank": parent.get("best_chunk_rank"),
        "retrieval_score": parent.get("retrieval_score"),
        "matched_chunks": [
            {key: chunk.get(key) for key in
             ("chunk_id", "chunk_index", "section", "score", "rank", "bm25_rank",
              "dense_rank", "rrf_rank", "rrf_score", "rerank_score",
              "matched_queries", "text")}
            for chunk in parent.get("matched_chunks", [])
        ],
    }


def run_search(opts: dict) -> dict:
    trace: dict = {}
    runtime = RunRuntime(opts["policy"])
    started = time.perf_counter()
    with use_runtime(runtime):
        parents, child_hits = retrieve_parents(
            opts["question"], top_k=opts["top_k"], candidate_k=opts["candidate_k"],
            retriever=opts["retriever"], pipeline=opts["pipeline"],
            rerank=opts["rerank"], rerank_k=opts["rerank_k"],
            trace=trace, resources=RESOURCES,
            parent_path=PATHS["parents"], child_path=PATHS["chunks"],
            index_path=PATHS["index"], dense_index_path=PATHS["dense_index"],
        )
    return {
        "status": "ok",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "trace": trace,
        "timings": runtime.timings,
        "degradations": runtime.degradations,
        "child_hit_count": len(child_hits),
        "parents": [trim_parent(parent) for parent in parents],
    }


def _trim_verification(verification: dict | None) -> dict | None:
    if not verification:
        return None
    claims = []
    for claim in verification.get("claims", []):
        claims.append({
            "claim": claim.get("claim", ""),
            "citations": list(claim.get("citations", [])),
            "verdict": claim.get("verdict", ""),
            "reason": claim.get("reason", ""),
            "evidence": [
                {"paper_id": ev.get("paper_id"), "chunk_id": ev.get("chunk_id"),
                 "section": ev.get("section"), "text": str(ev.get("text", ""))[:240]}
                for ev in claim.get("evidence", [])
            ],
        })
    return {"claims": claims,
            "unsupported_citations": verification.get("unsupported_citations", []),
            "safe": verification.get("safe", False)}


def run_ask(opts: dict) -> dict:
    report = run_fulltext_ask(
        opts["question"], top_k=opts["top_k"], candidate_k=opts["candidate_k"],
        retriever=opts["retriever"], pipeline=opts["pipeline"],
        rerank=opts["rerank"], rerank_k=opts["rerank_k"],
        runtime_policy=opts["policy"], resources=RESOURCES,
        parent_path=PATHS["parents"], child_path=PATHS["chunks"],
        index_path=PATHS["index"], dense_index_path=PATHS["dense_index"],
    )
    cited = sorted({number for number in _cited_numbers(report.get("answer", ""))
                    if 1 <= number <= len(report.get("contexts", []))})
    return {
        "status": report["status"],
        "answer": report.get("answer", ""),
        "cited": cited,
        "contexts": report.get("contexts", []),
        "candidate_parents": report.get("candidate_parents", []),
        "retrieval_trace": report.get("retrieval_trace", {}),
        "verification": _trim_verification(report.get("verification")),
        "timings": report.get("timings", []),
        "api_call_count": len(report.get("api_calls", [])),
        "cost": report.get("cost"),
        "context_chars": report.get("context_chars", 0),
        "degradations": report.get("degradations", []),
        "degraded": report.get("degraded", False),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "report_path": report.get("report_path"),
    }


def _cited_numbers(text: str) -> list[int]:
    from litagent.evidence import citation_numbers
    return citation_numbers(text)


class Handler(BaseHTTPRequestHandler):
    server_version = "LitSearchUI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静默常规访问日志，错误仍会打印
        return

    # ---------- 工具 ----------
    def _send_json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_page(self) -> None:
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大")
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ValueError("请求体不是合法 JSON")

    def _handle_query(self, mode: str) -> None:
        try:
            opts = parse_options(self._read_body())
        except ValueError as exc:
            self._send_json({"status": "error", "error": str(exc)}, code=400)
            return
        if mode == "ask" and not api_key_configured():
            self._send_json({"status": "error",
                             "error": "未配置 DEEPSEEK_API_KEY，无法生成回答；"
                                      "可切换到“仅检索”模式离线使用。"}, code=400)
            return
        if not LOCK.acquire(blocking=False):
            self._send_json({"status": "error",
                             "error": "正在处理上一个请求，请稍候再试。"}, code=409)
            return
        try:
            result = run_search(opts) if mode == "search" else run_ask(opts)
            self._send_json(result)
        except BudgetExceeded as exc:
            self._send_json({"status": "error", "error": f"超出运行预算：{exc}"}, code=429)
        except FileNotFoundError as exc:
            self._send_json({"status": "error", "error": str(exc)}, code=409)
        except (ValueError, RuntimeError, OSError) as exc:
            self._send_json({"status": "error", "error": str(exc)}, code=500)
        except Exception as exc:  # 意外错误不暴露堆栈，只给类型
            self._send_json({"status": "error",
                             "error": f"未预期的错误（{type(exc).__name__}）：{exc}"[:300]}, code=500)
        finally:
            LOCK.release()

    # ---------- 路由 ----------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_page()
        elif self.path == "/api/status":
            self._send_json(status_payload())
        else:
            self._send_json({"status": "error", "error": "not found"}, code=404)

    def do_POST(self) -> None:
        if self.path == "/api/search":
            self._handle_query("search")
        elif self.path == "/api/ask":
            self._handle_query("ask")
        else:
            self._send_json({"status": "error", "error": "not found"}, code=404)


def main() -> int:
    parser = argparse.ArgumentParser(description="LitSearch 全文 RAG 本地网页界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parents", type=Path, default=PARENT_DATA, help="全文父文档 jsonl")
    parser.add_argument("--chunks", type=Path, default=CHILD_DATA, help="全文子块 jsonl")
    parser.add_argument("--index", type=Path, default=CHILD_INDEX, help="BM25 子块索引")
    parser.add_argument("--dense-index", type=Path, default=DENSE_CHILD_INDEX, help="BGE/FAISS 子块索引")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动时的索引/模型预热")
    args = parser.parse_args()

    PATHS.update({"parents": args.parents, "chunks": args.chunks,
                  "index": args.index, "dense_index": args.dense_index})
    if not PATHS["chunks"].is_file() or not PATHS["parents"].is_file():
        print("未找到全文语料，请先按 README 运行 prepare 与 chunk 命令。", file=sys.stderr)
        return 1
    if not args.no_warmup:
        threading.Thread(target=warm_up, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"LitSearch 全文 RAG 网页界面已启动：{url}")
    print("首次查询前正在后台预热索引与模型；Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
