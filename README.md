# LitAgent：计算机论文全文 RAG

一个面向计算机论文的中英双语检索增强生成项目。用户可以用中文或英文提问；当前论文语料以英文全文为主。

系统包括全文解析与父子分块、BGE 向量嵌入、FAISS Dense 检索、BM25、RRF 融合、查询规划与子问题拆解、Cross Encoder 重排，以及带证据核验的回答生成。

## 仓库内容

本上传版保留源码、测试、依赖清单、100 篇 arXiv 摘要级小样本和项目报告。**LitSearch 全量全文、完整检索索引、模型缓存、虚拟环境和 API 密钥均不随仓库提供。** 全文语料与索引可按下方说明在本地重新获取和构建；其全文分发权利需要逐篇确认。

| 路径 | 用途 |
| --- | --- |
| `litsearch_fulltext.py`、`litagent/` | 全文检索、混合融合、重排、生成和评测逻辑 |
| `lit.py`、`data/` | 100 篇摘要级轻量样本及早期检索基线 |
| `rag/`、`main.py`、`docs/sample.md` | 不依赖外部语料的通用文档 RAG 学习基线 |
| `tests/` | 离线单元与回归测试 |
| `DEVELOPMENT_GUIDE.md`、`eval/RESULTS.md` | 架构、开发记录和评测边界 |

## 安装依赖

CPU 默认只需一个依赖清单：`requirements.txt`，覆盖通用文档 RAG、论文摘要检索和全文流程。PDF 导入、FAISS、BGE embedding 和 LitSearch 数据读取所需的库都在其中。

如果使用已配置好兼容 CUDA 运行时的环境，可用 `requirements-litsearch-gpu.txt` 替代 `requirements.txt`，不要同时安装两份；GPU 清单将 CPU 版 FastEmbed 换成 GPU 版。

## 快速运行：通用文档 RAG

需要 Python 3.11 或更新版本。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py index
.venv\Scripts\python.exe main.py search "这个项目使用什么检索方式？"
```

该基线使用 `docs/sample.md`，可离线索引和检索，不需要模型 API Key。

## 运行：摘要级论文检索

已包含 100 篇论文的标题与摘要、BM25 索引和 arXiv Atom 原始响应快照。该样本只支持摘要级检索，不代表全文检索。

```powershell
.venv\Scripts\python.exe lit.py search "corrective retrieval" --retriever bm25
```

## 构建：100 篇全文试点

全文数据和索引不会放进 Git 仓库。以下步骤会从固定版本的 LitSearch/S2ORC 数据源准备试点语料，并在本地生成父文档、子块及索引。首次运行会下载数据和模型，需预留磁盘空间与网络时间。

```powershell
.venv\Scripts\python.exe litsearch.py prepare
.venv\Scripts\python.exe litsearch_fulltext.py prepare --pilot-100
.venv\Scripts\python.exe litsearch_fulltext.py chunk
.venv\Scripts\python.exe litsearch_fulltext.py index-bm25
.venv\Scripts\python.exe litsearch_fulltext.py index-dense
.venv\Scripts\python.exe litsearch_fulltext.py search "How does corrective RAG handle poor retrieval?" --pipeline classic --no-rerank
```

试点并非随机抽样，不能用来推断全量语料表现。全量 64,183 篇语料和全量索引构建会占用大量磁盘、内存和时间；请先阅读 [数据说明](data/README.md) 与[评测说明](eval/RESULTS.md)。

## 问答与网页界面

复制模板后，把自己的密钥填入本地 `.env`：

```powershell
Copy-Item .env.example .env
notepad .env
```

在编辑器里把 `DEEPSEEK_API_KEY=` 后面填成你自己的 Key。`.env` 已加入 Git 忽略规则；模板不含任何 Key。不要把密钥写入代码、报告或提交记录。命令行也支持在没有配置时隐藏提示输入，交互输入不会写入项目文件。全文问答和 Agentic 查询需要可用的 DeepSeek API；英文 Classic 检索可离线运行。

在完成全文数据和索引构建后，可启动网页界面：

```powershell
.venv\Scripts\python.exe webui\server.py --no-warmup
```

然后打开 <http://127.0.0.1:8000>。当前副本没有内置全文索引，因此完成本地建库前，全文页面无法执行检索。

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试和离线检索不需要 API Key。项目有些测试依赖仓库内的小型摘要样本，但不依赖 LitSearch 全量全文或 `.rag/` 模型缓存。

## 许可与数据来源

代码仓库没有预设开源许可证；发布前请确定代码许可证。全量论文正文的分发授权未在本项目中统一核实，因此本仓库不包含正文或 PDF。摘要样本与外部数据的使用应遵守各自来源条款。
