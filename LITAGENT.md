# LitAgent 项目说明

LitAgent 面向计算机论文检索与问答。当前主线是 `litsearch_fulltext.py` 和 `litagent/`：英文论文全文作为父文档，解析后切分为带稳定 ID 的子块；Dense 使用 BGE 与 FAISS，稀疏检索使用 BM25，候选通过 RRF 融合并由本地 Cross Encoder 重排。查询规划和子问题拆解支持中文、英文输入；回答会附带原文证据并进行引用核验。

完整数据和模型缓存位于本地生成目录，不属于 Git 仓库内容。新环境的入口、数据准备命令和测试方式见 [README.md](README.md)；架构流程见 [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)，评测口径见 [eval/RESULTS.md](eval/RESULTS.md)。

API 密钥只从进程环境或本地 `.env` 读取。模板见 `.env.example`；不得在文档或源代码中写入密钥。
