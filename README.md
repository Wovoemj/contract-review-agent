# 合同审查 Agent (Contract Review Agent)

基于 LangGraph 的智能合同风险分析与问答系统。上传合同文件（PDF/TXT/MD/DOC/DOCX），自动识别合同类型、提取关键条款、评估法律风险，支持多轮追问和批量分析。

## 核心功能

- **多格式解析** — PDF、TXT、Markdown、DOCX 等格式自动解析
- **合同类型识别** — 自动识别劳动合同、租房合同、采购合同、技术服务合同、NDA、实习协议等
- **风险条款标注** — 逐条审查合同，标注风险等级（高/中/低），给出修改建议和法律依据
- **RAG 知识增强** — 基于法律知识库的检索增强生成，风险评估有法可依
- **多轮对话** — 分析完成后可针对具体条款追问，SSE 流式返回
- **批量分析** — 支持文件夹上传，并发分析多份合同（并发度=3）
- **历史记录** — 分析结果持久化，支持查看和删除历史会话

## 技术架构

```
┌─────────────────────────────────────────────────┐
│                  前端 (单文件 HTML)                │
│   拖拽上传 / 文件夹上传 / 历史面板 / SSE 流式展示    │
└────────────────────┬────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────┐
│              FastAPI 后端 (main.py)               │
│   /api/contract/upload  /api/chat  /api/sessions │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│           LangGraph 工作流 (graph.py)             │
│                                                  │
│  Parser → Classifier → RAG → Extractor →         │
│          Assessor → Generator → END              │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Mimo LLM │  │ ChromaDB │  │SiliconFlow│      │
│  │ (mimo-v2 │  │ (向量存储) │  │(Embedding)│      │
│  │  .5-pro) │  │          │  │ bge-large │      │
│  └──────────┘  └──────────┘  │ -zh-v1.5  │      │
│                               └──────────┘       │
└──────────────────────────────────────────────────┘
```

### 工作流节点说明

| 节点 | 作用 | 模型/工具 |
|------|------|-----------|
| **Parser** | 按文件格式提取纯文本 | pdfplumber / python-docx / open() |
| **Classifier** | 识别合同类型标签 | Mimo LLM (temperature=0) |
| **RAG Retriever** | 从法律知识库检索相关条文 | ChromaDB + BAAI/bge-large-zh-v1.5 |
| **Extractor** | 提取关键条款（期限、薪资、违约等） | Mimo LLM (temperature=0) |
| **Assessor** | 逐条评估风险，结合 RAG 法律知识 | Mimo LLM (temperature=0) |
| **Generator** | 格式化分析报告 / 流式问答 | Mimo LLM (temperature=0.7) |

## 项目结构

```
合同审查/
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置管理 (Pydantic Settings)
│   ├── main.py            # FastAPI 入口 + API 路由
│   ├── graph.py           # LangGraph 工作流定义
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py     # Pydantic 数据模型
│   └── rag/
│       ├── __init__.py
│       └── retriever.py   # RAG 文档检索器
├── data/
│   ├── contracts/         # 合同样本 (10份)
│   ├── annotations/       # 风险标注 (10份)
│   ├── knowledge_base/    # 法律知识库
│   ├── uploads/           # 上传文件存储
│   └── chroma_db/         # ChromaDB 持久化
├── static/
│   └── index.html         # 前端页面
├── tests/                 # 测试用例
├── .env.example           # 环境变量模板
├── requirements.txt       # Python 依赖
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 API 密钥

复制 `.env.example` 为 `.env`，填入：

```env
# LLM (Mimo API)
LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_API_KEY=your-mimo-api-key
LLM_MODEL=mimo-v2.5-pro

# Embedding (SiliconFlow)
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=your-siliconflow-key
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 3. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 `http://localhost:8000`。

### 4. API 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 上传合同
curl -F "file=@contract.pdf" http://localhost:8000/api/contract/upload

# 追问
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"xxx","message":"这个竞业限制条款合理吗？"}'
```

## 技术栈

- **后端**: FastAPI + LangGraph + LangChain
- **LLM**: Mimo v2.5-pro (小米 MiMo)
- **Embedding**: BAAI/bge-large-zh-v1.5 (SiliconFlow)
- **向量数据库**: ChromaDB
- **文档解析**: pdfplumber + python-docx
- **前端**: 原生 HTML/CSS/JS (单文件)
- **流式传输**: SSE (Server-Sent Events)

## License

MIT
