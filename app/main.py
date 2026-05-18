"""FastAPI主应用"""
import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from app.config import settings
from app.models.schemas import (
    ChatRequest, ContractAnalysis, ContractUploadResponse,
    HealthResponse, ConversationSession, Message
)
from app.rag.retriever import DocumentRetriever
from app.graph import build_contract_review_graph, build_conversation_graph

# 加载环境变量
load_dotenv()

# 创建应用
app = FastAPI(
    title="合同审查Agent",
    description="智能合同风险分析与问答系统",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = Path(__file__).parent.parent / "static"

# 全局变量
review_graph = None
conversation_graph = None
sessions: Dict[str, ConversationSession] = {}
upload_dir = Path(settings.PROJECT_ROOT / "data" / "uploads")
upload_dir.mkdir(parents=True, exist_ok=True)
history_file = Path(settings.PROJECT_ROOT / "data" / "history.json")

def load_sessions():
    """从磁盘加载历史会话"""
    global sessions
    if history_file.exists():
        try:
            raw = json.loads(history_file.read_text(encoding="utf-8"))
            for sid, data in raw.items():
                session = ConversationSession(**data)
                sessions[sid] = session
            print(f"[History] 加载了 {len(sessions)} 个历史会话")
        except Exception as e:
            print(f"[History] 加载历史失败: {e}")

def save_sessions():
    """保存所有会话到磁盘"""
    try:
        data = {}
        for sid, s in sessions.items():
            data[sid] = s.model_dump(mode="json")
        history_file.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        print(f"[History] 保存失败: {e}")


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global review_graph, conversation_graph
    
    print("正在初始化RAG知识库...")
    retriever = DocumentRetriever(
        knowledge_base_dir=str(settings.PROJECT_ROOT / "data" / "knowledge_base"),
        chroma_persist_dir=str(settings.CHROMA_PERSIST_DIR),
        collection_name=settings.CHROMA_COLLECTION_NAME
    )
    retriever.initialize()
    
    print("正在构建工作流图...")
    review_graph = build_contract_review_graph(retriever)
    conversation_graph = build_conversation_graph(retriever)
    
    load_sessions()
    print(f"初始化完成! 已加载 {len(sessions)} 个历史会话")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """提供前端页面"""
    html_path = static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        components={
            "review_graph": review_graph is not None,
            "conversation_graph": conversation_graph is not None,
            "sessions_count": len(sessions)
        }
    )


@app.post("/api/contract/upload")
async def upload_contract(file: UploadFile = File(...)):
    """上传合同文件进行分析"""
    # 支持的文件类型
    allowed_extensions = {".pdf", ".txt", ".md", ".doc", ".docx", ".json", ".csv", ".rtf"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持: {', '.join(allowed_extensions)}"
        )
    
    # 保存文件 (sanitize: folder upload sends full path like "data/contracts/08_xxx.md")
    session_id = str(uuid.uuid4())
    safe_filename = Path(file.filename).name  # only keep base filename
    file_path = upload_dir / f"{session_id}_{safe_filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 创建会话
    session = ConversationSession(
        session_id=session_id,
        contract_filename=file.filename
    )
    sessions[session_id] = session
    
    # 执行合同分析
    initial_state = {
        "session_id": session_id,
        "user_message": "",
        "contract_text": None,
        "contract_filename": file.filename,
        "contract_type": None,
        "extracted_clauses": None,
        "risk_clauses": None,
        "retrieved_docs": None,
        "analysis": None,
        "response": "",
        "stream_chunks": [],
        "metadata": {"filepath": str(file_path)},
        "error": None
    }
    
    try:
        start_time = time.time()
        result = await review_graph.ainvoke(initial_state)
        elapsed = time.time() - start_time
        print(f"[Langfuse] 合同分析完成，耗时: {elapsed:.2f}s")
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 更新会话
        session.contract_analysis = result.get("analysis")
        session.contract_text = result.get("contract_text", "")
        session.messages.append(Message(
            role="assistant",
            content=result.get("response", "")
        ))
        save_sessions()
        
        return ContractUploadResponse(
            session_id=session_id,
            filename=file.filename,
            analysis=result.get("analysis")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE流式对话"""
    session_id = request.session_id
    user_message = request.message
    
    # 获取会话
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 记录用户消息
    session.messages.append(Message(role="user", content=user_message))
    session.updated_at = datetime.now()
    
    async def event_generator():
        """SSE事件生成器"""
        yield {
            "event": "start",
            "data": json.dumps({"session_id": session_id})
        }
        
        # 构建状态
        state = {
            "session_id": session_id,
            "user_message": user_message,
            "contract_text": session.contract_text or "",
            "contract_filename": session.contract_filename,
            "contract_type": session.contract_analysis.contract_type if session.contract_analysis else None,
            "extracted_clauses": None,
            "risk_clauses": None,
            "retrieved_docs": None,
            "analysis": None,
            "response": "",
            "stream_chunks": [],
            "metadata": {},
            "error": None
        }
        
        try:
            start_time = time.time()
            # 使用对话图
            result = await conversation_graph.ainvoke(state)
            elapsed = time.time() - start_time
            print(f"[Langfuse] 对话响应完成，耗时: {elapsed:.2f}s")
            
            if result.get("error"):
                yield {
                    "event": "error",
                    "data": json.dumps({"error": result["error"]})
                }
            else:
                response = result.get("response", "")
                
                # 分块发送，模拟流式
                chunk_size = 20
                for i in range(0, len(response), chunk_size):
                    chunk = response[i:i + chunk_size]
                    yield {
                        "event": "message",
                        "data": json.dumps({"content": chunk})
                    }
                
                # 记录助手回复
                session.messages.append(Message(role="assistant", content=response))
                save_sessions()
            
            yield {
                "event": "end",
                "data": json.dumps({
                    "session_id": session_id,
                    "message_count": len(session.messages)
                })
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话信息"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话（含分析摘要）"""
    result = []
    for s in sessions.values():
        entry = {
            "session_id": s.session_id,
            "filename": s.contract_filename,
            "message_count": len(s.messages),
            "created_at": s.created_at.isoformat()
        }
        if s.contract_analysis:
            a = s.contract_analysis
            entry["contract_type_cn"] = a.contract_type_cn
            entry["overall_risk_level"] = a.overall_risk_level
            entry["risk_count"] = len(a.risk_clauses)
            entry["risk_summary"] = a.risk_summary
        result.append(entry)
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    del sessions[session_id]
    save_sessions()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)
