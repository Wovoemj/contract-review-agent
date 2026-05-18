"""数据模型定义"""
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# ============ 合同分析相关模型 ============

class RiskClause(BaseModel):
    """风险条款"""
    clause_id: str = Field(description="条款编号，如'第5条'")
    clause_title: str = Field(description="条款标题")
    clause_text: str = Field(description="条款原文")
    risk_level: Literal["high", "medium", "low"] = Field(description="风险等级")
    risk_type: str = Field(description="风险类型，如'竞业限制过严'")
    description: str = Field(description="风险说明")
    suggestion: str = Field(description="修改建议")
    legal_basis: str = Field(description="法律依据")


class ContractAnalysis(BaseModel):
    """合同分析结果"""
    contract_type: str = Field(description="合同类型")
    contract_type_cn: str = Field(description="合同类型中文名")
    parties: dict = Field(description="合同双方", default_factory=dict)
    key_terms: dict = Field(description="关键条款摘要", default_factory=dict)
    risk_clauses: List[RiskClause] = Field(description="风险条款列表", default_factory=list)
    risk_summary: str = Field(description="风险总结")
    overall_risk_level: Literal["high", "medium", "low"] = Field(description="整体风险等级")
    analyzed_at: datetime = Field(default_factory=datetime.now)


# ============ 对话相关模型 ============

class Message(BaseModel):
    """对话消息"""
    role: Literal["user", "assistant"] = Field(description="角色")
    content: str = Field(description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now)


class ConversationSession(BaseModel):
    """对话会话"""
    session_id: str = Field(description="会话ID")
    contract_filename: Optional[str] = Field(description="合同文件名", default=None)
    contract_analysis: Optional[ContractAnalysis] = Field(description="合同分析结果", default=None)
    messages: List[Message] = Field(description="对话历史", default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ============ API请求/响应模型 ============

class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str = Field(description="会话ID")
    message: str = Field(description="用户消息")


class ContractUploadResponse(BaseModel):
    """合同上传响应"""
    session_id: str = Field(description="会话ID")
    filename: str = Field(description="文件名")
    analysis: ContractAnalysis = Field(description="分析结果")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    components: dict = Field(default_factory=dict)
