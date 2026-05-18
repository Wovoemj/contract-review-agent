"""LangGraph工作流定义"""
import os
import json
import uuid
import time
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import time

from app.models.schemas import ContractAnalysis, RiskClause, Message
from app.rag.retriever import DocumentRetriever




# ============ 状态定义 ============

class ContractReviewState(TypedDict):
    """合同审查状态"""
    # 输入
    session_id: str
    user_message: str
    contract_text: Optional[str]
    contract_filename: Optional[str]
    
    # 中间结果
    contract_type: Optional[str]
    extracted_clauses: Optional[Dict[str, str]]
    risk_clauses: Optional[List[Dict[str, Any]]]
    retrieved_docs: Optional[str]
    
    # 输出
    analysis: Optional[ContractAnalysis]
    response: str
    stream_chunks: List[str]
    
    # 元数据
    metadata: Dict[str, Any]
    error: Optional[str]


# ============ 节点实现 ============

class ContractParserNode:
    """合同解析节点 - 从多种格式提取文本"""
    
    def parse_file(self, state: ContractReviewState) -> Dict[str, Any]:
        """解析文件，支持多种格式"""
        filepath = state.get("metadata", {}).get("filepath")
        if not filepath:
            return {"error": "未找到文件路径"}
        
        path = Path(filepath)
        ext = path.suffix.lower()
        
        try:
            if ext == ".pdf":
                text = self._parse_pdf(filepath)
            elif ext in [".txt", ".md", ".json", ".csv", ".rtf"]:
                text = self._parse_text(filepath)
            elif ext in [".doc", ".docx"]:
                text = self._parse_docx(filepath)
            else:
                return {"error": f"不支持的文件格式: {ext}"}
            
            if not text or len(text.strip()) < 10:
                return {"error": "文件内容为空或过短"}
            
            print(f"[Parser] 文件解析完成 - 格式: {ext}, 文本长度: {len(text)}")
            return {"contract_text": text}
        except Exception as e:
            return {"error": f"文件解析失败: {str(e)}"}
    
    def _parse_pdf(self, filepath: str) -> str:
        """解析PDF文件"""
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    
    def _parse_text(self, filepath: str) -> str:
        """解析纯文本文件 (txt, md, json, csv, rtf)"""
        # 尝试多种编码
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("无法识别文件编码")
    
    def _parse_docx(self, filepath: str) -> str:
        """解析Word文档 (docx/doc)"""
        try:
            import docx
            doc = docx.Document(filepath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            # 如果没有python-docx，尝试作为纯文本读取
            return self._parse_text(filepath)


class ContractTypeClassifierNode:
    """合同类型识别节点"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy"),
            model=os.getenv("LLM_MODEL", "mimo-v2.5-pro"),
            temperature=0
        )
    
    async def classify(self, state: ContractReviewState) -> Dict[str, Any]:
        """识别合同类型"""
        contract_text = state.get("contract_text", "")
        if not contract_text:
            return {"error": "合同文本为空"}
        
        prompt = f"""请识别以下合同的类型，只返回类型标识符，不要其他内容。

可选类型：
- labor: 劳动合同
- rental: 租房合同
- procurement: 采购合同
- service: 技术服务合同
- nda: 保密协议/NDA
- internship: 实习协议
- unknown: 无法识别

合同内容：
{contract_text[:2000]}

类型标识符："""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            contract_type = response.content.strip().lower()
            
            # 验证类型是否有效
            valid_types = ["labor", "rental", "procurement", "service", "nda", "internship", "unknown"]
            if contract_type not in valid_types:
                contract_type = "unknown"
            
            print(f"[Langfuse] 合同类型识别完成 - 类型: {contract_type}")
            return {"contract_type": contract_type}
        except Exception as e:
            return {"error": f"合同类型识别失败: {str(e)}"}


class ClauseExtractorNode:
    """条款提取节点"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy"),
            model=os.getenv("LLM_MODEL", "mimo-v2.5-pro"),
            temperature=0
        )
    
    async def extract(self, state: ContractReviewState) -> Dict[str, Any]:
        """提取关键条款"""
        contract_text = state.get("contract_text", "")
        contract_type = state.get("contract_type", "unknown")
        
        type_names = {
            "labor": "劳动合同", "rental": "租房合同", "procurement": "采购合同",
            "service": "技术服务合同", "nda": "保密协议", "internship": "实习协议"
        }   
        
        prompt = f"""你是一个专业的合同审查助手。请从以下{type_names.get(contract_type, '合同')}中提取关键条款。

要求：
1. 提取以下类型的条款（如果存在）：
   - 合同期限
   - 报酬/薪资/费用
   - 工作内容/服务内容
   - 违约责任
   - 保密条款
   - 竞业限制
   - 解除/终止条件
   - 争议解决方式

2. 对于每个条款，请用JSON格式返回，key为条款类型，value为条款原文摘要

合同内容：
{contract_text[:4000]}

请直接返回JSON格式，不要其他内容："""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # 尝试解析JSON
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            
            extracted_clauses = json.loads(content)
            print(f"[Langfuse] 条款提取完成 - 条款数: {len(extracted_clauses)}, 合同类型: {contract_type}")
            return {"extracted_clauses": extracted_clauses}
        except json.JSONDecodeError:
            # 如果解析失败，返回原始文本
            return {"extracted_clauses": {"raw": response.content}}
        except Exception as e:
            return {"error": f"条款提取失败: {str(e)}"}


class RiskAssessorNode:
    """风险评估节点"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy"),
            model=os.getenv("LLM_MODEL", "mimo-v2.5-pro"),
            temperature=0
        )
    
    async def assess(self, state: ContractReviewState) -> Dict[str, Any]:
        """评估风险条款"""
        contract_text = state.get("contract_text", "")
        contract_type = state.get("contract_type", "unknown")
        retrieved_docs = state.get("retrieved_docs", "")
        
        prompt = f"""你是一个专业的合同审查律师。请逐条审查以下合同，找出其中所有存在法律风险的条款。

## 合同类型
{contract_type}

## 合同内容
{contract_text[:8000]}

## 相关法律知识
{retrieved_docs[:3000] if retrieved_docs else "无"}

## 要求
1. 逐条仔细审查合同，识别所有存在风险的条款，不要遗漏（没有上限，有多少标多少）
2. 对每个风险条款，评估风险等级（high/medium/low）
3. 给出具体的修改建议
4. 引用相关法律依据
5. 如果某个条款存在多个风险点，分别列出

请以JSON数组格式返回，每个元素包含：
- clause_id: 条款编号
- clause_title: 条款标题
- clause_text: 条款原文
- risk_level: high/medium/low
- risk_type: 风险类型
- description: 风险说明
- suggestion: 修改建议
- legal_basis: 法律依据

直接返回JSON数组，不要其他内容："""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            
            risk_clauses_raw = json.loads(content)
            
            # 转换为RiskClause对象
            risk_clauses = []
            for rc in risk_clauses_raw:
                risk_clauses.append(RiskClause(**rc))
            
            # 计算整体风险等级
            risk_levels = [rc.risk_level for rc in risk_clauses]
            if "high" in risk_levels:
                overall_risk = "high"
            elif "medium" in risk_levels:
                overall_risk = "medium"
            else:
                overall_risk = "low"
            
            # 生成风险总结
            type_names = {
                "labor": "劳动合同", "rental": "租房合同", "procurement": "采购合同",
                "service": "技术服务合同", "nda": "保密协议", "internship": "实习协议"
            }
            
            analysis = ContractAnalysis(
                contract_type=contract_type,
                contract_type_cn=type_names.get(contract_type, "未知类型"),
                risk_clauses=risk_clauses,
                risk_summary=f"共发现 {len(risk_clauses)} 个风险条款，整体风险等级为{overall_risk}",
                overall_risk_level=overall_risk
            )
            
            print(f"[Langfuse] 风险评估完成 - 风险条款数: {len(risk_clauses)}, 整体风险: {overall_risk}")
            return {"risk_clauses": risk_clauses_raw, "analysis": analysis}
        except Exception as e:
            return {"error": f"风险评估失败: {str(e)}"}


class RAGRetrieverNode:
    """RAG检索节点"""
    
    def __init__(self, retriever: DocumentRetriever):
        self.retriever = retriever
    
    async def retrieve(self, state: ContractReviewState) -> Dict[str, Any]:
        """检索相关法律知识"""
        contract_text = state.get("contract_text", "")
        user_message = state.get("user_message", "")
        
        # 构建查询
        query = f"{user_message} {contract_text[:500]}" if user_message else contract_text[:500]
        
        try:
            context = self.retriever.get_context(query, k=3)
            print(f"[Langfuse] RAG检索完成 - 查询长度: {len(query)}, 上下文长度: {len(context)}")
            return {"retrieved_docs": context}
        except Exception as e:
            return {"retrieved_docs": "", "error": f"RAG检索失败: {str(e)}"}


class ResponseGeneratorNode:
    """响应生成节点"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
            api_key=os.getenv("LLM_API_KEY", "dummy"),
            model=os.getenv("LLM_MODEL", "mimo-v2.5-pro"),
            temperature=0.7,
            streaming=True
        )
    
    async def generate(self, state: ContractReviewState) -> Dict[str, Any]:
        """生成回复"""
        analysis = state.get("analysis")
        user_message = state.get("user_message", "")
        retrieved_docs = state.get("retrieved_docs", "")
        
        if analysis:
            # 合同分析结果回复
            response = self._format_analysis_response(analysis)
            print(f"[Langfuse] 响应生成完成 - 类型: analysis, 长度: {len(response)}")
            return {"response": response}
        else:
            # 对话问答回复
            contract_text = state.get("contract_text", "")
            prompt = f"""你是一个专业的合同审查助手。请根据以下信息回答用户的问题。

## 合同内容
{contract_text[:2000] if contract_text else "无"}

## 相关法律知识
{retrieved_docs[:1000] if retrieved_docs else "无"}

## 用户问题
{user_message}

请用专业但易懂的语言回答，必要时引用法律条文。"""
            
            try:
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                print(f"[Langfuse] 响应生成完成 - 类型: conversation, 长度: {len(response.content)}")
                return {"response": response.content}
            except Exception as e:
                return {"response": f"抱歉，回答时出现错误: {str(e)}"}
    
    def _format_analysis_response(self, analysis: ContractAnalysis) -> str:
        """格式化分析结果"""
        lines = [
            f"## 合同审查报告",
            f"",
            f"**合同类型**: {analysis.contract_type_cn}",
            f"**整体风险等级**: {analysis.overall_risk_level.upper()}",
            f"",
            f"### 风险条款 ({len(analysis.risk_clauses)}个)",
            ""
        ]
        
        for i, rc in enumerate(analysis.risk_clauses, 1):
            risk_level_map = {"high": "🔴 高风险", "medium": "🟡 中风险", "low": "🟢 低风险"}
            lines.extend([
                f"**{i}. {rc.clause_title}** {risk_level_map.get(rc.risk_level, '')}",
                f"条款: {rc.clause_text[:100]}...",
                f"风险: {rc.description}",
                f"建议: {rc.suggestion}",
                f"依据: {rc.legal_basis}",
                ""
            ])
        
        lines.extend([
            f"### 总结",
            f"{analysis.risk_summary}",
            "",
            f"---",
            f"*如需了解具体条款的详细分析，请告诉我。*"
        ])
        
        return "\n".join(lines)


# ============ 图构建 ============

def build_contract_review_graph(retriever: DocumentRetriever) -> StateGraph:
    """构建合同审查工作流图"""
    
    # 初始化节点
    parser = ContractParserNode()
    classifier = ContractTypeClassifierNode()
    extractor = ClauseExtractorNode()
    assessor = RiskAssessorNode()
    rag_retriever = RAGRetrieverNode(retriever)
    generator = ResponseGeneratorNode()
    
    # 创建图
    workflow = StateGraph(ContractReviewState)
    
    # 添加节点
    workflow.add_node("parser", parser.parse_file)
    workflow.add_node("classifier", classifier.classify)
    workflow.add_node("rag_retriever", rag_retriever.retrieve)
    workflow.add_node("extractor", extractor.extract)
    workflow.add_node("assessor", assessor.assess)
    workflow.add_node("generator", generator.generate)
    
    # 定义边
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "classifier")
    workflow.add_edge("classifier", "rag_retriever")
    workflow.add_edge("rag_retriever", "extractor")
    workflow.add_edge("extractor", "assessor")
    workflow.add_edge("assessor", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()


def build_conversation_graph(retriever: DocumentRetriever) -> StateGraph:
    """构建对话问答图（用于合同上传后的追问）"""
    
    rag_retriever = RAGRetrieverNode(retriever)
    generator = ResponseGeneratorNode()
    
    workflow = StateGraph(ContractReviewState)
    
    workflow.add_node("rag_retriever", rag_retriever.retrieve)
    workflow.add_node("generator", generator.generate)
    
    workflow.set_entry_point("rag_retriever")
    workflow.add_edge("rag_retriever", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()
