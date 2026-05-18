"""应用配置"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # 项目路径
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # LLM配置
    LLM_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "mimo-v2.5-pro"
    
    # Embedding配置
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    
    # 服务配置
    APP_PORT: int = 8000
    APP_HOST: str = "0.0.0.0"
    
    # ChromaDB配置
    CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "contract_knowledge_base"
    
    # Langfuse配置
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # 合同类型定义
    CONTRACT_TYPES: dict = {
        "labor": "劳动合同",
        "rental": "租房合同",
        "procurement": "采购合同",
        "service": "技术服务合同",
        "nda": "保密协议",
        "internship": "实习协议",
        "unknown": "未知类型"
    }
    
    # 风险等级定义
    RISK_LEVELS: dict = {
        "high": {"label": "高风险", "color": "#ef4444", "description": "违反强制性法律规定，建议修改或拒绝签署"},
        "medium": {"label": "中风险", "color": "#f59e0b", "description": "存在不合理条款，建议协商修改"},
        "low": {"label": "低风险", "color": "#22c55e", "description": "风险较低，可选择性优化"}
    }
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
