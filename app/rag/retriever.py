"""RAG知识库检索"""
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class DocumentRetriever:
    """文档检索器"""
    
    def __init__(self, knowledge_base_dir: str, chroma_persist_dir: str, collection_name: str):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.collection_name = collection_name
        self.vector_store = None
        self.embeddings = None
    
    def _init_embeddings(self):
        """初始化Embedding模型"""
        self.embeddings = OpenAIEmbeddings(
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.getenv("EMBEDDING_API_KEY", "dummy"),
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        )
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """加载知识库文档"""
        documents = []
        
        for md_file in self.knowledge_base_dir.glob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 按章节拆分
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，"]
            )
            
            chunks = splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "page_content": chunk,
                    "metadata": {
                        "source": md_file.name,
                        "chunk_index": i,
                        "file_type": "knowledge_base"
                    }
                })
        
        return documents
    
    def initialize(self):
        """初始化向量存储"""
        if self.embeddings is None:
            self._init_embeddings()
        
        # 加载文档
        documents = self._load_documents()
        
        if not documents:
            print("警告: 知识库为空")
            return
        
        # 创建或加载向量存储
        if self.chroma_persist_dir.exists() and any(self.chroma_persist_dir.iterdir()):
            print("加载已有向量数据库...")
            self.vector_store = Chroma(
                persist_directory=str(self.chroma_persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            print(f"创建新向量数据库，共 {len(documents)} 个文档片段...")
            texts = [doc["page_content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            self.vector_store = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_persist_dir),
                collection_name=self.collection_name
            )
            print("向量数据库创建完成")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            # ChromaDB返回L2距离，转换为相似度
            similarity = 1 / (1 + score)
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "similarity": round(similarity, 4)
            })
        
        return formatted_results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """获取检索上下文"""
        results = self.search(query, k=k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[参考文档{i}] ({r['source']})\n{r['content']}")
        
        return "\n\n".join(context_parts)
