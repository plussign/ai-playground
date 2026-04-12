"""
VectorStore.py
使用 sentence_transformers + 'BAAI/bge-m3' 模型和 chromadb，
递归遍历当前目录下的所有py脚本，使用ast合理分块后，将代码段信息存入向量数据库。
"""

import os
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings
import json

# 设置 ModelScope 作为模型下载源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'

# 配置
EMBEDDING_MODEL = 'BAAI/bge-m3'
CHROMA_DB_PATH = "./code_vector_db"
COLLECTION_NAME = "python_code_blocks"


class PythonCodeChunker:
    """使用AST对Python代码进行合理分块"""

    def __init__(self, min_chunk_lines=5, max_chunk_lines=200):
        self.min_chunk_lines = min_chunk_lines
        self.max_chunk_lines = max_chunk_lines

    def chunk_file(self, file_path: str) -> list[dict]:
        """对单个Python文件进行分块"""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            print(f"  读取文件失败 {file_path}: {e}")
            return chunks

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"  语法错误 {file_path}: {e}")
            # 如果无法解析，返回整个文件作为一个块
            chunks.append({
                'code': source,
                'type': 'file',
                'name': os.path.basename(file_path),
                'start_line': 1,
                'end_line': len(source.splitlines()),
                'path': file_path
            })
            return chunks

        # 遍历AST节点进行分块
        self._traverse_and_chunk(tree, source, file_path, chunks)

        # 如果没有分块成功，将整个文件作为一块
        if not chunks:
            chunks.append({
                'code': source,
                'type': 'file',
                'name': os.path.basename(file_path),
                'start_line': 1,
                'end_line': len(source.splitlines()),
                'path': file_path
            })

        return chunks

    def _traverse_and_chunk(self, tree: ast.AST, source: str, file_path: str, chunks: list):
        """遍历AST节点并提取代码块"""
        lines = source.splitlines()

        for node in ast.walk(tree):
            # 只处理顶层或二层的定义节点
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # 获取节点的起始和结束行号
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                if end_line is None:
                    end_line = start_line

                line_count = end_line - start_line + 1

                # 跳过太小的块
                if line_count < self.min_chunk_lines:
                    continue

                # 提取代码文本
                if start_line <= len(lines):
                    code_lines = lines[start_line - 1:end_line]
                    code_text = '\n'.join(code_lines)

                    chunks.append({
                        'code': code_text,
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'name': node.name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'path': file_path,
                        'docstring': ast.get_docstring(node) or ''
                    })

    def chunk_directory(self, directory: str) -> list[dict]:
        """递归遍历目录，_chunk所有Python文件"""
        all_chunks = []
        directory = Path(directory)

        for py_file in directory.rglob("*.py"):
            # 排除自身
            if py_file.name == 'VectorStore.py' or py_file.name == 'VectorSearch.py':
                continue

            print(f"处理文件: {py_file}")
            file_chunks = self.chunk_file(str(py_file))
            print(f"  提取到 {len(file_chunks)} 个代码块")
            all_chunks.extend(file_chunks)

        return all_chunks


class VectorStore:
    """向量数据库管理器"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, db_path: str = CHROMA_DB_PATH):
        print(f"加载模型: {model_name}")
        # 使用 modelscope 镜像
        self.model = SentenceTransformer(
            model_name,
            cache_folder="./models"  # 本地缓存目录
        )

        print(f"初始化ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Python代码块向量数据库"}
        )

    def add_chunks(self, chunks: list[dict]):
        """将代码块添加到向量数据库"""
        if not chunks:
            print("没有代码块需要添加")
            return

        print(f"开始向量化 {len(chunks)} 个代码块...")

        # 准备数据
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            ids.append(chunk_id)

            # 文档内容 = 代码 + 描述信息
            doc_content = f"[{chunk['type']}] {chunk['name']}\n{chunk['code']}"
            documents.append(doc_content)

            # 元数据
            metadatas.append({
                'path': chunk['path'],
                'name': chunk['name'],
                'type': chunk['type'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line'],
                'docstring': chunk.get('docstring', '')[:500]  # 限制长度
            })

        # 批量生成embeddings
        print("生成embeddings...")
        embeddings = self.model.encode(documents, show_progress_bar=True)

        # 转换为list格式
        embeddings_list = embeddings.tolist()

        # 添加到数据库
        print("写入向量数据库...")
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

        print(f"成功添加 {len(chunks)} 个代码块到向量数据库")

    def build_index(self, directory: str = "."):
        """构建向量索引"""
        print("=" * 50)
        print("开始构建代码向量索引")
        print("=" * 50)

        # 分块代码
        chunker = PythonCodeChunker()
        chunks = chunker.chunk_directory(directory)
        print(f"\n总共提取 {len(chunks)} 个代码块\n")

        # 存入向量数据库
        self.add_chunks(chunks)

        print("=" * 50)
        print("索引构建完成!")
        print("=" * 50)

    def clear(self):
        """清空向量数据库"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Python代码块向量数据库"}
        )
        print("向量数据库已清空")


def main():
    """主函数"""
    # 创建向量存储实例
    vector_store = VectorStore()

    # 可以选择先清空现有数据
    # vector_store.clear()

    # 构建索引
    vector_store.build_index(".")


if __name__ == "__main__":
    main()