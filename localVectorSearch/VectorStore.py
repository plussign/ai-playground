"""
VectorStore.py
使用 sentence_transformers + 'BAAI/bge-m3' 模型和 chromadb，
递归遍历当前目录下的所有py脚本和json文件，使用合理分块后，将内容存入向量数据库。
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

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置
EMBEDDING_MODEL = 'BAAI/bge-m3'
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "../code_vector_db")
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
            chunks.append({
                'code': source,
                'type': 'file',
                'name': os.path.basename(file_path),
                'start_line': 1,
                'end_line': len(source.splitlines()),
                'path': file_path
            })
            return chunks

        self._traverse_and_chunk(tree, source, file_path, chunks)

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
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                if end_line is None:
                    end_line = start_line

                line_count = end_line - start_line + 1

                if line_count < self.min_chunk_lines:
                    continue

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
            if py_file.name == 'VectorStore.py' or py_file.name == 'VectorSearch.py':
                continue

            print(f"处理文件: {py_file}")
            file_chunks = self.chunk_file(str(py_file))
            print(f"  提取到 {len(file_chunks)} 个代码块")
            all_chunks.extend(file_chunks)

        return all_chunks


class JSONChunker:
    """对JSON文件进行智能分块"""

    def __init__(self, max_chunk_size=2000, min_chunk_size=50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, file_path: str) -> list[dict]:
        """对单个JSON文件进行分块"""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON解析错误 {file_path}: {e}")
            chunks.append({
                'code': content,
                'type': 'json_file',
                'name': os.path.basename(file_path),
                'start_line': 1,
                'end_line': len(content.splitlines()),
                'path': file_path
            })
            return chunks
        except Exception as e:
            print(f"  读取文件失败 {file_path}: {e}")
            return chunks

        self._chunk_json_value(data, file_path, os.path.basename(file_path), "", chunks, 1)

        if not chunks:
            chunks.append({
                'code': content,
                'type': 'json_file',
                'name': os.path.basename(file_path),
                'start_line': 1,
                'end_line': len(content.splitlines()),
                'path': file_path
            })

        return chunks

    def _chunk_json_value(self, value, file_path: str, name: str, path: str, chunks: list, depth: int):
        """递归处理JSON值并创建块"""
        current_path = path if path else name

        if isinstance(value, dict):
            self._chunk_json_object(value, file_path, current_path, chunks, depth)
        elif isinstance(value, list):
            self._chunk_json_array(value, file_path, current_path, chunks, depth)
        else:
            json_str = json.dumps(value, ensure_ascii=False, indent=2)
            if len(json_str) >= self.min_chunk_size:
                chunks.append({
                    'code': f'"{current_path}": {json_str}',
                    'type': 'json_value',
                    'name': current_path,
                    'start_line': 1,
                    'end_line': len(json_str.splitlines()),
                    'path': file_path,
                    'json_path': current_path
                })

    def _chunk_json_object(self, obj: dict, file_path: str, path: str, chunks: list, depth: int):
        """处理JSON对象"""
        obj_str = json.dumps(obj, ensure_ascii=False, indent=2)

        if len(obj_str) <= self.max_chunk_size or depth > 5:
            chunks.append({
                'code': f'"{path}": {obj_str}',
                'type': 'json_object',
                'name': path,
                'start_line': 1,
                'end_line': len(obj_str.splitlines()),
                'path': file_path,
                'json_path': path
            })
        else:
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._chunk_json_value(value, file_path, key, new_path, chunks, depth + 1)

    def _chunk_json_array(self, arr: list, file_path: str, path: str, chunks: list, depth: int):
        """处理JSON数组"""
        arr_str = json.dumps(arr, ensure_ascii=False, indent=2)

        if len(arr_str) <= self.max_chunk_size or depth > 5 or len(arr) <= 3:
            chunks.append({
                'code': f'"{path}": {arr_str}',
                'type': 'json_array',
                'name': path,
                'start_line': 1,
                'end_line': len(arr_str.splitlines()),
                'path': file_path,
                'json_path': path
            })
        else:
            batch_size = max(1, self.max_chunk_size // max(100, len(arr_str) // len(arr)))
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i + batch_size]
                batch_str = json.dumps(batch, ensure_ascii=False, indent=2)
                new_path = f"{path}[{i}:{i + len(batch)}]"
                chunks.append({
                    'code': f'"{new_path}": {batch_str}',
                    'type': 'json_array_batch',
                    'name': new_path,
                    'start_line': 1,
                    'end_line': len(batch_str.splitlines()),
                    'path': file_path,
                    'json_path': new_path
                })


class VectorStore:
    """向量数据库管理器"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, db_path: str = CHROMA_DB_PATH):
        print(f"加载模型: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            cache_folder=os.path.join(SCRIPT_DIR, "../models")
        )

        print(f"初始化ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Python代码和JSON数据向量数据库"}
        )

    def add_chunks(self, chunks: list[dict]):
        """将块添加到向量数据库"""
        if not chunks:
            print("没有块需要添加")
            return

        print(f"开始向量化 {len(chunks)} 个块...")

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            ids.append(chunk_id)

            doc_content = f"[{chunk['type']}] {chunk['name']}\n{chunk['code']}"
            documents.append(doc_content)

            metadata = {
                'path': chunk['path'],
                'name': chunk['name'],
                'type': chunk['type'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line']
            }
            if 'docstring' in chunk:
                metadata['docstring'] = chunk.get('docstring', '')[:500]
            if 'json_path' in chunk:
                metadata['json_path'] = chunk.get('json_path', '')
            metadatas.append(metadata)

        print("生成embeddings...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        embeddings_list = embeddings.tolist()

        print("写入向量数据库...")
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

        print(f"成功添加 {len(chunks)} 个块到向量数据库")

    def build_index(self, directory: str = "."):
        """构建向量索引"""
        print("=" * 50)
        print("开始构建代码向量索引")
        print("=" * 50)

        all_chunks = []

        print("\n--- 处理Python文件 ---")
        py_chunker = PythonCodeChunker()
        py_chunks = py_chunker.chunk_directory(directory)
        print(f"Python文件: 提取到 {len(py_chunks)} 个代码块")
        all_chunks.extend(py_chunks)

        print("\n--- 处理JSON文件 ---")
        json_chunker = JSONChunker()
        directory_path = Path(directory)
        for json_file in directory_path.rglob("*.json"):
            print(f"处理文件: {json_file}")
            file_chunks = json_chunker.chunk_file(str(json_file))
            print(f"  提取到 {len(file_chunks)} 个JSON块")
            all_chunks.extend(file_chunks)

        print(f"\n总共提取 {len(all_chunks)} 个块\n")

        self.add_chunks(all_chunks)

        print("=" * 50)
        print("索引构建完成!")
        print("=" * 50)

    def clear(self):
        """清空向量数据库"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Python代码和JSON数据向量数据库"}
        )
        print("向量数据库已清空")


def main():
    """主函数"""
    vector_store = VectorStore()
    vector_store.build_index(".")


if __name__ == "__main__":
    main()
