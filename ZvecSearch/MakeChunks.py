"""
MakeChunks.py
负责将 Python 代码和 JSON 数据分块。
"""

import ast
import json
import os
from pathlib import Path


class PythonCodeChunker:
    """使用 AST 对 Python 代码进行合理分块"""

    def __init__(self, min_chunk_lines=5, max_chunk_lines=200):
        self.min_chunk_lines = min_chunk_lines
        self.max_chunk_lines = max_chunk_lines

    def chunk_file(self, file_path: str) -> list[dict]:
        """对单个 Python 文件进行分块"""
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
                'path': file_path,
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
                'path': file_path,
            })

        return chunks

    def _traverse_and_chunk(self, tree: ast.AST, source: str, file_path: str, chunks: list):
        """遍历 AST 节点并提取代码块"""
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
                        'docstring': ast.get_docstring(node) or '',
                    })

    def chunk_directory(self, directory: str) -> list[dict]:
        """递归遍历目录，分块所有 Python 文件"""
        all_chunks = []
        directory = Path(directory)

        for py_file in directory.rglob("*.py"):
            if py_file.name in {'VectorStore.py', 'VectorSearch.py', 'MakeChunks.py'}:
                continue

            print(f"处理文件: {py_file}")
            file_chunks = self.chunk_file(str(py_file))
            print(f"  提取到 {len(file_chunks)} 个代码块")
            all_chunks.extend(file_chunks)

        return all_chunks


class JSONChunker:
    """对 JSON 文件进行智能分块"""

    def __init__(self, max_chunk_size=2000, min_chunk_size=50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, file_path: str) -> list[dict]:
        """对单个 JSON 文件进行分块"""
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
                'path': file_path,
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
                'path': file_path,
            })

        return chunks

    def _chunk_json_value(self, value, file_path: str, name: str, path: str, chunks: list, depth: int):
        """递归处理 JSON 值并创建块"""
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
                    'json_path': current_path,
                })

    def _chunk_json_object(self, obj: dict, file_path: str, path: str, chunks: list, depth: int):
        """处理 JSON 对象"""
        obj_str = json.dumps(obj, ensure_ascii=False, indent=2)

        if len(obj_str) <= self.max_chunk_size or depth > 5:
            chunks.append({
                'code': f'"{path}": {obj_str}',
                'type': 'json_object',
                'name': path,
                'start_line': 1,
                'end_line': len(obj_str.splitlines()),
                'path': file_path,
                'json_path': path,
            })
        else:
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._chunk_json_value(value, file_path, key, new_path, chunks, depth + 1)

    def _chunk_json_array(self, arr: list, file_path: str, path: str, chunks: list, depth: int):
        """处理 JSON 数组"""
        arr_str = json.dumps(arr, ensure_ascii=False, indent=2)

        if len(arr_str) <= self.max_chunk_size or depth > 5 or len(arr) <= 3:
            chunks.append({
                'code': f'"{path}": {arr_str}',
                'type': 'json_array',
                'name': path,
                'start_line': 1,
                'end_line': len(arr_str.splitlines()),
                'path': file_path,
                'json_path': path,
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
                    'json_path': new_path,
                })
