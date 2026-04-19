"""
MakeChunks.py
负责将 JSON 数据按行分块。
"""

import json
import os
from pathlib import Path


class JSONChunker:
    """对 JSON 文件按行分块"""

    def chunk_file(self, file_path: str) -> list[dict]:
        """对单个 JSON 文件按行分块"""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON解析错误 {file_path}: {e}")
            return chunks
        except Exception as e:
            print(f"  读取文件失败 {file_path}: {e}")
            return chunks

        # 处理 JSON 数据
        if isinstance(data, dict):
            # 对于键值对，key 存入"番号"字段，value 存入"标题"字段
            line_num = 1
            for key, value in data.items():
                # 将 value 转为字符串
                if isinstance(value, (dict, list)):
                    title = json.dumps(value, ensure_ascii=False)
                else:
                    title = str(value)

                chunks.append({
                    'code': f'"{key}": {title}',
                    'type': 'json_entry',
                    'name': key,
                    'start_line': line_num,
                    'end_line': line_num,
                    'path': file_path,
                    'film_code': key,
                    'film_title': title,
                })
                line_num += 1
        elif isinstance(data, list):
            # 对于数组，按值存入"番号"字段，"标题"为空
            line_num = 1
            for item in data:
                # 将 item 转为字符串
                if isinstance(item, (dict, list)):
                    film_code = json.dumps(item, ensure_ascii=False)
                else:
                    film_code = str(item)

                chunks.append({
                    'code': film_code,
                    'type': 'json_array_item',
                    'name': f'item_{line_num}',
                    'start_line': line_num,
                    'end_line': line_num,
                    'path': file_path,
                    'film_code': film_code,
                    'film_title': '',
                })
                line_num += 1
        else:
            # 对于单个值
            chunks.append({
                'code': str(data),
                'type': 'json_single_value',
                'name': os.path.basename(file_path),
                'start_line': 1,
                'end_line': 1,
                'path': file_path,
                'film_code': str(data),
                'film_title': '',
            })

        return chunks

    def chunk_directory(self, directory: str) -> list[dict]:
        """递归遍历目录，分块所有 JSON 文件"""
        all_chunks = []
        directory_path = Path(directory)

        for json_file in directory_path.rglob("*.json"):
            print(f"处理文件: {json_file}")
            file_chunks = self.chunk_file(str(json_file))
            print(f"  提取到 {len(file_chunks)} 个JSON块")
            all_chunks.extend(file_chunks)

        return all_chunks
