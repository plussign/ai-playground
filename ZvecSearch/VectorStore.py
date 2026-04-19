"""
VectorStore.py
使用 Ollama 嵌入模型和 zvec，
递归遍历当前目录下的所有 json 文件，按行分块后，将内容存入向量数据库。
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
import zvec

from MakeChunks import JSONChunker

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置
EMBEDDING_MODEL = 'qwen3-embed:8b-q8'
ZVEC_DB_PATH = os.path.join(SCRIPT_DIR, "../code_zvec_db")
COLLECTION_NAME = "json_data_blocks"
VECTOR_FIELD = "embedding"
MAX_WRITE_BATCH_SIZE = 1024


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    使用 Ollama SDK 获取文本的向量嵌入

    Args:
        text: 输入文本
        model: Ollama 模型名称

    Returns:
        向量嵌入列表
    """
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']


class VectorStore:
    """向量数据库管理器"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, db_path: str = ZVEC_DB_PATH):
        self.model_name = model_name
        self.db_path = db_path
        print(f"使用 Ollama 模型: {model_name}")
        print("请确保 Ollama 服务已启动，并且模型已通过 'ollama pull' 命令拉取")

        print(f"初始化 Zvec: {db_path}")
        self.collection = self._open_or_create_collection()

    def _build_schema(self, dimension: int) -> zvec.CollectionSchema:
        """构建集合 schema。"""
        return zvec.CollectionSchema(
            name=COLLECTION_NAME,
            fields=[
                zvec.FieldSchema(name="path", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="name", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="type", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="start_line", data_type=zvec.DataType.INT32),
                zvec.FieldSchema(name="end_line", data_type=zvec.DataType.INT32),
                zvec.FieldSchema(name="film_code", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="film_title", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="document", data_type=zvec.DataType.STRING),
            ],
            vectors=[
                zvec.VectorSchema(
                    name=VECTOR_FIELD,
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=dimension,
                    index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
                )
            ],
        )

    def _open_or_create_collection(self):
        """优先打开已有集合，不存在时按当前 embedding 维度创建。"""
        try:
            collection = zvec.open(self.db_path)
            print("已打开现有 Zvec 集合")
            return collection
        except Exception:
            probe_embedding = get_embedding("dimension probe", self.model_name)
            dimension = len(probe_embedding)
            print(f"创建新的 Zvec 集合，向量维度: {dimension}")
            schema = self._build_schema(dimension)
            return zvec.create_and_open(path=self.db_path, schema=schema)

    def add_chunks(self, chunks: list[dict]):
        """将块添加到向量数据库"""
        if not chunks:
            print("没有块需要添加")
            return

        print(f"开始向量化 {len(chunks)} 个块 (并发数: 5)...")

        docs = []
        embeddings_list = [None] * len(chunks)  # 预分配列表保持顺序

        # 准备所有文档和元数据
        tasks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"

            doc_content = f"[{chunk['type']}] 番号: {chunk['film_code']}\n标题: {chunk['film_title']}"

            fields = {
                'path': chunk['path'],
                'name': chunk['name'],
                'type': chunk['type'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line'],
                'film_code': chunk.get('film_code', ''),
                'film_title': chunk.get('film_title', ''),
                'document': doc_content,
            }
            docs.append({'id': chunk_id, 'fields': fields})

            # 保存索引以便后续按顺序填充
            tasks.append((i, doc_content))

        # 使用线程池并发处理
        completed_count = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(get_embedding, doc_content, self.model_name): idx
                for idx, doc_content in tasks
            }

            # 收集结果
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings_list[idx] = embedding
                    completed_count += 1
                    if completed_count % 50 == 0 or completed_count == len(chunks):
                        print(f"  已处理 {completed_count}/{len(chunks)} 个块...")
                except Exception as e:
                    print(f"  处理第 {idx} 个块时出错: {e}")
                    raise

        print("写入向量数据库...")
        zvec_docs = []
        for i, item in enumerate(docs):
            zvec_docs.append(
                zvec.Doc(
                    id=item['id'],
                    fields=item['fields'],
                    vectors={VECTOR_FIELD: embeddings_list[i]},
                )
            )

        total_docs = len(zvec_docs)
        for start in range(0, total_docs, MAX_WRITE_BATCH_SIZE):
            end = min(start + MAX_WRITE_BATCH_SIZE, total_docs)
            self.collection.upsert(zvec_docs[start:end])
            print(f"  已写入 {end}/{total_docs} 个块...")

        self.collection.optimize()
        self.collection.flush()

        print(f"成功添加 {len(chunks)} 个块到向量数据库")

    def build_index(self, directory: str = "."):
        """构建向量索引"""
        print("=" * 50)
        print("开始构建JSON数据向量索引")
        print("=" * 50)

        all_chunks = []

        print("\n--- 处理JSON文件 ---")
        json_chunker = JSONChunker()
        json_chunks = json_chunker.chunk_directory(directory)
        print(f"JSON文件: 提取到 {len(json_chunks)} 个数据块")
        all_chunks.extend(json_chunks)

        print(f"\n总共提取 {len(all_chunks)} 个块\n")

        self.add_chunks(all_chunks)

        print("=" * 50)
        print("索引构建完成!")
        print("=" * 50)

    def clear(self):
        """清空向量数据库"""
        vector_schema = self.collection.schema.vector(VECTOR_FIELD)
        dimension = vector_schema.dimension if vector_schema else len(get_embedding("dimension probe", self.model_name))
        self.collection.destroy()
        schema = self._build_schema(dimension)
        self.collection = zvec.create_and_open(path=self.db_path, schema=schema)
        print("向量数据库已清空")


def main():
    """主函数"""
    vector_store = VectorStore()
    vector_store.build_index(".")


if __name__ == "__main__":
    main()
