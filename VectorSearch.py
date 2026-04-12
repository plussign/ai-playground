"""
VectorSearch.py
使用关键词对存入的代码信息进行检索
"""

import os

# 设置 ModelScope 作为模型下载源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json

# 配置
EMBEDDING_MODEL = 'BAAI/bge-m3'
CHROMA_DB_PATH = "./code_vector_db"
COLLECTION_NAME = "python_code_blocks"


class VectorSearch:
    """向量检索器"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, db_path: str = CHROMA_DB_PATH):
        print(f"加载模型: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            cache_folder="./models"
        )

        print(f"连接向量数据库: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Python代码块向量数据库"}
        )

        print(f"当前数据库共有 {self.collection.count()} 个代码块\n")

    def search(self, query: str, top_k: int = 5, filter_type: str = None, filter_path: str = None):
        """
        搜索代码块

        Args:
            query: 搜索关键词
            top_k: 返回结果数量
            filter_type: 可选，按类型过滤 (function/class/file)
            filter_path: 可选，按文件路径过滤

        Returns:
            搜索结果列表
        """
        # 构建过滤条件
        where = {}
        if filter_type:
            where["type"] = filter_type
        if filter_path:
            where["path"] = filter_path

        # 生成查询的embedding
        query_embedding = self.model.encode(query).tolist()

        # 执行搜索
        if where:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'score': 1 - results['distances'][0][i],  # 转换为相似度
                'distance': results['distances'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })

        return formatted_results

    def search_by_keyword(self, keyword: str, top_k: int = 10):
        """
        使用关键词搜索（语义搜索 + 关键词匹配）

        Args:
            keyword: 关键词
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        print(f"\n搜索关键词: {keyword}")
        print("-" * 50)

        # 语义搜索
        results = self.search(keyword, top_k=top_k)

        # 打印结果
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            print(f"\n【结果 {i}】相似度: {result['score']:.4f}")
            print(f"  类型: {meta['type']}")
            print(f"  名称: {meta['name']}")
            print(f"  文件: {meta['path']}")
            print(f"  行号: {meta['start_line']}-{meta['end_line']}")

            if meta.get('docstring'):
                print(f"  文档: {meta['docstring'][:100]}...")

            print(f"  代码片段:")
            code_lines = result['document'].split('\n')
            # 只显示前15行
            for line in code_lines[:15]:
                print(f"    {line}")
            if len(code_lines) > 15:
                print(f"    ... (共 {len(code_lines)} 行)")

        return results


def interactive_search():
    """交互式搜索"""
    searcher = VectorSearch()

    print("\n" + "=" * 50)
    print("Python代码向量检索工具")
    print("=" * 50)
    print("输入关键词进行搜索，输入 'quit' 或 'exit' 退出")
    print("支持按类型过滤: function / class / file")
    print("示例: 搜索 'tts' 或 '搜索 tts 类型 function'")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("\n请输入搜索关键词: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            # 解析命令
            # 格式: "关键词" 或 "关键词 类型 function"
            parts = query.split()
            keyword = parts[0]

            filter_type = None
            if len(parts) > 1 and parts[1] in ['function', 'class', 'file']:
                filter_type = parts[1]

            # 执行搜索
            results = searcher.search(keyword, top_k=10, filter_type=filter_type)

            if not results:
                print("未找到相关结果")
                continue

            # 打印结果
            print("\n" + "=" * 50)
            for i, result in enumerate(results, 1):
                meta = result['metadata']
                print(f"\n【结果 {i}】相似度: {result['score']:.4f}")
                print(f"  类型: {meta['type']}")
                print(f"  名称: {meta['name']}")
                print(f"  文件: {meta['path']}")
                print(f"  行号: {meta['start_line']}-{meta['end_line']}")

                # 显示代码
                code_lines = result['document'].split('\n')
                for line in code_lines[:10]:
                    print(f"    {line}")
                if len(code_lines) > 10:
                    print(f"    ... (共 {len(code_lines)} 行)")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"搜索出错: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Python代码向量检索工具')
    parser.add_argument('keyword', nargs='?', help='搜索关键词')
    parser.add_argument('--top-k', '-k', type=int, default=10, help='返回结果数量')
    parser.add_argument('--type', '-t', choices=['function', 'class', 'file'], help='按类型过滤')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式搜索模式')

    args = parser.parse_args()

    if args.interactive or not args.keyword:
        interactive_search()
    else:
        searcher = VectorSearch()
        searcher.search_by_keyword(args.keyword, top_k=args.top_k)


if __name__ == "__main__":
    main()