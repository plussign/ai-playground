"""
VectorSearch.py
使用关键词对存入的代码信息和JSON数据进行检索
"""

import os
import threading

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import ollama
import zvec

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置
EMBEDDING_MODEL = 'qwen3-embed:8b-q8 '
ZVEC_DB_PATH = os.path.join(SCRIPT_DIR, "../code_zvec_db")
VECTOR_FIELD = "embedding"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    使用 Ollama SDK 获取文本的向量嵌入

    Args:
        text: 输入文本
        model: Ollama 模型名称

    Returns:
        向量嵌入列表
    """
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    query = get_detailed_instruct(task, text)

    response = ollama.embeddings(model=model, prompt=query)
    return response['embedding']


class VectorSearch:
    """向量检索器"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, db_path: str = ZVEC_DB_PATH, status_callback=None):
        self.status_callback = status_callback
        self.model_name = model_name
        self._update_status(f"使用 Ollama 模型: {model_name}")
        self._update_status("请确保 Ollama 服务已启动，并且模型已通过 'ollama pull' 命令拉取")

        self._update_status(f"连接向量数据库: {db_path}")
        self.collection = zvec.open(db_path)

        count = self.collection.stats.doc_count
        self._update_status(f"就绪 - 数据库共有 {count} 个块")

    def _update_status(self, message):
        """更新状态（如果有回调函数）"""
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)

    def search(self, query: str, top_k: int = 5, filter_type: str = None, filter_path: str = None):
        """
        搜索代码块或JSON数据

        Args:
            query: 搜索关键词
            top_k: 返回结果数量
            filter_type: 可选，按类型过滤 (function/class/file/json_file/json_object/json_array/json_value/json_array_batch)
            filter_path: 可选，按文件路径过滤

        Returns:
            搜索结果列表
        """
        # 构建过滤条件
        filters = []
        if filter_type:
            safe_type = filter_type.replace("'", "\\'")
            filters.append(f"type == '{safe_type}'")
        if filter_path:
            safe_path = filter_path.replace("'", "\\'")
            filters.append(f"path == '{safe_path}'")
        filter_expr = " and ".join(filters) if filters else None

        # 生成查询的embedding
        query_embedding = get_embedding(query, self.model_name)

        # 执行搜索
        results = self.collection.query(
            vectors=zvec.VectorQuery(field_name=VECTOR_FIELD, vector=query_embedding),
            topk=top_k,
            filter=filter_expr,
            output_fields=[
                "path",
                "name",
                "type",
                "start_line",
                "end_line",
                "docstring",
                "json_path",
                "document",
            ],
        )

        # 格式化结果
        formatted_results = []
        for doc in results:
            fields = doc.fields or {}
            formatted_results.append({
                'id': doc.id,
                'score': doc.score if doc.score is not None else 0.0,
                'distance': None,
                'document': fields.get('document', ''),
                'metadata': {
                    'path': fields.get('path', ''),
                    'name': fields.get('name', ''),
                    'type': fields.get('type', ''),
                    'start_line': fields.get('start_line', 0),
                    'end_line': fields.get('end_line', 0),
                    'docstring': fields.get('docstring', ''),
                    'json_path': fields.get('json_path', ''),
                }
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

            # 显示JSON路径（如果有）
            if meta.get('json_path'):
                print(f"  JSON路径: {meta['json_path']}")
            else:
                print(f"  行号: {meta['start_line']}-{meta['end_line']}")

            if meta.get('docstring'):
                print(f"  文档: {meta['docstring'][:100]}...")

            content_type = "JSON内容" if meta['type'].startswith('json') else "代码片段"
            print(f"  {content_type}:")
            code_lines = result['document'].split('\n')
            # 只显示前15行
            for line in code_lines[:15]:
                print(f"    {line}")
            if len(code_lines) > 15:
                print(f"    ... (共 {len(code_lines)} 行)")

        return results


class SearchWindow:
    """搜索窗口GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("向量检索工具 - Vector Search")
        self.root.geometry("1600x1200")

        self.searcher = None
        self.results = []

        self._setup_ui()

        # 在后台线程加载模型
        self.status_var.set("正在初始化...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _setup_ui(self):
        """设置UI界面"""
        # 设置全局字体样式
        style = ttk.Style()
        style.configure('TLabel', font=('Microsoft YaHei', 11))
        style.configure('TButton', font=('Microsoft YaHei', 11))
        style.configure('TCombobox', font=('Microsoft YaHei', 11))
        style.configure('Treeview', font=('Microsoft YaHei', 11))
        style.configure('Treeview.Heading', font=('Microsoft YaHei', 12, 'bold'))
        style.configure('TLabelframe', font=('Microsoft YaHei', 11))
        style.configure('TLabelframe.Label', font=('Microsoft YaHei', 11))

        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 顶部搜索区域框架
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(0, weight=1)

        # 搜索输入框
        ttk.Label(search_frame, text="搜索关键词:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.query_entry = ttk.Entry(search_frame, font=('Microsoft YaHei', 13))
        self.query_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.query_entry.bind('<Return>', lambda e: self._on_search())

        # Top-K 选择
        ttk.Label(search_frame, text="Top-K:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.topk_var = tk.StringVar(value="10")
        topk_combo = ttk.Combobox(search_frame, textvariable=self.topk_var, values=["10", "20", "30", "50", "100"], state="readonly", width=8)
        topk_combo.grid(row=0, column=3, padx=(0, 10))

        # 搜索按钮
        self.search_btn = ttk.Button(search_frame, text="开始查找", command=self._on_search, state=tk.DISABLED)
        self.search_btn.grid(row=0, column=4)

        # 结果列表框架
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # 结果列表（Treeview）
        columns = ("score", "type", "name", "path", "location", "docstring")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", selectmode="browse")

        # 配置列
        self.tree.heading("score", text="相似度")
        self.tree.heading("type", text="类型")
        self.tree.heading("name", text="名称")
        self.tree.heading("path", text="文件路径")
        self.tree.heading("location", text="位置")
        self.tree.heading("docstring", text="文档/说明")

        self.tree.column("score", width=100, anchor=tk.CENTER)
        self.tree.column("type", width=120, anchor=tk.CENTER)
        self.tree.column("name", width=200)
        self.tree.column("path", width=400)
        self.tree.column("location", width=150)
        self.tree.column("docstring", width=400)

        # 滚动条
        tree_scroll_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        # 布局
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # 底部详情面板
        detail_frame = ttk.LabelFrame(main_frame, text="内容详情", padding="10")
        detail_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        self.detail_text = scrolledtext.ScrolledText(detail_frame, wrap=tk.WORD, font=('Microsoft YaHei', 11))
        self.detail_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=('Microsoft YaHei', 10))
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # 绑定事件
        self.tree.bind('<<TreeviewSelect>>', self._on_select_item)

    def _update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def _load_model(self):
        """加载模型（在后台线程）"""
        try:
            self.root.after(0, lambda: self._update_status("正在加载模型..."))
            self.searcher = VectorSearch(status_callback=lambda msg: self.root.after(0, lambda: self._update_status(msg)))
            self.root.after(0, lambda: self.search_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.query_entry.focus())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"加载失败: {str(e)}"))
            self.root.after(0, lambda: self._update_status("加载失败"))

    def _on_search(self):
        """执行搜索"""
        if not self.searcher:
            messagebox.showwarning("警告", "模型尚未加载完成")
            return

        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("警告", "请输入搜索关键词")
            return

        try:
            top_k = int(self.topk_var.get())
            self._update_status(f"正在搜索: {query}...")
            self.search_btn.config(state=tk.DISABLED)

            # 在后台线程执行搜索
            threading.Thread(target=self._do_search, args=(query, top_k), daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", f"搜索出错: {str(e)}")
            self.search_btn.config(state=tk.NORMAL)

    def _do_search(self, query, top_k):
        """执行实际搜索"""
        try:
            results = self.searcher.search(query, top_k=top_k)
            self.root.after(0, lambda: self._display_results(results))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"搜索出错: {str(e)}"))
            self.root.after(0, lambda: self.search_btn.config(state=tk.NORMAL))

    def _display_results(self, results):
        """显示搜索结果"""
        # 清空现有结果
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.results = results

        # 插入新结果
        for result in results:
            meta = result['metadata']
            score = f"{result['score']:.4f}"
            type_ = meta['type']
            name = meta['name']
            path = meta['path']

            # 确定位置信息
            if meta.get('json_path'):
                location = meta['json_path']
            else:
                location = f"{meta['start_line']}-{meta['end_line']}"

            docstring = meta.get('docstring', '')[:100]

            self.tree.insert("", tk.END, values=(score, type_, name, path, location, docstring))

        self._update_status(f"找到 {len(results)} 个结果")
        self.search_btn.config(state=tk.NORMAL)

    def _on_select_item(self, event):
        """选中结果项时显示详情"""
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        index = self.tree.index(item)
        if 0 <= index < len(self.results):
            result = self.results[index]
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(tk.END, result['document'])


def interactive_search():
    """交互式搜索"""
    searcher = VectorSearch()

    print("\n" + "=" * 50)
    print("Python代码和JSON数据向量检索工具")
    print("=" * 50)
    print("输入关键词进行搜索，输入 'quit' 或 'exit' 退出")
    print("支持按类型过滤: function / class / file / json_file / json_object / json_array")
    print("示例: 搜索 '配置' 或 '搜索 config 类型 json_object'")
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
            valid_types = ['function', 'class', 'file', 'json_file', 'json_object', 'json_array', 'json_value', 'json_array_batch']
            if len(parts) > 1 and parts[1] in valid_types:
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

                # 显示JSON路径（如果有）
                if meta.get('json_path'):
                    print(f"  JSON路径: {meta['json_path']}")
                else:
                    print(f"  行号: {meta['start_line']}-{meta['end_line']}")

                if meta.get('docstring'):
                    print(f"  文档: {meta['docstring'][:100]}...")

                # 显示内容
                content_type = "JSON内容" if meta['type'].startswith('json') else "代码"
                print(f"  {content_type}:")
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

    parser = argparse.ArgumentParser(description='Python代码和JSON数据向量检索工具')
    parser.add_argument('keyword', nargs='?', help='搜索关键词')
    parser.add_argument('--top-k', '-k', type=int, default=10, help='返回结果数量')
    parser.add_argument('--type', '-t', choices=['function', 'class', 'file', 'json_file', 'json_object', 'json_array', 'json_value', 'json_array_batch'], help='按类型过滤')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式搜索模式')
    parser.add_argument('--window', '-w', action='store_true', help='窗口GUI交互模式')

    args = parser.parse_args()

    if args.window:
        # 启动窗口模式
        root = tk.Tk()
        app = SearchWindow(root)
        root.mainloop()
    elif args.interactive or not args.keyword:
        interactive_search()
    else:
        searcher = VectorSearch()
        searcher.search_by_keyword(args.keyword, top_k=args.top_k)


if __name__ == "__main__":
    main()
