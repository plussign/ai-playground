import sys
import os
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD


def parse_srt_time(time_str):
    """Parse SRT time string 'HH:MM:SS,mmm' to milliseconds."""
    h, m, s = time_str.strip().replace(',', '.').split(':')
    s, ms = s.split('.')
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def format_srt_time(ms):
    """Format milliseconds to SRT time string 'HH:MM:SS,mmm'."""
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(filepath):
    """Parse an SRT file into a list of subtitle entries."""
    entries = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_parts = lines[1].strip().split(' --> ')
        if len(time_parts) != 2:
            continue
        start = parse_srt_time(time_parts[0])
        end = parse_srt_time(time_parts[1])
        text = '\n'.join(lines[2:])
        entries.append({'index': idx, 'start': start, 'end': end, 'text': text})
    return entries


def write_srt(filepath, entries):
    """Write subtitle entries to an SRT file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries):
            f.write(f"{i + 1}\n")
            f.write(f"{format_srt_time(e['start'])} --> {format_srt_time(e['end'])}\n")
            f.write(f"{e['text']}\n")
            if i < len(entries) - 1:
                f.write('\n')


class TextEditDialog(tk.Toplevel):
    def __init__(self, parent, current_text):
        super().__init__(parent)
        self.title("编辑字幕文本")
        self.resizable(True, True)
        self.result = None
        self.grab_set()

        frame = ttk.Frame(self, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="修改字幕后点击确定：").pack(anchor=tk.W, pady=(0, 5))

        self.text = tk.Text(frame, width=50, height=8, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.text.insert("1.0", current_text)
        self.text.focus_set()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack()
        ttk.Button(btn_frame, text="确定", command=self.on_ok, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.on_cancel, width=8).pack(side=tk.LEFT, padx=5)

        self.bind('<Escape>', lambda e: self.on_cancel())

        self.transient(parent)
        self.wait_window()

    def on_ok(self):
        self.result = self.text.get("1.0", "end-1c")
        self.destroy()

    def on_cancel(self):
        self.destroy()


class OffsetDialog(tk.Toplevel):
    def __init__(self, parent, title, label):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self.grab_set()

        frame = ttk.Frame(self, padding=15)
        frame.pack()

        ttk.Label(frame, text=label).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.entry = ttk.Entry(frame, width=15)
        self.entry.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        self.entry.focus_set()

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2)
        ttk.Button(btn_frame, text="确定", command=self.on_ok, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.on_cancel, width=8).pack(side=tk.LEFT, padx=5)

        self.entry.bind('<Return>', lambda e: self.on_ok())
        self.bind('<Escape>', lambda e: self.on_cancel())

        self.transient(parent)
        self.wait_window()

    def on_ok(self):
        try:
            val = int(self.entry.get())
            self.result = val
            self.destroy()
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的整数（毫秒）", parent=self)

    def on_cancel(self):
        self.destroy()


class SRTOffsetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SRT Offset Tool")
        self.root.geometry("800x500")
        self.root.minsize(600, 400)

        self.filepath = None
        self.entries = []
        self.undo_stack = []
        self.redo_stack = []

        self._build_ui()

    def _build_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="打开文件", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.btn_minus = ttk.Button(toolbar, text="时间-", command=self.offset_minus, state=tk.DISABLED)
        self.btn_minus.pack(side=tk.LEFT, padx=2)
        self.btn_plus = ttk.Button(toolbar, text="时间+", command=self.offset_plus, state=tk.DISABLED)
        self.btn_plus.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.btn_undo = ttk.Button(toolbar, text="撤销", command=self.undo, state=tk.DISABLED)
        self.btn_undo.pack(side=tk.LEFT, padx=2)
        self.btn_redo = ttk.Button(toolbar, text="重做", command=self.redo, state=tk.DISABLED)
        self.btn_redo.pack(side=tk.LEFT, padx=2)

        self.file_label = ttk.Label(toolbar, text="", foreground="gray")
        self.file_label.pack(side=tk.RIGHT, padx=5)

        # Treeview
        columns = ("index", "start", "end", "text")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", selectmode="extended")

        self.tree.heading("index", text="编号")
        self.tree.heading("start", text="开始时间")
        self.tree.heading("end", text="结束时间")
        self.tree.heading("text", text="文本内容")

        self.tree.column("index", width=50, minwidth=40, anchor=tk.CENTER, stretch=False)
        self.tree.column("start", width=120, minwidth=100, anchor=tk.CENTER, stretch=False)
        self.tree.column("end", width=120, minwidth=100, anchor=tk.CENTER, stretch=False)
        self.tree.column("text", width=500, minwidth=200, anchor=tk.W, stretch=True)

        vsb = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_selection_change)
        self.tree.bind("<Double-1>", self._on_double_click)

        # Drag and drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self._on_drop)

    def _on_drop(self, event):
        path = event.data.strip('{}').strip()
        if path.lower().endswith('.srt') and os.path.isfile(path):
            self.load_file(path)
        else:
            messagebox.showwarning("不支持", "请拖入 .srt 文件")

    def open_file(self):
        path = filedialog.askopenfilename(
            title="选择 SRT 文件",
            filetypes=[("SRT 字幕文件", "*.srt"), ("所有文件", "*.*")]
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            entries = parse_srt(path)
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败:\n{e}")
            return
        self.filepath = path
        self.entries = entries
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_tree()
        self._update_buttons()
        self.file_label.config(text=os.path.basename(path))

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for e in self.entries:
            self.tree.insert("", tk.END, values=(
                e['index'],
                format_srt_time(e['start']),
                format_srt_time(e['end']),
                e['text']
            ))

    def _on_selection_change(self, event=None):
        self._update_offset_buttons()

    def _on_double_click(self, event=None):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        values = self.tree.item(item, 'values')
        idx = int(values[0]) - 1
        dlg = TextEditDialog(self.root, self.entries[idx]['text'])
        if dlg.result is not None and dlg.result != self.entries[idx]['text']:
            self.undo_stack.append(copy.deepcopy(self.entries))
            self.redo_stack.clear()
            self.entries[idx]['text'] = dlg.result
            self._refresh_tree()
            self._save()
            self._update_buttons()
            # Re-select the edited row
            children = self.tree.get_children()
            if idx < len(children):
                self.tree.selection_set(children[idx])

    def _update_offset_buttons(self):
        has_sel = len(self.tree.selection()) > 0
        self.btn_minus.config(state=tk.NORMAL if has_sel else tk.DISABLED)
        self.btn_plus.config(state=tk.NORMAL if has_sel else tk.DISABLED)

    def _update_buttons(self):
        self._update_offset_buttons()
        self.btn_undo.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)
        self.btn_redo.config(state=tk.NORMAL if self.redo_stack else tk.DISABLED)

    def _get_selected_indices(self):
        indices = []
        for item in self.tree.selection():
            values = self.tree.item(item, 'values')
            indices.append(int(values[0]) - 1)
        return sorted(indices)

    def _apply_offset(self, delta):
        if not self.entries:
            return
        selected = self._get_selected_indices()
        if not selected:
            return

        # Save current state for undo
        self.undo_stack.append(copy.deepcopy(self.entries))
        self.redo_stack.clear()

        for i in selected:
            self.entries[i]['start'] = max(0, self.entries[i]['start'] + delta)
            self.entries[i]['end'] = max(0, self.entries[i]['end'] + delta)

        self._refresh_tree()
        self._save()
        self._update_buttons()

        # Reselect items
        for i in selected:
            children = self.tree.get_children()
            if i < len(children):
                self.tree.selection_add(children[i])

    def offset_minus(self):
        dlg = OffsetDialog(self.root, "时间偏移", "输入要减少的毫秒数：")
        if dlg.result is not None:
            self._apply_offset(-dlg.result)

    def offset_plus(self):
        dlg = OffsetDialog(self.root, "时间偏移", "输入要增加的毫秒数：")
        if dlg.result is not None:
            self._apply_offset(dlg.result)

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.entries))
        self.entries = self.undo_stack.pop()
        self._refresh_tree()
        self._save()
        self._update_buttons()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.entries))
        self.entries = self.redo_stack.pop()
        self._refresh_tree()
        self._save()
        self._update_buttons()

    def _save(self):
        if self.filepath and self.entries:
            write_srt(self.filepath, self.entries)


def main():
    root = TkinterDnD.Tk()
    app = SRTOffsetApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
