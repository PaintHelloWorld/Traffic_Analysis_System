# ui_components.py - 界面组件

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np


class DataTable(ttk.Frame):
    """数据表格组件 - 显示DataFrame数据"""

    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.tree = None
        self.scrollbar_y = None
        self.scrollbar_x = None

        self.setup_table()

    def setup_table(self):
        """设置表格框架"""
        # 创建表格框架
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 创建Treeview（表格）
        self.tree = ttk.Treeview(self, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        # 垂直滚动条
        self.scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=self.scrollbar_y.set)

        # 水平滚动条
        self.scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=self.scrollbar_x.set)

    def load_data(self):
        """加载数据到表格"""
        if self.data_manager.display_data is None:
            return False, "没有数据可显示"

        try:
            # 清空现有数据
            for item in self.tree.get_children():
                self.tree.delete(item)

            # 获取数据
            data = self.data_manager.display_data
            columns = self.data_manager.get_column_names()

            # 设置表格列
            self.tree["columns"] = columns
            for col in columns:
                self.tree.heading(col, text=col)
                # 根据内容自动调整列宽
                max_len = max([len(str(val)) for val in data[col].head(20).astype(str)]) if len(data) > 0 else 10
                width = min(max_len * 8, 200)  # 最大200像素
                self.tree.column(col, width=width, minwidth=50)

            # 插入数据
            for idx, row in data.iterrows():
                values = [str(row[col])[:100] for col in columns]  # 限制显示长度
                self.tree.insert("", tk.END, values=values, iid=str(idx))

            return True, f"显示 {len(data)} 条记录"

        except Exception as e:
            return False, f"加载数据到表格失败: {str(e)}"

    def get_selected_indices(self):
        """获取选中的行索引"""
        selected_items = self.tree.selection()
        return [int(item) for item in selected_items]

    def clear_selection(self):
        """清除选择"""
        self.tree.selection_remove(self.tree.selection())


class ControlPanel(ttk.LabelFrame):
    """控制面板 - 筛选、搜索、操作按钮（修改版）"""

    def __init__(self, parent, data_manager, table, status_callback):
        super().__init__(parent, text="控制面板", padding=10)
        self.data_manager = data_manager
        self.table = table
        self.status_callback = status_callback

        self.setup_controls()

    def setup_controls(self):
        """设置控制组件"""
        # 文件操作按钮
        file_frame = ttk.Frame(self)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="导入CSV", command=self.open_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导入Excel", command=self.open_excel).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导出CSV", command=self.save_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导出Excel", command=self.export_excel).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="示例数据", command=self.generate_sample).pack(side=tk.LEFT, padx=2)

        # 确保这行代码存在且正确
        ttk.Separator(self, orient='horizontal').pack(fill=tk.X, pady=10)

        # 筛选控制 - 确保这个框架被正确创建和打包
        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="筛选列:").pack(side=tk.LEFT, padx=2)
        self.filter_column = ttk.Combobox(filter_frame, width=15, state="readonly")
        self.filter_column.pack(side=tk.LEFT, padx=2)

        ttk.Label(filter_frame, text="条件:").pack(side=tk.LEFT, padx=2)
        self.filter_value = ttk.Entry(filter_frame, width=15)
        self.filter_value.pack(side=tk.LEFT, padx=2)

        ttk.Button(filter_frame, text="应用筛选", command=self.apply_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_frame, text="清除筛选", command=self.clear_filter).pack(side=tk.LEFT, padx=2)

        # 搜索框
        search_frame = ttk.Frame(self)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT, padx=2)
        self.search_entry = ttk.Entry(search_frame, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=2)
        self.search_entry.bind("<Return>", lambda e: self.search_data())

        ttk.Button(search_frame, text="搜索", command=self.search_data).pack(side=tk.LEFT, padx=2)

        # 排序控制
        sort_frame = ttk.Frame(self)
        sort_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(sort_frame, text="排序列:").pack(side=tk.LEFT, padx=2)
        self.sort_column = ttk.Combobox(sort_frame, width=15, state="readonly")
        self.sort_column.pack(side=tk.LEFT, padx=2)

        self.sort_ascending = tk.BooleanVar(value=True)
        ttk.Radiobutton(sort_frame, text="升序", variable=self.sort_ascending, value=True).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sort_frame, text="降序", variable=self.sort_ascending, value=False).pack(side=tk.LEFT, padx=2)

        ttk.Button(sort_frame, text="排序", command=self.sort_data).pack(side=tk.LEFT, padx=2)

        # 数据操作按钮
        data_frame = ttk.Frame(self)
        data_frame.pack(fill=tk.X)

        ttk.Button(data_frame, text="添加记录", command=self.add_record).pack(side=tk.LEFT, padx=2)
        ttk.Button(data_frame, text="删除选中", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(data_frame, text="刷新表格", command=self.refresh_table).pack(side=tk.LEFT, padx=2)

        # 更新列选项
        self.update_column_options()

    def update_column_options(self):
        """更新列选项"""
        columns = self.data_manager.get_column_names()
        self.filter_column['values'] = columns
        self.sort_column['values'] = columns

        if columns:
            self.filter_column.current(0)
            self.sort_column.current(0)


    def open_csv(self):
        """打开数据文件（支持CSV）"""
        filepath = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("CSV文件", "*.csv"),
              #  ("Excel文件", "*.xlsx *.xls"),
                ("所有文件", "*.*")
            ]
        )

        if filepath:
            # 根据文件扩展名选择加载方法
            file_ext = filepath.lower().split('.')[-1]

            if file_ext in ['xlsx', 'xls']:
                success, message = self.data_manager.load_excel(filepath)
            else:
                success, message = self.data_manager.load_csv(filepath)

            if success:
                # 检查是否有警告信息
                if hasattr(self.data_manager, 'validation_warning') and self.data_manager.validation_warning:
                    # 显示警告对话框，让用户选择是否继续
                    response = messagebox.askyesno(
                        "数据格式警告",
                        f"{self.data_manager.validation_warning}\n\n是否继续导入？"
                    )

                    if not response:
                        # 用户选择不继续，重置数据管理器
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                # 数据量检查
                data_size = len(self.data_manager.display_data)
                if data_size > 100:
                    response = messagebox.askyesno(
                        "数据量警告",
                        f"加载了 {data_size} 条数据。\n\n"
                        f"数据量超过100条，使用可视化功能可能导致程序卡顿\n"
                        f"是否继续导入？"
                    )

                    if not response:
                        # 用户选择不继续，重置数据管理器
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                self.refresh_table()
                self.update_column_options()
            else:
                # 显示错误消息框
                messagebox.showerror(
                    "文件格式错误",
                    f"无法导入文件:\n\n{message}"
                )

            self.status_callback(message)

    # ui_components.py - 在 ControlPanel 类中添加 open_excel 方法
    def open_excel(self):
        """打开Excel文件"""
        filepath = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[
                ("Excel文件", "*.xlsx *.xls"),
                ("所有文件", "*.*")
            ]
        )

        if filepath:
            success, message = self.data_manager.load_excel(filepath)

            if success:
                # 检查是否有警告信息
                if hasattr(self.data_manager, 'validation_warning') and self.data_manager.validation_warning:
                    # 显示警告对话框，让用户选择是否继续
                    response = messagebox.askyesno(
                        "数据格式警告",
                        f"{self.data_manager.validation_warning}\n\n是否继续导入？"
                    )

                    if not response:
                        # 用户选择不继续，重置数据管理器
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                # ============ 数据量检查 ============
                data_size = len(self.data_manager.display_data)
                if data_size > 100:
                    response = messagebox.askyesno(
                        "数据量警告",
                        f"加载了 {data_size} 条数据。\n\n"
                        f"数据量超过100条，使用可视化功能可能导致程序卡顿\n"
                        f"是否继续导入？"
                    )

                    if not response:
                        # 用户选择不继续，重置数据管理器
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return
                # ============ 数据量检查结束 ============

                self.refresh_table()
                self.update_column_options()
            else:
                # 显示错误消息框
                messagebox.showerror(
                    "文件格式错误",
                    f"无法导入Excel文件:\n\n{message}"
                )

            self.status_callback(message)

    def save_csv(self):
        """保存CSV文件"""
        if self.data_manager.display_data is None:
            messagebox.showwarning("无数据", "请先加载数据")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv")]
        )

        if filepath:
            success, message = self.data_manager.save_to_csv(filepath)
            self.status_callback(message)

        # 导出到Excel方法

    def export_excel(self):
        """导出到Excel"""
        if self.data_manager.display_data is None:
            messagebox.showwarning("无数据", "请先加载数据")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[
                ("Excel文件", "*.xlsx"),
                ("Excel 97-2003", "*.xls"),
                ("所有文件", "*.*")
            ]
        )

        if filepath:
            success, message = self.data_manager.export_to_excel(filepath)
            if success:
                messagebox.showinfo("导出成功", message)
            else:
                messagebox.showerror("导出失败", message)
            self.status_callback(message)

    def generate_sample(self):
        """生成示例数据"""
        success, message = self.data_manager.generate_sample_data(100)
        if success:
            self.refresh_table()
            self.update_column_options()
        self.status_callback(message)

    def apply_filter(self):
        """应用筛选条件"""
        column = self.filter_column.get()
        condition = self.filter_value.get()

        if not column or not condition:
            messagebox.showwarning("输入错误", "请选择列名并输入条件")
            return

        success, message = self.data_manager.apply_filter(column, condition)
        if success:
            self.refresh_table()
        self.status_callback(message)

    def clear_filter(self):
        """清除筛选"""
        success, message = self.data_manager.clear_all_filters()
        if success:
            self.refresh_table()
        self.status_callback(message)

    def search_data(self):
        """搜索数据"""
        keyword = self.search_entry.get()
        if not keyword:
            messagebox.showwarning("输入错误", "请输入搜索关键词")
            return

        results, message = self.data_manager.search_data(keyword)
        if results is not None:
            # 临时显示搜索结果
            self.data_manager.display_data = results
            self.refresh_table()
        self.status_callback(message)

    def sort_data(self):
        """排序数据"""
        column = self.sort_column.get()
        if not column:
            messagebox.showwarning("输入错误", "请选择排序列")
            return

        success, message = self.data_manager.sort_data(column, self.sort_ascending.get())
        if success:
            self.refresh_table()
        self.status_callback(message)

    def add_record(self):
        """添加新记录（简化版对话框）"""
        if self.data_manager.display_data is None:
            messagebox.showwarning("无数据", "请先加载数据")
            return

        # 创建添加记录对话框
        dialog = tk.Toplevel(self)
        dialog.title("添加新记录")
        dialog.geometry("400x800")

        columns = self.data_manager.get_column_names()
        entries = {}

        for i, col in enumerate(columns):
            ttk.Label(dialog, text=f"{col}:").grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(dialog, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            entries[col] = entry

        def save_record():
            record_dict = {}
            for col, entry in entries.items():
                record_dict[col] = entry.get() or None

            success, message = self.data_manager.add_record(record_dict)
            if success:
                self.refresh_table()
                dialog.destroy()
            self.status_callback(message)

        ttk.Button(dialog, text="保存", command=save_record).grid(row=len(columns), column=0, columnspan=2, pady=20)

    def delete_selected(self):
        """删除选中记录"""
        selected_indices = self.table.get_selected_indices()
        if not selected_indices:
            messagebox.showwarning("无选择", "请先选择要删除的记录")
            return

        if messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_indices)} 条记录吗？"):
            success, message = self.data_manager.delete_records(selected_indices)
            if success:
                self.refresh_table()
            self.status_callback(message)

    def refresh_table(self):
        """刷新表格显示"""
        success, message = self.table.load_data()
        if not success:
            self.status_callback(message)


class InfoPanel(ttk.LabelFrame):
    """信息面板 - 显示数据统计信息"""

    def __init__(self, parent, data_manager):
        super().__init__(parent, text="数据信息", padding=10)
        self.data_manager = data_manager

        self.info_text = tk.Text(self, height=15, width=30, state="disabled")
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.update_info()

    def update_info(self):
        """更新信息显示"""
        if self.data_manager.display_data is None:
            info = "请导入数据..."
        else:
            stats = self.data_manager.get_basic_stats()
            info = f"📊 数据概览\n{'=' * 30}\n"
            info += f"总记录数: {stats['total_records']}\n"
            info += f"总列数: {stats['total_columns']}\n\n"

            info += "📈 列信息:\n"
            for col_info in stats['column_details']:
                info += f"\n{col_info['name']}:\n"
                info += f"  类型: {col_info['type']}\n"
                info += f"  非空值: {col_info['non_null']}\n"
                info += f"  唯一值: {col_info['unique_values']}\n"

        self.info_text.config(state="normal")
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state="disabled")



class IntegratedMainWindow:
    """集成版主窗口 """

    def __init__(self, root, data_manager):
        self.root = root
        self.data_manager = data_manager

        # 初始化预测器（延迟加载）
        self.predictor = None
        # 初始化状态变量
        self.status_var = tk.StringVar(value="就绪")

        self.setup_window()
        self.setup_status_bar()  # 先设置状态栏
        self.setup_notebook()

        # 初始状态
        self.update_status("就绪 - 城市交通事故分析与预警系统")

    def setup_window(self):
        """设置窗口属性"""
        self.root.title("城市交通事故分析与预警系统")
        self.root.geometry("1200x700")

        # 使窗口可调整大小
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def setup_notebook(self):
        """设置选项卡控件"""
        # 创建Notebook（选项卡容器）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建各个选项卡
        self.setup_data_tab()  # 数据管理
        self.setup_viz_tab()  # 可视化分析
        self.setup_pred_tab()  # 风险预测
        self.setup_help_tab()  # 帮助（新增）

    def setup_data_tab(self):
        """设置数据管理选项卡"""
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 数据管理")

        # 数据管理页布局
        self.setup_data_tab_layout()

    def setup_data_tab_layout(self):
        """数据管理页的具体布局"""
        # 主框架
        main_frame = ttk.Frame(self.data_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：信息面板
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        self.info_panel = InfoPanel(left_panel, self.data_manager)
        self.info_panel.pack(fill=tk.BOTH, expand=True)

        # 右侧：数据表格和控制面板
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 数据表格
        self.data_table = DataTable(right_panel, self.data_manager)
        self.data_table.pack(fill=tk.BOTH, expand=True, padx=(0, 5))

        # 控制面板（在表格下面）
        control_frame = ttk.Frame(right_panel)
        control_frame.pack(fill=tk.X, pady=(5, 0))

        self.control_panel = ControlPanel(
            control_frame,
            self.data_manager,
            self.data_table,
            self.update_status
        )
        self.control_panel.pack(fill=tk.X)

    def setup_viz_tab(self):
        """设置可视化分析选项卡"""
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="📈 可视化分析")

        # 创建主容器
        viz_container = ttk.Frame(self.viz_tab)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 导入可视化器
        from visualizer import TrafficVisualizer

        # 初始化可视化器
        self.visualizer = TrafficVisualizer(self.data_manager, viz_container)

        # 绑定选项卡切换事件
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event=None):
        """处理选项卡切换事件"""
        try:
            if not hasattr(self, 'notebook') or not self.notebook.winfo_exists():
                return

            # 获取当前选中的选项卡
            current_tab_id = self.notebook.select()
            if not current_tab_id:
                return

            # 获取选项卡索引
            current_tab_index = self.notebook.index(current_tab_id)

            # 获取选项卡文本
            tab_text = self.notebook.tab(current_tab_index, "text")

            if tab_text == "📈 可视化分析":
                # ============ 新增：数据量检查 ============
                if hasattr(self, 'data_manager') and self.data_manager.display_data is not None:
                    data_size = len(self.data_manager.display_data)
                    if data_size > 100:
                        # 显示警告消息框，使用 askokcancel 提供选择
                        response = messagebox.askokcancel(
                            "数据量警告",
                            f"当前数据有 {data_size} 条记录。\n\n"
                            f"⚠️ 数据量超过100条，使用可视化功能可能导致程序卡顿。\n\n"
                            f"【确定】继续使用可视化分析\n"
                            f"【取消】返回数据管理页面进行筛选"
                        )

                        # 如果用户点击取消，切换到数据管理选项卡
                        if not response:  # 用户点击了取消
                            # 查找数据管理选项卡
                            for i in range(self.notebook.index("end")):
                                if self.notebook.tab(i, "text") == "📊 数据管理":
                                    self.notebook.select(i)
                                    self.update_status(f"已返回数据管理页面 (数据量: {data_size} 条)")
                                    return
                # ============ 新增结束 ============

                # 切换到可视化选项卡
                if hasattr(self, 'visualizer') and self.visualizer:
                    # 延迟一点时间，确保界面完全加载
                    self.root.after(300, self.refresh_visualizer)

            elif tab_text == "📊 数据管理":
                # 切换到数据管理选项卡
                if hasattr(self, 'data_table'):
                    self.data_table.load_data()
                    if hasattr(self, 'info_panel'):
                        self.info_panel.update_info()

            elif tab_text == "⚠️ 风险预测":
                # 切换到风险预测选项卡
                pass

        except Exception as e:
            print(f"选项卡切换错误: {e}")

    def refresh_visualizer(self):
        """刷新可视化器"""
        if hasattr(self, 'visualizer') and self.visualizer:
            # 更新轴选项
            self.visualizer.update_axis_options()

            # 如果有数据，尝试生成图表
            if self.data_manager.display_data is not None:
                try:
                    # 检查是否有有效的轴选择
                    x_axis = self.visualizer.x_axis_var.get()
                    y_axis = self.visualizer.y_axis_var.get()

                    if x_axis and y_axis:
                        # 延迟生成图表，给用户一点时间看到选项更新
                        self.root.after(500, self.visualizer.generate_chart)
                    else:
                        # 如果没有有效选择，设置默认值
                        columns = self.data_manager.get_column_names()
                        if columns:
                            self.visualizer.x_axis_var.set(columns[0])

                            # 查找数值列
                            numeric_cols = []
                            data = self.data_manager.display_data
                            for col in columns:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    numeric_cols.append(col)

                            if numeric_cols:
                                self.visualizer.y_axis_var.set(numeric_cols[0])
                                self.root.after(500, self.visualizer.generate_chart)

                except Exception as e:
                    self.update_status(f"刷新可视化器失败: {str(e)}")

    def setup_pred_tab(self):
        """设置风险预测选项卡"""
        self.pred_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_tab, text="⚠️ 风险预测")

        # 创建预测界面
        self.setup_prediction_ui()

    def setup_training_panel(self, parent):
        """设置模型训练面板"""
        frame = ttk.LabelFrame(parent, text="模型训练", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 按钮行
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="训练预测模型",
                   command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导入模型文件",
                   command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导出当前模型",
                   command=self.save_model).pack(side=tk.LEFT, padx=5)

        # 状态显示
        self.model_status_var = tk.StringVar(value="模型状态: 未训练")
        ttk.Label(frame, textvariable=self.model_status_var).pack(anchor=tk.W)

    def setup_prediction_ui(self):
        """设置预测用户界面 - 改进版布局"""
        # 主框架
        main_frame = ttk.Frame(self.pred_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1. 模型训练面板（最上方，保持不变）
        self.setup_training_panel(main_frame)

        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=20)

        # 创建左右两列的容器
        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 配置网格权重，使左右两列平分空间
        columns_frame.columnconfigure(0, weight=1)
        columns_frame.columnconfigure(1, weight=1)
        columns_frame.rowconfigure(0, weight=1)

        # 2. 左侧：单条预测面板
        left_frame = ttk.Frame(columns_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.setup_single_prediction_panel(left_frame)

        # 3. 右侧：批量预测和特征重要性面板
        right_frame = ttk.Frame(columns_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # 配置右侧框架的网格
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # 批量预测面板（右上）
        batch_frame = ttk.LabelFrame(right_frame, text="批量风险预测", padding=10)
        batch_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.setup_batch_prediction_panel_content(batch_frame)

        # 特征重要性面板（右下）
        feature_frame = ttk.LabelFrame(right_frame, text="特征重要性分析", padding=10)
        feature_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.setup_feature_importance_panel_content(feature_frame)

    def setup_single_prediction_panel(self, parent):
        """设置单条预测面板（左侧）"""
        frame = ttk.LabelFrame(parent, text="单条事故风险预测", padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动区域以容纳所有输入字段
        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 打包滚动组件
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 输入表单网格
        form_frame = ttk.Frame(scrollable_frame)
        form_frame.pack(fill=tk.X, pady=10, padx=5)

        # 常见字段输入
        fields = [
            ("事故时间", "2024-01-01 08:30"),
            ("所在区域", "朝阳区"),
            ("事故类型", "追尾"),
            ("受伤人数", "1"),
            ("死亡人数", "0"),
            ("温度(℃)", "25.5"),
            ("湿度(%)", "65"),
            ("能见度(km)", "10.5"),
            ("风速(m/s)", "3.2")
        ]

        self.pred_inputs = {}
        for i, (label, default) in enumerate(fields):
            row_frame = ttk.Frame(form_frame)
            row_frame.pack(fill=tk.X, pady=3)

            lbl = ttk.Label(row_frame, text=f"{label}:", width=15, anchor="e")
            lbl.pack(side=tk.LEFT, padx=(0, 5))

            entry = ttk.Entry(row_frame)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.pred_inputs[label] = entry

        # 预测按钮和结果显示
        result_frame = ttk.Frame(scrollable_frame)
        result_frame.pack(fill=tk.X, pady=15)

        btn_frame = ttk.Frame(result_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="预测风险", command=self.predict_single,
                   style="Accent.TButton" if hasattr(ttk, 'Style') else None).pack()

        # 结果显示区域
        result_display = ttk.Frame(result_frame, relief="solid", borderwidth=1)
        result_display.pack(fill=tk.X, pady=5)

        self.pred_result_var = tk.StringVar(value="等待预测...")
        self.pred_result_label = ttk.Label(
            result_display,
            textvariable=self.pred_result_var,
            font=("Arial", 14, "bold"),
            anchor="center",
            padding=10
        )
        self.pred_result_label.pack(fill=tk.X)

        self.pred_prob_var = tk.StringVar(value="")
        ttk.Label(
            result_display,
            textvariable=self.pred_prob_var,
            anchor="center",
            padding=(0, 5, 0, 10)
        ).pack(fill=tk.X)

    def setup_batch_prediction_panel_content(self, parent):
        """设置批量预测面板内容（右侧上部）"""
        # 按钮框架
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="对当前数据批量预测",
                   command=self.predict_batch,
                   width=20).pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="刷新预测结果",
                   command=self.refresh_predictions,
                   width=15).pack(side=tk.LEFT, padx=2)

        # 导出按钮框架
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill=tk.X, pady=5)

        ttk.Label(export_frame, text="导出结果:").pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(export_frame, text="CSV",
                   command=self.export_predictions_csv,
                   width=10).pack(side=tk.LEFT, padx=2)

        ttk.Button(export_frame, text="Excel",
                   command=self.export_predictions_excel,
                   width=10).pack(side=tk.LEFT, padx=2)

        # 批量预测状态显示
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        self.batch_status_var = tk.StringVar(value="未进行批量预测")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.batch_status_var,
            relief="sunken",
            anchor="w",
            padding=5,
            background="#f0f0f0"
        )
        status_label.pack(fill=tk.X)

        # 预测结果统计
        self.batch_stats_text = tk.Text(parent, height=8, width=30, state="disabled")
        self.batch_stats_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def setup_feature_importance_panel_content(self, parent):
        """设置特征重要性面板内容（右侧下部）"""
        # 按钮框架
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="查看特征重要性",
                   command=self.show_feature_importance,
                   width=20).pack()

        # 特征重要性显示区域
        self.feature_text = tk.Text(parent, height=12, state="disabled")
        self.feature_text.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(parent, command=self.feature_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_text.config(yscrollcommand=scrollbar.set)

    def refresh_predictions(self):
        """刷新预测结果显示"""
        if self.data_manager.display_data is not None and '预测风险等级' in self.data_manager.display_data.columns:
            # 统计结果
            predictions = self.data_manager.display_data['预测风险等级']
            unique, counts = np.unique(predictions, return_counts=True)

            stats_text = "📊 预测结果统计:\n"
            stats_text += "=" * 30 + "\n"

            total = len(predictions)
            for level, count in zip(unique, counts):
                risk_labels = ['低风险', '中风险', '高风险']
                label = risk_labels[level] if level < 3 else f"等级{level}"
                percentage = count / total * 100
                stats_text += f"{label}: {count} 条 ({percentage:.1f}%)\n"

            self.batch_stats_text.config(state="normal")
            self.batch_stats_text.delete(1.0, tk.END)
            self.batch_stats_text.insert(1.0, stats_text)
            self.batch_stats_text.config(state="disabled")

            self.batch_status_var.set(f"已预测 {total} 条记录")
        else:
            self.batch_status_var.set("未进行批量预测")
            self.batch_stats_text.config(state="normal")
            self.batch_stats_text.delete(1.0, tk.END)
            self.batch_stats_text.config(state="disabled")

    def setup_help_tab(self):
        """设置帮助选项卡"""
        self.help_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.help_tab, text="❓ 使用帮助")

        # 创建帮助内容
        self.setup_help_content()

    def setup_help_content(self):
        """设置帮助内容"""
        # 主框架
        main_frame = ttk.Frame(self.help_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 创建文本区域
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 文本区域
        help_text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                            font=("宋体", 11), padx=10, pady=10)
        help_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=help_text.yview)

        # 帮助内容
        help_content = """
城市交通事故分析与预警系统 - 使用帮助
=========================================

📊 数据管理
------------
• 导入CSV：加载交通事故数据文件
• 导出CSV：将当前数据保存为CSV格式
• 导出Excel：将数据导出为Excel文件，包含数据表和统计信息
• 生成示例：快速生成测试数据
• 筛选数据：按条件筛选数据
• 搜索数据：搜索包含关键词的记录
• 排序数据：按指定列排序
• 添加记录：手动添加新的事故记录
• 删除记录：删除选中的记录

📈 可视化分析
------------
• 图表类型：支持柱状图、折线图、饼图、散点图、热力图、箱线图
• 轴选择：选择X轴和Y轴的数据列
• 生成图表：根据选择生成可视化图表
• 导出图片：将图表导出为PNG、JPG、PDF等格式
• 工具栏：图表缩放、平移、保存等操作

⚠️ 风险预测
------------
• 训练模型：使用当前数据训练风险预测模型
• 单条预测：输入事故信息，预测风险等级
• 批量预测：对当前所有数据进行风险预测
• 保存模型：将训练好的模型保存为文件
• 加载模型：从文件加载已训练的模型
• 导出结果：将预测结果导出为CSV或Excel
• 特征重要性：查看影响风险预测的主要因素

系统特点
--------
1. 一体化界面：数据管理、可视化、预测在一个界面中完成
2. 智能预测：基于机器学习的事故风险预测
3. 多种导出：支持CSV、Excel、图片等多种格式导出
4. 用户友好：简洁直观的操作界面

技术支持
--------
Github仓库地址：https://github.com/PaintHelloWorld/Traffic_Analysis_System
© 2026 交通数据分析项目 - 版本 1.3.1
        """

        help_text.insert(1.0, help_content)
        help_text.config(state="disabled")

    def setup_status_bar(self):
        """设置状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        # 同时更新信息面板
        if hasattr(self, 'info_panel'):
            self.info_panel.update_info()

    # ============ 核心功能方法 ============

    def init_predictor(self):
        """初始化预测器"""
        if self.predictor is None:
            from predictor import TrafficPredictor
            self.predictor = TrafficPredictor()
        return self.predictor

    def train_model(self):
        """训练预测模型"""
        if self.data_manager.display_data is None:
            self.update_status("请先加载数据")
            tk.messagebox.showwarning("无数据", "请先导入数据")
            return

        predictor = self.init_predictor()

        try:
            success, result = predictor.train_model(self.data_manager.display_data)

            if success:
                self.model_status_var.set(f"模型状态: 已训练 (准确率: {result['accuracy']:.2%})")
                self.update_status(f"模型训练成功，准确率: {result['accuracy']:.2%}")

                # 显示详细报告
                report_dialog = tk.Toplevel(self.root)
                report_dialog.title("模型训练报告")
                report_dialog.geometry("500x400")

                text = tk.Text(report_dialog, wrap=tk.WORD)
                text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                report_text = f"模型训练完成！\n\n"
                report_text += f"准确率: {result['accuracy']:.2%}\n"
                report_text += f"使用特征数: {result['feature_count']}\n"
                report_text += f"训练集大小: {result['train_size']}\n"
                report_text += f"测试集大小: {result['test_size']}\n\n"
                report_text += "分类报告:\n" + result['report']

                text.insert(1.0, report_text)
                text.config(state="disabled")

                ttk.Button(report_dialog, text="确定",
                           command=report_dialog.destroy).pack(pady=10)

            else:
                self.update_status(f"模型训练失败: {result}")
                tk.messagebox.showerror("训练失败", result)
        except Exception as e:
            self.update_status(f"模型训练异常: {str(e)}")

    def predict_single(self):
        """单条预测"""
        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("请先训练模型")
            tk.messagebox.showwarning("模型未训练", "请先训练预测模型")
            return

        try:
            # 收集输入数据
            input_dict = {}
            for label, entry in self.pred_inputs.items():
                value = entry.get()
                # 尝试转换为数值
                try:
                    if label in ['受伤人数', '死亡人数', '温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)']:
                        value = float(value)
                except:
                    pass
                input_dict[label] = value

            # 进行预测
            risk_label, prob_dict, message = self.predictor.predict_single(input_dict)

            if risk_label:
                # 更新结果显示
                self.pred_result_var.set(f"预测结果: {risk_label}")

                # 设置颜色
                color_map = {
                    '低风险': 'green',
                    '中风险': 'orange',
                    '高风险': 'red'
                }
                color = color_map.get(risk_label, 'black')

                # 创建带颜色的标签
                for widget in self.pred_result_var._widgets:
                    widget.config(foreground=color)

                # 显示概率
                if prob_dict:
                    prob_text = " | ".join([f"{k}: {v:.1%}" for k, v in prob_dict.items()])
                    self.pred_prob_var.set(f"概率分布: {prob_text}")

                self.update_status(f"预测完成: {risk_label}")
            else:
                self.update_status(f"预测失败: {message}")

        except Exception as e:
            self.update_status(f"预测出错: {str(e)}")

    def predict_batch(self):
        """批量预测"""
        if self.data_manager.display_data is None:
            self.update_status("请先加载数据")
            return

        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("请先训练模型")
            tk.messagebox.showwarning("模型未训练", "请先训练预测模型")
            return

        try:
            # 进行批量预测
            predictions, probabilities, message = self.predictor.predict(
                self.data_manager.display_data
            )

            if predictions is not None:
                # 添加预测结果到数据
                self.data_manager.display_data['预测风险等级'] = predictions

                # 更新状态和显示
                self.batch_status_var.set(f"批量预测完成，共 {len(predictions)} 条记录")
                self.update_status(f"批量预测完成，共 {len(predictions)} 条记录")

                # 刷新预测结果显示
                self.refresh_predictions()

                # 刷新表格显示
                if hasattr(self, 'data_table'):
                    self.data_table.load_data()
                if hasattr(self, 'info_panel'):
                    self.info_panel.update_info()

                tk.messagebox.showinfo("批量预测完成", f"已完成 {len(predictions)} 条记录的预测")
            else:
                self.update_status(f"批量预测失败: {message}")

        except Exception as e:
            self.update_status(f"批量预测出错: {str(e)}")

    def export_predictions_csv(self):
        """导出预测结果为CSV"""
        self.export_predictions('csv')

    def export_predictions_excel(self):
        """导出预测结果为Excel"""
        self.export_predictions('excel')

    def export_predictions(self, file_type='csv'):
        """导出预测结果（通用方法）"""
        if self.data_manager.display_data is None or '预测风险等级' not in self.data_manager.display_data.columns:
            self.update_status("没有预测结果可导出")
            tk.messagebox.showwarning("无结果", "请先进行批量预测")
            return

        # 设置文件类型
        if file_type == 'excel':
            default_ext = ".xlsx"
            filetypes = [
                ("Excel文件", "*.xlsx"),
                ("Excel 97-2003", "*.xls"),
                ("所有文件", "*.*")
            ]
        else:  # csv
            default_ext = ".csv"
            filetypes = [("CSV文件", "*.csv"), ("所有文件", "*.*")]

        filepath = tk.filedialog.asksaveasfilename(
            defaultextension=default_ext,
            filetypes=filetypes
        )

        if filepath:
            if file_type == 'excel':
                success, message = self.data_manager.export_to_excel(filepath)
            else:
                success, message = self.data_manager.save_to_csv(filepath)

            if success:
                self.update_status(f"预测结果已导出到: {filepath}")
                tk.messagebox.showinfo("导出成功", f"预测结果已成功导出到:\n{filepath}")
            else:
                self.update_status(message)
                tk.messagebox.showerror("导出失败", message)

    def show_feature_importance(self):
        """显示特征重要性"""
        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("请先训练模型")
            return

        importance_df = self.predictor.get_feature_importance()

        if importance_df is not None:
            self.feature_text.config(state="normal")
            self.feature_text.delete(1.0, tk.END)

            text = "特征重要性排序:\n"
            text += "=" * 40 + "\n\n"

            for idx, row in importance_df.iterrows():
                text += f"{row['feature']}: {row['importance']:.3f}\n"

            self.feature_text.insert(1.0, text)
            self.feature_text.config(state="disabled")
            self.update_status("特征重要性已显示")
        else:
            self.update_status("无法获取特征重要性")

    def load_model(self):
        """加载模型文件"""
        filepath = tk.filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("模型文件", "*.pkl"), ("所有文件", "*.*")]
        )

        if filepath:
            predictor = self.init_predictor()
            success, message = predictor.load_model(filepath)

            if success:
                self.model_status_var.set("模型状态: 已加载")
                self.update_status(message)
                tk.messagebox.showinfo("加载成功", "模型加载成功")
            else:
                self.update_status(message)
                tk.messagebox.showerror("加载失败", message)

    def save_model(self):
        """保存模型文件"""
        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("没有训练好的模型可保存")
            tk.messagebox.showwarning("无模型", "请先训练模型")
            return

        filepath = tk.filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("模型文件", "*.pkl")]
        )

        if filepath:
            success, message = self.predictor.save_model(filepath)
            self.update_status(message)

            if success:
                tk.messagebox.showinfo("保存成功", "模型保存成功")
            else:
                tk.messagebox.showerror("保存失败", message)
