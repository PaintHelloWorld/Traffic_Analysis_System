# visualizer.py - 数据可视化模块
"""
数据可视化模块 - TrafficVisualizer
===================================

图表支持：
    1. 柱状图 - 分类数据对比
    2. 折线图 - 趋势分析
    3. 饼图 - 比例分布
    4. 散点图 - 相关性分析
    5. 热力图 - 特征相关性
    6. 箱线图 - 数据分布

技术实现：
    1. 图表嵌入：FigureCanvasTkAgg将matplotlib嵌入tkinter
    2. 动态切换：根据数据类型智能推荐图表类型
    3. 工具栏：集成matplotlib导航工具栏（缩放、保存）
    4. 导出功能：支持PNG、JPG、PDF、SVG格式

设计特点：
    1. 智能适配：饼图自动禁用Y轴选择
    2. 性能优化：大数据集时限制显示项数
    3. 用户友好：清晰的坐标轴标签和图例
    4. 中文支持：配置中文字体避免乱码
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#选择FigureCanvasTkAgg的原因：在初版代码中，我使用了plt.show()
#结果会弹独立窗口，破坏UI统一性
#我向ai寻求帮助，找到了解决方案
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrafficVisualizer:
    """交通事故可视化器 """

    def __init__(self, data_manager, parent_frame):
        """
        初始化可视化器

        Args:
            data_manager: TrafficDataManager实例
            parent_frame: 父框架（用于显示图表）
        Returns:
            无。。。
        """
        self.data_manager = data_manager
        self.parent_frame = parent_frame
        self.current_figure = None
        self.canvas = None
        self.toolbar = None
        self.chart_type = "柱状图"

        # 创建控制面板
        self.setup_control_panel()

        # 创建图表显示区域
        self.setup_chart_area()

        # 初始状态
        self.update_status("可视化器就绪")

    def setup_control_panel(self):
        """创建图表控制面板"""
        control_frame = ttk.LabelFrame(self.parent_frame, text="图表设置", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：图表类型
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(row1, text="图表类型:").pack(side=tk.LEFT, padx=5)
        self.chart_type_var = tk.StringVar(value="柱状图")
        chart_types = ["柱状图", "折线图", "饼图", "散点图", "热力图", "箱线图"]
        chart_combo = ttk.Combobox(row1, textvariable=self.chart_type_var,
                                   values=chart_types, state="readonly", width=12)
        chart_combo.pack(side=tk.LEFT, padx=5)
        chart_combo.bind("<<ComboboxSelected>>", lambda e: self.on_chart_type_changed())

        # 第二行：轴选择和按钮
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X)

        # X轴选择
        ttk.Label(row2, text="X轴:").pack(side=tk.LEFT, padx=5)
        self.x_axis_var = tk.StringVar()
        self.x_axis_combo = ttk.Combobox(row2, textvariable=self.x_axis_var, width=15)
        self.x_axis_combo.pack(side=tk.LEFT, padx=5)

        # Y轴选择
        ttk.Label(row2, text="Y轴:").pack(side=tk.LEFT, padx=5)
        self.y_axis_var = tk.StringVar()
        self.y_axis_combo = ttk.Combobox(row2, textvariable=self.y_axis_var, width=15)
        self.y_axis_combo.pack(side=tk.LEFT, padx=5)

        # 按钮
        ttk.Button(row2, text="生成图表", command=self.generate_chart).pack(side=tk.LEFT, padx=10)
        ttk.Button(row2, text="导出图片", command=self.export_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="刷新数据", command=self.refresh_data).pack(side=tk.LEFT, padx=5)

        # 更新轴选项
        self.update_axis_options()

    def setup_chart_area(self):
        """创建图表显示区域"""
        # 图表容器框架
        chart_container = ttk.Frame(self.parent_frame)
        chart_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 图表框架
        self.chart_frame = ttk.Frame(chart_container)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(chart_container, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def update_axis_options(self):
        """更新坐标轴选项"""
        if self.data_manager.display_data is not None:
            data = self.data_manager.display_data
            columns = list(data.columns)

            # 数值列（用于Y轴）
            numeric_cols = []
            for col in columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_cols.append(col)

            # 更新下拉框
            self.x_axis_combo['values'] = columns
            self.y_axis_combo['values'] = numeric_cols

            # 设置智能默认值
            if columns:
                # 尝试找到时间列
                time_cols = [col for col in columns if any(kw in col.lower()
                                                           for kw in ['time', 'date', '时间', '日期'])]
                if time_cols:
                    self.x_axis_var.set(time_cols[0])
                else:
                    self.x_axis_var.set(columns[0])

            if numeric_cols:
                # 尝试找到数值列
                num_col = numeric_cols[0] if len(numeric_cols) > 0 else ""
                self.y_axis_var.set(num_col)

    def on_chart_type_changed(self):
        """图表类型改变时的处理"""
        chart_type = self.chart_type_var.get()

        # 根据图表类型启用/禁用xy轴选择
        if chart_type == "饼图":
            self.x_axis_combo.config(state="normal")
            self.y_axis_combo.config(state="disabled")
        elif chart_type == "热力图":
            self.x_axis_combo.config(state="disabled")
            self.y_axis_combo.config(state="disabled")
        else:
            self.x_axis_combo.config(state="normal")
            self.y_axis_combo.config(state="normal")

        # 自动生成图表
        self.generate_chart()

    def refresh_data(self):
        """刷新数据"""
        self.update_axis_options()
        self.update_status("数据已刷新")
        self.generate_chart()

    def clear_chart(self):
        """清除当前图表"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None

    def generate_chart(self):
        """生成图表"""
        if self.data_manager.display_data is None:
            self.update_status("请先加载数据")
            messagebox.showwarning("无数据", "请先加载数据")
            return

        chart_type = self.chart_type_var.get()

        try:
            self.clear_chart()

            if chart_type == "柱状图":
                self.create_bar_chart()
            elif chart_type == "折线图":
                self.create_line_chart()
            elif chart_type == "饼图":
                self.create_pie_chart()
            elif chart_type == "散点图":
                self.create_scatter_plot()
            elif chart_type == "热力图":
                self.create_heatmap()
            elif chart_type == "箱线图":
                self.create_box_plot()

        except Exception as e:
            self.update_status(f"图表生成失败: {str(e)}")
            messagebox.showerror("图表错误", f"生成图表时出错:\n{str(e)}")

    def create_bar_chart(self):
        """创建柱状图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            self.update_status("请选择X轴和Y轴")
            return

        data = self.data_manager.display_data

        # 创建图形
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # 分组统计
        if data[x_col].dtype == 'object' or data[x_col].nunique() < 15:
            # 分类数据：按类别分组
            group_data = data.groupby(x_col)[y_col].mean().sort_values(ascending=False)

            # 限制显示数量
            if len(group_data) > 15:
                group_data = group_data.head(15)

            x_pos = range(len(group_data))
            bars = ax.bar(x_pos, group_data.values, color='steelblue', alpha=0.8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_data.index, rotation=45, ha='right')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            # 数值数据：直接绘制
            sorted_data = data.sort_values(x_col)
            ax.bar(sorted_data[x_col].astype(str), sorted_data[y_col],
                   color='steelblue', alpha=0.8)
            ax.tick_params(axis='x', rotation=45)

        ax.set_title(f'{y_col} 按 {x_col} 分布', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.display_figure(fig)
        self.update_status(f"柱状图: {y_col} vs {x_col}")

    def create_line_chart(self):
        """创建折线图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            self.update_status("请选择X轴和Y轴")
            return

        data = self.data_manager.display_data.copy()

        # 尝试转换为时间序列
        try:
            data[x_col] = pd.to_datetime(data[x_col])
            data = data.sort_values(x_col)
            is_time_series = True
        except:
            data = data.sort_values(x_col)
            is_time_series = False

        # 创建图形
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # 绘制折线图
        ax.plot(data[x_col], data[y_col], marker='o', linewidth=2,
                markersize=5, color='coral', alpha=0.8, label=y_col)

        ax.set_title(f'{y_col} 趋势图', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()

        # 格式化时间轴
        if is_time_series:
            fig.autofmt_xdate()

        self.display_figure(fig)
        self.update_status(f"折线图: {y_col} 趋势")

    def create_pie_chart(self):
        """创建饼图"""
        x_col = self.x_axis_var.get()

        if not x_col:
            self.update_status("请选择X轴")
            return

        data = self.data_manager.display_data

        # 统计各类别
        value_counts = data[x_col].value_counts()

        # 限制类别数量
        if len(value_counts) > 10:
            top_data = value_counts.head(10)
            others = value_counts[10:].sum()
            top_data['其他'] = others
            value_counts = top_data

        # 创建图形
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)

        # 颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))

        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(edgecolor='white', linewidth=1)
        )

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title(f'{x_col} 分布比例', fontsize=14, fontweight='bold')

        self.display_figure(fig)
        self.update_status(f"饼图: {x_col} 分布")

    def create_scatter_plot(self):
        """创建散点图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            self.update_status("请选择X轴和Y轴")
            return

        data = self.data_manager.display_data

        # 创建图形
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # 绘制散点图
        scatter = ax.scatter(data[x_col], data[y_col],
                             c=data[y_col],  # 使用y值作为颜色
                             cmap='viridis',
                             alpha=0.7,
                             edgecolors='w',
                             linewidth=0.5,
                             s=100)

        ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label=y_col)

        self.display_figure(fig)
        self.update_status(f"散点图: {y_col} vs {x_col}")

    def create_heatmap(self):
        """创建热力图"""
        data = self.data_manager.display_data

        # 选择数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            self.update_status("需要至少2个数值列")
            return

        # 创建图形
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)

        # 计算相关性
        correlation = data[numeric_cols].corr()

        # 绘制热力图
        sns.heatmap(correlation,
                    ax=ax,
                    annot=True,
                    fmt=".2f",
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})

        ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')

        self.display_figure(fig)
        self.update_status("热力图: 特征相关性")

    def create_box_plot(self):
        """创建箱线图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            self.update_status("请选择X轴和Y轴")
            return

        data = self.data_manager.display_data

        # 创建图形
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # 限制类别数量
        if data[x_col].nunique() > 10:
            top_categories = data[x_col].value_counts().head(10).index
            filtered_data = data[data[x_col].isin(top_categories)]
            plot_data = [filtered_data[filtered_data[x_col] == cat][y_col]
                         for cat in top_categories]
            labels = top_categories
        else:
            categories = data[x_col].unique()
            plot_data = [data[data[x_col] == cat][y_col] for cat in categories]
            labels = categories

        # 绘制箱线图
        box = ax.boxplot(plot_data,
                         labels=labels,
                         patch_artist=True,
                         showmeans=True,
                         meanline=True)

        # 设置颜色
        colors = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'{y_col} 按 {x_col} 分布', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)

        self.display_figure(fig)
        self.update_status(f"箱线图: {y_col} 分布")

    def display_figure(self, figure):
        """显示图形"""
        self.current_figure = figure

        # 创建Canvas
        self.canvas = FigureCanvasTkAgg(figure, self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_status(self, message):
        """更新状态"""
        self.status_var.set(message)

    def export_image(self):
        """导出图表为图片"""
        if self.current_figure is None:
            messagebox.showwarning("无图表", "请先生成图表")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG图片", "*.png"),
                ("JPEG图片", "*.jpg"),
                ("PDF文件", "*.pdf"),
                ("SVG矢量图", "*.svg")
            ]
        )

        if filepath:
            try:
                self.current_figure.savefig(filepath, dpi=300, bbox_inches='tight')
                self.update_status(f"图表已导出到: {filepath}")
                messagebox.showinfo("导出成功", f"图表已成功导出到:\n{filepath}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出图表时出错:\n{str(e)}")


# ==================== 集成到主界面 ====================

def create_visualization_tab(notebook, data_manager):
    """
    创建可视化选项卡

    Args:
        notebook: ttk.Notebook 实例
        data_manager: TrafficDataManager 实例

    Returns:
        ttk.Frame: 可视化选项卡框架
    """
    # 创建选项卡框架
    viz_frame = ttk.Frame(notebook)
    notebook.add(viz_frame, text="📈 可视化分析")

    # 创建可视化器
    TrafficVisualizer(data_manager, viz_frame)

    return viz_frame