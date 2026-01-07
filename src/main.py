# main.py - 程序入口
# Github仓库地址：
# https://github.com/PaintHelloWorld/Traffic_Analysis_System
"""
说明：本项目分为5个单独的.py文件。
为满足作业提交需求，分在同一个文本文件。
"""

import tkinter as tk
from data_manager import TrafficDataManager
from ui_components import IntegratedMainWindow


def main():
    root = tk.Tk()
    root.title("城市交通事故分析与预警系统")
    root.state('zoomed')
    root.geometry("1200x800")

    # 初始化数据管理器
    data_manager = TrafficDataManager()

    IntegratedMainWindow(root, data_manager)
    root.mainloop()


if __name__ == "__main__":
    main()