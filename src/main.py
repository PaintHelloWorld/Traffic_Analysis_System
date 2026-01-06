# main.py - 程序入口
# Github仓库地址：
# https://github.com/PaintHelloWorld/Traffic_Analysis_System
# 真诚邀请各位老师前来查看！❤
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

"""
作为大一非计算机专业学生，这个项目让我深刻体会到：
1. 编程不仅是写代码，更是解决问题的系统思维；
2. 好的项目需要清晰的设计文档和注释；
3. 学会使用工具（包括AI）是现代编程的必备能力；
4. 从需求分析到产品实现的完整流程比代码本身更重要。
感谢老师的指导！
"""