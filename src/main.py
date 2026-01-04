# main.py - 主程序入口（集成版）
import tkinter as tk
from data_manager import TrafficDataManager
from ui_components import IntegratedMainWindow


def main():
    """主函数 - 启动应用程序"""
    root = tk.Tk()
    root.title("城市交通事故分析与预警系统")
    root.geometry("1200x700")

    # 设置应用程序图标（可选）
    try:
        root.iconbitmap('traffic_icon.ico')
    except:
        pass

    # 初始化数据管理器
    data_manager = TrafficDataManager()

    # 创建主窗口
    app = IntegratedMainWindow(root, data_manager)

    # 启动程序
    root.mainloop()


if __name__ == "__main__":
    main()