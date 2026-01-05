# main.py - 程序入口
import tkinter as tk
from data_manager import TrafficDataManager
from ui_components import IntegratedMainWindow


def main():
    root = tk.Tk()
    root.title("城市交通事故分析与预警系统")
    root.state('zoomed')
    root.geometry("1200x800")

    # TODO: 设计一个应用程序图标.ico
    '''
    try:
        root.iconbitmap('traffic_icon.ico')
    except:
        pass
    '''

    # 初始化数据管理器
    data_manager = TrafficDataManager()

    IntegratedMainWindow(root, data_manager)
    root.mainloop()


if __name__ == "__main__":
    main()