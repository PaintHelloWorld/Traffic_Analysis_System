# utils.py - 工具函数
import os
import colorsys
from datetime import datetime
import tkinter as tk

# ==================== 颜色工具 ====================

def generate_colors(n, saturation=0.7, value=0.9):
    """
    生成一组美观的颜色

    Args:
        n: 颜色数量
        saturation: 饱和度 (0-1)
        value: 明度 (0-1)

    Returns:
        list: 十六进制颜色代码列表
    """
    colors = []
    for i in range(n):
        hue = i / n  # 均匀分布色相
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def get_risk_color(risk_level, alpha=1.0):
    """
    根据风险等级获取颜色

    Args:
        risk_level: 风险等级 (0:低, 1:中, 2:高)
        alpha: 透明度 (0-1)

    Returns:
        str: 颜色代码
    """
    colors = {
        0: f'rgba(76, 175, 80, {alpha})',  # 绿色 - 低风险
        1: f'rgba(255, 152, 0, {alpha})',  # 橙色 - 中风险
        2: f'rgba(244, 67, 54, {alpha})'  # 红色 - 高风险
    }
    return colors.get(risk_level, f'rgba(158, 158, 158, {alpha})')  # 灰色 - 默认


# ==================== 时间工具 ====================

def format_datetime(dt, format_str='%Y-%m-%d %H:%M:%S'):
    """
    格式化日期时间

    Args:
        dt: datetime对象或字符串
        format_str: 格式字符串

    Returns:
        str: 格式化后的字符串
    """
    if isinstance(dt, str):
        try:
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        except:
            return dt

    if isinstance(dt, datetime):
        return dt.strftime(format_str)

    return str(dt)


def parse_time_range(time_str):
    """
    解析时间范围字符串

    Args:
        time_str: 时间字符串，如 "08:00-18:00"

    Returns:
        tuple: (开始时间, 结束时间) 的datetime.time对象
    """
    try:
        start_str, end_str = time_str.split('-')
        start_time = datetime.strptime(start_str.strip(), '%H:%M').time()
        end_time = datetime.strptime(end_str.strip(), '%H:%M').time()
        return start_time, end_time
    except:
        return None, None


def get_time_period(hour):
    """
    根据小时获取时间段

    Args:
        hour: 小时 (0-23)

    Returns:
        str: 时间段描述
    """
    if 5 <= hour < 8:
        return "清晨"
    elif 8 <= hour < 12:
        return "上午"
    elif 12 <= hour < 14:
        return "中午"
    elif 14 <= hour < 18:
        return "下午"
    elif 18 <= hour < 22:
        return "晚上"
    else:
        return "深夜"


# ==================== 文件工具 ====================

def ensure_directory(directory):
    """
    确保目录存在，不存在则创建

    Args:
        directory: 目录路径

    Returns:
        bool: 是否成功
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except:
        return False


def get_file_extension(filepath):
    """
    获取文件扩展名（不含点）

    Args:
        filepath: 文件路径

    Returns:
        str: 扩展名
    """
    return os.path.splitext(filepath)[1][1:].lower()


def is_valid_csv(filepath):
    """
    检查文件是否是有效的CSV

    Args:
        filepath: 文件路径

    Returns:
        bool: 是否有效
    """
    if not os.path.exists(filepath):
        return False

    ext = get_file_extension(filepath)
    if ext not in ['csv', 'txt']:
        return False

    # 检查文件大小（不超过10MB）
    if os.path.getsize(filepath) > 10 * 1024 * 1024:
        return False

    return True


# ==================== 数据验证工具 ====================

def validate_number(value, min_val=None, max_val=None):
    """
    验证数字

    Args:
        value: 要验证的值
        min_val: 最小值
        max_val: 最大值

    Returns:
        tuple: (是否有效, 错误信息)
    """
    try:
        num = float(value)

        if min_val is not None and num < min_val:
            return False, f"值不能小于 {min_val}"

        if max_val is not None and num > max_val:
            return False, f"值不能大于 {max_val}"

        return True, None
    except:
        return False, "请输入有效的数字"


def validate_date(date_str, format_str='%Y-%m-%d'):
    """
    验证日期字符串

    Args:
        date_str: 日期字符串
        format_str: 日期格式

    Returns:
        tuple: (是否有效, 错误信息)
    """
    try:
        datetime.strptime(date_str, format_str)
        return True, None
    except ValueError:
        return False, f"日期格式应为 {format_str}"


def validate_time(time_str, format_str='%H:%M'):
    """
    验证时间字符串

    Args:
        time_str: 时间字符串
        format_str: 时间格式

    Returns:
        tuple: (是否有效, 错误信息)
    """
    try:
        datetime.strptime(time_str, format_str)
        return True, None
    except ValueError:
        return False, f"时间格式应为 {format_str}"


# ==================== 界面工具 ====================

def center_window(window, width=None, height=None):
    """
    将窗口居中显示

    Args:
        window: tkinter窗口
        width: 窗口宽度（默认使用当前宽度）
        height: 窗口高度（默认使用当前高度）
    """
    window.update_idletasks()

    if width is None:
        width = window.winfo_width()
    if height is None:
        height = window.winfo_height()

    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)

    window.geometry(f'{width}x{height}+{x}+{y}')


def create_tooltip(widget, text):
    """
    为控件创建工具提示

    Args:
        widget: tkinter控件
        text: 提示文本
    """

    def show_tooltip(event):
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

        label = tk.Label(tooltip, text=text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("宋体", 10))
        label.pack()

        def hide_tooltip():
            tooltip.destroy()

        widget.bind('<Leave>', lambda e: hide_tooltip())

    widget.bind('<Enter>', show_tooltip)


def show_loading_dialog(parent, title="处理中", message="请稍候..."):
    """
    显示加载对话框

    Args:
        parent: 父窗口
        title: 对话框标题
        message: 消息文本

    Returns:
        tk.Toplevel: 加载对话框
    """
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.geometry("300x100")
    dialog.transient(parent)
    dialog.grab_set()

    # 居中显示
    center_window(dialog, 300, 100)

    # 消息标签
    label = tk.Label(dialog, text=message, font=("宋体", 11))
    label.pack(pady=20)

    # 进度条
    progress = tk.ttk.Progressbar(dialog, mode='indeterminate')
    progress.pack(pady=10)
    progress.start()

    dialog.update()
    return dialog


# ==================== 文本处理工具 ====================

def truncate_text(text, max_length=50, suffix="..."):
    """
    截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        str: 截断后的文本
    """
    if not isinstance(text, str):
        text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def format_number(number, decimal_places=2):
    """
    格式化数字（添加千位分隔符）

    Args:
        number: 数字
        decimal_places: 小数位数

    Returns:
        str: 格式化后的字符串
    """
    try:
        num = float(number)
        if decimal_places == 0:
            return f"{int(num):,}"
        else:
            return f"{num:,.{decimal_places}f}"
    except:
        return str(number)


# ==================== 测试函数 ====================

def test_utils():
    """测试工具函数"""
    print("=== 测试 utils ===")

    # 1. 颜色生成
    colors = generate_colors(5)
    print(f"1. 生成5种颜色: {colors[:3]}... ✓")

    # 2. 风险颜色
    risk_colors = [get_risk_color(i) for i in range(3)]
    print(f"2. 风险等级颜色: {risk_colors} ✓")

    # 3. 时间格式化
    now = datetime.now()
    formatted = format_datetime(now)
    print(f"3. 时间格式化: {formatted} ✓")

    # 4. 时间段
    period = get_time_period(14)
    print(f"4. 14点的时段: {period} ✓")

    # 5. 数字验证
    valid, msg = validate_number("123.5", min_val=0, max_val=200)
    print(f"5. 数字验证 (123.5): {'有效' if valid else '无效'} - {msg or 'OK'} ✓")

    # 6. 日期验证
    valid, msg = validate_date("2024-01-01")
    print(f"6. 日期验证 (2024-01-01): {'有效' if valid else '无效'} ✓")

    # 7. 文本截断
    truncated = truncate_text("这是一个很长的文本需要被截断显示", 10)
    print(f"7. 文本截断: {truncated} ✓")

    # 8. 数字格式化
    formatted_num = format_number(1234567.891, 2)
    print(f"8. 数字格式化: {formatted_num} ✓")

    print("\n=== 所有测试通过 ===")


if __name__ == "__main__":
    test_utils()