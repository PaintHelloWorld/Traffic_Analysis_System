# utils.py - 工具函数
import os
import colorsys
from datetime import datetime
import tkinter as tk

# ==================== 颜色工具 ====================

def generate_colors(n, saturation=0.7, value=0.9):
    """生成n种好看的颜色"""
    colors = []
    for i in range(n):
        # 在色相环上均匀取色
        hue = i / n
        # HSV转RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # 转成十六进制颜色码
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
    gba(R, G, B, A)，支持透明度
    比上一个函数多了一个A，更适合需要透明度的场景

    Args:
        risk_level: 风险等级 (0:低, 1:中, 2:高)
        alpha: 透明度 (0-1)

    Returns:
        str: 颜色rgb
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
        format_str: 目标格式字符串
    Returns:
        str: 格式化后的字符串
    Raises:
        ValueError: 如果字符串无法解析为datetime
    """
    if isinstance(dt, str):
        # 先尝试默认格式
        try:
            dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        except ValueError as e1:
            # 再尝试其他常见格式
            common_formats = [
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
                '%Y-%m-%d',
                '%Y/%m/%d'
            ]

            for fmt in common_formats:
                try:
                    dt = datetime.strptime(dt, fmt)
                    break
                except ValueError:
                    continue
            else:
                # 所有格式都失败，那就给用户提示。
                raise ValueError(
                    f"无法解析时间字符串 '{dt}'。支持的格式："
                    f"YYYY-MM-DD HH:MM:SS, YYYY-MM-DD HH:MM, "
                    f"YYYY/MM/DD HH:MM:SS, YYYY/MM/DD HH:MM, "
                    f"YYYY-MM-DD, YYYY/MM/DD"
                ) from e1

    if isinstance(dt, datetime):
        return dt.strftime(format_str)

    # 最后尝试转换为字符串
    return str(dt)


def parse_time_range(time_str):
    """把'08:00-18:00'这种字符串拆成两个时间对象"""
    try:
        start_str, end_str = time_str.split('-')
        start_time = datetime.strptime(start_str.strip(), '%H:%M').time()
        end_time = datetime.strptime(end_str.strip(), '%H:%M').time()
        return start_time, end_time
    except:
        return None, None


def get_time_period(hour):
    """根据小时数判断是上午，下午，还是晚上"""
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

    # 扩展名要是csv或txt
    ext = os.path.splitext(filepath)[1][1:].lower()
    if ext not in ['csv', 'txt']:
        return False

    # 检查文件大小（不超过10MB，否则应用会卡炸。）
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
        tuple: (是否有效bool, 错误信息)
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


def validate_date(date_str):
    """检查日期格式是否正确"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True, None
    except ValueError:
        return False, "日期格式应该是 YYYY-MM-DD"

def validate_time(time_str):
    """检查时间格式是否正确"""
    try:
        datetime.strptime(time_str, '%H:%M')
        return True, None
    except ValueError:
        return False, "时间格式应该是 HH:MM"


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


# ==================== 文本处理工具 ====================

def truncate_text(text, max_length=50):
    """文字太长时截断并加省略号"""
    if not isinstance(text, str):
        text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..." #3是“...”的长度


def format_number(number, decimal_places=2):
    """给数字加千位分隔符"""
    try:
        num = float(number)
        if decimal_places == 0:
            return f"{int(num):,}"
        else:
            return f"{num:,.{decimal_places}f}"
    except:
        return str(number)


# ==================== 测试函数 ====================
def test_all_functions():
    """测试一下各个函数"""
    print("=== 测试工具函数 ===")

    # 测试颜色函数
    colors = generate_colors(3)
    print(f"1. 生成3种颜色: {colors}")

    # 测试风险颜色
    for i in range(3):
        print(f"   风险等级{i}: {get_risk_color(i)}")

    # 测试时间函数
    now = datetime.now()
    print(f"2. 当前时间: {format_datetime(now)}")

    # 测试时间段
    print(f"   14点是: {get_time_period(14)}")

    # 测试数字验证
    print("3. 验证数字:")
    print(f"   123.5: {validate_number('123.5', 0, 200)}")
    print(f"   -10: {validate_number('-10', 0, 100)}")

    # 测试日期验证
    print("4. 验证日期:")
    print(f"   2024-01-01: {validate_date('2024-01-01')}")
    print(f"   2024/01/01: {validate_date('2024/01/01')}")

    # 测试文本截断
    long_text = "这是一段非常非常长的文字需要被截断"
    print(f"5. 截断文本: '{truncate_text(long_text, 10)}'")

    # 测试数字格式化
    print(f"6. 格式化数字: {format_number(1234567.891)}")

if __name__ == "__main__":
    test_all_functions()