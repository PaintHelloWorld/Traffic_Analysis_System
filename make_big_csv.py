# make_big_csv.py
"""
简单粗暴的大数据集生成器 - 直接生成超大CSV文件
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

print("🚀 开始生成超大数据集...")

# 用户选择大小
print("\n请选择数据集大小：")
print("1. 小型测试 (1万条)")
print("2. 中型测试 (10万条)")
print("3. 大型测试 (50万条)")
print("4. 超大型测试 (100万条)")
print("5. 自定义大小")

choice = input("请输入选择 (1-5): ").strip()

if choice == '1':
    num_records = 10000
elif choice == '2':
    num_records = 100000
elif choice == '3':
    num_records = 500000
elif choice == '4':
    num_records = 1000000
elif choice == '5':
    try:
        num_records = int(input("请输入记录条数: "))
    except:
        num_records = 100000
        print("输入无效，使用默认10万条")
else:
    num_records = 100000
    print("默认使用10万条")

print(f"\n将要生成 {num_records:,} 条记录...")

# 开始计时
start_time = time.time()

# 生成数据
print("正在生成数据...")

# 1. 生成时间数据（分批生成避免内存溢出）
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 5)  # 5年范围

# 时间列 - 使用numpy的向量化操作生成
time_stamps = pd.date_range(start=start_date, end=end_date, periods=min(num_records, 100000))
if num_records > len(time_stamps):
    # 如果数量太多，随机选择
    time_data = np.random.choice(time_stamps, num_records)
else:
    time_data = np.random.choice(time_stamps, num_records, replace=False)

# 2. 生成其他数据（使用numpy批量生成，超快！）
print("批量生成字段...")

# 固定列表选项
areas = ['朝阳区', '海淀区', '东城区', '西城区', '丰台区', '石景山区']
road_types = ['高速公路', '城市主干道', '城市次干道', '支路']
accident_types = ['追尾', '侧碰', '刮擦', '单车事故', '多车连环']
weathers = ['晴天', '雨天', '阴天', '雾天', '雪天']
severity_levels = ['轻微', '一般', '严重']

# 批量生成所有数据
data = {
    '事故ID': np.arange(1, num_records + 1),
    '事故时间': time_data,
    '所在区域': np.random.choice(areas, num_records),
    '道路类型': np.random.choice(road_types, num_records),
    '事故类型': np.random.choice(accident_types, num_records),
    '天气情况': np.random.choice(weathers, num_records),
    '受伤人数': np.random.randint(0, 5, num_records),
    '死亡人数': np.random.randint(0, 2, num_records),
    '温度(℃)': np.random.uniform(-10, 40, num_records).round(1),
    '湿度(%)': np.random.randint(20, 95, num_records),
    '能见度(km)': np.random.uniform(0.1, 20, num_records).round(1),
    '风速(m/s)': np.random.uniform(0, 20, num_records).round(1),
    '事故等级': np.random.choice(severity_levels, num_records, p=[0.7, 0.25, 0.05])
}

print("创建DataFrame...")
df = pd.DataFrame(data)

# 保存到CSV
filename = f"traffic_bigdata_{num_records}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print(f"\n正在保存到 {filename}...")

# 分批写入CSV以避免内存问题
chunk_size = 100000  # 每批10万条
num_chunks = (num_records + chunk_size - 1) // chunk_size

with open(filename, 'w', encoding='utf-8', newline='') as f:
    # 写入表头
    df.head(0).to_csv(f, index=False)

    # 分批写入数据
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_records)
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_df.to_csv(f, header=False, index=False, encoding='utf-8')
        print(f"  已写入 {end_idx:,}/{num_records:,} 条记录")
        del chunk_df  # 释放内存

elapsed_time = time.time() - start_time

print(f"\n✅ 完成！")
print(f"📊 生成记录数: {num_records:,}")
print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
print(f"📁 文件大小: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
print(f"📍 文件位置: {os.path.abspath(filename)}")

# 显示前几行数据
print("\n🔍 数据预览（前5行）:")
print(df.head())
print(f"\n📋 列信息: {list(df.columns)}")