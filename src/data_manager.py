# data_manager.py - 数据管理核心
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TrafficDataManager:
    """交通事故数据管理器 - 负责所有数据相关操作"""

    def __init__(self):
        """初始化数据管理器"""
        self.raw_data = None  # 原始数据（只读）
        self.display_data = None  # 当前显示的数据（筛选/排序后）
        self.current_file = None  # 当前文件路径
        self.filters = {}  # 当前筛选条件
        self.sort_column = None  # 当前排序列
        self.sort_ascending = True  # 排序方向

    # ==================== 基础数据操作 ====================

    def load_csv(self, filepath):
        """加载CSV文件"""
        try:
            # 1. 读取CSV文件
            self.raw_data = pd.read_csv(filepath, encoding='utf-8')

            # 2. 自动识别和转换数据类型
            self._auto_detect_types()

            # 3. 设置显示数据（初始为原始数据副本）
            self.display_data = self.raw_data.copy()
            self.current_file = filepath

            # 4. 重置筛选和排序状态
            self.filters = {}
            self.sort_column = None
            self.sort_ascending = True

            return True, f"成功加载 {len(self.raw_data)} 条记录，{len(self.raw_data.columns)} 个字段"

        except FileNotFoundError:
            return False, f"文件不存在: {filepath}"
        except Exception as e:
            return False, f"加载失败: {e}"

    def _auto_detect_types(self):
        """自动识别和转换数据类型"""
        if self.raw_data is None:
            return

        for col in self.raw_data.columns:
            # 尝试转换为datetime（列名包含time/date）
            if any(keyword in col.lower() for keyword in ['time', 'date', '日期', '时间']):
                try:
                    self.raw_data[col] = pd.to_datetime(self.raw_data[col], errors='coerce')
                except:
                    pass

            # 尝试转换为数值类型
            try:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='ignore')
            except:
                pass

    def save_to_csv(self, filepath=None):
        """保存数据到CSV文件"""
        if self.display_data is None:
            return False, "没有可保存的数据"

        try:
            # 如果没有指定路径，使用当前文件路径
            if filepath is None:
                if self.current_file:
                    filepath = self.current_file
                else:
                    return False, "请指定保存路径"

            # 保存数据
            self.display_data.to_csv(filepath, index=False, encoding='utf-8')
            self.current_file = filepath
            return True, f"数据已保存到: {filepath}"

        except Exception as e:
            return False, f"保存失败: {str(e)}"

    # ==================== 数据查询与统计 ====================

    def get_data_preview(self, n=5):
        """获取数据预览（前n行）"""
        if self.display_data is not None:
            return self.display_data.head(n)
        return None

    def get_data_tail(self, n=5):
        """获取数据尾部（后n行）"""
        if self.display_data is not None:
            return self.display_data.tail(n)
        return None

    def get_column_names(self):
        """获取所有列名"""
        if self.display_data is not None:
            return list(self.display_data.columns)
        return []

    def get_column_types(self):
        """获取列数据类型"""
        if self.display_data is not None:
            type_dict = {}
            for col in self.display_data.columns:
                dtype = self.display_data[col].dtype
                # 简化类型显示
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    type_dict[col] = '日期时间'
                elif pd.api.types.is_numeric_dtype(dtype):
                    type_dict[col] = '数值'
                else:
                    type_dict[col] = '文本'
            return type_dict
        return {}

    def get_basic_stats(self):
        """获取基本统计信息"""
        if self.display_data is None:
            return None

        stats = {
            'total_records': len(self.display_data),
            'total_columns': len(self.display_data.columns),
            'column_details': []
        }

        for col in self.display_data.columns:
            col_stats = {
                'name': col,
                'type': str(self.display_data[col].dtype),
                'non_null': self.display_data[col].count(),
                'null_count': self.display_data[col].isnull().sum(),
                'null_percentage': f"{self.display_data[col].isnull().sum() / len(self.display_data) * 100:.1f}%",
                'unique_values': self.display_data[col].nunique()
            }

            # 数值列的额外统计
            if pd.api.types.is_numeric_dtype(self.display_data[col]):
                col_stats.update({
                    'min': float(self.display_data[col].min()),
                    'max': float(self.display_data[col].max()),
                    'mean': float(self.display_data[col].mean()),
                    'std': float(self.display_data[col].std())
                })

            stats['column_details'].append(col_stats)

        return stats

    def get_unique_values(self, column, limit=20):
        """获取某列的唯一值列表（限制数量）"""
        if self.display_data is None or column not in self.display_data.columns:
            return []

        unique_vals = self.display_data[column].dropna().unique()
        # 转换为字符串并排序
        unique_vals = sorted([str(val) for val in unique_vals])

        # 限制返回数量
        if len(unique_vals) > limit:
            return unique_vals[:limit] + [f"... 共 {len(unique_vals)} 个值"]

        return unique_vals

    # ==================== 数据筛选功能 ====================

    def apply_filter(self, column, condition):
        """应用筛选条件"""
        if self.raw_data is None:
            return False, "请先加载数据"

        try:
            # 保存筛选条件
            self.filters[column] = condition

            # 应用所有筛选条件
            filtered_data = self.raw_data.copy()

            for filter_col, filter_cond in self.filters.items():
                if filter_col not in filtered_data.columns:
                    continue

                if isinstance(filter_cond, tuple) and len(filter_cond) == 2:
                    # 范围筛选 [min, max]
                    min_val, max_val = filter_cond
                    filtered_data = filtered_data[
                        (filtered_data[filter_col] >= min_val) &
                        (filtered_data[filter_col] <= max_val)
                        ]
                elif isinstance(filter_cond, list):
                    # 多值筛选
                    filtered_data = filtered_data[filtered_data[filter_col].isin(filter_cond)]
                else:
                    # 文本包含筛选
                    filtered_data = filtered_data[
                        filtered_data[filter_col].astype(str).str.contains(str(filter_cond), na=False)
                    ]

            self.display_data = filtered_data
            return True, f"筛选后剩余 {len(filtered_data)} 条记录"

        except Exception as e:
            return False, f"筛选失败: {str(e)}"

    def remove_filter(self, column):
        """移除某个筛选条件"""
        if column in self.filters:
            del self.filters[column]
            # 重新应用剩余筛选条件
            if self.filters:
                # 暂时清空，然后重新应用
                temp_filters = self.filters.copy()
                self.filters = {}
                for col, cond in temp_filters.items():
                    self.apply_filter(col, cond)
            else:
                self.display_data = self.raw_data.copy()
            return True, f"已移除 {column} 的筛选条件"
        return False, f"{column} 没有筛选条件"

    def clear_all_filters(self):
        """清除所有筛选条件"""
        self.filters = {}
        if self.raw_data is not None:
            self.display_data = self.raw_data.copy()
        return True, "已清除所有筛选条件"

    # ==================== 数据排序功能 ====================

    def sort_data(self, column, ascending=True):
        """对数据进行排序"""
        if self.display_data is None:
            return False, "请先加载数据"

        if column not in self.display_data.columns:
            return False, f"列 {column} 不存在"

        try:
            self.display_data = self.display_data.sort_values(
                by=column,
                ascending=ascending,
                na_position='last'  # 空值放在最后
            )
            self.sort_column = column
            self.sort_ascending = ascending
            return True, f"已按 {column} {'升序' if ascending else '降序'} 排序"

        except Exception as e:
            return False, f"排序失败: {str(e)}"

    # ==================== 数据搜索功能 ====================

    def search_data(self, keyword):
        """搜索包含关键词的数据"""
        if self.display_data is None:
            return None, "请先加载数据"

        try:
            # 在所有列中搜索关键词
            mask = pd.Series(False, index=self.display_data.index)

            for col in self.display_data.columns:
                # 将列转换为字符串进行搜索
                col_mask = self.display_data[col].astype(str).str.contains(
                    keyword, case=False, na=False
                )
                mask = mask | col_mask

            search_results = self.display_data[mask]
            return search_results, f"找到 {len(search_results)} 条匹配记录"

        except Exception as e:
            return None, f"搜索失败: {str(e)}"

    # ==================== 数据编辑功能 ====================

    def add_record(self, record_dict):
        """添加新记录"""
        if self.display_data is None:
            return False, "请先加载数据"

        try:
            # 创建新记录DataFrame
            new_record = pd.DataFrame([record_dict])

            # 确保列匹配
            for col in self.display_data.columns:
                if col not in new_record.columns:
                    new_record[col] = None

            # 重新排列列顺序
            new_record = new_record[self.display_data.columns]

            # 添加到数据
            self.display_data = pd.concat([self.display_data, new_record], ignore_index=True)
            return True, "记录添加成功"

        except Exception as e:
            return False, f"添加记录失败: {str(e)}"

    def delete_records(self, indices):
        """删除指定索引的记录"""
        if self.display_data is None:
            return False, "请先加载数据"

        try:
            # 保留不在删除列表中的记录
            self.display_data = self.display_data.drop(indices).reset_index(drop=True)
            return True, f"已删除 {len(indices)} 条记录"

        except Exception as e:
            return False, f"删除记录失败: {str(e)}"

    def update_record(self, index, column, value):
        """更新单个记录的值"""
        if self.display_data is None:
            return False, "请先加载数据"

        if index >= len(self.display_data):
            return False, "索引超出范围"

        if column not in self.display_data.columns:
            return False, f"列 {column} 不存在"

        try:
            # 更新值
            self.display_data.at[index, column] = value
            return True, "记录更新成功"

        except Exception as e:
            return False, f"更新记录失败: {str(e)}"

    # ==================== 数据导出功能 ====================

    def export_to_excel(self, filepath):
        """导出数据到Excel"""
        if self.display_data is None:
            return False, "没有可导出的数据"

        try:
            self.display_data.to_excel(filepath, index=False)
            return True, f"数据已导出到: {filepath}"

        except Exception as e:
            return False, f"导出失败: {str(e)}"

    # ==================== 示例数据生成 ====================

    def generate_sample_data(self, n=100):
        """生成示例数据（用于测试）"""
        np.random.seed(42)

        # 生成时间数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, periods=n)

        # 生成交通事故数据
        sample_data = {
            '事故ID': range(1, n + 1),
            '事故时间': np.random.choice(date_range, n),
            '所在区域': np.random.choice(['朝阳区', '海淀区', '东城区', '西城区', '丰台区'], n),
            '道路名称': np.random.choice(['长安街', '三环路', '四环路', '中关村大街', '王府井大街'], n),
            '事故类型': np.random.choice(['追尾', '侧碰', '刮擦', '单车事故', '多车连环'], n),
            '天气情况': np.random.choice(['晴天', '雨天', '阴天', '雾天', '雪天'], n),
            '道路类型': np.random.choice(['高速公路', '城市主干道', '城市次干道', '支路'], n),
            '照明条件': np.random.choice(['白天', '夜间有照明', '夜间无照明'], n),
            '受伤人数': np.random.randint(0, 3, n),
            '死亡人数': np.random.randint(0, 2, n),
            '温度(℃)': np.random.uniform(-5, 35, n).round(1),
            '湿度(%)': np.random.uniform(30, 95, n).round(0),
            '能见度(km)': np.random.uniform(0.5, 20, n).round(1),
            '风速(m/s)': np.random.uniform(0, 15, n).round(1),
            '事故等级': np.random.choice(['轻微', '一般', '严重'], n, p=[0.6, 0.3, 0.1])
        }

        self.raw_data = pd.DataFrame(sample_data)
        self.display_data = self.raw_data.copy()
        self.current_file = None

        return True, f"已生成 {n} 条示例数据"


# ==================== 测试函数 ====================

def test_data_manager():
    """测试数据管理器功能"""
    print("=== 测试 TrafficDataManager ===")

    # 1. 创建管理器
    manager = TrafficDataManager()
    print("1. 创建数据管理器 ✓")

    # 2. 生成示例数据
    success, message = manager.generate_sample_data(50)
    print(f"2. {message} ✓")

    # 3. 获取基本信息
    print(f"3. 总记录数: {len(manager.display_data)}")
    print(f"   总列数: {len(manager.get_column_names())}")
    print(f"   列名: {manager.get_column_names()}")

    # 4. 数据预览
    preview = manager.get_data_preview(3)
    print(f"4. 数据预览:\n{preview}")

    # 5. 统计信息
    stats = manager.get_basic_stats()
    print(f"5. 统计信息 - 总记录: {stats['total_records']}")

    # 6. 筛选测试
    success, message = manager.apply_filter('所在区域', ['朝阳区', '海淀区'])
    print(f"6. {message} ✓")

    # 7. 排序测试
    success, message = manager.sort_data('事故时间', ascending=False)
    print(f"7. {message} ✓")

    # 8. 搜索测试
    results, message = manager.search_data('追尾')
    print(f"8. {message} ✓")

    print("=== 所有测试完成 ===")


if __name__ == "__main__":
    test_data_manager()