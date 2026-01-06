# data_manager.py - 数据管理核心
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TrafficDataManager:
    """交通事故数据管理器"""

    def __init__(self):
        """初始化数据管理器"""
        self.raw_data = None  # 原始数据（只读）
        self.display_data = None  # 当前显示的数据（筛选/排序后）
        self.current_file = None  # 当前文件路径
        self.filters = {}  # 当前筛选条件
        self.sort_column = None  # 当前排序列
        self.sort_ascending = True  # 排序方向

    # 在 TrafficDataManager 类的 __init__ 方法后添加
    EXPECTED_FIELDS = {
        'required': ['事故时间', '所在区域', '事故类型', '受伤人数', '死亡人数'],
        'optional': ['温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)', '事故等级', '道路名称', '天气情况', '道路类型',
                     '照明条件']
    }

    def validate_csv_structure(self, df):
        """验证CSV文件的结构是否符合预期

        Args:
            df: pandas DataFrame

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if df is None or df.empty:
            return False, "文件为空或无法读取"

        # 获取实际列名
        actual_columns = set(df.columns)

        # 检查必需字段
        missing_required = []
        for field in self.EXPECTED_FIELDS['required']:
            if field not in actual_columns:
                missing_required.append(field)

        # 检查是否有足够的字段（至少有部分必需字段）
        if len(missing_required) == len(self.EXPECTED_FIELDS['required']):
            # 尝试匹配相似的列名（大小写不敏感）
            column_lower_map = {col.lower().replace(' ', '').replace('(', '').replace(')', ''): col
                                for col in df.columns}

            # 相似性匹配
            field_mapping = {}
            for expected in self.EXPECTED_FIELDS['required']:
                expected_simple = expected.lower().replace(' ', '').replace('(', '').replace(')', '')
                if expected_simple in column_lower_map:
                    field_mapping[expected] = column_lower_map[expected_simple]

            if field_mapping:
                return True, f"检测到相似字段: {field_mapping}"
            else:
                return False, "文件格式不匹配：未找到必需字段"

        # 部分必需字段缺失
        if missing_required:
            error_msg = "文件格式不匹配：缺少以下必需字段:\n"
            error_msg += "\n".join(f"  • {field}" for field in missing_required)
            error_msg += f"\n\n当前文件包含的字段:\n"
            error_msg += "\n".join(f"  • {col}" for col in sorted(df.columns))
            return False, error_msg

        # 验证数据类型
        data_issues = []

        # 检查数值字段
        numerical_fields = ['受伤人数', '死亡人数', '温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)']
        for field in numerical_fields:
            if field in df.columns:
                try:
                    # 尝试转换为数值
                    pd.to_numeric(df[field], errors='raise')
                except:
                    data_issues.append(f"字段 '{field}' 包含非数值数据")

        # 检查时间字段
        time_fields = ['事故时间']
        for field in time_fields:
            if field in df.columns:
                try:
                    pd.to_datetime(df[field], errors='raise')
                except:
                    data_issues.append(f"字段 '{field}' 不是有效的时间格式")

        if data_issues:
            warning_msg = "数据格式警告:\n"
            warning_msg += "\n".join(f"  • {issue}" for issue in data_issues)
            warning_msg += "\n\n是否继续导入？"
            return True, warning_msg  # 警告但允许继续

        return True, "文件格式验证通过"

    # ==================== 基础数据操作 ====================

    def load_csv(self, filepath):
        """加载CSV文件"""
        try:
            # 1. 读取CSV文件
            self.raw_data = pd.read_csv(filepath, encoding='utf-8')

            # 2. 验证文件格式
            is_valid, validation_msg = self.validate_csv_structure(self.raw_data)

            if not is_valid:
                # 格式严重不匹配，不允许导入
                self.raw_data = None
                return False, validation_msg

            # 3. 如果是警告信息，需要用户确认
            if "警告" in validation_msg:
                # 这里需要在UI层显示确认对话框
                # 暂时记录为需要用户确认
                self.validation_warning = validation_msg
            else:
                self.validation_warning = None

            # 4. 自动识别和转换数据类型
            self.auto_detect_types()

            # 5. 设置显示数据（初始为原始数据副本）
            self.display_data = self.raw_data.copy()
            self.current_file = filepath

            # 6. 重置筛选和排序状态
            self.filters = {}
            self.sort_column = None
            self.sort_ascending = True

            # 7. 如果有警告信息，包含在返回消息中
            success_msg = f"成功加载 {len(self.raw_data)} 条记录，{len(self.raw_data.columns)} 个字段"
            if self.validation_warning:
                success_msg += f"\n\n警告：{self.validation_warning}"

            return True, success_msg

        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                self.raw_data = pd.read_csv(filepath, encoding='gbk')

                # 验证文件格式
                is_valid, validation_msg = self.validate_csv_structure(self.raw_data)

                if not is_valid:
                    self.raw_data = None
                    return False, validation_msg

                # 继续处理...
                self.auto_detect_types()
                self.display_data = self.raw_data.copy()
                self.current_file = filepath
                self.filters = {}
                self.sort_column = None
                self.sort_ascending = True

                return True, f"成功加载 {len(self.raw_data)} 条记录（使用GBK编码），{len(self.raw_data.columns)} 个字段"

            except Exception as e2:
                return False, f"加载失败: 无法识别的文件编码或格式\n错误详情: {e2}"

        except Exception as e:
            return False, f"加载失败: {str(e)}"

    # data_manager.py - 在 TrafficDataManager 类中添加

    def load_excel(self, filepath):
        """加载Excel文件"""
        try:
            # 1. 尝试不同的引擎读取Excel
            try:
                self.raw_data = pd.read_excel(filepath, engine='openpyxl')
            except ImportError:
                try:
                    self.raw_data = pd.read_excel(filepath, engine='xlrd')
                except ImportError:
                    return False, "需要安装openpyxl或xlrd库，请运行: pip install openpyxl xlrd"
            except Exception as e:
                # 尝试其他引擎
                try:
                    self.raw_data = pd.read_excel(filepath, engine='odf')
                except:
                    try:
                        # 尝试自动检测引擎
                        self.raw_data = pd.read_excel(filepath)
                    except Exception as e2:
                        return False, f"读取Excel失败: {str(e2)}"

            # 2. 验证文件格式
            is_valid, validation_msg = self.validate_csv_structure(self.raw_data)

            if not is_valid:
                # 格式严重不匹配，不允许导入
                self.raw_data = None
                return False, validation_msg

            # 3. 如果是警告信息，需要用户确认
            if "警告" in validation_msg:
                self.validation_warning = validation_msg
            else:
                self.validation_warning = None

            # 4. 自动识别和转换数据类型
            self.auto_detect_types()

            # 5. 设置显示数据
            self.display_data = self.raw_data.copy()
            self.current_file = filepath

            # 6. 重置筛选和排序状态
            self.filters = {}
            self.sort_column = None
            self.sort_ascending = True

            # 7. 返回成功消息
            success_msg = f"成功加载Excel文件：{len(self.raw_data)} 条记录，{len(self.raw_data.columns)} 个字段"
            if self.validation_warning:
                success_msg += f"\n\n警告：{self.validation_warning}"

            return True, success_msg

        except Exception as e:
            return False, f"加载Excel失败: {str(e)}"

    def auto_detect_types(self):
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

    # 在 TrafficDataManager 类中添加：
    def export_to_excel(self, filepath):
        """导出数据到Excel文件"""
        if self.display_data is None:
            return False, "没有可导出的数据"

        try:
            # 使用ExcelWriter支持多个sheet
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 导出主要数据
                self.display_data.to_excel(
                    writer,
                    index=False,
                    sheet_name='事故数据'
                )

                # 可选：导出统计信息
                stats = self.get_basic_stats()
                if stats:
                    stats_df = pd.DataFrame(stats['column_details'])
                    stats_df.to_excel(
                        writer,
                        index=False,
                        sheet_name='数据统计'
                    )

            return True, f"数据已成功导出到: {filepath}"
        except ImportError:
            return False, "导出Excel需要安装openpyxl库，请运行: pip install openpyxl"
        except Exception as e:
            return False, f"导出失败: {str(e)}"

    # ==================== 示例数据生成 ====================
    # TODO: 让生成的数据变得更合理

    def generate_sample_data(self, n=100):
        """生成示例数据（用于测试）"""
        np.random.seed(42)

        # 生成时间数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, periods=n)

        # 基础变量
        regions = ['朝阳区', '海淀区', '东城区', '西城区', '丰台区']
        road_names = ['长安街', '三环路', '四环路', '中关村大街', '王府井大街']
        accident_types = ['追尾', '侧碰', '刮擦', '单车事故', '多车连环']
        weather_conditions = ['晴天', '雨天', '阴天', '雾天', '雪天']
        road_types = ['高速公路', '城市主干道', '城市次干道', '支路']
        lighting_conditions = ['白天', '夜间有照明', '夜间无照明']

        # 初始化数据容器
        data_dict = {
            '事故ID': range(1, n + 1),
            '事故时间': np.random.choice(date_range, n),
            '所在区域': np.random.choice(regions, n),
            '道路名称': np.random.choice(road_names, n),
            '事故类型': np.random.choice(accident_types, n),
            '天气情况': np.random.choice(weather_conditions, n),
            '道路类型': np.random.choice(road_types, n),
            '照明条件': np.random.choice(lighting_conditions, n),
        }

        # 生成气象数据 - 加强相关性
        temperature = np.random.uniform(-5, 35, n).round(1)

        # 温度与湿度的负相关（但加入噪声）
        humidity_base = 75 - 0.7 * temperature + np.random.normal(0, 8, n)
        humidity = np.clip(humidity_base.round(0), 30, 95)

        # 能见度与天气的强相关
        visibility = np.zeros(n)
        for i in range(n):
            base_vis = float(np.random.uniform(1, 20))  # 明确转换为float
            if data_dict['天气情况'][i] == '雾天':
                base_vis *= float(np.random.uniform(0.1, 0.4))  # 雾天能见度极低
            elif data_dict['天气情况'][i] == '雪天':
                base_vis *= float(np.random.uniform(0.2, 0.6))
            elif data_dict['天气情况'][i] == '雨天':
                base_vis *= float(np.random.uniform(0.4, 0.8))
            elif data_dict['天气情况'][i] == '阴天':
                base_vis *= float(np.random.uniform(0.6, 0.9))
            visibility[i] = np.clip(round(base_vis, 1), 0.1, 20)  # 使用Python内置的round函数

        # 风速与天气相关
        wind_speed = np.zeros(n)
        for i in range(n):
            base_wind = float(np.random.uniform(0, 12))
            if data_dict['天气情况'][i] in ['雨天', '雪天']:
                base_wind *= float(np.random.uniform(1.3, 2.0))
            elif data_dict['天气情况'][i] == '雾天':
                base_wind *= float(np.random.uniform(0.8, 1.2))  # 雾天风速通常不大
            wind_speed[i] = np.clip(round(base_wind, 1), 0, 20)

        # 生成伤亡数据 - 建立更清晰的关联
        injured = np.zeros(n, dtype=int)
        deaths = np.zeros(n, dtype=int)

        for i in range(n):
            # 基础风险分数 - 更强调关键因素
            risk_score = 0

            # 1. 天气影响（权重最高）
            if data_dict['天气情况'][i] in ['雾天', '雪天']:
                risk_score += float(np.random.uniform(0.4, 0.7))
            elif data_dict['天气情况'][i] == '雨天':
                risk_score += float(np.random.uniform(0.2, 0.4))

            # 2. 照明条件（权重高）
            if data_dict['照明条件'][i] == '夜间无照明':
                risk_score += float(np.random.uniform(0.3, 0.6))

            # 3. 道路类型（权重中）
            if data_dict['道路类型'][i] == '高速公路':
                risk_score += float(np.random.uniform(0.3, 0.5))
            elif data_dict['道路类型'][i] == '城市主干道':
                risk_score += float(np.random.uniform(0.1, 0.3))

            # 4. 能见度影响（非线性，但关系更清晰）
            if visibility[i] < 0.5:
                risk_score += float(np.random.uniform(0.5, 0.7))
            elif visibility[i] < 2:
                risk_score += float(np.random.uniform(0.3, 0.5))
            elif visibility[i] < 5:
                risk_score += float(np.random.uniform(0.1, 0.3))

            # 5. 风速影响（中等）
            if wind_speed[i] > 10:
                risk_score += float(np.random.uniform(0.3, 0.5))
            elif wind_speed[i] > 5:
                risk_score += float(np.random.uniform(0.1, 0.2))

            # 6. 事故类型影响
            if data_dict['事故类型'][i] in ['多车连环', '侧碰']:
                risk_score += float(np.random.uniform(0.2, 0.4))
            elif data_dict['事故类型'][i] == '追尾':
                risk_score += float(np.random.uniform(0.1, 0.3))

            # 添加随机噪声（但减少噪声幅度）
            risk_score += float(np.random.normal(0, 0.1))
            risk_score = max(0, min(1, risk_score))  # 限制在0-1之间

            # 根据风险分数生成伤亡数据（关系更明确）
            if risk_score > 0.7:  # 高风险
                # 死亡概率高
                death_prob = np.random.random()
                if death_prob > 0.6:
                    deaths[i] = np.random.randint(1, 4)
                    injured[i] = np.random.randint(deaths[i] + 1, deaths[i] + 5)
                else:
                    deaths[i] = np.random.randint(0, 2)
                    injured[i] = np.random.randint(3, 7)

            elif risk_score > 0.4:  # 中风险
                # 可能有死亡，但概率较低
                death_prob = np.random.random()
                if death_prob > 0.8:
                    deaths[i] = 1
                    injured[i] = np.random.randint(2, 5)
                else:
                    deaths[i] = 0
                    injured[i] = np.random.randint(1, 4)

            else:  # 低风险
                # 基本无死亡
                deaths[i] = 0
                injured_prob = np.random.random()
                if injured_prob > 0.3:
                    injured[i] = np.random.randint(0, 2)
                else:
                    injured[i] = 0

            # 确保逻辑合理性
            if deaths[i] > 0 and injured[i] < deaths[i]:
                injured[i] = deaths[i] + np.random.randint(0, 3)

        # 根据伤亡情况生成事故等级（关系更直接）
        accident_level = []
        for i in range(n):
            total_severity = deaths[i] * 3 + injured[i]  # 死亡权重3倍

            if deaths[i] >= 2 or total_severity >= 8:
                level = '严重'
            elif deaths[i] == 1 or total_severity >= 4:
                level = '一般'
            else:
                level = '轻微'

            # 10%的概率等级不完全匹配（减少异常比例）
            if np.random.random() < 0.1:
                if level == '严重' and np.random.random() < 0.5:
                    level = '一般'
                elif level == '一般' and np.random.random() < 0.5:
                    level = '轻微' if np.random.random() < 0.5 else '严重'
                elif level == '轻微' and np.random.random() < 0.5:
                    level = '一般'

            accident_level.append(level)

        # 组合所有数据
        sample_data = {
            **data_dict,
            '受伤人数': injured,
            '死亡人数': deaths,
            '温度(℃)': temperature,
            '湿度(%)': humidity,
            '能见度(km)': visibility,
            '风速(m/s)': wind_speed,
            '事故等级': accident_level
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