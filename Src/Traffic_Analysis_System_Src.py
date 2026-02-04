# 城市交通事故分析与预警系统 - 单文件整合版

"""
系统结构说明：
1. TrafficDataManager - 数据管理核心
2. TrafficPredictor - 机器学习预测模型
3. TrafficVisualizer - 数据可视化模块
4. UI Components - 界面组件
5. IntegratedMainWindow - 主窗口整合
原始项目地址：
https://github.com/PaintHelloWorld/Traffic_Analysis_System

"""

# ==================== 导入所有需要的库 ====================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# ==================== 1. 数据管理核心 ====================

class TrafficDataManager:
    """交通事故数据管理器"""

    def __init__(self):
        """初始化数据管理器"""
        self.raw_data = None
        self.display_data = None
        self.current_file = None
        self.filters = {}
        self.sort_column = None
        self.sort_ascending = True

    EXPECTED_FIELDS = {
        'required': ['事故时间', '所在区域', '事故类型', '受伤人数', '死亡人数'],
        'optional': ['温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)', '事故等级', '道路名称', '天气情况', '道路类型',
                     '照明条件']
    }

    def validate_csv_structure(self, df):
        """验证CSV文件的结构是否符合预期"""
        if df is None or df.empty:
            return False, "文件为空或无法读取"

        actual_columns = set(df.columns)
        missing_required = []

        for field in self.EXPECTED_FIELDS['required']:
            if field not in actual_columns:
                missing_required.append(field)

        if len(missing_required) == len(self.EXPECTED_FIELDS['required']):
            column_lower_map = {col.lower().replace(' ', '').replace('(', '').replace(')', ''): col for col in
                                df.columns}
            field_mapping = {}

            for expected in self.EXPECTED_FIELDS['required']:
                expected_simple = expected.lower().replace(' ', '').replace('(', '').replace(')', '')
                if expected_simple in column_lower_map:
                    field_mapping[expected] = column_lower_map[expected_simple]

            if field_mapping:
                return True, f"检测到相似字段: {field_mapping}"
            else:
                return False, "文件格式不匹配：未找到必需字段"

        if missing_required:
            error_msg = "文件格式不匹配：缺少以下必需字段:\n"
            error_msg += "\n".join(f"  • {field}" for field in missing_required)
            error_msg += f"\n\n当前文件包含的字段:\n"
            error_msg += "\n".join(f"  • {col}" for col in sorted(df.columns))
            return False, error_msg

        data_issues = []
        numerical_fields = ['受伤人数', '死亡人数', '温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)']

        for field in numerical_fields:
            if field in df.columns:
                try:
                    pd.to_numeric(df[field], errors='raise')
                except:
                    data_issues.append(f"字段 '{field}' 包含非数值数据")

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
            return True, warning_msg

        return True, "文件格式验证通过"

    def load_csv(self, filepath):
        """加载CSV文件"""
        try:
            self.raw_data = pd.read_csv(filepath, encoding='utf-8')
            is_valid, validation_msg = self.validate_csv_structure(self.raw_data)

            if not is_valid:
                self.raw_data = None
                return False, validation_msg

            if "警告" in validation_msg:
                self.validation_warning = validation_msg
            else:
                self.validation_warning = None

            self.auto_detect_types()
            self.display_data = self.raw_data.copy()
            self.current_file = filepath
            self.filters = {}
            self.sort_column = None
            self.sort_ascending = True

            success_msg = f"成功加载 {len(self.raw_data)} 条记录，{len(self.raw_data.columns)} 个字段"
            if self.validation_warning:
                success_msg += f"\n\n警告：{self.validation_warning}"

            return True, success_msg

        except UnicodeDecodeError:
            try:
                self.raw_data = pd.read_csv(filepath, encoding='gbk')
                is_valid, validation_msg = self.validate_csv_structure(self.raw_data)

                if not is_valid:
                    self.raw_data = None
                    return False, validation_msg

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

    def load_excel(self, filepath):
        """加载Excel文件"""
        try:
            try:
                self.raw_data = pd.read_excel(filepath, engine='openpyxl')
            except ImportError:
                try:
                    self.raw_data = pd.read_excel(filepath, engine='xlrd')
                except ImportError:
                    return False, "需要安装openpyxl或xlrd库，请运行: pip install openpyxl或pip install xlrd"

            is_valid, validation_msg = self.validate_csv_structure(self.raw_data)

            if not is_valid:
                self.raw_data = None
                return False, validation_msg

            if "警告" in validation_msg:
                self.validation_warning = validation_msg
            else:
                self.validation_warning = None

            self.auto_detect_types()
            self.display_data = self.raw_data.copy()
            self.current_file = filepath
            self.filters = {}
            self.sort_column = None
            self.sort_ascending = True

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
            if any(keyword in col.lower() for keyword in ['time', 'date', '日期', '时间']):
                try:
                    self.raw_data[col] = pd.to_datetime(self.raw_data[col], errors='coerce')
                except:
                    pass

            try:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='ignore')
            except:
                pass

    def save_to_csv(self, filepath=None):
        """保存数据到CSV文件"""
        if self.display_data is None:
            return False, "没有可保存的数据"

        try:
            if filepath is None:
                if self.current_file:
                    filepath = self.current_file
                else:
                    return False, "请指定保存路径"

            self.display_data.to_csv(filepath, index=False, encoding='utf-8')
            self.current_file = filepath
            return True, f"数据已保存到: {filepath}"
        except Exception as e:
            return False, f"保存失败: {str(e)}"

    def get_column_names(self):
        """获取所有列名"""
        if self.display_data is not None:
            return list(self.display_data.columns)
        return []


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

            if pd.api.types.is_numeric_dtype(self.display_data[col]):
                col_stats.update({
                    'min': float(self.display_data[col].min()),
                    'max': float(self.display_data[col].max()),
                    'mean': float(self.display_data[col].mean()),
                    'std': float(self.display_data[col].std())
                })

            stats['column_details'].append(col_stats)

        return stats

    def apply_filter(self, column, condition):
        """应用筛选条件"""
        if self.raw_data is None:
            return False, "请先加载数据"

        try:
            self.filters[column] = condition
            filtered_data = self.raw_data.copy()

            for filter_col, filter_cond in self.filters.items():
                if filter_col not in filtered_data.columns:
                    continue

                if isinstance(filter_cond, tuple) and len(filter_cond) == 2:
                    min_val, max_val = filter_cond
                    filtered_data = filtered_data[
                        (filtered_data[filter_col] >= min_val) &
                        (filtered_data[filter_col] <= max_val)
                        ]
                elif isinstance(filter_cond, list):
                    filtered_data = filtered_data[filtered_data[filter_col].isin(filter_cond)]
                else:
                    filtered_data = filtered_data[
                        filtered_data[filter_col].astype(str).str.contains(str(filter_cond), na=False)
                    ]

            self.display_data = filtered_data
            return True, f"筛选后剩余 {len(filtered_data)} 条记录"
        except Exception as e:
            return False, f"筛选失败: {str(e)}"

    def clear_all_filters(self):
        """清除所有筛选条件"""
        self.filters = {}
        if self.raw_data is not None:
            self.display_data = self.raw_data.copy()
        return True, "已清除所有筛选条件"

    def sort_data(self, column, ascending=True):
        """对数据进行排序"""
        if self.display_data is None:
            return False, "请先加载数据"

        if column not in self.display_data.columns:
            return False, f"列 {column} 不存在"

        try:
            self.display_data = self.display_data.sort_values(
                by=column, ascending=ascending, na_position='last'
            )
            self.sort_column = column
            self.sort_ascending = ascending
            return True, f"已按 {column} {'升序' if ascending else '降序'} 排序"
        except Exception as e:
            return False, f"排序失败: {e}"

    def search_data(self, keyword):
        """搜索包含关键词的数据"""
        if self.display_data is None:
            return None, "请先加载数据"

        try:
            mask = pd.Series(False, index=self.display_data.index)
            for col in self.display_data.columns:
                col_mask = self.display_data[col].astype(str).str.contains(keyword, case=False, na=False)
                mask = mask | col_mask

            search_results = self.display_data[mask]
            return search_results, f"找到 {len(search_results)} 条匹配记录"
        except Exception as e:
            return None, f"搜索失败: {str(e)}"

    def add_record(self, record_dict):
        """添加新记录"""
        if self.display_data is None:
            return False, "请先加载数据"

        try:
            new_record = pd.DataFrame([record_dict])
            for col in self.display_data.columns:
                if col not in new_record.columns:
                    new_record[col] = None

            new_record = new_record[self.display_data.columns]
            self.display_data = pd.concat([self.display_data, new_record], ignore_index=True)
            return True, "记录添加成功"
        except Exception as e:
            return False, f"添加记录失败: {str(e)}"

    def delete_records(self, indices):
        """删除指定索引的记录"""
        if self.display_data is None:
            return False, "请先加载数据"

        try:
            self.display_data = self.display_data.drop(indices).reset_index(drop=True)
            return True, f"已删除 {len(indices)} 条记录"
        except Exception as e:
            return False, f"删除记录失败: {str(e)}"

    def export_to_excel(self, filepath):
        """导出数据到Excel文件"""
        if self.display_data is None:
            return False, "没有可导出的数据"

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.display_data.to_excel(writer, index=False, sheet_name='事故数据')

                stats = self.get_basic_stats()
                if stats:
                    stats_df = pd.DataFrame(stats['column_details'])
                    stats_df.to_excel(writer, index=False, sheet_name='数据统计')

            return True, f"数据已成功导出到: {filepath}"
        except ImportError:
            return False, "导出Excel需要安装openpyxl库，请运行: pip install openpyxl"
        except Exception as e:
            return False, f"导出失败: {str(e)}"

    def generate_sample_data(self, n=100):
        """生成示例数据"""
        np.random.seed(42)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, periods=n)

        regions = ['朝阳区', '海淀区', '东城区', '西城区', '丰台区']
        road_names = ['长安街', '三环路', '四环路', '中关村大街', '王府井大街']
        accident_types = ['追尾', '侧碰', '刮擦', '单车事故', '多车连环']
        weather_conditions = ['晴天', '雨天', '阴天', '雾天', '雪天']
        road_types = ['高速公路', '城市主干道', '城市次干道', '支路']
        lighting_conditions = ['白天', '夜间有照明', '夜间无照明']

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

        temperature = np.random.uniform(-5, 35, n).round(1)
        humidity_base = 75 - 0.7 * temperature + np.random.normal(0, 8, n)
        humidity = np.clip(humidity_base.round(0), 30, 95)

        visibility = np.zeros(n)
        for i in range(n):
            base_vis = float(np.random.uniform(1, 20))
            if data_dict['天气情况'][i] == '雾天':
                base_vis *= float(np.random.uniform(0.1, 0.4))
            elif data_dict['天气情况'][i] == '雪天':
                base_vis *= float(np.random.uniform(0.2, 0.6))
            elif data_dict['天气情况'][i] == '雨天':
                base_vis *= float(np.random.uniform(0.4, 0.8))
            elif data_dict['天气情况'][i] == '阴天':
                base_vis *= float(np.random.uniform(0.6, 0.9))
            visibility[i] = np.clip(round(base_vis, 1), 0.1, 20)

        wind_speed = np.zeros(n)
        for i in range(n):
            base_wind = float(np.random.uniform(0, 12))
            if data_dict['天气情况'][i] in ['雨天', '雪天']:
                base_wind *= float(np.random.uniform(1.3, 2.0))
            elif data_dict['天气情况'][i] == '雾天':
                base_wind *= float(np.random.uniform(0.8, 1.2))
            wind_speed[i] = np.clip(round(base_wind, 1), 0, 20)

        injured = np.zeros(n, dtype=int)
        deaths = np.zeros(n, dtype=int)

        for i in range(n):
            risk_score = 0

            if data_dict['天气情况'][i] in ['雾天', '雪天']:
                risk_score += float(np.random.uniform(0.4, 0.7))
            elif data_dict['天气情况'][i] == '雨天':
                risk_score += float(np.random.uniform(0.2, 0.4))

            if data_dict['照明条件'][i] == '夜间无照明':
                risk_score += float(np.random.uniform(0.3, 0.6))

            if data_dict['道路类型'][i] == '高速公路':
                risk_score += float(np.random.uniform(0.3, 0.5))
            elif data_dict['道路类型'][i] == '城市主干道':
                risk_score += float(np.random.uniform(0.1, 0.3))

            if visibility[i] < 0.5:
                risk_score += float(np.random.uniform(0.5, 0.7))
            elif visibility[i] < 2:
                risk_score += float(np.random.uniform(0.3, 0.5))
            elif visibility[i] < 5:
                risk_score += float(np.random.uniform(0.1, 0.3))

            if wind_speed[i] > 10:
                risk_score += float(np.random.uniform(0.3, 0.5))
            elif wind_speed[i] > 5:
                risk_score += float(np.random.uniform(0.1, 0.2))

            if data_dict['事故类型'][i] in ['多车连环', '侧碰']:
                risk_score += float(np.random.uniform(0.2, 0.4))
            elif data_dict['事故类型'][i] == '追尾':
                risk_score += float(np.random.uniform(0.1, 0.3))

            risk_score += float(np.random.normal(0, 0.1))
            risk_score = max(0, min(1, risk_score))

            if risk_score > 0.7:
                death_prob = np.random.random()
                if death_prob > 0.6:
                    deaths[i] = np.random.randint(1, 4)
                    injured[i] = np.random.randint(deaths[i] + 1, deaths[i] + 5)
                else:
                    deaths[i] = np.random.randint(0, 2)
                    injured[i] = np.random.randint(3, 7)
            elif risk_score > 0.4:
                death_prob = np.random.random()
                if death_prob > 0.8:
                    deaths[i] = 1
                    injured[i] = np.random.randint(2, 5)
                else:
                    deaths[i] = 0
                    injured[i] = np.random.randint(1, 4)
            else:
                deaths[i] = 0
                injured_prob = np.random.random()
                if injured_prob > 0.3:
                    injured[i] = np.random.randint(0, 2)
                else:
                    injured[i] = 0

            if deaths[i] > 0 and injured[i] < deaths[i]:
                injured[i] = deaths[i] + np.random.randint(0, 3)

        accident_level = []
        for i in range(n):
            total_severity = deaths[i] * 3 + injured[i]

            if deaths[i] >= 2 or total_severity >= 8:
                level = '严重'
            elif deaths[i] == 1 or total_severity >= 4:
                level = '一般'
            else:
                level = '轻微'

            if np.random.random() < 0.1:
                if level == '严重' and np.random.random() < 0.5:
                    level = '一般'
                elif level == '一般' and np.random.random() < 0.5:
                    level = '轻微' if np.random.random() < 0.5 else '严重'
                elif level == '轻微' and np.random.random() < 0.5:
                    level = '一般'

            accident_level.append(level)

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


# ==================== 2. 机器学习预测模型 ====================

class TrafficPredictor:
    """交通事故预测器"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'risk_level'
        self.is_trained = False

    def prepare_features(self, data):
        """从原始数据提取特征"""
        df = data.copy()

        time_cols = [col for col in df.columns if any(kw in col.lower()
                                                      for kw in ['time', 'date', '时间', '日期'])]

        if time_cols:
            time_col = time_cols[0]
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df['hour'] = df[time_col].dt.hour
                df['day_of_week'] = df[time_col].dt.dayofweek
                df['month'] = df[time_col].dt.month
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18, 19]).astype(int)
            except:
                pass

        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                categorical_cols.append(col)

        for col in categorical_cols[:5]:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            except:
                pass

        feature_candidates = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            '受伤人数', '死亡人数', '温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)'
        ]

        available_features = []
        for feat in feature_candidates:
            if feat in df.columns:
                available_features.append(feat)

        for col in categorical_cols[:3]:
            if col not in available_features and col in df.columns:
                available_features.append(col)

        if len(available_features) < 3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols[:8]

        return df[available_features]

    def create_target_variable(self, data):
        """创建目标变量（风险等级）"""
        severity_cols = [col for col in data.columns if any(kw in col.lower()
                                                            for kw in ['severity', 'level', '等级', '严重'])]

        if severity_cols:
            severity_col = severity_cols[0]
            if data[severity_col].dtype == 'object':
                level_mapping = {
                    '轻微': 0, '一般': 1, '严重': 2,
                    '低': 0, '中': 1, '高': 2,
                    'low': 0, 'medium': 1, 'high': 2
                }
                target = data[severity_col].map(level_mapping).fillna(1).astype(int)
            else:
                target = data[severity_col].astype(int)
        else:
            if '受伤人数' in data.columns and '死亡人数' in data.columns:
                target = np.zeros(len(data), dtype=int)
                target[data['死亡人数'] > 0] = 2
                target[(data['受伤人数'] >= 2) & (target == 0)] = 1
                target[(data['受伤人数'] == 1) & (target == 0)] = 0
            else:
                np.random.seed(42)
                target = np.random.choice([0, 1, 2], size=len(data), p=[0.6, 0.3, 0.1])

        return target

    def train_model(self, data):
        """训练预测模型"""
        try:
            if len(data) < 50:
                return False, "数据量不足，至少需要50条"

            features = self.prepare_features(data)
            if len(features.columns) < 2:
                return False, "特征不足，无法训练模型"

            target = self.create_target_variable(data)

            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            self.feature_columns = list(features.columns)
            self.is_trained = True

            report = classification_report(y_test, y_pred, target_names=['低风险', '中风险', '高风险'])
            confusion = confusion_matrix(y_test, y_pred)

            result = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': confusion,
                'feature_count': len(self.feature_columns),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

            return True, result
        except Exception as e:
            return False, f"模型训练失败: {str(e)}"

    def predict(self, data):
        """对新数据进行预测"""
        if not self.is_trained or self.model is None:
            return None, None, "请先训练模型"

        try:
            features = self.prepare_features(data)
            missing_cols = set(self.feature_columns) - set(features.columns)

            for col in missing_cols:
                features[col] = 0

            features = features[self.feature_columns]
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)
                return predictions, probabilities, "预测成功"
            else:
                return predictions, None, "预测成功"
        except Exception as e:
            return None, None, f"预测失败: {str(e)}"

    def save_model(self, filepath):
        """保存模型到文件"""
        try:
            if not self.is_trained:
                return False, "没有训练好的模型可保存"

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'is_trained': self.is_trained
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            return True, f"模型已保存到: {filepath}"
        except Exception as e:
            return False, f"保存模型失败: {str(e)}"

    def load_model(self, filepath):
        """从文件加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data.get('label_encoders', {})
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data.get('target_column', 'risk_level')
            self.is_trained = model_data['is_trained']

            return True, f"模型已从 {filepath} 加载"
        except Exception as e:
            return False, f"加载模型失败: {str(e)}"

    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_trained or self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_columns

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            return importance_df

        return None

    def predict_single(self, input_dict):
        """预测单个记录"""
        if not self.is_trained:
            return None, None, "请先训练模型"

        try:
            input_df = pd.DataFrame([input_dict])
            predictions, probabilities, message = self.predict(input_df)

            if predictions is not None:
                risk_level = predictions[0]
                risk_labels = ['低风险', '中风险', '高风险']
                risk_label = risk_labels[risk_level] if risk_level < 3 else f"等级{risk_level}"

                if probabilities is not None:
                    prob_dict = {
                        '低风险': float(probabilities[0][0]) if len(probabilities[0]) > 0 else 0,
                        '中风险': float(probabilities[0][1]) if len(probabilities[0]) > 1 else 0,
                        '高风险': float(probabilities[0][2]) if len(probabilities[0]) > 2 else 0
                    }
                else:
                    prob_dict = None

                return risk_label, prob_dict, "预测成功"

            return None, None, message
        except Exception as e:
            return None, None, f"单条预测失败: {str(e)}"


# ==================== 3. 数据可视化模块 ====================

class TrafficVisualizer:
    """交通事故可视化器"""

    def __init__(self, data_manager, parent_frame):
        self.data_manager = data_manager
        self.parent_frame = parent_frame
        self.current_figure = None
        self.canvas = None
        self.toolbar = None
        self.chart_type = "柱状图"

        self.setup_control_panel()
        self.setup_chart_area()

    def setup_control_panel(self):
        """创建图表控制面板"""
        control_frame = ttk.LabelFrame(self.parent_frame, text="图表设置", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(row1, text="图表类型:").pack(side=tk.LEFT, padx=5)
        self.chart_type_var = tk.StringVar(value="柱状图")
        chart_types = ["柱状图", "折线图", "饼图", "散点图", "热力图", "箱线图"]
        chart_combo = ttk.Combobox(row1, textvariable=self.chart_type_var,
                                   values=chart_types, state="readonly", width=12)
        chart_combo.pack(side=tk.LEFT, padx=5)
        chart_combo.bind("<<ComboboxSelected>>", lambda e: self.on_chart_type_changed())

        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X)

        ttk.Label(row2, text="X轴:").pack(side=tk.LEFT, padx=5)
        self.x_axis_var = tk.StringVar()
        self.x_axis_combo = ttk.Combobox(row2, textvariable=self.x_axis_var, width=15)
        self.x_axis_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Y轴:").pack(side=tk.LEFT, padx=5)
        self.y_axis_var = tk.StringVar()
        self.y_axis_combo = ttk.Combobox(row2, textvariable=self.y_axis_var, width=15)
        self.y_axis_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(row2, text="生成图表", command=self.generate_chart).pack(side=tk.LEFT, padx=10)
        ttk.Button(row2, text="导出图片", command=self.export_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="刷新数据", command=self.refresh_data).pack(side=tk.LEFT, padx=5)

        self.update_axis_options()

    def setup_chart_area(self):
        """创建图表显示区域"""
        chart_container = ttk.Frame(self.parent_frame)
        chart_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chart_frame = ttk.Frame(chart_container)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(chart_container, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def update_axis_options(self):
        """更新坐标轴选项"""
        if self.data_manager.display_data is not None:
            data = self.data_manager.display_data
            columns = list(data.columns)

            numeric_cols = []
            for col in columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_cols.append(col)

            self.x_axis_combo['values'] = columns
            self.y_axis_combo['values'] = numeric_cols

            if columns:
                time_cols = [col for col in columns if any(kw in col.lower()
                                                           for kw in ['time', 'date', '时间', '日期'])]
                if time_cols:
                    self.x_axis_var.set(time_cols[0])
                else:
                    self.x_axis_var.set(columns[0])

            if numeric_cols:
                num_col = numeric_cols[0] if len(numeric_cols) > 0 else ""
                self.y_axis_var.set(num_col)

    def on_chart_type_changed(self):
        """图表类型改变时的处理"""
        chart_type = self.chart_type_var.get()

        if chart_type == "饼图":
            self.x_axis_combo.config(state="normal")
            self.y_axis_combo.config(state="disabled")
        elif chart_type == "热力图":
            self.x_axis_combo.config(state="disabled")
            self.y_axis_combo.config(state="disabled")
        else:
            self.x_axis_combo.config(state="normal")
            self.y_axis_combo.config(state="normal")

        self.generate_chart()

    def refresh_data(self):
        """刷新数据"""
        self.update_axis_options()
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
            messagebox.showerror("图表错误", f"生成图表时出错:\n{str(e)}")

    def create_bar_chart(self):
        """创建柱状图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            return

        data = self.data_manager.display_data
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        if data[x_col].dtype == 'object' or data[x_col].nunique() < 15:
            group_data = data.groupby(x_col)[y_col].mean().sort_values(ascending=False)

            if len(group_data) > 15:
                group_data = group_data.head(15)

            x_pos = range(len(group_data))
            bars = ax.bar(x_pos, group_data.values, color='steelblue', alpha=0.8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_data.index, rotation=45, ha='right')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            sorted_data = data.sort_values(x_col)
            ax.bar(sorted_data[x_col].astype(str), sorted_data[y_col],
                   color='steelblue', alpha=0.8)
            ax.tick_params(axis='x', rotation=45)

        ax.set_title(f'{y_col} 按 {x_col} 分布', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.display_figure(fig)

    def create_line_chart(self):
        """创建折线图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            return

        data = self.data_manager.display_data.copy()

        try:
            data[x_col] = pd.to_datetime(data[x_col])
            data = data.sort_values(x_col)
            is_time_series = True
        except:
            data = data.sort_values(x_col)
            is_time_series = False

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(data[x_col], data[y_col], marker='o', linewidth=2,
                markersize=5, color='coral', alpha=0.8, label=y_col)

        ax.set_title(f'{y_col} 趋势图', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()

        if is_time_series:
            fig.autofmt_xdate()

        self.display_figure(fig)

    def create_pie_chart(self):
        """创建饼图"""
        x_col = self.x_axis_var.get()

        if not x_col:
            return

        data = self.data_manager.display_data
        value_counts = data[x_col].value_counts()

        if len(value_counts) > 10:
            top_data = value_counts.head(10)
            others = value_counts[10:].sum()
            top_data['其他'] = others
            value_counts = top_data

        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)

        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))

        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(edgecolor='white', linewidth=1)
        )

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title(f'{x_col} 分布比例', fontsize=14, fontweight='bold')
        self.display_figure(fig)

    def create_scatter_plot(self):
        """创建散点图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            return

        data = self.data_manager.display_data
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        scatter = ax.scatter(data[x_col], data[y_col],
                             c=data[y_col],
                             cmap='viridis',
                             alpha=0.7,
                             edgecolors='w',
                             linewidth=0.5,
                             s=100)

        ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.colorbar(scatter, ax=ax, label=y_col)
        self.display_figure(fig)

    def create_heatmap(self):
        """创建热力图"""
        data = self.data_manager.display_data
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return

        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)

        correlation = data[numeric_cols].corr()
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

    def create_box_plot(self):
        """创建箱线图"""
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()

        if not x_col or not y_col:
            return

        data = self.data_manager.display_data
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

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

        box = ax.boxplot(plot_data,
                         labels=labels,
                         patch_artist=True,
                         showmeans=True,
                         meanline=True)

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

    def display_figure(self, figure):
        """显示图形"""
        self.current_figure = figure
        self.canvas = FigureCanvasTkAgg(figure, self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
                messagebox.showinfo("导出成功", f"图表已成功导出到:\n{filepath}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出图表时出错:\n{str(e)}")


# ==================== 4. UI组件 ====================

class DataTable(ttk.Frame):
    """数据表格组件"""

    def __init__(self, parent, data_manager):
        super().__init__(parent)
        self.data_manager = data_manager
        self.tree = None
        self.setup_table()

    def setup_table(self):
        """设置表格框架"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=scrollbar_x.set)

    def load_data(self):
        """加载数据到表格"""
        if self.data_manager.display_data is None:
            return False, "没有数据可显示"

        try:
            for item in self.tree.get_children():
                self.tree.delete(item)

            data = self.data_manager.display_data
            columns = self.data_manager.get_column_names()

            self.tree["columns"] = columns
            for col in columns:
                self.tree.heading(col, text=col)
                max_len = max([len(str(val)) for val in data[col].head(20).astype(str)]) if len(data) > 0 else 10
                width = min(max_len * 8, 200)
                self.tree.column(col, width=width, minwidth=50)

            for idx, row in data.iterrows():
                values = [str(row[col])[:100] for col in columns]
                self.tree.insert("", tk.END, values=values, iid=str(idx))

            return True, f"显示 {len(data)} 条记录"
        except Exception as e:
            return False, f"加载数据到表格失败: {str(e)}"

    def get_selected_indices(self):
        """获取选中的行索引"""
        selected_items = self.tree.selection()
        return [int(item) for item in selected_items]

    def clear_selection(self):
        """清除选择"""
        self.tree.selection_remove(self.tree.selection())


class ControlPanel(ttk.LabelFrame):
    """控制面板"""

    def __init__(self, parent, data_manager, table, status_callback):
        super().__init__(parent, text="控制面板", padding=10)
        self.data_manager = data_manager
        self.table = table
        self.status_callback = status_callback
        self.setup_controls()

    def setup_controls(self):
        """设置控制组件"""
        file_frame = ttk.Frame(self)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="导入CSV", command=self.open_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导入Excel", command=self.open_excel).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导出CSV", command=self.save_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导出Excel", command=self.export_excel).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="示例数据", command=self.generate_sample).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self, orient='horizontal').pack(fill=tk.X, pady=10)

        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="筛选列:").pack(side=tk.LEFT, padx=2)
        self.filter_column = ttk.Combobox(filter_frame, width=15, state="readonly")
        self.filter_column.pack(side=tk.LEFT, padx=2)

        ttk.Label(filter_frame, text="条件:").pack(side=tk.LEFT, padx=2)
        self.filter_value = ttk.Entry(filter_frame, width=15)
        self.filter_value.pack(side=tk.LEFT, padx=2)

        ttk.Button(filter_frame, text="应用筛选", command=self.apply_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_frame, text="清除筛选", command=self.clear_filter).pack(side=tk.LEFT, padx=2)

        search_frame = ttk.Frame(self)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT, padx=2)
        self.search_entry = ttk.Entry(search_frame, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=2)
        self.search_entry.bind("<Return>", lambda e: self.search_data())

        ttk.Button(search_frame, text="搜索", command=self.search_data).pack(side=tk.LEFT, padx=2)

        sort_frame = ttk.Frame(self)
        sort_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(sort_frame, text="排序列:").pack(side=tk.LEFT, padx=2)
        self.sort_column = ttk.Combobox(sort_frame, width=15, state="readonly")
        self.sort_column.pack(side=tk.LEFT, padx=2)

        self.sort_ascending = tk.BooleanVar(value=True)
        ttk.Radiobutton(sort_frame, text="升序", variable=self.sort_ascending, value=True).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sort_frame, text="降序", variable=self.sort_ascending, value=False).pack(side=tk.LEFT, padx=2)

        ttk.Button(sort_frame, text="排序", command=self.sort_data).pack(side=tk.LEFT, padx=2)

        data_frame = ttk.Frame(self)
        data_frame.pack(fill=tk.X)

        ttk.Button(data_frame, text="添加记录", command=self.add_record).pack(side=tk.LEFT, padx=2)
        ttk.Button(data_frame, text="删除选中", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(data_frame, text="刷新表格", command=self.refresh_table).pack(side=tk.LEFT, padx=2)

        self.update_column_options()

    def update_column_options(self):
        """更新列选项"""
        columns = self.data_manager.get_column_names()
        self.filter_column['values'] = columns
        self.sort_column['values'] = columns

        if columns:
            self.filter_column.current(0)
            self.sort_column.current(0)

    def open_csv(self):
        """打开CSV文件"""
        filepath = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )

        if filepath:
            success, message = self.data_manager.load_csv(filepath)

            if success:
                if hasattr(self.data_manager, 'validation_warning') and self.data_manager.validation_warning:
                    response = messagebox.askyesno(
                        "数据格式警告",
                        f"{self.data_manager.validation_warning}\n\n是否继续导入？"
                    )

                    if not response:
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                data_size = len(self.data_manager.display_data)
                if data_size > 100:
                    response = messagebox.askyesno(
                        "数据量警告",
                        f"加载了 {data_size} 条数据。\n\n数据量超过100条，使用可视化功能可能导致程序卡顿\n是否继续导入？"
                    )

                    if not response:
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                self.refresh_table()
                self.update_column_options()
            else:
                messagebox.showerror("文件格式错误", f"无法导入文件:\n\n{message}")

            self.status_callback(message)

    def open_excel(self):
        """打开Excel文件"""
        filepath = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )

        if filepath:
            success, message = self.data_manager.load_excel(filepath)

            if success:
                if hasattr(self.data_manager, 'validation_warning') and self.data_manager.validation_warning:
                    response = messagebox.askyesno(
                        "数据格式警告",
                        f"{self.data_manager.validation_warning}\n\n是否继续导入？"
                    )

                    if not response:
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                data_size = len(self.data_manager.display_data)
                if data_size > 100:
                    response = messagebox.askyesno(
                        "数据量警告",
                        f"加载了 {data_size} 条数据。\n\n数据量超过100条，使用可视化功能可能导致程序卡顿\n是否继续导入？"
                    )

                    if not response:
                        self.data_manager.raw_data = None
                        self.data_manager.display_data = None
                        self.data_manager.current_file = None
                        self.status_callback("导入已取消")
                        return

                self.refresh_table()
                self.update_column_options()
            else:
                messagebox.showerror("文件格式错误", f"无法导入Excel文件:\n\n{message}")

            self.status_callback(message)

    def save_csv(self):
        """保存CSV文件"""
        if self.data_manager.display_data is None:
            messagebox.showwarning("无数据", "请先加载数据")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv")]
        )

        if filepath:
            success, message = self.data_manager.save_to_csv(filepath)
            self.status_callback(message)

    def export_excel(self):
        """导出到Excel"""
        if self.data_manager.display_data is None:
            messagebox.showwarning("无数据", "请先加载数据")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("Excel 97-2003", "*.xls"), ("所有文件", "*.*")]
        )

        if filepath:
            success, message = self.data_manager.export_to_excel(filepath)
            if success:
                messagebox.showinfo("导出成功", message)
            else:
                messagebox.showerror("导出失败", message)
            self.status_callback(message)

    def generate_sample(self):
        """生成示例数据"""
        success, message = self.data_manager.generate_sample_data(100)
        if success:
            self.refresh_table()
            self.update_column_options()
        self.status_callback(message)

    def apply_filter(self):
        """应用筛选条件"""
        column = self.filter_column.get()
        condition = self.filter_value.get()

        if not column or not condition:
            messagebox.showwarning("输入错误", "请选择列名并输入条件")
            return

        success, message = self.data_manager.apply_filter(column, condition)
        if success:
            self.refresh_table()
        self.status_callback(message)

    def clear_filter(self):
        """清除筛选"""
        success, message = self.data_manager.clear_all_filters()
        if success:
            self.refresh_table()
        self.status_callback(message)

    def search_data(self):
        """搜索数据"""
        keyword = self.search_entry.get()
        if not keyword:
            messagebox.showwarning("输入错误", "请输入搜索关键词")
            return

        results, message = self.data_manager.search_data(keyword)
        if results is not None:
            self.data_manager.display_data = results
            self.refresh_table()
        self.status_callback(message)

    def sort_data(self):
        """排序数据"""
        column = self.sort_column.get()
        if not column:
            messagebox.showwarning("输入错误", "请选择排序列")
            return

        success, message = self.data_manager.sort_data(column, self.sort_ascending.get())
        if success:
            self.refresh_table()
        self.status_callback(message)

    def add_record(self):
        """添加新记录"""
        if self.data_manager.display_data is None:
            messagebox.showwarning("无数据", "请先加载数据")
            return

        dialog = tk.Toplevel(self)
        dialog.title("添加新记录")
        dialog.geometry("400x800")

        columns = self.data_manager.get_column_names()
        entries = {}

        for i, col in enumerate(columns):
            ttk.Label(dialog, text=f"{col}:").grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(dialog, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            entries[col] = entry

        def save_record():
            record_dict = {}
            for col, entry in entries.items():
                record_dict[col] = entry.get() or None

            success, message = self.data_manager.add_record(record_dict)
            if success:
                self.refresh_table()
                dialog.destroy()
            self.status_callback(message)

        ttk.Button(dialog, text="保存", command=save_record).grid(row=len(columns), column=0, columnspan=2, pady=20)

    def delete_selected(self):
        """删除选中记录"""
        selected_indices = self.table.get_selected_indices()
        if not selected_indices:
            messagebox.showwarning("无选择", "请先选择要删除的记录")
            return

        if messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_indices)} 条记录吗？"):
            success, message = self.data_manager.delete_records(selected_indices)
            if success:
                self.refresh_table()
            self.status_callback(message)

    def refresh_table(self):
        """刷新表格显示"""
        success, message = self.table.load_data()
        if not success:
            self.status_callback(message)


class InfoPanel(ttk.LabelFrame):
    """信息面板"""

    def __init__(self, parent, data_manager):
        super().__init__(parent, text="数据信息", padding=10)
        self.data_manager = data_manager

        self.info_text = tk.Text(self, height=15, width=30, state="disabled")
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.update_info()

    def update_info(self):
        """更新信息显示"""
        if self.data_manager.display_data is None:
            info = "请导入数据..."
        else:
            stats = self.data_manager.get_basic_stats()
            info = f"📊 数据概览\n{'=' * 30}\n"
            info += f"总记录数: {stats['total_records']}\n"
            info += f"总列数: {stats['total_columns']}\n\n"

            info += "📈 列信息:\n"
            for col_info in stats['column_details']:
                info += f"\n{col_info['name']}:\n"
                info += f"  类型: {col_info['type']}\n"
                info += f"  非空值: {col_info['non_null']}\n"
                info += f"  唯一值: {col_info['unique_values']}\n"

        self.info_text.config(state="normal")
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state="disabled")


# ==================== 5. 主窗口整合 ====================

class IntegratedMainWindow:
    """集成版主窗口"""

    def __init__(self, root, data_manager):
        self.root = root
        self.data_manager = data_manager
        self.predictor = None
        self.status_var = tk.StringVar(value="就绪")

        self.setup_window()
        self.setup_status_bar()
        self.setup_notebook()
        self.update_status("就绪 - 城市交通事故分析与预警系统")

    def setup_window(self):
        """设置窗口属性"""
        self.root.title("城市交通事故分析与预警系统")
        self.root.geometry("1200x700")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def setup_notebook(self):
        """设置选项卡控件"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.setup_data_tab()
        self.setup_viz_tab()
        self.setup_pred_tab()
        self.setup_help_tab()

    def setup_data_tab(self):
        """设置数据管理选项卡"""
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 数据管理")
        self.setup_data_tab_layout()

    def setup_data_tab_layout(self):
        """数据管理页的具体布局"""
        main_frame = ttk.Frame(self.data_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        self.info_panel = InfoPanel(left_panel, self.data_manager)
        self.info_panel.pack(fill=tk.BOTH, expand=True)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.data_table = DataTable(right_panel, self.data_manager)
        self.data_table.pack(fill=tk.BOTH, expand=True, padx=(0, 5))

        control_frame = ttk.Frame(right_panel)
        control_frame.pack(fill=tk.X, pady=(5, 0))

        self.control_panel = ControlPanel(
            control_frame,
            self.data_manager,
            self.data_table,
            self.update_status
        )
        self.control_panel.pack(fill=tk.X)

    def setup_viz_tab(self):
        """设置可视化分析选项卡"""
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="📈 可视化分析")

        viz_container = ttk.Frame(self.viz_tab)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.visualizer = TrafficVisualizer(self.data_manager, viz_container)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event=None):
        """处理选项卡切换事件"""
        try:
            if not hasattr(self, 'notebook') or not self.notebook.winfo_exists():
                return

            current_tab_id = self.notebook.select()
            if not current_tab_id:
                return

            current_tab_index = self.notebook.index(current_tab_id)
            tab_text = self.notebook.tab(current_tab_index, "text")

            if tab_text == "📈 可视化分析":
                if hasattr(self, 'data_manager') and self.data_manager.display_data is not None:
                    data_size = len(self.data_manager.display_data)
                    if data_size > 100:
                        response = messagebox.askokcancel(
                            "数据量警告",
                            f"当前数据有 {data_size} 条记录。\n\n⚠️ 数据量超过100条，使用可视化功能可能导致程序卡顿。\n\n【确定】继续使用可视化分析\n【取消】返回数据管理页面进行筛选"
                        )

                        if not response:
                            for i in range(self.notebook.index("end")):
                                if self.notebook.tab(i, "text") == "📊 数据管理":
                                    self.notebook.select(i)
                                    self.update_status(f"已返回数据管理页面 (数据量: {data_size} 条)")
                                    return

                if hasattr(self, 'visualizer') and self.visualizer:
                    self.root.after(300, self.refresh_visualizer)

            elif tab_text == "📊 数据管理":
                if hasattr(self, 'data_table'):
                    self.data_table.load_data()
                    if hasattr(self, 'info_panel'):
                        self.info_panel.update_info()

            elif tab_text == "⚠️ 风险预测":
                pass

        except Exception as e:
            print(f"选项卡切换错误: {e}")

    def refresh_visualizer(self):
        """刷新可视化器"""
        if hasattr(self, 'visualizer') and self.visualizer:
            self.visualizer.update_axis_options()

            if self.data_manager.display_data is not None:
                try:
                    x_axis = self.visualizer.x_axis_var.get()
                    y_axis = self.visualizer.y_axis_var.get()

                    if x_axis and y_axis:
                        self.root.after(500, self.visualizer.generate_chart)
                    else:
                        columns = self.data_manager.get_column_names()
                        if columns:
                            self.visualizer.x_axis_var.set(columns[0])

                            numeric_cols = []
                            data = self.data_manager.display_data
                            for col in columns:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    numeric_cols.append(col)

                            if numeric_cols:
                                self.visualizer.y_axis_var.set(numeric_cols[0])
                                self.root.after(500, self.visualizer.generate_chart)
                except Exception as e:
                    self.update_status(f"刷新可视化器失败: {str(e)}")

    def setup_pred_tab(self):
        """设置风险预测选项卡"""
        self.pred_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_tab, text="⚠️ 风险预测")
        self.setup_prediction_ui()

    def setup_training_panel(self, parent):
        """设置模型训练面板"""
        frame = ttk.LabelFrame(parent, text="模型训练", padding=10)
        frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="训练预测模型", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导入模型文件", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导出当前模型", command=self.save_model).pack(side=tk.LEFT, padx=5)

        self.model_status_var = tk.StringVar(value="模型状态: 未训练")
        ttk.Label(frame, textvariable=self.model_status_var).pack(anchor=tk.W)

    def setup_prediction_ui(self):
        """设置预测用户界面"""
        main_frame = ttk.Frame(self.pred_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_training_panel(main_frame)
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=20)

        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        columns_frame.columnconfigure(0, weight=1)
        columns_frame.columnconfigure(1, weight=1)
        columns_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(columns_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.setup_single_prediction_panel(left_frame)

        right_frame = ttk.Frame(columns_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        batch_frame = ttk.LabelFrame(right_frame, text="批量风险预测", padding=10)
        batch_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.setup_batch_prediction_panel_content(batch_frame)

        feature_frame = ttk.LabelFrame(right_frame, text="特征重要性分析", padding=10)
        feature_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.setup_feature_importance_panel_content(feature_frame)

    def setup_single_prediction_panel(self, parent):
        """设置单条预测面板"""
        frame = ttk.LabelFrame(parent, text="单条事故风险预测", padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        form_frame = ttk.Frame(scrollable_frame)
        form_frame.pack(fill=tk.X, pady=10, padx=5)

        fields = [
            ("事故时间", "2024-01-01 08:30"),
            ("所在区域", "朝阳区"),
            ("事故类型", "追尾"),
            ("受伤人数", "1"),
            ("死亡人数", "0"),
            ("温度(℃)", "25.5"),
            ("湿度(%)", "65"),
            ("能见度(km)", "10.5"),
            ("风速(m/s)", "3.2")
        ]

        self.pred_inputs = {}
        for i, (label, default) in enumerate(fields):
            row_frame = ttk.Frame(form_frame)
            row_frame.pack(fill=tk.X, pady=3)

            lbl = ttk.Label(row_frame, text=f"{label}:", width=15, anchor="e")
            lbl.pack(side=tk.LEFT, padx=(0, 5))

            entry = ttk.Entry(row_frame)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.pred_inputs[label] = entry

        result_frame = ttk.Frame(scrollable_frame)
        result_frame.pack(fill=tk.X, pady=15)

        btn_frame = ttk.Frame(result_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="预测风险", command=self.predict_single).pack()

        result_display = ttk.Frame(result_frame, relief="solid", borderwidth=1)
        result_display.pack(fill=tk.X, pady=5)

        self.pred_result_var = tk.StringVar(value="等待预测...")
        self.pred_result_label = ttk.Label(
            result_display,
            textvariable=self.pred_result_var,
            font=("Arial", 14, "bold"),
            anchor="center",
            padding=10
        )
        self.pred_result_label.pack(fill=tk.X)

        self.pred_prob_var = tk.StringVar(value="")
        ttk.Label(
            result_display,
            textvariable=self.pred_prob_var,
            anchor="center",
            padding=(0, 5, 0, 10)
        ).pack(fill=tk.X)

    def setup_batch_prediction_panel_content(self, parent):
        """设置批量预测面板内容"""
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="对当前数据批量预测", command=self.predict_batch, width=20).pack(side=tk.LEFT,
                                                                                                    padx=2)
        ttk.Button(btn_frame, text="刷新预测结果", command=self.refresh_predictions, width=15).pack(side=tk.LEFT,
                                                                                                    padx=2)

        export_frame = ttk.Frame(parent)
        export_frame.pack(fill=tk.X, pady=5)

        ttk.Label(export_frame, text="导出结果:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="CSV", command=self.export_predictions_csv, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(export_frame, text="Excel", command=self.export_predictions_excel, width=10).pack(side=tk.LEFT,
                                                                                                     padx=2)

        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        self.batch_status_var = tk.StringVar(value="未进行批量预测")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.batch_status_var,
            relief="sunken",
            anchor="w",
            padding=5,
            background="#f0f0f0"
        )
        status_label.pack(fill=tk.X)

        self.batch_stats_text = tk.Text(parent, height=8, width=30, state="disabled")
        self.batch_stats_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def setup_feature_importance_panel_content(self, parent):
        """设置特征重要性面板内容"""
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="查看特征重要性", command=self.show_feature_importance, width=20).pack()

        self.feature_text = tk.Text(parent, height=12, state="disabled")
        self.feature_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(parent, command=self.feature_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_text.config(yscrollcommand=scrollbar.set)

    def refresh_predictions(self):
        """刷新预测结果显示"""
        if self.data_manager.display_data is not None and '预测风险等级' in self.data_manager.display_data.columns:
            predictions = self.data_manager.display_data['预测风险等级']
            unique, counts = np.unique(predictions, return_counts=True)

            stats_text = "📊 预测结果统计:\n"
            stats_text += "=" * 30 + "\n"

            total = len(predictions)
            for level, count in zip(unique, counts):
                risk_labels = ['低风险', '中风险', '高风险']
                label = risk_labels[level] if level < 3 else f"等级{level}"
                percentage = count / total * 100
                stats_text += f"{label}: {count} 条 ({percentage:.1f}%)\n"

            self.batch_stats_text.config(state="normal")
            self.batch_stats_text.delete(1.0, tk.END)
            self.batch_stats_text.insert(1.0, stats_text)
            self.batch_stats_text.config(state="disabled")

            self.batch_status_var.set(f"已预测 {total} 条记录")
        else:
            self.batch_status_var.set("未进行批量预测")
            self.batch_stats_text.config(state="normal")
            self.batch_stats_text.delete(1.0, tk.END)
            self.batch_stats_text.config(state="disabled")

    def setup_help_tab(self):
        """设置帮助选项卡"""
        self.help_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.help_tab, text="❓ 使用帮助")
        self.setup_help_content()

    def setup_help_content(self):
        """设置帮助内容"""
        main_frame = ttk.Frame(self.help_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        help_text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                            font=("宋体", 11), padx=10, pady=10)
        help_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=help_text.yview)

        help_content = """
城市交通事故分析与预警系统 - 使用帮助
=========================================

📊 数据管理
------------
• 导入CSV：加载交通事故数据文件
• 导出CSV：将当前数据保存为CSV格式
• 导出Excel：将数据导出为Excel文件，包含数据表和统计信息
• 生成示例：快速生成测试数据
• 筛选数据：按条件筛选数据
• 搜索数据：搜索包含关键词的记录
• 排序数据：按指定列排序
• 添加记录：手动添加新的事故记录
• 删除记录：删除选中的记录

📈 可视化分析
------------
• 图表类型：支持柱状图、折线图、饼图、散点图、热力图、箱线图
• 轴选择：选择X轴和Y轴的数据列
• 生成图表：根据选择生成可视化图表
• 导出图片：将图表导出为PNG、JPG、PDF等格式
• 工具栏：图表缩放、平移、保存等操作

⚠️ 风险预测
------------
• 训练模型：使用当前数据训练风险预测模型
• 单条预测：输入事故信息，预测风险等级
• 批量预测：对当前所有数据进行风险预测
• 保存模型：将训练好的模型保存为文件
• 加载模型：从文件加载已训练的模型
• 导出结果：将预测结果导出为CSV或Excel
• 特征重要性：查看影响风险预测的主要因素

系统特点
--------
1. 一体化界面：数据管理、可视化、预测在一个界面中完成
2. 智能预测：基于机器学习的事故风险预测
3. 多种导出：支持CSV、Excel、图片等多种格式导出
4. 用户友好：简洁直观的操作界面

技术支持
--------
Github仓库地址：https://github.com/PaintHelloWorld/Traffic_Analysis_System
© 2026 交通数据分析项目 - 版本 1.3.1
        """

        help_text.insert(1.0, help_content)
        help_text.config(state="disabled")

    def setup_status_bar(self):
        """设置状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        if hasattr(self, 'info_panel'):
            self.info_panel.update_info()

    def init_predictor(self):
        """初始化预测器"""
        if self.predictor is None:
            self.predictor = TrafficPredictor()
        return self.predictor

    def train_model(self):
        """训练预测模型"""
        if self.data_manager.display_data is None:
            self.update_status("请先加载数据")
            tk.messagebox.showwarning("无数据", "请先导入数据")
            return

        predictor = self.init_predictor()

        try:
            success, result = predictor.train_model(self.data_manager.display_data)

            if success:
                self.model_status_var.set(f"模型状态: 已训练 (准确率: {result['accuracy']:.2%})")
                self.update_status(f"模型训练成功，准确率: {result['accuracy']:.2%}")

                report_dialog = tk.Toplevel(self.root)
                report_dialog.title("模型训练报告")
                report_dialog.geometry("500x400")

                text = tk.Text(report_dialog, wrap=tk.WORD)
                text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                report_text = f"模型训练完成！\n\n"
                report_text += f"准确率: {result['accuracy']:.2%}\n"
                report_text += f"使用特征数: {result['feature_count']}\n"
                report_text += f"训练集大小: {result['train_size']}\n"
                report_text += f"测试集大小: {result['test_size']}\n\n"
                report_text += "分类报告:\n" + result['report']

                text.insert(1.0, report_text)
                text.config(state="disabled")

                ttk.Button(report_dialog, text="确定", command=report_dialog.destroy).pack(pady=10)
            else:
                self.update_status(f"模型训练失败: {result}")
                tk.messagebox.showerror("训练失败", result)
        except Exception as e:
            self.update_status(f"模型训练异常: {str(e)}")

    def predict_single(self):
        """单条预测"""
        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("请先训练模型")
            tk.messagebox.showwarning("模型未训练", "请先训练预测模型")
            return

        try:
            input_dict = {}
            for label, entry in self.pred_inputs.items():
                value = entry.get()
                try:
                    if label in ['受伤人数', '死亡人数', '温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)']:
                        value = float(value)
                except:
                    pass
                input_dict[label] = value

            risk_label, prob_dict, message = self.predictor.predict_single(input_dict)

            if risk_label:
                self.pred_result_var.set(f"预测结果: {risk_label}")

                color_map = {
                    '低风险': 'green',
                    '中风险': 'orange',
                    '高风险': 'red'
                }
                color = color_map.get(risk_label, 'black')

                for widget in self.pred_result_var._widgets:
                    widget.config(foreground=color)

                if prob_dict:
                    prob_text = " | ".join([f"{k}: {v:.1%}" for k, v in prob_dict.items()])
                    self.pred_prob_var.set(f"概率分布: {prob_text}")

                self.update_status(f"预测完成: {risk_label}")
            else:
                self.update_status(f"预测失败: {message}")
        except Exception as e:
            self.update_status(f"预测出错: {str(e)}")

    def predict_batch(self):
        """批量预测"""
        if self.data_manager.display_data is None:
            self.update_status("请先加载数据")
            return

        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("请先训练模型")
            tk.messagebox.showwarning("模型未训练", "请先训练预测模型")
            return

        try:
            predictions, probabilities, message = self.predictor.predict(self.data_manager.display_data)

            if predictions is not None:
                self.data_manager.display_data['预测风险等级'] = predictions
                self.batch_status_var.set(f"批量预测完成，共 {len(predictions)} 条记录")
                self.update_status(f"批量预测完成，共 {len(predictions)} 条记录")
                self.refresh_predictions()

                if hasattr(self, 'data_table'):
                    self.data_table.load_data()
                if hasattr(self, 'info_panel'):
                    self.info_panel.update_info()

                tk.messagebox.showinfo("批量预测完成", f"已完成 {len(predictions)} 条记录的预测")
            else:
                self.update_status(f"批量预测失败: {message}")
        except Exception as e:
            self.update_status(f"批量预测出错: {str(e)}")

    def export_predictions_csv(self):
        """导出预测结果为CSV"""
        self.export_predictions('csv')

    def export_predictions_excel(self):
        """导出预测结果为Excel"""
        self.export_predictions('excel')

    def export_predictions(self, file_type='csv'):
        """导出预测结果"""
        if self.data_manager.display_data is None or '预测风险等级' not in self.data_manager.display_data.columns:
            self.update_status("没有预测结果可导出")
            tk.messagebox.showwarning("无结果", "请先进行批量预测")
            return

        if file_type == 'excel':
            default_ext = ".xlsx"
            filetypes = [("Excel文件", "*.xlsx"), ("Excel 97-2003", "*.xls"), ("所有文件", "*.*")]
        else:
            default_ext = ".csv"
            filetypes = [("CSV文件", "*.csv"), ("所有文件", "*.*")]

        filepath = tk.filedialog.asksaveasfilename(
            defaultextension=default_ext,
            filetypes=filetypes
        )

        if filepath:
            if file_type == 'excel':
                success, message = self.data_manager.export_to_excel(filepath)
            else:
                success, message = self.data_manager.save_to_csv(filepath)

            if success:
                self.update_status(f"预测结果已导出到: {filepath}")
                tk.messagebox.showinfo("导出成功", f"预测结果已成功导出到:\n{filepath}")
            else:
                self.update_status(message)
                tk.messagebox.showerror("导出失败", message)

    def show_feature_importance(self):
        """显示特征重要性"""
        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("请先训练模型")
            return

        importance_df = self.predictor.get_feature_importance()

        if importance_df is not None:
            self.feature_text.config(state="normal")
            self.feature_text.delete(1.0, tk.END)

            text = "特征重要性排序:\n"
            text += "=" * 40 + "\n\n"

            for idx, row in importance_df.iterrows():
                text += f"{row['feature']}: {row['importance']:.3f}\n"

            self.feature_text.insert(1.0, text)
            self.feature_text.config(state="disabled")
            self.update_status("特征重要性已显示")
        else:
            self.update_status("无法获取特征重要性")

    def load_model(self):
        """加载模型文件"""
        filepath = tk.filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("模型文件", "*.pkl"), ("所有文件", "*.*")]
        )

        if filepath:
            predictor = self.init_predictor()
            success, message = predictor.load_model(filepath)

            if success:
                self.model_status_var.set("模型状态: 已加载")
                self.update_status(message)
                tk.messagebox.showinfo("加载成功", "模型加载成功")
            else:
                self.update_status(message)
                tk.messagebox.showerror("加载失败", message)

    def save_model(self):
        """保存模型文件"""
        if self.predictor is None or not self.predictor.is_trained:
            self.update_status("没有训练好的模型可保存")
            tk.messagebox.showwarning("无模型", "请先训练模型")
            return

        filepath = tk.filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("模型文件", "*.pkl")]
        )

        if filepath:
            success, message = self.predictor.save_model(filepath)
            self.update_status(message)

            if success:
                tk.messagebox.showinfo("保存成功", "模型保存成功")
            else:
                tk.messagebox.showerror("保存失败", message)


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    root = tk.Tk()
    root.title("城市交通事故分析与预警系统")
    root.state('zoomed')
    root.geometry("1200x800")

    data_manager = TrafficDataManager()
    IntegratedMainWindow(root, data_manager)
    root.mainloop()


if __name__ == "__main__":
    main()