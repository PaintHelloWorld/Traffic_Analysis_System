# predictor.py - 事故预测模型
"""
机器学习预测模块 - TrafficPredictor
====================================

模型介绍：
    使用随机森林(RandomForest)算法预测交通事故风险等级
    风险等级：0=低风险, 1=中风险, 2=高风险

特征工程：
    1. 时间特征提取：从时间字段提取小时、星期、月份等
    2. 分类变量编码：区域、事故类型等转为数值
    3. 特征选择：自动选择数值列和重要分类列

模型流程：
    1. 数据准备 → 2. 特征提取 → 3. 训练测试分割 →
    4. 数据标准化 → 5. 模型训练 → 6. 评估预测

关键技术点：
    1. 特征缩放：StandardScaler标准化，提高模型收敛速度
    2. 分层抽样：train_test_split的stratify参数保持类别平衡
    3. 模型保存：使用pickle持久化，方便复用
    4. 概率输出：predict_proba提供风险概率分布

学习收获：
    1. 理解了机器学习的基本流程（特征-模型-评估）
    2. 学会了sklearn的基本使用方法
    3. 认识到特征工程的重要性
    4. 了解了模型评估指标（准确率、混淆矩阵）

模型限制：
    1. 数据量要求：至少50条记录才能训练
    2. 特征依赖：需要包含基本的事故信息字段
    3. 预测解释：可提供概率但不能解释具体原因
"""

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class TrafficPredictor:
    """交通事故预测器"""

    def __init__(self):
        """初始化预测器"""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'risk_level'
        self.is_trained = False

    def prepare_features(self, data):
        """从原始数据提取特征"""
        df = data.copy()

        # 1. 提取时间特征（如果有时间列）
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

        # 2. 处理分类变量
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                categorical_cols.append(col)

        for col in categorical_cols[:5]:  # 限制处理前5个分类变量
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            except:
                pass

        # 3. 选择特征列
        feature_candidates = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            '受伤人数', '死亡人数', '温度(℃)', '湿度(%)', '能见度(km)', '风速(m/s)'
        ]

        available_features = []
        for feat in feature_candidates:
            if feat in df.columns:
                available_features.append(feat)

        # 添加分类变量
        for col in categorical_cols[:3]:
            if col not in available_features and col in df.columns:
                available_features.append(col)

        # 确保有足够的特征
        if len(available_features) < 3:
            # 如果特征太少，使用所有数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols[:8]  # 取前8个数值列

        return df[available_features]

    def create_target_variable(self, data):
        """
        创建目标变量（风险等级：0:低风险, 1:中风险, 2:高风险）
        """
        # 如果有事故等级列，使用它
        severity_cols = [col for col in data.columns if any(kw in col.lower()
                                                            for kw in ['severity', 'level', '等级', '严重'])]

        if severity_cols:
            severity_col = severity_cols[0]
            if data[severity_col].dtype == 'object':
                # 文本等级转换为数值
                level_mapping = {
                    '轻微': 0, '一般': 1, '严重': 2,
                    '低': 0, '中': 1, '高': 2,
                    'low': 0, 'medium': 1, 'high': 2
                }
                target = data[severity_col].map(level_mapping).fillna(1).astype(int)
            else:
                target = data[severity_col].astype(int)
        else:
            # 如果没有等级列，基于受伤人数和死亡人数创建
            if '受伤人数' in data.columns and '死亡人数' in data.columns:
                target = np.zeros(len(data), dtype=int)

                # 规则1：有死亡人数 -> 高风险
                target[data['死亡人数'] > 0] = 2

                # 规则2：受伤人数 >= 2 -> 中风险
                target[(data['受伤人数'] >= 2) & (target == 0)] = 1

                # 规则3：受伤人数 = 1 -> 低风险（默认）
                target[(data['受伤人数'] == 1) & (target == 0)] = 0
            else:
                # 随机生成目标变量（仅用于演示）
                np.random.seed(42)
                target = np.random.choice([0, 1, 2], size=len(data), p=[0.6, 0.3, 0.1])

        return target

    def train_model(self, data):
        """
        训练预测模型
        """
        try:
            # 检查数据量
            if len(data) < 50:
                return False, "数据量不足，至少需要50条"

            # 1. 准备特征
            features = self.prepare_features(data)

            if len(features.columns) < 2:
                return False, "特征不足，无法训练模型"

            # 2. 创建目标变量
            target = self.create_target_variable(data)

            # 3. 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )

            # 4. 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 5. 训练模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(X_train_scaled, y_train)

            # 6. 评估模型
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # 7. 保存特征信息
            self.feature_columns = list(features.columns)
            self.is_trained = True

            # 生成评估报告
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
        """
        对新数据进行预测
        """
        if not self.is_trained or self.model is None:
            return None, "请先训练模型"

        try:
            # 1. 准备特征
            features = self.prepare_features(data)

            # 2. 确保特征列匹配
            missing_cols = set(self.feature_columns) - set(features.columns)
            for col in missing_cols:
                features[col] = 0

            # 确保列顺序一致
            features = features[self.feature_columns]

            # 3. 标准化
            features_scaled = self.scaler.transform(features)

            # 4. 预测
            predictions = self.model.predict(features_scaled)

            # 5. 添加概率
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)
                return predictions, probabilities, "预测成功"
            else:
                return predictions, None, "预测成功"

        except Exception as e:
            return None, None, f"预测失败: {str(e)}"

    def save_model(self, filepath):
        """
        保存模型到文件
        """
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
        """
        从文件加载模型
        """
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
        """
        获取特征重要性

        Returns:
            DataFrame: 特征重要性排序
        """
        if not self.is_trained or self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_columns

            # 创建DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            return importance_df

        return None

    def predict_single(self, input_dict):
        """
        预测单个记录

        Args:
            input_dict: 字典，包含特征值

        Returns:
            tuple: (预测结果, 风险概率, 消息)
        """
        if not self.is_trained:
            return None, None, "请先训练模型"

        try:
            # 创建DataFrame
            input_df = pd.DataFrame([input_dict])

            # 预测
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
