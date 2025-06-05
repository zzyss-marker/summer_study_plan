"""
表格数据机器学习Baseline脚本
支持分类和回归任务，自动化特征工程和模型选择
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class TabularBaseline:
    """表格数据机器学习Baseline类"""
    
    def __init__(self, task_type='auto'):
        """
        初始化
        task_type: 'classification', 'regression', 'auto'
        """
        self.task_type = task_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, data_path=None, X=None, y=None):
        """加载数据"""
        if data_path:
            self.data = pd.read_csv(data_path)
            print(f"数据加载完成，形状: {self.data.shape}")
        elif X is not None and y is not None:
            self.X = X
            self.y = y
            print(f"数据加载完成，X形状: {X.shape}, y形状: {y.shape}")
        else:
            # 生成示例数据
            self._generate_sample_data()
    
    def _generate_sample_data(self):
        """生成示例数据"""
        np.random.seed(42)
        n_samples = 1000
        
        # 生成特征
        age = np.random.randint(18, 80, n_samples)
        income = np.random.normal(50000, 20000, n_samples)
        education = np.random.choice(['高中', '本科', '硕士', '博士'], n_samples)
        experience = np.random.randint(0, 40, n_samples)
        
        # 生成目标变量
        if self.task_type == 'classification' or self.task_type == 'auto':
            # 分类任务：预测是否高收入
            target = ((income > 60000) & (age > 30) & (experience > 5)).astype(int)
            self.task_type = 'classification'
        else:
            # 回归任务：预测收入
            target = income + age * 500 + experience * 1000 + np.random.normal(0, 5000, n_samples)
        
        self.data = pd.DataFrame({
            'age': age,
            'income': income,
            'education': education,
            'experience': experience,
            'target': target
        })
        
        print(f"生成示例数据完成，形状: {self.data.shape}")
        print(f"任务类型: {self.task_type}")
    
    def preprocess_data(self, target_column='target'):
        """数据预处理"""
        print("开始数据预处理...")
        
        # 分离特征和目标
        if hasattr(self, 'data'):
            X = self.data.drop(target_column, axis=1)
            y = self.data[target_column]
        else:
            X, y = self.X, self.y
        
        # 自动检测任务类型
        if self.task_type == 'auto':
            if len(np.unique(y)) <= 10 and y.dtype in ['int64', 'object']:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        
        # 处理分类特征
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        X_processed = X.copy()
        
        # 编码分类特征
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, 
            stratify=y if self.task_type == 'classification' else None
        )
        
        # 标准化数值特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"预处理完成，训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
        print(f"分类特征: {list(categorical_columns)}")
        print(f"数值特征: {list(numerical_columns)}")
    
    def setup_models(self):
        """设置模型"""
        if self.task_type == 'classification':
            self.models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True)
            }
        else:
            self.models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'SVR': SVR()
            }
    
    def train_models(self):
        """训练所有模型"""
        print(f"开始训练{self.task_type}模型...")
        
        for name, model in self.models.items():
            print(f"训练 {name}...")
            
            # 选择是否使用标准化数据
            if name in ['LogisticRegression', 'SVM', 'SVR', 'Ridge', 'Lasso']:
                X_train, X_test = self.X_train_scaled, self.X_test_scaled
            else:
                X_train, X_test = self.X_train, self.X_test
            
            # 训练模型
            model.fit(X_train, self.y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            if self.task_type == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='accuracy')
                
                self.results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  准确率: {accuracy:.4f}")
                print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='r2')
                
                self.results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
                print(f"  交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def evaluate_models(self):
        """评估和比较模型"""
        print(f"\n模型评估结果:")
        print("=" * 50)
        
        if self.task_type == 'classification':
            # 找到最佳模型
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['accuracy'])
            self.best_model = self.results[best_model_name]['model']
            
            # 打印结果
            for name, result in self.results.items():
                print(f"{name}:")
                print(f"  准确率: {result['accuracy']:.4f}")
                print(f"  交叉验证: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
                if name == best_model_name:
                    print("  ⭐ 最佳模型")
                print()
            
            # 详细评估最佳模型
            print(f"最佳模型 ({best_model_name}) 详细评估:")
            best_pred = self.results[best_model_name]['predictions']
            print(classification_report(self.y_test, best_pred))
            
        else:
            # 找到最佳模型
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['r2'])
            self.best_model = self.results[best_model_name]['model']
            
            # 打印结果
            for name, result in self.results.items():
                print(f"{name}:")
                print(f"  RMSE: {result['rmse']:.4f}")
                print(f"  R²: {result['r2']:.4f}")
                print(f"  交叉验证R²: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
                if name == best_model_name:
                    print("  ⭐ 最佳模型")
                print()
        
        return best_model_name
    
    def visualize_results(self):
        """可视化结果"""
        plt.figure(figsize=(15, 10))
        
        if self.task_type == 'classification':
            # 准确率比较
            plt.subplot(2, 3, 1)
            names = list(self.results.keys())
            accuracies = [self.results[name]['accuracy'] for name in names]
            
            bars = plt.bar(names, accuracies, alpha=0.7)
            plt.title('模型准确率比较')
            plt.ylabel('准确率')
            plt.xticks(rotation=45)
            
            # 标注最佳模型
            best_idx = accuracies.index(max(accuracies))
            bars[best_idx].set_color('red')
            
            # 混淆矩阵
            plt.subplot(2, 3, 2)
            best_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            best_pred = self.results[best_name]['predictions']
            
            cm = confusion_matrix(self.y_test, best_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'混淆矩阵 ({best_name})')
            plt.ylabel('真实值')
            plt.xlabel('预测值')
            
        else:
            # R²比较
            plt.subplot(2, 3, 1)
            names = list(self.results.keys())
            r2_scores = [self.results[name]['r2'] for name in names]
            
            bars = plt.bar(names, r2_scores, alpha=0.7)
            plt.title('模型R²比较')
            plt.ylabel('R²得分')
            plt.xticks(rotation=45)
            
            # 标注最佳模型
            best_idx = r2_scores.index(max(r2_scores))
            bars[best_idx].set_color('red')
            
            # 预测vs真实值
            plt.subplot(2, 3, 2)
            best_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
            best_pred = self.results[best_name]['predictions']
            
            plt.scatter(self.y_test, best_pred, alpha=0.6)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'预测vs真实 ({best_name})')
        
        # 交叉验证得分
        plt.subplot(2, 3, 3)
        cv_means = [self.results[name]['cv_mean'] for name in names]
        cv_stds = [self.results[name]['cv_std'] for name in names]
        
        plt.errorbar(range(len(names)), cv_means, yerr=cv_stds, 
                    fmt='o', capsize=5, capthick=2)
        plt.xticks(range(len(names)), names, rotation=45)
        plt.title('交叉验证得分')
        plt.ylabel('CV得分')
        
        plt.tight_layout()
        plt.show()
    
    def run_baseline(self, data_path=None, X=None, y=None, target_column='target'):
        """运行完整的baseline流程"""
        print("🚀 开始表格数据机器学习Baseline")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data(data_path, X, y)
        
        # 2. 数据预处理
        self.preprocess_data(target_column)
        
        # 3. 设置模型
        self.setup_models()
        
        # 4. 训练模型
        self.train_models()
        
        # 5. 评估模型
        best_model_name = self.evaluate_models()
        
        # 6. 可视化结果
        self.visualize_results()
        
        print(f"\n🎉 Baseline完成！最佳模型: {best_model_name}")
        return self.best_model, self.results

def main():
    """主函数 - 演示用法"""
    # 分类任务示例
    print("=" * 60)
    print("分类任务示例")
    print("=" * 60)
    
    classifier = TabularBaseline(task_type='classification')
    best_clf, clf_results = classifier.run_baseline()
    
    print("\n" + "=" * 60)
    print("回归任务示例")
    print("=" * 60)
    
    # 回归任务示例
    regressor = TabularBaseline(task_type='regression')
    best_reg, reg_results = regressor.run_baseline()
    
    print("\n💡 使用说明:")
    print("1. 自定义数据: baseline.run_baseline(data_path='your_data.csv')")
    print("2. 指定目标列: baseline.run_baseline(target_column='your_target')")
    print("3. 使用numpy数组: baseline.run_baseline(X=X_array, y=y_array)")

if __name__ == "__main__":
    main()
