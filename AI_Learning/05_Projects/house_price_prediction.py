"""
房价预测完整项目
从数据生成到模型部署的完整机器学习项目流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class HousePricePrediction:
    """房价预测完整项目类"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        print("🏠 房价预测完整项目")
        print("=" * 50)
    
    def generate_synthetic_data(self, n_samples=1000):
        """生成合成房价数据"""
        print("📊 生成合成房价数据")
        print("=" * 30)
        
        np.random.seed(42)
        
        # 基础特征
        area = np.random.normal(120, 40, n_samples)  # 面积
        area = np.clip(area, 50, 300)  # 限制范围
        
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                  p=[0.1, 0.2, 0.4, 0.25, 0.05])
        
        bathrooms = np.random.choice([1, 2, 3, 4], n_samples,
                                   p=[0.2, 0.5, 0.25, 0.05])
        
        age = np.random.randint(0, 50, n_samples)  # 房龄
        
        # 地理位置（影响价格的重要因素）
        districts = ['市中心', '商业区', '住宅区', '郊区', '远郊']
        district = np.random.choice(districts, n_samples,
                                  p=[0.15, 0.25, 0.35, 0.2, 0.05])
        
        # 楼层
        floor = np.random.randint(1, 31, n_samples)
        
        # 朝向
        orientations = ['南', '东南', '东', '西南', '北', '西', '东北', '西北']
        orientation = np.random.choice(orientations, n_samples)
        
        # 装修状况
        decoration = np.random.choice(['毛坯', '简装', '精装', '豪装'], n_samples,
                                    p=[0.2, 0.4, 0.3, 0.1])
        
        # 计算价格（基于特征的复杂关系）
        # 基础价格
        base_price = area * 50  # 每平米50元基础价
        
        # 区域调整
        district_multiplier = {'市中心': 3.0, '商业区': 2.5, '住宅区': 2.0, 
                             '郊区': 1.5, '远郊': 1.0}
        district_adj = np.array([district_multiplier[d] for d in district])
        
        # 房间数调整
        bedroom_adj = bedrooms * 5000
        bathroom_adj = bathrooms * 3000
        
        # 房龄调整（新房更贵）
        age_adj = -age * 200
        
        # 楼层调整（中间楼层更贵）
        floor_adj = np.where((floor >= 6) & (floor <= 20), 2000, 0)
        
        # 朝向调整
        orientation_adj = np.where(np.isin(orientation, ['南', '东南']), 5000, 0)
        
        # 装修调整
        decoration_multiplier = {'毛坯': 0.8, '简装': 1.0, '精装': 1.2, '豪装': 1.5}
        decoration_adj = np.array([decoration_multiplier[d] for d in decoration])
        
        # 最终价格计算
        price = (base_price * district_adj + bedroom_adj + bathroom_adj + 
                age_adj + floor_adj + orientation_adj) * decoration_adj
        
        # 添加噪声
        noise = np.random.normal(0, price * 0.1)
        price = price + noise
        price = np.maximum(price, 50000)  # 最低价格限制
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'district': district,
            'floor': floor,
            'orientation': orientation,
            'decoration': decoration,
            'price': price
        })
        
        print(f"生成数据集大小: {self.data.shape}")
        print(f"特征数量: {self.data.shape[1] - 1}")
        print(f"价格范围: {self.data['price'].min():.0f} - {self.data['price'].max():.0f}")
        
        return self.data
    
    def exploratory_data_analysis(self):
        """探索性数据分析"""
        print("\n🔍 探索性数据分析")
        print("=" * 30)
        
        # 基本统计信息
        print("1. 数据基本信息:")
        print(self.data.info())
        print(f"\n数值特征统计:")
        print(self.data.describe())
        
        # 可视化分析
        plt.figure(figsize=(20, 15))
        
        # 价格分布
        plt.subplot(3, 4, 1)
        plt.hist(self.data['price'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('房价分布')
        plt.xlabel('价格')
        plt.ylabel('频数')
        
        # 面积vs价格
        plt.subplot(3, 4, 2)
        plt.scatter(self.data['area'], self.data['price'], alpha=0.6)
        plt.title('面积 vs 价格')
        plt.xlabel('面积 (平米)')
        plt.ylabel('价格')
        
        # 房龄vs价格
        plt.subplot(3, 4, 3)
        plt.scatter(self.data['age'], self.data['price'], alpha=0.6)
        plt.title('房龄 vs 价格')
        plt.xlabel('房龄 (年)')
        plt.ylabel('价格')
        
        # 区域价格箱线图
        plt.subplot(3, 4, 4)
        self.data.boxplot(column='price', by='district', ax=plt.gca())
        plt.title('不同区域的价格分布')
        plt.xticks(rotation=45)
        
        # 卧室数量vs价格
        plt.subplot(3, 4, 5)
        bedroom_price = self.data.groupby('bedrooms')['price'].mean()
        plt.bar(bedroom_price.index, bedroom_price.values, alpha=0.7)
        plt.title('卧室数量 vs 平均价格')
        plt.xlabel('卧室数量')
        plt.ylabel('平均价格')
        
        # 装修状况vs价格
        plt.subplot(3, 4, 6)
        decoration_price = self.data.groupby('decoration')['price'].mean()
        plt.bar(decoration_price.index, decoration_price.values, alpha=0.7)
        plt.title('装修状况 vs 平均价格')
        plt.xlabel('装修状况')
        plt.ylabel('平均价格')
        plt.xticks(rotation=45)
        
        # 相关性热力图
        plt.subplot(3, 4, 7)
        numeric_cols = ['area', 'bedrooms', 'bathrooms', 'age', 'floor', 'price']
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=plt.gca())
        plt.title('特征相关性热力图')
        
        # 楼层vs价格
        plt.subplot(3, 4, 8)
        floor_bins = pd.cut(self.data['floor'], bins=5)
        floor_price = self.data.groupby(floor_bins)['price'].mean()
        plt.bar(range(len(floor_price)), floor_price.values, alpha=0.7)
        plt.title('楼层 vs 平均价格')
        plt.xlabel('楼层区间')
        plt.ylabel('平均价格')
        plt.xticks(range(len(floor_price)), 
                  [f'{int(interval.left)}-{int(interval.right)}' 
                   for interval in floor_price.index], rotation=45)
        
        # 朝向vs价格
        plt.subplot(3, 4, 9)
        orientation_price = self.data.groupby('orientation')['price'].mean().sort_values(ascending=False)
        plt.bar(orientation_price.index, orientation_price.values, alpha=0.7)
        plt.title('朝向 vs 平均价格')
        plt.xlabel('朝向')
        plt.ylabel('平均价格')
        plt.xticks(rotation=45)
        
        # 价格对数分布
        plt.subplot(3, 4, 10)
        plt.hist(np.log(self.data['price']), bins=50, alpha=0.7, edgecolor='black')
        plt.title('价格对数分布')
        plt.xlabel('log(价格)')
        plt.ylabel('频数')
        
        # 面积分布
        plt.subplot(3, 4, 11)
        plt.hist(self.data['area'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('面积分布')
        plt.xlabel('面积 (平米)')
        plt.ylabel('频数')
        
        # 价格vs面积（按区域着色）
        plt.subplot(3, 4, 12)
        districts = self.data['district'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(districts)))
        
        for district, color in zip(districts, colors):
            mask = self.data['district'] == district
            plt.scatter(self.data[mask]['area'], self.data[mask]['price'], 
                       alpha=0.6, label=district, color=color)
        
        plt.title('面积 vs 价格 (按区域)')
        plt.xlabel('面积 (平米)')
        plt.ylabel('价格')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # 统计分析
        print(f"\n2. 关键发现:")
        print(f"• 面积与价格相关系数: {self.data['area'].corr(self.data['price']):.3f}")
        print(f"• 房龄与价格相关系数: {self.data['age'].corr(self.data['price']):.3f}")
        print(f"• 最贵区域: {self.data.groupby('district')['price'].mean().idxmax()}")
        print(f"• 最便宜区域: {self.data.groupby('district')['price'].mean().idxmin()}")
        
        return self.data
    
    def feature_engineering(self):
        """特征工程"""
        print("\n🔧 特征工程")
        print("=" * 30)
        
        # 创建新特征
        self.data['price_per_sqm'] = self.data['price'] / self.data['area']
        self.data['room_ratio'] = self.data['bathrooms'] / self.data['bedrooms']
        self.data['is_new'] = (self.data['age'] < 5).astype(int)
        self.data['is_high_floor'] = (self.data['floor'] > 15).astype(int)
        self.data['is_good_orientation'] = self.data['orientation'].isin(['南', '东南']).astype(int)
        
        # 对分类变量进行编码
        le_district = LabelEncoder()
        le_orientation = LabelEncoder()
        le_decoration = LabelEncoder()
        
        self.data['district_encoded'] = le_district.fit_transform(self.data['district'])
        self.data['orientation_encoded'] = le_orientation.fit_transform(self.data['orientation'])
        self.data['decoration_encoded'] = le_decoration.fit_transform(self.data['decoration'])
        
        # 保存编码器
        self.encoders = {
            'district': le_district,
            'orientation': le_orientation,
            'decoration': le_decoration
        }
        
        print("新增特征:")
        print("• price_per_sqm: 每平米价格")
        print("• room_ratio: 卫生间与卧室比例")
        print("• is_new: 是否为新房 (房龄<5年)")
        print("• is_high_floor: 是否为高楼层 (>15层)")
        print("• is_good_orientation: 是否为好朝向 (南/东南)")
        
        # 选择建模特征
        self.feature_columns = [
            'area', 'bedrooms', 'bathrooms', 'age', 'floor',
            'district_encoded', 'orientation_encoded', 'decoration_encoded',
            'room_ratio', 'is_new', 'is_high_floor', 'is_good_orientation'
        ]
        
        print(f"\n建模特征数量: {len(self.feature_columns)}")
        
        return self.data
    
    def prepare_data(self):
        """准备训练数据"""
        print("\n📋 准备训练数据")
        print("=" * 30)
        
        # 准备特征和目标变量
        X = self.data[self.feature_columns]
        y = self.data['price']
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print(f"特征已标准化")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """训练多个模型"""
        print("\n🤖 训练多个模型")
        print("=" * 30)
        
        # 定义模型
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # 训练和评估模型
        for name, model in self.models.items():
            print(f"\n训练 {name}...")
            
            # 训练模型
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # 计算评估指标
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # 交叉验证
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='r2')
            
            # 保存结果
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        return self.results
    
    def model_comparison(self):
        """模型比较和可视化"""
        print("\n📊 模型比较")
        print("=" * 30)
        
        # 创建结果对比表
        comparison_df = pd.DataFrame({
            name: {
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2'],
                'CV R²': results['cv_mean']
            }
            for name, results in self.results.items()
        }).T
        
        print("模型性能对比:")
        print(comparison_df.round(4))
        
        # 找到最佳模型
        best_model_name = comparison_df['R²'].idxmax()
        self.best_model = self.results[best_model_name]['model']
        print(f"\n最佳模型: {best_model_name}")
        
        # 可视化模型比较
        plt.figure(figsize=(15, 10))
        
        # R²得分比较
        plt.subplot(2, 3, 1)
        r2_scores = [results['r2'] for results in self.results.values()]
        model_names = list(self.results.keys())
        
        bars = plt.bar(model_names, r2_scores, alpha=0.7)
        plt.title('R²得分比较')
        plt.ylabel('R²得分')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 标注最佳模型
        best_idx = r2_scores.index(max(r2_scores))
        bars[best_idx].set_color('red')
        
        # RMSE比较
        plt.subplot(2, 3, 2)
        rmse_scores = [results['rmse'] for results in self.results.values()]
        plt.bar(model_names, rmse_scores, alpha=0.7)
        plt.title('RMSE比较')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 预测vs真实值（最佳模型）
        plt.subplot(2, 3, 3)
        best_predictions = self.results[best_model_name]['predictions']
        plt.scatter(self.y_test, best_predictions, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('真实价格')
        plt.ylabel('预测价格')
        plt.title(f'{best_model_name} - 预测vs真实')
        plt.grid(True, alpha=0.3)
        
        # 残差分析
        plt.subplot(2, 3, 4)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测价格')
        plt.ylabel('残差')
        plt.title(f'{best_model_name} - 残差分析')
        plt.grid(True, alpha=0.3)
        
        # 特征重要性（如果是树模型）
        plt.subplot(2, 3, 5)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.feature_columns
            
            # 排序
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(importances)), importances[indices], alpha=0.7)
            plt.title('特征重要性')
            plt.ylabel('重要性')
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], rotation=45)
        else:
            plt.text(0.5, 0.5, '该模型不支持\n特征重要性分析', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('特征重要性')
        
        # 交叉验证得分分布
        plt.subplot(2, 3, 6)
        cv_means = [results['cv_mean'] for results in self.results.values()]
        cv_stds = [results['cv_std'] for results in self.results.values()]
        
        plt.errorbar(range(len(model_names)), cv_means, yerr=cv_stds, 
                    fmt='o', capsize=5, capthick=2)
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.title('交叉验证R²得分')
        plt.ylabel('CV R²得分')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def run_complete_project(self):
        """运行完整项目"""
        print("🏠 房价预测完整项目流程")
        print("=" * 60)
        
        # 1. 数据生成
        self.generate_synthetic_data()
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 特征工程
        self.feature_engineering()
        
        # 4. 数据准备
        self.prepare_data()
        
        # 5. 模型训练
        self.train_models()
        
        # 6. 模型比较
        comparison_results = self.model_comparison()
        
        print(f"\n🎉 项目完成！")
        print(f"\n📊 项目总结:")
        print(f"• 数据集大小: {self.data.shape[0]} 样本")
        print(f"• 特征数量: {len(self.feature_columns)}")
        print(f"• 最佳模型: {comparison_results['R²'].idxmax()}")
        print(f"• 最佳R²得分: {comparison_results['R²'].max():.4f}")
        
        print(f"\n💡 项目亮点:")
        print("1. 完整的数据科学流程")
        print("2. 合成数据生成技术")
        print("3. 全面的探索性数据分析")
        print("4. 系统的特征工程")
        print("5. 多模型比较和评估")
        print("6. 可视化分析和解释")

def main():
    """主函数"""
    project = HousePricePrediction()
    project.run_complete_project()
    
    print("\n🚀 下一步建议:")
    print("1. 尝试更多特征工程技术")
    print("2. 实验不同的模型和超参数")
    print("3. 添加模型解释性分析")
    print("4. 部署模型为Web服务")

if __name__ == "__main__":
    main()
