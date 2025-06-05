"""
AI学习完整演示
包含所有模块的核心概念和代码示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class CompleteAILearningDemo:
    """AI学习完整演示类"""
    
    def __init__(self):
        print("🎓 AI学习完整演示")
        print("=" * 60)
        print("这个演示包含了AI学习的所有核心概念")
        print("包括：Python基础、数学基础、机器学习、深度学习、项目实战")
        print("=" * 60)
    
    def python_fundamentals_demo(self):
        """Python基础演示"""
        print("\n🐍 Python基础强化演示")
        print("=" * 50)
        
        # 1. 装饰器示例
        def timer(func):
            import time
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                print(f"函数 {func.__name__} 执行时间: {end-start:.4f}秒")
                return result
            return wrapper
        
        @timer
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        print("1. 装饰器示例 - 计时器:")
        result = fibonacci(10)
        print(f"fibonacci(10) = {result}")
        
        # 2. 生成器示例
        def number_generator(n):
            for i in range(n):
                yield i ** 2
        
        print(f"\n2. 生成器示例:")
        squares = list(number_generator(5))
        print(f"前5个平方数: {squares}")
        
        # 3. NumPy基础
        print(f"\n3. NumPy基础操作:")
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        print(f"数组:\n{arr}")
        print(f"形状: {arr.shape}")
        print(f"转置:\n{arr.T}")
        print(f"统计: 均值={np.mean(arr):.2f}, 标准差={np.std(arr):.2f}")
        
        # 4. Pandas基础
        print(f"\n4. Pandas基础操作:")
        data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [85, 92, 78]
        }
        df = pd.DataFrame(data)
        print(f"DataFrame:\n{df}")
        print(f"年龄大于25的记录:\n{df[df['age'] > 25]}")
        
        print("✅ Python基础演示完成")
    
    def math_foundations_demo(self):
        """数学基础演示"""
        print("\n📐 数学基础演示")
        print("=" * 50)
        
        # 1. 线性代数
        print("1. 线性代数基础:")
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        print(f"矩阵A:\n{A}")
        print(f"矩阵B:\n{B}")
        print(f"矩阵乘法 A@B:\n{A @ B}")
        
        # 特征值分解
        eigenvals, eigenvecs = np.linalg.eig(A)
        print(f"A的特征值: {eigenvals}")
        print(f"A的特征向量:\n{eigenvecs}")
        
        # 2. 概率统计
        print(f"\n2. 概率统计基础:")
        # 生成正态分布数据
        data = np.random.normal(100, 15, 1000)
        print(f"正态分布数据统计:")
        print(f"均值: {np.mean(data):.2f}")
        print(f"标准差: {np.std(data):.2f}")
        print(f"95%置信区间: [{np.percentile(data, 2.5):.2f}, {np.percentile(data, 97.5):.2f}]")
        
        # 3. 可视化数学概念
        plt.figure(figsize=(12, 4))
        
        # 特征向量可视化
        plt.subplot(1, 3, 1)
        origin = [0, 0]
        plt.quiver(*origin, eigenvecs[0, 0], eigenvecs[1, 0], scale=1, scale_units='xy', angles='xy', color='red', label='特征向量1')
        plt.quiver(*origin, eigenvecs[0, 1], eigenvecs[1, 1], scale=1, scale_units='xy', angles='xy', color='blue', label='特征向量2')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid(True)
        plt.legend()
        plt.title('特征向量可视化')
        
        # 正态分布
        plt.subplot(1, 3, 2)
        plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
        x = np.linspace(data.min(), data.max(), 100)
        y = (1/np.sqrt(2*np.pi*15**2)) * np.exp(-0.5*((x-100)/15)**2)
        plt.plot(x, y, 'r-', linewidth=2, label='理论正态分布')
        plt.title('正态分布')
        plt.legend()
        
        # 函数图像
        plt.subplot(1, 3, 3)
        x = np.linspace(-5, 5, 100)
        sigmoid = 1 / (1 + np.exp(-x))
        plt.plot(x, sigmoid, label='Sigmoid')
        plt.plot(x, np.maximum(0, x), label='ReLU')
        plt.title('激活函数')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ 数学基础演示完成")
    
    def machine_learning_demo(self):
        """机器学习演示"""
        print("\n🤖 机器学习演示")
        print("=" * 50)
        
        # 1. 监督学习 - 回归
        print("1. 线性回归演示:")
        X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"R²得分: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
        # 2. 监督学习 - 分类
        print(f"\n2. 逻辑回归演示:")
        X_clf, y_clf = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                         n_informative=2, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        log_reg = LogisticRegression()
        log_reg.fit(X_train_scaled, y_train)
        y_pred_clf = log_reg.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred_clf)
        print(f"分类准确率: {accuracy:.4f}")
        
        # 3. 无监督学习 - 聚类
        print(f"\n3. K-means聚类演示:")
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X_clf)
        
        print(f"聚类中心:\n{kmeans.cluster_centers_}")
        
        # 4. 降维
        print(f"\n4. PCA降维演示:")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_clf)
        
        print(f"主成分方差解释比例: {pca.explained_variance_ratio_}")
        print(f"累计方差解释比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # 可视化机器学习结果
        plt.figure(figsize=(15, 5))
        
        # 回归结果
        plt.subplot(1, 3, 1)
        plt.scatter(X_test, y_test, alpha=0.6, label='真实值')
        plt.scatter(X_test, y_pred, alpha=0.6, label='预测值')
        plt.title(f'线性回归 (R²={r2:.3f})')
        plt.legend()
        
        # 分类结果
        plt.subplot(1, 3, 2)
        plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='viridis', alpha=0.6)
        plt.title(f'逻辑回归分类 (准确率={accuracy:.3f})')
        
        # 聚类结果
        plt.subplot(1, 3, 3)
        plt.scatter(X_clf[:, 0], X_clf[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        plt.title('K-means聚类')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ 机器学习演示完成")
    
    def deep_learning_demo(self):
        """深度学习基础演示"""
        print("\n🧠 深度学习基础演示")
        print("=" * 50)
        
        # 1. 激活函数
        print("1. 激活函数演示:")
        x = np.linspace(-5, 5, 100)
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def relu(x):
            return np.maximum(0, x)
        
        def tanh(x):
            return np.tanh(x)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
        plt.title('Sigmoid激活函数')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(x, relu(x), label='ReLU', linewidth=2, color='red')
        plt.title('ReLU激活函数')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(x, tanh(x), label='Tanh', linewidth=2, color='green')
        plt.title('Tanh激活函数')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 2. 简单神经网络概念
        print(f"\n2. 神经网络概念:")
        print("神经网络的基本组成:")
        print("- 输入层：接收数据")
        print("- 隐藏层：特征提取和变换")
        print("- 输出层：产生最终结果")
        print("- 激活函数：引入非线性")
        print("- 损失函数：衡量预测误差")
        print("- 优化器：更新网络参数")
        
        # 3. 梯度下降可视化
        print(f"\n3. 梯度下降可视化:")
        
        def cost_function(w):
            return (w - 2)**2 + 1
        
        def gradient(w):
            return 2 * (w - 2)
        
        # 梯度下降过程
        w = 0.0
        learning_rate = 0.1
        history = [w]
        
        for _ in range(20):
            grad = gradient(w)
            w = w - learning_rate * grad
            history.append(w)
        
        # 可视化
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        w_range = np.linspace(-1, 5, 100)
        cost_values = cost_function(w_range)
        plt.plot(w_range, cost_values, 'b-', linewidth=2, label='损失函数')
        plt.plot(history, [cost_function(w) for w in history], 'ro-', label='梯度下降路径')
        plt.xlabel('权重 w')
        plt.ylabel('损失')
        plt.title('梯度下降优化过程')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history, 'go-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('权重值')
        plt.title('权重收敛过程')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"最终权重: {history[-1]:.4f} (目标: 2.0)")
        print("✅ 深度学习基础演示完成")
    
    def project_demo(self):
        """项目实战演示"""
        print("\n🏠 项目实战演示 - 简化版房价预测")
        print("=" * 50)
        
        # 1. 生成模拟数据
        np.random.seed(42)
        n_samples = 500
        
        # 特征：面积、房间数、年龄
        area = np.random.normal(100, 30, n_samples)
        rooms = np.random.randint(1, 6, n_samples)
        age = np.random.randint(0, 30, n_samples)
        
        # 目标：价格（基于特征的线性组合加噪声）
        price = (area * 50 + rooms * 10000 - age * 500 + 
                np.random.normal(0, 5000, n_samples))
        price = np.maximum(price, 10000)  # 确保价格为正
        
        # 创建DataFrame
        data = pd.DataFrame({
            'area': area,
            'rooms': rooms,
            'age': age,
            'price': price
        })
        
        print("1. 数据探索:")
        print(data.describe())
        
        # 2. 特征工程
        data['price_per_sqm'] = data['price'] / data['area']
        data['is_new'] = (data['age'] < 5).astype(int)
        
        print(f"\n2. 特征工程完成，新增特征:")
        print("- price_per_sqm: 每平米价格")
        print("- is_new: 是否为新房")
        
        # 3. 模型训练
        features = ['area', 'rooms', 'age', 'is_new']
        X = data[features]
        y = data['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练多个模型
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print(f"\n3. 模型训练结果:")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {'r2': r2, 'rmse': rmse, 'predictions': y_pred}
            print(f"{name}: R²={r2:.4f}, RMSE={rmse:.0f}")
        
        # 4. 结果可视化
        plt.figure(figsize=(15, 5))
        
        # 数据分布
        plt.subplot(1, 3, 1)
        plt.hist(data['price'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('房价分布')
        plt.xlabel('价格')
        plt.ylabel('频数')
        
        # 面积vs价格
        plt.subplot(1, 3, 2)
        plt.scatter(data['area'], data['price'], alpha=0.6)
        plt.xlabel('面积')
        plt.ylabel('价格')
        plt.title('面积vs价格关系')
        
        # 预测vs真实
        plt.subplot(1, 3, 3)
        best_model = max(results.keys(), key=lambda k: results[k]['r2'])
        best_pred = results[best_model]['predictions']
        
        plt.scatter(y_test, best_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实价格')
        plt.ylabel('预测价格')
        plt.title(f'{best_model} - 预测vs真实')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n4. 项目总结:")
        print(f"最佳模型: {best_model}")
        print(f"模型性能: R²={results[best_model]['r2']:.4f}")
        print("✅ 项目实战演示完成")
    
    def run_complete_demo(self):
        """运行完整演示"""
        self.python_fundamentals_demo()
        self.math_foundations_demo()
        self.machine_learning_demo()
        self.deep_learning_demo()
        self.project_demo()
        
        print(f"\n🎉 AI学习完整演示结束！")
        print(f"\n📚 学习路径总结:")
        print("1. ✅ Python基础强化 - 掌握高级特性和数据处理")
        print("2. ✅ 数学基础复习 - 线性代数和概率统计")
        print("3. ✅ 机器学习基础 - 监督学习和无监督学习")
        print("4. ✅ 深度学习入门 - 神经网络基本概念")
        print("5. ✅ 项目实战练习 - 完整的机器学习项目")
        
        print(f"\n💡 下一步学习建议:")
        print("1. 深入学习每个模块的详细内容")
        print("2. 完成更多实际项目练习")
        print("3. 学习深度学习框架 (PyTorch/TensorFlow)")
        print("4. 参与开源项目和竞赛")
        print("5. 持续关注AI领域最新发展")

def main():
    """主函数"""
    demo = CompleteAILearningDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
