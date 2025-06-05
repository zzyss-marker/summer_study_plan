"""
监督学习算法实现
从零开始实现主要的监督学习算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

class SupervisedLearning:
    """监督学习算法实现类"""
    
    def __init__(self):
        self.algorithms_implemented = []
        print("🤖 监督学习算法实现系统")
        print("=" * 50)
    
    def linear_regression_implementation(self):
        """线性回归实现"""
        print("📈 线性回归算法实现")
        print("=" * 30)
        
        class LinearRegression:
            def __init__(self, learning_rate=0.01, max_iterations=1000):
                self.learning_rate = learning_rate
                self.max_iterations = max_iterations
                self.weights = None
                self.bias = None
                self.cost_history = []
            
            def fit(self, X, y):
                n_samples, n_features = X.shape
                self.weights = np.zeros(n_features)
                self.bias = 0
                
                for i in range(self.max_iterations):
                    # 前向传播
                    y_pred = self.predict(X)
                    
                    # 计算损失
                    cost = np.mean((y - y_pred) ** 2)
                    self.cost_history.append(cost)
                    
                    # 计算梯度
                    dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
                    db = (1/n_samples) * np.sum(y_pred - y)
                    
                    # 更新参数
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            
            def predict(self, X):
                return np.dot(X, self.weights) + self.bias
        
        # 生成回归数据
        X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"均方误差: {mse:.4f}")
        print(f"R²得分: {r2:.4f}")
        print(f"权重: {model.weights[0]:.4f}")
        print(f"偏置: {model.bias:.4f}")
        
        # 可视化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(X_test, y_test, alpha=0.6, label='真实值')
        plt.scatter(X_test, y_pred, alpha=0.6, label='预测值')
        plt.xlabel('特征')
        plt.ylabel('目标值')
        plt.title('线性回归预测结果')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(model.cost_history)
        plt.title('损失函数变化')
        plt.xlabel('迭代次数')
        plt.ylabel('均方误差')
        
        plt.subplot(1, 3, 3)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测vs真实')
        
        plt.tight_layout()
        plt.show()
        
        self.algorithms_implemented.append("线性回归")
    
    def logistic_regression_implementation(self):
        """逻辑回归实现"""
        print("\n🎯 逻辑回归算法实现")
        print("=" * 30)
        
        class LogisticRegression:
            def __init__(self, learning_rate=0.01, max_iterations=1000):
                self.learning_rate = learning_rate
                self.max_iterations = max_iterations
                self.weights = None
                self.bias = None
                self.cost_history = []
            
            def sigmoid(self, z):
                z = np.clip(z, -500, 500)
                return 1 / (1 + np.exp(-z))
            
            def fit(self, X, y):
                n_samples, n_features = X.shape
                self.weights = np.zeros(n_features)
                self.bias = 0
                
                for i in range(self.max_iterations):
                    # 前向传播
                    linear_pred = np.dot(X, self.weights) + self.bias
                    predictions = self.sigmoid(linear_pred)
                    
                    # 计算损失
                    cost = self.compute_cost(y, predictions)
                    self.cost_history.append(cost)
                    
                    # 计算梯度
                    dw = (1/n_samples) * np.dot(X.T, (predictions - y))
                    db = (1/n_samples) * np.sum(predictions - y)
                    
                    # 更新参数
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            
            def compute_cost(self, y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
            def predict(self, X):
                linear_pred = np.dot(X, self.weights) + self.bias
                y_pred = self.sigmoid(linear_pred)
                return [0 if y <= 0.5 else 1 for y in y_pred]
            
            def predict_proba(self, X):
                linear_pred = np.dot(X, self.weights) + self.bias
                return self.sigmoid(linear_pred)
        
        # 生成分类数据
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
        model.fit(X_train_scaled, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"权重: {model.weights}")
        print(f"偏置: {model.bias:.4f}")
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 真实分类
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('真实分类')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        
        # 预测概率
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_proba, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('预测概率')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        
        # 损失函数变化
        plt.subplot(1, 3, 3)
        plt.plot(model.cost_history)
        plt.title('损失函数变化')
        plt.xlabel('迭代次数')
        plt.ylabel('交叉熵损失')
        
        plt.tight_layout()
        plt.show()
        
        self.algorithms_implemented.append("逻辑回归")
    
    def knn_implementation(self):
        """K近邻算法实现"""
        print("\n👥 K近邻算法实现")
        print("=" * 30)
        
        class KNearestNeighbors:
            def __init__(self, k=3):
                self.k = k
                self.X_train = None
                self.y_train = None
            
            def fit(self, X, y):
                self.X_train = X
                self.y_train = y
            
            def euclidean_distance(self, x1, x2):
                return np.sqrt(np.sum((x1 - x2) ** 2))
            
            def predict(self, X):
                predictions = []
                for sample in X:
                    # 计算到所有训练样本的距离
                    distances = []
                    for i, train_sample in enumerate(self.X_train):
                        distance = self.euclidean_distance(sample, train_sample)
                        distances.append((distance, self.y_train[i]))
                    
                    # 找到k个最近邻
                    distances.sort(key=lambda x: x[0])
                    k_nearest = distances[:self.k]
                    
                    # 投票决定类别
                    k_nearest_labels = [label for _, label in k_nearest]
                    prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
                    predictions.append(prediction)
                
                return predictions
        
        # 生成分类数据
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 测试不同的k值
        k_values = [1, 3, 5, 7, 9]
        accuracies = []
        
        for k in k_values:
            knn = KNearestNeighbors(k=k)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"k={k}, 准确率: {accuracy:.4f}")
        
        # 可视化k值对准确率的影响
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('k值')
        plt.ylabel('准确率')
        plt.title('k值对KNN准确率的影响')
        plt.grid(True, alpha=0.3)
        
        # 可视化最佳k值的结果
        best_k = k_values[np.argmax(accuracies)]
        knn_best = KNearestNeighbors(k=best_k)
        knn_best.fit(X_train_scaled, y_train)
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                            c=y_train, cmap='viridis', alpha=0.6)
        plt.xlabel('特征1 (标准化)')
        plt.ylabel('特征2 (标准化)')
        plt.title(f'KNN分类 (k={best_k})')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        self.algorithms_implemented.append("K近邻")
    
    def run_all_algorithms(self):
        """运行所有监督学习算法"""
        print("🎓 监督学习算法完整实现")
        print("=" * 60)
        
        self.linear_regression_implementation()
        self.logistic_regression_implementation()
        self.knn_implementation()
        
        print(f"\n🎉 监督学习算法实现完成！")
        print(f"已实现的算法: {', '.join(self.algorithms_implemented)}")
        
        print(f"\n📚 算法总结:")
        print("1. 线性回归 - 连续值预测，最小二乘法")
        print("2. 逻辑回归 - 二分类，sigmoid函数")
        print("3. K近邻 - 基于距离的分类，懒惰学习")

def main():
    """主函数"""
    supervised = SupervisedLearning()
    supervised.run_all_algorithms()
    
    print("\n💡 算法选择指南:")
    print("1. 线性回归 - 线性关系，可解释性强")
    print("2. 逻辑回归 - 线性分类边界，概率输出")
    print("3. K近邻 - 非参数，局部模式，需要大量数据")

if __name__ == "__main__":
    main()
