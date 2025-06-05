"""
线性回归算法实现演示
从零开始实现线性回归，理解梯度下降原理
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示

class LinearRegression:
    """从零实现线性回归"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """训练模型"""
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iterations):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算损失
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, y_true, y_pred):
        """计算均方误差损失"""
        return np.mean((y_true - y_pred) ** 2)
    
    def plot_cost_history(self):
        """绘制损失函数变化"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('损失函数变化')
        plt.xlabel('迭代次数')
        plt.ylabel('均方误差')
        plt.grid(True)
        plt.show()

def demo_linear_regression():
    """线性回归演示"""
    print("🤖 线性回归算法演示")
    print("=" * 50)
    
    # 生成示例数据
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"均方误差: {mse:.4f}")
    print(f"R²得分: {r2:.4f}")
    print(f"权重: {model.weights[0]:.4f}")
    print(f"偏置: {model.bias:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 训练数据
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
    plt.xlabel('特征')
    plt.ylabel('目标值')
    plt.title('训练数据分布')
    plt.legend()
    
    # 预测结果
    plt.subplot(1, 3, 2)
    plt.scatter(X_test, y_test, alpha=0.6, label='真实值')
    plt.scatter(X_test, y_pred, alpha=0.6, label='预测值')
    plt.xlabel('特征')
    plt.ylabel('目标值')
    plt.title('预测结果对比')
    plt.legend()
    
    # 损失函数
    plt.subplot(1, 3, 3)
    plt.plot(model.cost_history)
    plt.title('损失函数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_linear_regression()
    print("\n🎉 线性回归学习完成！")
    print("💡 学习要点：")
    print("   - 理解梯度下降算法原理")
    print("   - 掌握损失函数的计算")
    print("   - 学会模型评估指标")
