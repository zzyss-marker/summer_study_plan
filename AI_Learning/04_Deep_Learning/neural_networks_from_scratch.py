"""
从零开始实现神经网络
理解神经网络的基本原理和反向传播算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetworkFromScratch:
    """从零开始实现神经网络"""
    
    def __init__(self):
        self.examples_completed = []
        print("🧠 神经网络从零实现系统")
        print("=" * 50)
    
    def activation_functions_demo(self):
        """激活函数演示"""
        print("🧠 激活函数演示")
        print("=" * 30)
        
        class ActivationFunctions:
            @staticmethod
            def sigmoid(x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
            @staticmethod
            def sigmoid_derivative(x):
                s = ActivationFunctions.sigmoid(x)
                return s * (1 - s)
            
            @staticmethod
            def tanh(x):
                return np.tanh(x)
            
            @staticmethod
            def tanh_derivative(x):
                return 1 - np.tanh(x)**2
            
            @staticmethod
            def relu(x):
                return np.maximum(0, x)
            
            @staticmethod
            def relu_derivative(x):
                return (x > 0).astype(float)
        
        # 可视化激活函数
        x = np.linspace(-5, 5, 1000)
        
        plt.figure(figsize=(15, 10))
        
        activations = [
            ('Sigmoid', ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            ('Tanh', ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            ('ReLU', ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        ]
        
        for i, (name, func, deriv_func) in enumerate(activations):
            # 激活函数
            plt.subplot(2, 3, i+1)
            y = func(x)
            plt.plot(x, y, linewidth=2)
            plt.title(f'{name}激活函数')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.grid(True, alpha=0.3)
            
            # 导数
            plt.subplot(2, 3, i+4)
            y_deriv = deriv_func(x)
            plt.plot(x, y_deriv, linewidth=2, color='red')
            plt.title(f'{name}导数')
            plt.xlabel('x')
            plt.ylabel("f'(x)")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("激活函数特点:")
        print("1. Sigmoid: 输出范围(0,1)，存在梯度消失问题")
        print("2. Tanh: 输出范围(-1,1)，零中心化")
        print("3. ReLU: 计算简单，解决梯度消失，但存在神经元死亡")
        
        self.examples_completed.append("激活函数")
    
    def simple_perceptron(self):
        """简单感知机实现"""
        print("\n🔗 简单感知机实现")
        print("=" * 30)
        
        class Perceptron:
            def __init__(self, learning_rate=0.01, max_iterations=1000):
                self.learning_rate = learning_rate
                self.max_iterations = max_iterations
                self.weights = None
                self.bias = None
                self.errors = []
            
            def activation(self, x):
                """阶跃激活函数"""
                return 1 if x >= 0 else 0
            
            def fit(self, X, y):
                n_samples, n_features = X.shape
                
                # 初始化权重和偏置
                self.weights = np.random.normal(0, 0.01, n_features)
                self.bias = 0
                
                for iteration in range(self.max_iterations):
                    errors = 0
                    
                    for i in range(n_samples):
                        # 前向传播
                        linear_output = np.dot(X[i], self.weights) + self.bias
                        prediction = self.activation(linear_output)
                        
                        # 计算误差
                        error = y[i] - prediction
                        
                        if error != 0:
                            errors += 1
                            # 更新权重和偏置
                            self.weights += self.learning_rate * error * X[i]
                            self.bias += self.learning_rate * error
                    
                    self.errors.append(errors)
                    
                    # 如果没有错误，提前停止
                    if errors == 0:
                        print(f"感知机在第{iteration+1}次迭代后收敛")
                        break
            
            def predict(self, X):
                linear_output = np.dot(X, self.weights) + self.bias
                return [self.activation(x) for x in linear_output]
        
        # 生成线性可分数据
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, 
                                 random_state=42)
        
        # 将标签转换为0和1
        y = (y > 0).astype(int)
        
        # 训练感知机
        perceptron = Perceptron(learning_rate=0.1, max_iterations=100)
        perceptron.fit(X, y)
        
        # 预测
        predictions = perceptron.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"感知机准确率: {accuracy:.4f}")
        print(f"最终权重: {perceptron.weights}")
        print(f"最终偏置: {perceptron.bias}")
        
        # 可视化结果
        plt.figure(figsize=(12, 5))
        
        # 决策边界
        plt.subplot(1, 2, 1)
        
        # 绘制数据点
        colors = ['red', 'blue']
        for i in range(2):
            mask = y == i
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                       label=f'类别 {i}', alpha=0.7)
        
        # 绘制决策边界
        if perceptron.weights[1] != 0:
            x_boundary = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            y_boundary = -(perceptron.weights[0] * x_boundary + perceptron.bias) / perceptron.weights[1]
            plt.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='决策边界')
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('感知机分类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 错误数量变化
        plt.subplot(1, 2, 2)
        plt.plot(perceptron.errors, 'bo-')
        plt.xlabel('迭代次数')
        plt.ylabel('错误数量')
        plt.title('感知机收敛过程')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.examples_completed.append("感知机")
    
    def multilayer_perceptron(self):
        """多层感知机实现"""
        print("\n🧠 多层感知机实现")
        print("=" * 30)
        
        class MLP:
            def __init__(self, layers, learning_rate=0.01, max_iterations=1000):
                self.layers = layers
                self.learning_rate = learning_rate
                self.max_iterations = max_iterations
                self.weights = []
                self.biases = []
                self.costs = []
                
                # 初始化权重和偏置
                for i in range(len(layers) - 1):
                    w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
                    b = np.zeros((1, layers[i+1]))
                    self.weights.append(w)
                    self.biases.append(b)
            
            def sigmoid(self, x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
            def sigmoid_derivative(self, x):
                s = self.sigmoid(x)
                return s * (1 - s)
            
            def forward_propagation(self, X):
                """前向传播"""
                activations = [X]
                z_values = []
                
                for i in range(len(self.weights)):
                    z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                    z_values.append(z)
                    a = self.sigmoid(z)
                    activations.append(a)
                
                return activations, z_values
            
            def compute_cost(self, y_true, y_pred):
                """计算损失函数"""
                m = y_true.shape[0]
                cost = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-8) + 
                                     (1 - y_true) * np.log(1 - y_pred + 1e-8))
                return cost
            
            def fit(self, X, y):
                for iteration in range(self.max_iterations):
                    # 前向传播
                    activations, z_values = self.forward_propagation(X)
                    
                    # 计算损失
                    cost = self.compute_cost(y, activations[-1])
                    self.costs.append(cost)
                    
                    # 反向传播（简化版本）
                    m = X.shape[0]
                    
                    # 输出层误差
                    output_error = activations[-1] - y
                    
                    # 更新输出层权重
                    dW_output = (1/m) * np.dot(activations[-2].T, output_error)
                    db_output = (1/m) * np.sum(output_error, axis=0, keepdims=True)
                    
                    self.weights[-1] -= self.learning_rate * dW_output
                    self.biases[-1] -= self.learning_rate * db_output
                    
                    if iteration % 100 == 0:
                        print(f"迭代 {iteration}, 损失: {cost:.6f}")
            
            def predict(self, X):
                activations, _ = self.forward_propagation(X)
                return activations[-1]
            
            def predict_classes(self, X):
                predictions = self.predict(X)
                return (predictions > 0.5).astype(int)
        
        # 生成非线性可分数据
        X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=42)
        y = y.reshape(-1, 1)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        
        # 创建和训练MLP
        mlp = MLP(layers=[2, 10, 1], learning_rate=0.1, max_iterations=1000)
        mlp.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = mlp.predict(X_test)
        y_pred_classes = mlp.predict_classes(X_test)
        accuracy = np.mean(y_pred_classes == y_test)
        
        print(f"\nMLP准确率: {accuracy:.4f}")
        
        # 可视化结果
        plt.figure(figsize=(15, 5))
        
        # 原始数据
        plt.subplot(1, 3, 1)
        colors = ['red', 'blue']
        for i in range(2):
            mask = y.flatten() == i
            plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=colors[i], 
                       label=f'类别 {i}', alpha=0.7)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('原始数据')
        plt.legend()
        
        # 决策边界
        plt.subplot(1, 3, 2)
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = mlp.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        
        for i in range(2):
            mask = y.flatten() == i
            plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=colors[i], 
                       label=f'类别 {i}', alpha=0.7, edgecolors='black')
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('MLP决策边界')
        plt.legend()
        
        # 损失函数变化
        plt.subplot(1, 3, 3)
        plt.plot(mlp.costs)
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.title('训练损失变化')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.examples_completed.append("多层感知机")
    
    def run_all_examples(self):
        """运行所有示例"""
        print("🧠 神经网络基础完整学习")
        print("=" * 60)
        
        self.activation_functions_demo()
        self.simple_perceptron()
        self.multilayer_perceptron()
        
        print(f"\n🎉 神经网络基础学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. 激活函数 - 引入非线性，各有优缺点")
        print("2. 感知机 - 最简单的神经网络，只能解决线性可分问题")
        print("3. 多层感知机 - 通过隐藏层解决非线性问题")

def main():
    """主函数"""
    nn = NeuralNetworkFromScratch()
    nn.run_all_examples()
    
    print("\n💡 深度学习发展:")
    print("1. 感知机 → 多层感知机 → 深度神经网络")
    print("2. 激活函数: Sigmoid → ReLU → 各种变体")
    print("3. 优化算法: SGD → Adam → 各种自适应方法")

if __name__ == "__main__":
    main()
