"""
线性代数基础
机器学习必备的线性代数概念和计算
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearAlgebraFundamentals:
    """线性代数基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("📐 线性代数基础学习系统")
        print("=" * 50)
    
    def vector_operations(self):
        """向量运算"""
        print("📐 向量运算基础")
        print("=" * 30)
        
        # 1. 向量创建和基本属性
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        
        print("1. 向量基础:")
        print(f"向量v1: {v1}")
        print(f"向量v2: {v2}")
        print(f"v1的维度: {v1.shape}")
        print(f"v1的长度(范数): {np.linalg.norm(v1):.4f}")
        
        # 2. 向量运算
        print(f"\n2. 向量运算:")
        print(f"向量加法 v1 + v2: {v1 + v2}")
        print(f"向量减法 v1 - v2: {v1 - v2}")
        print(f"标量乘法 2 * v1: {2 * v1}")
        
        # 3. 点积(内积)
        dot_product = np.dot(v1, v2)
        print(f"\n3. 点积:")
        print(f"v1 · v2 = {dot_product}")
        
        # 4. 向量夹角
        cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        print(f"\n4. 向量夹角:")
        print(f"cos(θ) = {cos_angle:.4f}")
        print(f"夹角 = {angle_deg:.2f}度")
        
        # 5. 叉积(仅适用于3D向量)
        cross_product = np.cross(v1, v2)
        print(f"\n5. 叉积:")
        print(f"v1 × v2 = {cross_product}")
        
        # 6. 单位向量
        unit_v1 = v1 / np.linalg.norm(v1)
        print(f"\n6. 单位向量:")
        print(f"v1的单位向量: {unit_v1}")
        print(f"单位向量的模长: {np.linalg.norm(unit_v1):.4f}")
        
        self.examples_completed.append("向量运算")
    
    def matrix_operations(self):
        """矩阵运算"""
        print("\n🔢 矩阵运算")
        print("=" * 30)
        
        # 1. 矩阵创建
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        
        B = np.array([[9, 8, 7],
                      [6, 5, 4],
                      [3, 2, 1]])
        
        print("1. 矩阵基础:")
        print(f"矩阵A:\n{A}")
        print(f"矩阵B:\n{B}")
        print(f"A的形状: {A.shape}")
        print(f"A的秩: {np.linalg.matrix_rank(A)}")
        
        # 2. 矩阵基本运算
        print(f"\n2. 矩阵基本运算:")
        print(f"矩阵加法 A + B:\n{A + B}")
        print(f"标量乘法 2 * A:\n{2 * A}")
        print(f"元素级乘法 A * B:\n{A * B}")
        
        # 3. 矩阵乘法
        print(f"\n3. 矩阵乘法:")
        C = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        D = np.array([[7, 8, 9],
                      [10, 11, 12]])
        
        print(f"矩阵C (3x2):\n{C}")
        print(f"矩阵D (2x3):\n{D}")
        print(f"C @ D (3x3):\n{C @ D}")
        
        # 4. 矩阵转置
        print(f"\n4. 矩阵转置:")
        print(f"A的转置:\n{A.T}")
        
        # 5. 特殊矩阵
        print(f"\n5. 特殊矩阵:")
        I = np.eye(3)  # 单位矩阵
        zeros = np.zeros((3, 3))  # 零矩阵
        
        print(f"3x3单位矩阵:\n{I}")
        print(f"3x3零矩阵:\n{zeros}")
        
        # 6. 矩阵的迹
        trace_A = np.trace(A)
        print(f"\n6. 矩阵的迹:")
        print(f"tr(A) = {trace_A}")
        
        self.examples_completed.append("矩阵运算")
    
    def eigenvalue_decomposition(self):
        """特征值分解"""
        print("\n🔍 特征值分解")
        print("=" * 30)
        
        # 创建一个对称矩阵
        A = np.array([[4, 2, 1],
                      [2, 5, 3],
                      [1, 3, 6]], dtype=float)
        
        print("1. 原矩阵:")
        print(f"矩阵A:\n{A}")
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(A)
        print(f"\n2. 特征值分解:")
        print(f"特征值: {eigenvalues}")
        print(f"特征向量:\n{eigenvectors}")
        
        # 验证特征值分解
        for i in range(len(eigenvalues)):
            lambda_i = eigenvalues[i]
            v_i = eigenvectors[:, i]
            Av = A @ v_i
            lambda_v = lambda_i * v_i
            print(f"验证 Av{i+1} = λ{i+1}v{i+1}: {np.allclose(Av, lambda_v)}")
        
        self.examples_completed.append("特征值分解")
    
    def linear_systems(self):
        """线性方程组求解"""
        print("\n📊 线性方程组求解")
        print("=" * 30)
        
        # 线性方程组: Ax = b
        # 2x + 3y + z = 1
        # x + 4y + 2z = 2  
        # 3x + y + 5z = 3
        
        A = np.array([[2, 3, 1],
                      [1, 4, 2],
                      [3, 1, 5]], dtype=float)
        
        b = np.array([1, 2, 3], dtype=float)
        
        print("1. 线性方程组:")
        print("2x + 3y + z = 1")
        print("x + 4y + 2z = 2")
        print("3x + y + 5z = 3")
        print(f"\n系数矩阵A:\n{A}")
        print(f"常数向量b: {b}")
        
        # 直接求解
        print(f"\n2. 直接求解:")
        x = np.linalg.solve(A, b)
        print(f"解向量x: {x}")
        
        # 验证解
        verification = A @ x
        print(f"验证 Ax: {verification}")
        print(f"验证 Ax = b: {np.allclose(verification, b)}")
        
        # 条件数分析
        print(f"\n3. 条件数分析:")
        cond_A = np.linalg.cond(A)
        print(f"矩阵A的条件数: {cond_A:.2f}")
        
        if cond_A < 100:
            print("矩阵条件良好")
        elif cond_A < 1000:
            print("矩阵条件一般")
        else:
            print("矩阵条件较差")
        
        self.examples_completed.append("线性方程组")
    
    def pca_application(self):
        """PCA应用示例"""
        print("\n🎯 PCA应用示例")
        print("=" * 30)
        
        # 生成相关的2D数据
        np.random.seed(42)
        mean = [0, 0]
        cov = [[3, 1.5], [1.5, 1]]
        data = np.random.multivariate_normal(mean, cov, 200)
        
        print(f"数据形状: {data.shape}")
        print(f"数据均值: {np.mean(data, axis=0)}")
        
        # 计算协方差矩阵
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        print(f"协方差矩阵:\n{cov_matrix}")
        
        # 特征值分解
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # 按特征值大小排序
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        print(f"特征值: {eigenvals}")
        print(f"方差解释比例: {eigenvals / np.sum(eigenvals)}")
        
        # 可视化PCA
        plt.figure(figsize=(12, 5))
        
        # 原始数据
        plt.subplot(1, 2, 1)
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
        
        # 绘制主成分方向
        mean_point = np.mean(data, axis=0)
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
            plt.arrow(mean_point[0], mean_point[1], 
                     eigenvec[0] * np.sqrt(eigenval) * 2, 
                     eigenvec[1] * np.sqrt(eigenval) * 2,
                     head_width=0.1, head_length=0.1, 
                     fc=f'C{i}', ec=f'C{i}', 
                     label=f'PC{i+1} (λ={eigenval:.2f})')
        
        plt.title('原始数据与主成分')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 投影到主成分
        plt.subplot(1, 2, 2)
        data_pca = data_centered @ eigenvecs
        
        plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6)
        plt.title('主成分空间中的数据')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        print("已生成PCA可视化图表")
        self.examples_completed.append("PCA应用")
    
    def run_all_examples(self):
        """运行所有示例"""
        print("📐 线性代数基础完整学习")
        print("=" * 60)
        
        self.vector_operations()
        self.matrix_operations()
        self.eigenvalue_decomposition()
        self.linear_systems()
        self.pca_application()
        
        print(f"\n🎉 线性代数基础学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. 向量运算 - 点积、叉积、投影等基础操作")
        print("2. 矩阵运算 - 加法、乘法、转置等矩阵操作")
        print("3. 特征值分解 - 理解矩阵的本质特性")
        print("4. 线性方程组 - 直接求解和条件数分析")
        print("5. PCA应用 - 降维和数据可视化")

def main():
    """主函数"""
    linear_algebra = LinearAlgebraFundamentals()
    linear_algebra.run_all_examples()
    
    print("\n💡 机器学习中的应用:")
    print("1. 向量运算 - 特征向量、梯度计算")
    print("2. 矩阵运算 - 权重矩阵、数据变换")
    print("3. 特征值分解 - PCA降维、谱聚类")
    print("4. 线性方程组 - 最小二乘回归")

if __name__ == "__main__":
    main()
