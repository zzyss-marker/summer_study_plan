"""
NumPy基础学习
数组操作、向量化计算、广播机制等核心概念
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class NumpyFundamentals:
    """NumPy基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("🔢 NumPy基础学习系统")
        print("=" * 50)
    
    def array_creation_examples(self):
        """数组创建示例"""
        print("🔢 NumPy数组创建")
        print("=" * 30)
        
        # 1. 从列表创建
        list_1d = [1, 2, 3, 4, 5]
        arr_1d = np.array(list_1d)
        print(f"1D数组: {arr_1d}")
        print(f"数据类型: {arr_1d.dtype}")
        print(f"形状: {arr_1d.shape}")
        
        # 2. 创建多维数组
        list_2d = [[1, 2, 3], [4, 5, 6]]
        arr_2d = np.array(list_2d)
        print(f"\n2D数组:\n{arr_2d}")
        print(f"形状: {arr_2d.shape}")
        print(f"维度: {arr_2d.ndim}")
        
        # 3. 特殊数组创建函数
        zeros = np.zeros((3, 4))
        ones = np.ones((2, 3))
        eye = np.eye(3)
        
        print(f"\n零数组:\n{zeros}")
        print(f"\n单位矩阵:\n{eye}")
        
        # 4. 数值范围数组
        arange_arr = np.arange(0, 10, 2)
        linspace_arr = np.linspace(0, 1, 5)
        
        print(f"\narange(0, 10, 2): {arange_arr}")
        print(f"linspace(0, 1, 5): {linspace_arr}")
        
        # 5. 随机数组
        np.random.seed(42)
        random_arr = np.random.random((2, 3))
        print(f"\n随机数组:\n{random_arr}")
        
        self.examples_completed.append("数组创建")
    
    def array_operations(self):
        """数组运算"""
        print("\n🧮 数组运算")
        print("=" * 30)
        
        # 创建示例数组
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        
        print(f"数组a: {a}")
        print(f"数组b: {b}")
        
        # 1. 基础算术运算
        print(f"\n基础算术运算:")
        print(f"a + b = {a + b}")
        print(f"a * b = {a * b}")  # 元素级乘法
        print(f"a ** 2 = {a ** 2}")
        
        # 2. 统计函数
        print(f"\n统计函数:")
        data = np.random.normal(0, 1, 100)
        print(f"均值: {np.mean(data):.4f}")
        print(f"标准差: {np.std(data):.4f}")
        print(f"最小值: {np.min(data):.4f}")
        print(f"最大值: {np.max(data):.4f}")
        
        self.examples_completed.append("数组运算")
    
    def broadcasting_examples(self):
        """广播机制示例"""
        print("\n📡 广播机制")
        print("=" * 30)
        
        # 1. 标量与数组的广播
        arr = np.array([1, 2, 3, 4])
        scalar = 10
        result1 = arr + scalar
        print(f"数组: {arr}")
        print(f"标量: {scalar}")
        print(f"广播结果: {result1}")
        
        # 2. 不同形状数组的广播
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
        arr_1d = np.array([10, 20, 30])
        
        print(f"\n2D数组 (2,3):\n{arr_2d}")
        print(f"1D数组 (3,): {arr_1d}")
        result2 = arr_2d + arr_1d
        print(f"广播结果:\n{result2}")
        
        self.examples_completed.append("广播机制")
    
    def vectorization_performance(self):
        """向量化性能对比"""
        print("\n⚡ 向量化性能对比")
        print("=" * 30)
        
        # 创建大数组
        size = 100000
        a = np.random.random(size)
        b = np.random.random(size)
        
        # 1. Python循环 vs NumPy向量化
        def python_loop_add(x, y):
            """Python循环实现"""
            result = []
            for i in range(len(x)):
                result.append(x[i] + y[i])
            return result
        
        def numpy_vectorized_add(x, y):
            """NumPy向量化实现"""
            return x + y
        
        # 性能测试
        print("测试数组大小:", size)
        
        # Python循环 (只测试小部分)
        start_time = time.time()
        result_python = python_loop_add(a[:1000], b[:1000])
        python_time = time.time() - start_time
        
        # NumPy向量化
        start_time = time.time()
        result_numpy = numpy_vectorized_add(a, b)
        numpy_time = time.time() - start_time
        
        print(f"Python循环时间 (1000元素): {python_time:.6f}秒")
        print(f"NumPy向量化时间 ({size}元素): {numpy_time:.6f}秒")
        
        if python_time > 0 and numpy_time > 0:
            speedup = (python_time * size / 1000) / numpy_time
            print(f"性能提升: ~{speedup:.0f}倍")
        
        self.examples_completed.append("向量化性能")
    
    def practical_examples(self):
        """实际应用示例"""
        print("\n🎯 实际应用示例")
        print("=" * 30)
        
        # 1. 数据分析示例
        print("1. 数据分析示例:")
        # 模拟学生成绩数据
        np.random.seed(42)
        students = 100
        subjects = 5
        scores = np.random.normal(75, 15, (students, subjects))
        scores = np.clip(scores, 0, 100)
        
        print(f"学生数: {students}, 科目数: {subjects}")
        
        # 统计分析
        avg_per_student = np.mean(scores, axis=1)
        avg_per_subject = np.mean(scores, axis=0)
        overall_avg = np.mean(scores)
        
        print(f"总体平均分: {overall_avg:.2f}")
        print(f"最高个人平均分: {np.max(avg_per_student):.2f}")
        print(f"最低个人平均分: {np.min(avg_per_student):.2f}")
        
        # 2. 数值积分示例
        print(f"\n2. 数值积分示例:")
        # 计算sin(x)在[0, π]的积分
        x = np.linspace(0, np.pi, 1000)
        y = np.sin(x)
        integral_approx = np.trapz(y, x)
        integral_exact = 2.0
        
        print(f"数值积分结果: {integral_approx:.6f}")
        print(f"精确结果: {integral_exact:.6f}")
        print(f"误差: {abs(integral_approx - integral_exact):.6f}")
        
        self.examples_completed.append("实际应用")
    
    def visualization_examples(self):
        """可视化示例"""
        print("\n📊 NumPy数据可视化")
        print("=" * 30)
        
        # 基础图表
        x = np.linspace(0, 2*np.pi, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        plt.figure(figsize=(12, 4))
        
        # 三角函数
        plt.subplot(1, 3, 1)
        plt.plot(x, y1, label='sin(x)', linewidth=2)
        plt.plot(x, y2, label='cos(x)', linewidth=2)
        plt.title('三角函数')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 随机数据分布
        plt.subplot(1, 3, 2)
        data = np.random.normal(0, 1, 1000)
        plt.hist(data, bins=30, alpha=0.7, density=True)
        plt.title('正态分布直方图')
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.grid(True, alpha=0.3)
        
        # 2D数据
        plt.subplot(1, 3, 3)
        x_2d = np.linspace(-2, 2, 50)
        y_2d = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x_2d, y_2d)
        Z = np.exp(-(X**2 + Y**2))
        
        plt.contour(X, Y, Z, levels=10)
        plt.title('2D高斯函数等高线')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.tight_layout()
        plt.show()
        
        print("已生成可视化图表")
        self.examples_completed.append("数据可视化")
    
    def run_all_examples(self):
        """运行所有示例"""
        print("🔢 NumPy基础完整学习")
        print("=" * 60)
        
        self.array_creation_examples()
        self.array_operations()
        self.broadcasting_examples()
        self.vectorization_performance()
        self.practical_examples()
        self.visualization_examples()
        
        print(f"\n🎉 NumPy基础学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. 数组创建 - 掌握各种数组创建方法")
        print("2. 数组运算 - 向量化计算的强大功能")
        print("3. 广播机制 - 不同形状数组间的运算")
        print("4. 性能优化 - 向量化带来的巨大性能提升")
        print("5. 实际应用 - 数据分析、科学计算")
        print("6. 数据可视化 - 结合matplotlib展示数据")

def main():
    """主函数"""
    numpy_tutorial = NumpyFundamentals()
    numpy_tutorial.run_all_examples()
    
    print("\n💡 下一步学习建议:")
    print("1. 深入学习NumPy的高级索引技巧")
    print("2. 掌握NumPy的线性代数函数")
    print("3. 学习NumPy与其他库的集成")
    print("4. 练习使用NumPy解决实际问题")

if __name__ == "__main__":
    main()
