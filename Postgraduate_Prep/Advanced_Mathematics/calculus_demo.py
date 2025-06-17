"""
高等数学计算练习演示
包含极限、导数、积分的计算和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, limit, diff, integrate, oo, sin, cos, exp, log, sqrt, pi
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

class CalculusDemo:
    """高等数学演示类"""
    
    def __init__(self):
        self.x = symbols('x')
        
    def limit_demo(self):
        """极限计算演示"""
        print("🔢 极限计算演示")
        print("=" * 50)
        
        # 定义一些经典极限题目
        problems = [
            {
                'function': sin(self.x) / self.x,
                'point': 0,
                'description': 'lim(x→0) sin(x)/x',
                'answer': 1
            },
            {
                'function': (1 + 1/self.x)**self.x,
                'point': oo,
                'description': 'lim(x→∞) (1 + 1/x)^x',
                'answer': sp.E
            },
            {
                'function': (exp(self.x) - 1) / self.x,
                'point': 0,
                'description': 'lim(x→0) (e^x - 1)/x',
                'answer': 1
            }
        ]
        
        for i, problem in enumerate(problems, 1):
            print(f"\n题目 {i}: {problem['description']}")
            
            # 计算极限
            result = limit(problem['function'], self.x, problem['point'])
            print(f"计算结果: {result}")
            print(f"标准答案: {problem['answer']}")
    
    def derivative_demo(self):
        """导数计算演示"""
        print("\n📈 导数计算演示")
        print("=" * 50)
        
        # 定义函数
        functions = [
            {
                'function': self.x**3 + 2*self.x**2 - 5*self.x + 1,
                'description': 'f(x) = x³ + 2x² - 5x + 1'
            },
            {
                'function': sin(self.x) * cos(self.x),
                'description': 'f(x) = sin(x)cos(x)'
            },
            {
                'function': exp(self.x**2),
                'description': 'f(x) = e^(x²)'
            }
        ]
        
        for i, func_info in enumerate(functions, 1):
            print(f"\n题目 {i}: {func_info['description']}")
            
            func = func_info['function']
            
            # 计算一阶导数
            first_derivative = diff(func, self.x)
            print(f"一阶导数: f'(x) = {first_derivative}")
            
            # 计算二阶导数
            second_derivative = diff(func, self.x, 2)
            print(f"二阶导数: f''(x) = {second_derivative}")
    
    def integration_demo(self):
        """积分计算演示"""
        print("\n∫ 积分计算演示")
        print("=" * 50)
        
        # 定义积分题目
        integrals = [
            {
                'function': self.x**2 + 3*self.x + 2,
                'description': '∫(x² + 3x + 2)dx'
            },
            {
                'function': sin(self.x) * cos(self.x),
                'description': '∫sin(x)cos(x)dx'
            },
            {
                'function': 1 / (self.x**2 + 1),
                'description': '∫1/(x² + 1)dx'
            }
        ]
        
        for i, integral_info in enumerate(integrals, 1):
            print(f"\n题目 {i}: {integral_info['description']}")
            
            func = integral_info['function']
            
            # 计算不定积分
            indefinite_integral = integrate(func, self.x)
            print(f"不定积分: {indefinite_integral} + C")
            
            # 计算定积分 (0 到 1)
            try:
                definite_integral = integrate(func, (self.x, 0, 1))
                print(f"定积分[0,1]: {definite_integral}")
                print(f"数值结果: {float(definite_integral.evalf()):.6f}")
            except:
                print("定积分计算复杂")
    
    def plot_function_demo(self):
        """函数图像演示"""
        print("\n📊 函数图像演示")
        print("=" * 50)
        
        # 定义函数
        func = self.x**3 - 3*self.x**2 + 2*self.x + 1
        first_deriv = diff(func, self.x)
        second_deriv = diff(func, self.x, 2)
        
        # 转换为numpy函数
        func_lambdified = sp.lambdify(self.x, func, 'numpy')
        first_deriv_lambdified = sp.lambdify(self.x, first_deriv, 'numpy')
        second_deriv_lambdified = sp.lambdify(self.x, second_deriv, 'numpy')
        
        x_vals = np.linspace(-2, 4, 1000)
        
        plt.figure(figsize=(15, 5))
        
        # 原函数
        plt.subplot(1, 3, 1)
        y_vals = func_lambdified(x_vals)
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.title(f'原函数: f(x) = {func}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        
        # 一阶导数
        plt.subplot(1, 3, 2)
        y_deriv1 = first_deriv_lambdified(x_vals)
        plt.plot(x_vals, y_deriv1, 'r-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.title(f"一阶导数: f'(x) = {first_deriv}")
        plt.xlabel('x')
        plt.ylabel("f'(x)")
        
        # 二阶导数
        plt.subplot(1, 3, 3)
        y_deriv2 = second_deriv_lambdified(x_vals)
        plt.plot(x_vals, y_deriv2, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.title(f'二阶导数: f\'\'(x) = {second_deriv}')
        plt.xlabel('x')
        plt.ylabel('f\'\'(x)')
        
        plt.tight_layout()
        plt.show()

def practice_problems():
    """练习题目"""
    print("\n📝 练习题目")
    print("=" * 50)
    
    problems = [
        {
            'type': '极限',
            'problem': 'lim(x→0) (1-cos(x))/x²',
            'hint': '使用洛必达法则或泰勒展开'
        },
        {
            'type': '导数',
            'problem': '求 y = ln(x² + 1) 的导数',
            'hint': '使用复合函数求导法则'
        },
        {
            'type': '积分',
            'problem': '∫ x·e^x dx',
            'hint': '使用分部积分法'
        },
        {
            'type': '应用',
            'problem': '求函数 f(x) = x³ - 3x + 1 的极值',
            'hint': '先求导数，令导数为0'
        }
    ]
    
    for i, prob in enumerate(problems, 1):
        print(f"\n练习 {i} ({prob['type']}):")
        print(f"  题目: {prob['problem']}")
        print(f"  提示: {prob['hint']}")

def study_tips():
    """学习建议"""
    print("\n💡 高等数学学习建议")
    print("=" * 50)
    
    tips = [
        "理解概念：重视数学概念的理解，不要死记硬背公式",
        "大量练习：每天至少做20道题，保持手感",
        "总结方法：整理常见题型的解题方法和技巧",
        "错题本：建立错题本，定期复习错题",
        "可视化：利用图像理解函数性质和几何意义",
        "循序渐进：从基础题开始，逐步提高难度",
        "定期测试：进行模拟考试，检验学习效果"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")

def main():
    """主函数"""
    print("📚 高等数学计算练习演示")
    print("=" * 60)
    
    # 创建演示实例
    demo = CalculusDemo()
    
    # 极限演示
    demo.limit_demo()
    
    # 导数演示
    demo.derivative_demo()
    
    # 积分演示
    demo.integration_demo()
    
    # 函数图像演示
    demo.plot_function_demo()
    
    # 练习题目
    practice_problems()
    
    # 学习建议
    study_tips()
    
    print("\n🎉 高等数学演示完成！")
    print("💪 继续努力，考研加油！")

if __name__ == "__main__":
    main()
