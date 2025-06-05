"""
高等数学综合学习系统
涵盖考研数学一的核心内容：极限、导数、积分、级数、微分方程等
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, limit, diff, integrate, oo, sin, cos, exp, log, sqrt, pi, E
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class AdvancedMathematics:
    """高等数学综合学习类"""
    
    def __init__(self):
        self.x, self.t, self.n = symbols('x t n')
        self.examples_completed = []
        print("📚 高等数学综合学习系统")
        print("=" * 50)
        print("涵盖考研数学一核心内容")
    
    def limits_comprehensive(self):
        """极限综合学习"""
        print("\n🎯 极限理论与计算")
        print("=" * 40)
        
        # 重要极限
        print("1. 重要极限:")
        important_limits = [
            (sin(self.x)/self.x, 0, "第一重要极限: lim(x→0) sin(x)/x = 1"),
            ((1 + 1/self.x)**self.x, oo, "第二重要极限: lim(x→∞) (1+1/x)^x = e"),
            ((1 + self.x)**(1/self.x), 0, "变形: lim(x→0) (1+x)^(1/x) = e"),
            ((exp(self.x) - 1)/self.x, 0, "等价无穷小: lim(x→0) (e^x-1)/x = 1")
        ]
        
        for expr, point, description in important_limits:
            result = limit(expr, self.x, point)
            print(f"  {description}")
            print(f"    计算结果: {result}")
        
        # 洛必达法则应用
        print(f"\n2. 洛必达法则应用:")
        lhopital_cases = [
            (self.x**2 / exp(self.x), oo, "∞/∞型: x²/e^x"),
            ((1 - cos(self.x)) / self.x**2, 0, "0/0型: (1-cos(x))/x²"),
            (self.x * log(self.x), 0, "0·∞型: x·ln(x)")
        ]
        
        for expr, point, description in lhopital_cases:
            result = limit(expr, self.x, point)
            print(f"  {description} = {result}")
        
        # 可视化极限概念
        self.visualize_limits()
        self.examples_completed.append("极限理论")
    
    def visualize_limits(self):
        """可视化极限概念"""
        plt.figure(figsize=(15, 10))
        
        # sin(x)/x 的图像
        plt.subplot(2, 3, 1)
        x_vals = np.linspace(-10, 10, 1000)
        x_vals = x_vals[x_vals != 0]
        y_vals = np.sin(x_vals) / x_vals
        
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='sin(x)/x')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='y=1')
        plt.scatter([0], [1], color='red', s=100, zorder=5)
        plt.xlim(-10, 10)
        plt.ylim(-0.5, 1.2)
        plt.title('第一重要极限')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # (1+1/x)^x 的图像
        plt.subplot(2, 3, 2)
        x_vals2 = np.linspace(0.1, 50, 1000)
        y_vals2 = (1 + 1/x_vals2) ** x_vals2
        plt.plot(x_vals2, y_vals2, 'g-', linewidth=2, label='(1+1/x)^x')
        plt.axhline(y=np.e, color='r', linestyle='--', alpha=0.7, label=f'y=e≈{np.e:.3f}')
        plt.xlim(0, 50)
        plt.ylim(2, 3)
        plt.title('第二重要极限')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 连续性与间断点
        plt.subplot(2, 3, 3)
        x_cont = np.linspace(-3, 3, 1000)
        
        # 可去间断点
        y1 = np.where(x_cont != 1, (x_cont**2 - 1)/(x_cont - 1), np.nan)
        plt.plot(x_cont, y1, 'b-', linewidth=2, label='可去间断点')
        plt.scatter([1], [2], color='red', s=100, zorder=5, facecolors='none', edgecolors='red')
        plt.scatter([1], [3], color='red', s=100, zorder=5)
        
        plt.xlim(-1, 3)
        plt.ylim(-1, 4)
        plt.title('间断点类型')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 导数的定义
        plt.subplot(2, 3, 4)
        x_deriv = np.linspace(-2, 2, 100)
        y_deriv = x_deriv**2
        
        plt.plot(x_deriv, y_deriv, 'b-', linewidth=2, label='f(x) = x²')
        
        # 割线到切线的过程
        x0 = 1
        for h in [0.5, 0.2, 0.05]:
            x1 = x0 + h
            y0, y1 = x0**2, x1**2
            slope = (y1 - y0) / h
            
            x_line = np.linspace(x0-0.5, x1+0.5, 100)
            y_line = slope * (x_line - x0) + y0
            
            alpha = 1 - h  # 透明度随h减小而增加
            plt.plot(x_line, y_line, '--', alpha=alpha, linewidth=1)
        
        # 真正的切线
        slope_true = 2 * x0
        x_tangent = np.linspace(0, 2, 100)
        y_tangent = slope_true * (x_tangent - x0) + x0**2
        plt.plot(x_tangent, y_tangent, 'r-', linewidth=2, label='切线')
        
        plt.xlim(-0.5, 2)
        plt.ylim(-0.5, 3)
        plt.title('导数的几何意义')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 积分的几何意义
        plt.subplot(2, 3, 5)
        x_int = np.linspace(0, 2, 100)
        y_int = x_int**2
        
        plt.plot(x_int, y_int, 'b-', linewidth=2, label='f(x) = x²')
        plt.fill_between(x_int, 0, y_int, alpha=0.3, color='blue', label='积分区域')
        
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 4.5)
        plt.title('定积分的几何意义')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 泰勒级数展开
        plt.subplot(2, 3, 6)
        x_taylor = np.linspace(-2, 2, 100)
        
        # e^x 的泰勒展开
        y_exact = np.exp(x_taylor)
        plt.plot(x_taylor, y_exact, 'b-', linewidth=2, label='e^x')
        
        # 不同阶数的泰勒多项式
        for n in [1, 2, 3, 5]:
            y_taylor = sum(x_taylor**k / np.math.factorial(k) for k in range(n+1))
            plt.plot(x_taylor, y_taylor, '--', alpha=0.7, label=f'T_{n}(x)')
        
        plt.xlim(-2, 2)
        plt.ylim(-2, 8)
        plt.title('泰勒级数展开')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def derivatives_comprehensive(self):
        """导数综合学习"""
        print("\n📈 导数理论与应用")
        print("=" * 40)
        
        # 基本求导公式
        print("1. 基本求导公式验证:")
        basic_functions = [
            (self.x**3, "幂函数: (x³)' = 3x²"),
            (sin(self.x), "三角函数: (sin x)' = cos x"),
            (exp(self.x), "指数函数: (e^x)' = e^x"),
            (log(self.x), "对数函数: (ln x)' = 1/x")
        ]
        
        for func, description in basic_functions:
            derivative = diff(func, self.x)
            print(f"  {description}")
            print(f"    计算结果: {derivative}")
        
        # 复合函数求导（链式法则）
        print(f"\n2. 链式法则应用:")
        composite_functions = [
            (sin(self.x**2), "sin(x²)"),
            (exp(2*self.x + 1), "e^(2x+1)"),
            (log(self.x**2 + 1), "ln(x²+1)"),
            ((self.x**2 + 1)**(1/3), "(x²+1)^(1/3)")
        ]
        
        for func, description in composite_functions:
            derivative = diff(func, self.x)
            print(f"  d/dx[{description}] = {derivative}")
        
        # 隐函数求导
        print(f"\n3. 隐函数求导:")
        print("  例：x² + y² = 1")
        print("  两边对x求导：2x + 2y·dy/dx = 0")
        print("  解得：dy/dx = -x/y")
        
        # 参数方程求导
        print(f"\n4. 参数方程求导:")
        print("  x = t², y = t³")
        print("  dx/dt = 2t, dy/dt = 3t²")
        print("  dy/dx = (dy/dt)/(dx/dt) = 3t²/(2t) = 3t/2")
        
        self.examples_completed.append("导数理论")
    
    def integrals_comprehensive(self):
        """积分综合学习"""
        print("\n∫ 积分理论与计算")
        print("=" * 40)
        
        # 基本积分公式
        print("1. 基本积分公式:")
        basic_integrals = [
            (self.x**2, "∫x² dx = x³/3 + C"),
            (sin(self.x), "∫sin x dx = -cos x + C"),
            (exp(self.x), "∫e^x dx = e^x + C"),
            (1/self.x, "∫(1/x) dx = ln|x| + C")
        ]
        
        for func, description in basic_integrals:
            integral = integrate(func, self.x)
            print(f"  {description}")
            print(f"    计算结果: {integral} + C")
        
        # 换元积分法
        print(f"\n2. 换元积分法:")
        substitution_integrals = [
            (2*self.x * exp(self.x**2), "∫2x·e^(x²) dx，令u = x²"),
            (sin(self.x) * cos(self.x), "∫sin x cos x dx，令u = sin x"),
            (self.x / sqrt(self.x**2 + 1), "∫x/√(x²+1) dx，令u = x²+1")
        ]
        
        for func, description in substitution_integrals:
            integral = integrate(func, self.x)
            print(f"  {description}")
            print(f"    结果: {integral} + C")
        
        # 分部积分法
        print(f"\n3. 分部积分法:")
        print("  公式: ∫u dv = uv - ∫v du")
        
        parts_integrals = [
            (self.x * exp(self.x), "∫x·e^x dx"),
            (self.x * sin(self.x), "∫x·sin x dx"),
            (log(self.x), "∫ln x dx")
        ]
        
        for func, description in parts_integrals:
            integral = integrate(func, self.x)
            print(f"  {description} = {integral} + C")
        
        # 定积分计算
        print(f"\n4. 定积分计算:")
        definite_integrals = [
            (self.x**2, 0, 2, "∫₀² x² dx"),
            (sin(self.x), 0, pi, "∫₀^π sin x dx"),
            (exp(-self.x), 0, oo, "∫₀^∞ e^(-x) dx")
        ]
        
        for func, a, b, description in definite_integrals:
            result = integrate(func, (self.x, a, b))
            print(f"  {description} = {result}")
        
        self.examples_completed.append("积分理论")
    
    def series_and_differential_equations(self):
        """级数与微分方程"""
        print("\n📊 级数与微分方程")
        print("=" * 40)
        
        # 数项级数
        print("1. 数项级数收敛性:")
        print("  几何级数: ∑(1/2)^n = 1/(1-1/2) = 2 (收敛)")
        print("  调和级数: ∑(1/n) 发散")
        print("  p级数: ∑(1/n^p) 当p>1时收敛，p≤1时发散")
        
        # 幂级数
        print(f"\n2. 幂级数展开:")
        power_series = [
            ("e^x", "1 + x + x²/2! + x³/3! + ..."),
            ("sin x", "x - x³/3! + x⁵/5! - ..."),
            ("cos x", "1 - x²/2! + x⁴/4! - ..."),
            ("1/(1-x)", "1 + x + x² + x³ + ... (|x|<1)")
        ]
        
        for func, series in power_series:
            print(f"  {func} = {series}")
        
        # 微分方程
        print(f"\n3. 常微分方程:")
        print("  一阶线性微分方程: dy/dx + P(x)y = Q(x)")
        print("  解法：积分因子法")
        print("  二阶常系数齐次方程: y'' + py' + qy = 0")
        print("  特征方程: r² + pr + q = 0")
        
        # 求解简单微分方程
        y = sp.Function('y')
        eq1 = sp.Eq(y(self.x).diff(self.x) - y(self.x), 0)
        sol1 = sp.dsolve(eq1, y(self.x))
        print(f"  例：dy/dx - y = 0")
        print(f"  解：{sol1}")
        
        self.examples_completed.append("级数与微分方程")
    
    def exam_problem_solving(self):
        """考研真题解题技巧"""
        print("\n🎯 考研真题解题技巧")
        print("=" * 40)
        
        print("1. 极限计算技巧:")
        print("  • 等价无穷小替换")
        print("  • 洛必达法则")
        print("  • 泰勒展开")
        print("  • 夹逼定理")
        
        print(f"\n2. 导数应用:")
        print("  • 函数单调性：f'(x) > 0 ⟹ 单调递增")
        print("  • 函数极值：f'(x₀) = 0 且 f''(x₀) ≠ 0")
        print("  • 函数凹凸性：f''(x) > 0 ⟹ 凹函数")
        print("  • 渐近线：垂直、水平、斜渐近线")
        
        print(f"\n3. 积分应用:")
        print("  • 面积计算：S = ∫[a,b] |f(x)| dx")
        print("  • 体积计算：V = π∫[a,b] [f(x)]² dx")
        print("  • 弧长计算：L = ∫[a,b] √(1 + [f'(x)]²) dx")
        
        print(f"\n4. 解题策略:")
        print("  • 先定性分析，再定量计算")
        print("  • 注意函数定义域")
        print("  • 检验答案合理性")
        print("  • 掌握常见题型模板")
        
        self.examples_completed.append("解题技巧")
    
    def run_comprehensive_study(self):
        """运行综合学习"""
        print("📚 高等数学综合学习")
        print("=" * 60)
        
        self.limits_comprehensive()
        self.derivatives_comprehensive()
        self.integrals_comprehensive()
        self.series_and_differential_equations()
        self.exam_problem_solving()
        
        print(f"\n🎉 高等数学学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📊 考研数学一知识点覆盖:")
        print("✅ 极限理论与计算")
        print("✅ 导数与微分")
        print("✅ 积分学")
        print("✅ 级数理论")
        print("✅ 微分方程")
        print("✅ 解题技巧")
        
        print(f"\n🎯 备考建议:")
        print("1. 每日练习基础计算题")
        print("2. 重点掌握重要定理")
        print("3. 多做历年真题")
        print("4. 总结常见题型")
        print("5. 注意计算准确性")

def main():
    """主函数"""
    math_study = AdvancedMathematics()
    math_study.run_comprehensive_study()
    
    print("\n💡 进一步学习资源:")
    print("1. 同济版《高等数学》教材")
    print("2. 张宇、汤家凤等名师视频")
    print("3. 历年考研真题集")
    print("4. 数学建模竞赛题目")

if __name__ == "__main__":
    main()
