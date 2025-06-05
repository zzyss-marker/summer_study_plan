"""
概率统计基础学习
机器学习必备的概率统计理论和实践
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import integrate
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class ProbabilityStatistics:
    """概率统计基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("📊 概率统计基础学习系统")
        print("=" * 50)
    
    def basic_probability(self):
        """基础概率概念"""
        print("🎲 基础概率概念")
        print("=" * 30)
        
        # 1. 条件概率和贝叶斯定理
        print("1. 条件概率和贝叶斯定理:")
        
        # 示例：医疗诊断问题
        # P(疾病) = 0.01 (患病率1%)
        # P(阳性|疾病) = 0.95 (敏感性95%)
        # P(阳性|无疾病) = 0.05 (假阳性率5%)
        
        p_disease = 0.01
        p_positive_given_disease = 0.95
        p_positive_given_no_disease = 0.05
        
        # 计算P(阳性)
        p_positive = (p_positive_given_disease * p_disease + 
                     p_positive_given_no_disease * (1 - p_disease))
        
        # 贝叶斯定理：P(疾病|阳性)
        p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
        
        print(f"  患病率: {p_disease:.3f}")
        print(f"  检测敏感性: {p_positive_given_disease:.3f}")
        print(f"  假阳性率: {p_positive_given_no_disease:.3f}")
        print(f"  检测阳性概率: {p_positive:.3f}")
        print(f"  阳性结果下患病概率: {p_disease_given_positive:.3f}")
        
        # 2. 独立性检验
        print(f"\n2. 独立性检验:")
        
        # 模拟两个变量的数据
        np.random.seed(42)
        n = 1000
        
        # 独立变量
        x_independent = np.random.normal(0, 1, n)
        y_independent = np.random.normal(0, 1, n)
        
        # 相关变量
        x_correlated = np.random.normal(0, 1, n)
        y_correlated = 0.7 * x_correlated + 0.3 * np.random.normal(0, 1, n)
        
        # 计算相关系数
        corr_independent = np.corrcoef(x_independent, y_independent)[0, 1]
        corr_correlated = np.corrcoef(x_correlated, y_correlated)[0, 1]
        
        print(f"  独立变量相关系数: {corr_independent:.4f}")
        print(f"  相关变量相关系数: {corr_correlated:.4f}")
        
        # 可视化
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(x_independent, y_independent, alpha=0.6)
        plt.title(f'独立变量 (r={corr_independent:.3f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(x_correlated, y_correlated, alpha=0.6)
        plt.title(f'相关变量 (r={corr_correlated:.3f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.examples_completed.append("基础概率")
    
    def probability_distributions(self):
        """概率分布"""
        print("\n📈 概率分布")
        print("=" * 30)
        
        # 1. 离散分布
        print("1. 离散概率分布:")
        
        # 二项分布
        n_trials = 20
        p_success = 0.3
        x_binomial = np.arange(0, n_trials + 1)
        y_binomial = stats.binom.pmf(x_binomial, n_trials, p_success)
        
        print(f"  二项分布 B({n_trials}, {p_success}):")
        print(f"    期望: {n_trials * p_success:.2f}")
        print(f"    方差: {n_trials * p_success * (1 - p_success):.2f}")
        
        # 泊松分布
        lambda_poisson = 3.5
        x_poisson = np.arange(0, 15)
        y_poisson = stats.poisson.pmf(x_poisson, lambda_poisson)
        
        print(f"  泊松分布 P({lambda_poisson}):")
        print(f"    期望: {lambda_poisson}")
        print(f"    方差: {lambda_poisson}")
        
        # 2. 连续分布
        print(f"\n2. 连续概率分布:")
        
        # 正态分布
        mu, sigma = 0, 1
        x_normal = np.linspace(-4, 4, 100)
        y_normal = stats.norm.pdf(x_normal, mu, sigma)
        
        print(f"  标准正态分布 N({mu}, {sigma}²):")
        print(f"    期望: {mu}")
        print(f"    方差: {sigma**2}")
        
        # 指数分布
        lambda_exp = 1.5
        x_exp = np.linspace(0, 5, 100)
        y_exp = stats.expon.pdf(x_exp, scale=1/lambda_exp)
        
        print(f"  指数分布 Exp({lambda_exp}):")
        print(f"    期望: {1/lambda_exp:.3f}")
        print(f"    方差: {1/lambda_exp**2:.3f}")
        
        # 可视化分布
        plt.figure(figsize=(15, 10))
        
        # 二项分布
        plt.subplot(2, 3, 1)
        plt.bar(x_binomial, y_binomial, alpha=0.7, color='blue')
        plt.title(f'二项分布 B({n_trials}, {p_success})')
        plt.xlabel('k')
        plt.ylabel('P(X=k)')
        plt.grid(True, alpha=0.3)
        
        # 泊松分布
        plt.subplot(2, 3, 2)
        plt.bar(x_poisson, y_poisson, alpha=0.7, color='green')
        plt.title(f'泊松分布 P({lambda_poisson})')
        plt.xlabel('k')
        plt.ylabel('P(X=k)')
        plt.grid(True, alpha=0.3)
        
        # 正态分布
        plt.subplot(2, 3, 3)
        plt.plot(x_normal, y_normal, linewidth=2, color='red')
        plt.fill_between(x_normal, y_normal, alpha=0.3, color='red')
        plt.title(f'正态分布 N({mu}, {sigma}²)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)
        
        # 指数分布
        plt.subplot(2, 3, 4)
        plt.plot(x_exp, y_exp, linewidth=2, color='orange')
        plt.fill_between(x_exp, y_exp, alpha=0.3, color='orange')
        plt.title(f'指数分布 Exp({lambda_exp})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)
        
        # 多个正态分布比较
        plt.subplot(2, 3, 5)
        for mu_i, sigma_i, color in [(0, 1, 'blue'), (0, 2, 'red'), (2, 1, 'green')]:
            x_i = np.linspace(-6, 6, 100)
            y_i = stats.norm.pdf(x_i, mu_i, sigma_i)
            plt.plot(x_i, y_i, linewidth=2, label=f'N({mu_i}, {sigma_i}²)', color=color)
        
        plt.title('不同参数的正态分布')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 累积分布函数
        plt.subplot(2, 3, 6)
        x_cdf = np.linspace(-3, 3, 100)
        y_cdf = stats.norm.cdf(x_cdf, 0, 1)
        plt.plot(x_cdf, y_cdf, linewidth=2, color='purple')
        plt.title('标准正态分布CDF')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.examples_completed.append("概率分布")
    
    def central_limit_theorem(self):
        """中心极限定理"""
        print("\n🎯 中心极限定理")
        print("=" * 30)
        
        # 演示中心极限定理
        print("中心极限定理演示:")
        print("无论原始分布如何，样本均值的分布趋向于正态分布")
        
        # 不同的原始分布
        distributions = {
            '均匀分布': lambda n: np.random.uniform(0, 1, n),
            '指数分布': lambda n: np.random.exponential(2, n),
            '二项分布': lambda n: np.random.binomial(10, 0.3, n),
            '泊松分布': lambda n: np.random.poisson(3, n)
        }
        
        sample_sizes = [1, 5, 30, 100]
        n_samples = 1000
        
        plt.figure(figsize=(16, 12))
        
        for i, (dist_name, dist_func) in enumerate(distributions.items()):
            for j, sample_size in enumerate(sample_sizes):
                plt.subplot(4, 4, i * 4 + j + 1)
                
                # 生成样本均值
                sample_means = []
                for _ in range(n_samples):
                    sample = dist_func(sample_size)
                    sample_means.append(np.mean(sample))
                
                # 绘制直方图
                plt.hist(sample_means, bins=30, density=True, alpha=0.7, 
                        color=f'C{i}', edgecolor='black')
                
                # 理论正态分布（如果适用）
                if sample_size >= 30:
                    mean_theory = np.mean(sample_means)
                    std_theory = np.std(sample_means)
                    x_theory = np.linspace(min(sample_means), max(sample_means), 100)
                    y_theory = stats.norm.pdf(x_theory, mean_theory, std_theory)
                    plt.plot(x_theory, y_theory, 'r-', linewidth=2, label='理论正态分布')
                    plt.legend()
                
                plt.title(f'{dist_name}\n样本大小={sample_size}')
                plt.xlabel('样本均值')
                plt.ylabel('密度')
                
                # 显示统计信息
                mean_val = np.mean(sample_means)
                std_val = np.std(sample_means)
                plt.text(0.05, 0.95, f'均值={mean_val:.3f}\n标准差={std_val:.3f}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n观察结果:")
        print("1. 随着样本大小增加，样本均值分布越来越接近正态分布")
        print("2. 样本均值的标准差随样本大小增加而减小")
        print("3. 无论原始分布形状如何，这个规律都成立")
        
        self.examples_completed.append("中心极限定理")
    
    def hypothesis_testing(self):
        """假设检验"""
        print("\n🔬 假设检验")
        print("=" * 30)
        
        # 1. 单样本t检验
        print("1. 单样本t检验:")
        
        # 生成样本数据
        np.random.seed(42)
        sample_data = np.random.normal(105, 15, 50)  # 真实均值105
        
        # 假设检验：H0: μ = 100, H1: μ ≠ 100
        null_mean = 100
        t_stat, p_value = stats.ttest_1samp(sample_data, null_mean)
        
        print(f"  样本均值: {np.mean(sample_data):.2f}")
        print(f"  样本标准差: {np.std(sample_data, ddof=1):.2f}")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  结论: {'拒绝' if p_value < 0.05 else '不拒绝'}原假设 (α=0.05)")
        
        # 2. 双样本t检验
        print(f"\n2. 双样本t检验:")
        
        # 生成两组数据
        group1 = np.random.normal(100, 15, 30)
        group2 = np.random.normal(110, 15, 35)
        
        # 独立样本t检验
        t_stat2, p_value2 = stats.ttest_ind(group1, group2)
        
        print(f"  组1均值: {np.mean(group1):.2f}")
        print(f"  组2均值: {np.mean(group2):.2f}")
        print(f"  t统计量: {t_stat2:.4f}")
        print(f"  p值: {p_value2:.4f}")
        print(f"  结论: 两组均值{'有' if p_value2 < 0.05 else '无'}显著差异")
        
        # 3. 卡方检验
        print(f"\n3. 卡方独立性检验:")
        
        # 创建列联表
        observed = np.array([[20, 30, 25],
                           [15, 35, 30]])
        
        chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(observed)
        
        print(f"  观察频数:\n{observed}")
        print(f"  期望频数:\n{expected}")
        print(f"  卡方统计量: {chi2_stat:.4f}")
        print(f"  自由度: {dof}")
        print(f"  p值: {p_value_chi2:.4f}")
        print(f"  结论: 变量间{'存在' if p_value_chi2 < 0.05 else '不存在'}关联")
        
        # 4. 功效分析
        print(f"\n4. 统计功效分析:")
        
        # 计算不同效应大小下的功效
        effect_sizes = np.linspace(0, 2, 100)
        powers = []
        
        for effect_size in effect_sizes:
            # 使用scipy计算功效
            power = stats.ttest_1samp_power(effect_size, nobs=30, alpha=0.05)
            powers.append(power)
        
        plt.figure(figsize=(10, 6))
        plt.plot(effect_sizes, powers, linewidth=2)
        plt.axhline(y=0.8, color='r', linestyle='--', label='功效=0.8')
        plt.axvline(x=0.5, color='g', linestyle='--', label='中等效应大小')
        plt.xlabel('效应大小 (Cohen\'s d)')
        plt.ylabel('统计功效')
        plt.title('统计功效曲线 (n=30, α=0.05)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        self.examples_completed.append("假设检验")
    
    def bayesian_inference(self):
        """贝叶斯推断"""
        print("\n🎲 贝叶斯推断")
        print("=" * 30)
        
        # 1. 贝叶斯参数估计
        print("1. 贝叶斯参数估计:")
        
        # 硬币投掷问题：估计正面概率
        # 先验：Beta(1, 1) - 均匀分布
        # 似然：二项分布
        # 后验：Beta(α + 成功次数, β + 失败次数)
        
        # 观察数据
        tosses = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # 1=正面, 0=反面
        n_heads = sum(tosses)
        n_tails = len(tosses) - n_heads
        
        print(f"  投掷次数: {len(tosses)}")
        print(f"  正面次数: {n_heads}")
        print(f"  反面次数: {n_tails}")
        
        # 先验参数
        alpha_prior = 1
        beta_prior = 1
        
        # 后验参数
        alpha_posterior = alpha_prior + n_heads
        beta_posterior = beta_prior + n_tails
        
        # 绘制先验、似然和后验分布
        p_values = np.linspace(0, 1, 100)
        
        prior = stats.beta.pdf(p_values, alpha_prior, beta_prior)
        likelihood = stats.binom.pmf(n_heads, len(tosses), p_values)
        posterior = stats.beta.pdf(p_values, alpha_posterior, beta_posterior)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(p_values, prior, linewidth=2, label='先验 Beta(1,1)')
        plt.title('先验分布')
        plt.xlabel('p (正面概率)')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(p_values, likelihood, linewidth=2, label=f'似然 Bin({len(tosses)},{n_heads})', color='orange')
        plt.title('似然函数')
        plt.xlabel('p (正面概率)')
        plt.ylabel('似然')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(p_values, posterior, linewidth=2, label=f'后验 Beta({alpha_posterior},{beta_posterior})', color='green')
        plt.title('后验分布')
        plt.xlabel('p (正面概率)')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 后验统计
        posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
        posterior_var = (alpha_posterior * beta_posterior) / \
                       ((alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1))
        
        print(f"  后验均值: {posterior_mean:.3f}")
        print(f"  后验方差: {posterior_var:.3f}")
        print(f"  95%可信区间: {stats.beta.interval(0.95, alpha_posterior, beta_posterior)}")
        
        # 2. 贝叶斯线性回归
        print(f"\n2. 贝叶斯线性回归概念:")
        print("  与频率派回归的区别:")
        print("  • 频率派：参数是固定但未知的常数")
        print("  • 贝叶斯：参数是随机变量，有先验分布")
        print("  • 贝叶斯方法提供参数的不确定性量化")
        print("  • 可以自然地处理小样本问题")
        
        self.examples_completed.append("贝叶斯推断")
    
    def run_all_examples(self):
        """运行所有示例"""
        print("📊 概率统计基础完整学习")
        print("=" * 60)
        
        self.basic_probability()
        self.probability_distributions()
        self.central_limit_theorem()
        self.hypothesis_testing()
        self.bayesian_inference()
        
        print(f"\n🎉 概率统计学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. 基础概率 - 条件概率、贝叶斯定理、独立性")
        print("2. 概率分布 - 离散分布、连续分布、分布特性")
        print("3. 中心极限定理 - 样本均值分布的收敛性")
        print("4. 假设检验 - t检验、卡方检验、功效分析")
        print("5. 贝叶斯推断 - 参数估计、不确定性量化")

def main():
    """主函数"""
    prob_stats = ProbabilityStatistics()
    prob_stats.run_all_examples()
    
    print("\n💡 机器学习中的应用:")
    print("1. 概率分布 - 数据建模、生成模型")
    print("2. 贝叶斯方法 - 贝叶斯网络、朴素贝叶斯")
    print("3. 假设检验 - 模型选择、A/B测试")
    print("4. 中心极限定理 - 置信区间、bootstrap")

if __name__ == "__main__":
    main()
