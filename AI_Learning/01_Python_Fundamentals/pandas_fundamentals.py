"""
Pandas数据处理基础学习
从基础操作到高级数据分析技术
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class PandasFundamentals:
    """Pandas基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("🐼 Pandas数据处理学习系统")
        print("=" * 50)
    
    def series_basics(self):
        """Series基础操作"""
        print("📊 Series基础操作")
        print("=" * 30)
        
        # 1. Series创建
        print("1. Series创建方法:")
        
        # 从列表创建
        s1 = pd.Series([1, 2, 3, 4, 5])
        print(f"从列表创建: {s1.values}")
        
        # 从字典创建
        s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
        print(f"从字典创建: {s2.to_dict()}")
        
        # 指定索引
        s3 = pd.Series([10, 20, 30], index=['x', 'y', 'z'])
        print(f"指定索引: {s3.to_dict()}")
        
        # 2. Series属性
        print(f"\n2. Series属性:")
        print(f"数据类型: {s1.dtype}")
        print(f"形状: {s1.shape}")
        print(f"大小: {s1.size}")
        print(f"索引: {s3.index.tolist()}")
        print(f"值: {s3.values}")
        
        # 3. Series索引和切片
        print(f"\n3. Series索引和切片:")
        print(f"按位置索引 s3[0]: {s3[0]}")
        print(f"按标签索引 s3['x']: {s3['x']}")
        print(f"切片 s3['x':'y']: {s3['x':'y'].to_dict()}")
        print(f"布尔索引 s1[s1 > 3]: {s1[s1 > 3].values}")
        
        # 4. Series运算
        print(f"\n4. Series运算:")
        s4 = pd.Series([1, 2, 3])
        s5 = pd.Series([4, 5, 6])
        
        print(f"加法: {(s4 + s5).values}")
        print(f"乘法: {(s4 * s5).values}")
        print(f"统计: 均值={s4.mean():.2f}, 标准差={s4.std():.2f}")
        
        self.examples_completed.append("Series基础")
    
    def dataframe_basics(self):
        """DataFrame基础操作"""
        print("\n📋 DataFrame基础操作")
        print("=" * 30)
        
        # 1. DataFrame创建
        print("1. DataFrame创建方法:")
        
        # 从字典创建
        data = {
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [25, 30, 35, 28],
            'city': ['北京', '上海', '广州', '深圳'],
            'salary': [8000, 12000, 15000, 9500]
        }
        df = pd.DataFrame(data)
        print("从字典创建DataFrame:")
        print(df)
        
        # 2. DataFrame基本信息
        print(f"\n2. DataFrame基本信息:")
        print(f"形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"索引: {df.index.tolist()}")
        print(f"数据类型:\n{df.dtypes}")
        
        print(f"\n详细信息:")
        print(df.info())
        
        print(f"\n统计描述:")
        print(df.describe())
        
        # 3. 数据选择
        print(f"\n3. 数据选择:")
        print(f"选择列 df['name']:")
        print(df['name'].tolist())
        
        print(f"\n选择多列 df[['name', 'age']]:")
        print(df[['name', 'age']])
        
        print(f"\n选择行 df.iloc[0]:")
        print(df.iloc[0])
        
        print(f"\n条件筛选 df[df['age'] > 30]:")
        print(df[df['age'] > 30])
        
        # 4. 数据修改
        print(f"\n4. 数据修改:")
        df_copy = df.copy()
        
        # 添加新列
        df_copy['bonus'] = df_copy['salary'] * 0.1
        print("添加bonus列:")
        print(df_copy[['name', 'salary', 'bonus']])
        
        # 修改数据
        df_copy.loc[df_copy['age'] > 30, 'level'] = 'Senior'
        df_copy.loc[df_copy['age'] <= 30, 'level'] = 'Junior'
        print(f"\n添加level列:")
        print(df_copy[['name', 'age', 'level']])
        
        self.examples_completed.append("DataFrame基础")
    
    def data_cleaning(self):
        """数据清洗"""
        print("\n🧹 数据清洗")
        print("=" * 30)
        
        # 创建包含缺失值的数据
        data_dirty = {
            'name': ['Alice', 'Bob', None, 'Diana', 'Eve'],
            'age': [25, None, 35, 28, 32],
            'score': [85.5, 92.0, 78.5, None, 88.0],
            'city': ['北京', '上海', '广州', '深圳', '北京']
        }
        df_dirty = pd.DataFrame(data_dirty)
        
        print("1. 原始数据（包含缺失值）:")
        print(df_dirty)
        print(f"\n缺失值统计:")
        print(df_dirty.isnull().sum())
        
        # 2. 处理缺失值
        print(f"\n2. 处理缺失值:")
        
        # 删除包含缺失值的行
        df_dropna = df_dirty.dropna()
        print(f"删除缺失值后的数据:")
        print(df_dropna)
        
        # 填充缺失值
        df_filled = df_dirty.copy()
        df_filled['name'].fillna('Unknown', inplace=True)
        df_filled['age'].fillna(df_filled['age'].mean(), inplace=True)
        df_filled['score'].fillna(df_filled['score'].median(), inplace=True)
        
        print(f"\n填充缺失值后的数据:")
        print(df_filled)
        
        # 3. 重复值处理
        print(f"\n3. 重复值处理:")
        
        # 添加重复行
        df_with_duplicates = pd.concat([df_filled, df_filled.iloc[[0]]], ignore_index=True)
        print(f"包含重复值的数据:")
        print(df_with_duplicates)
        
        print(f"重复值检查:")
        print(df_with_duplicates.duplicated())
        
        # 删除重复值
        df_no_duplicates = df_with_duplicates.drop_duplicates()
        print(f"\n删除重复值后:")
        print(df_no_duplicates)
        
        # 4. 数据类型转换
        print(f"\n4. 数据类型转换:")
        df_types = df_filled.copy()
        
        print(f"原始数据类型:")
        print(df_types.dtypes)
        
        # 转换数据类型
        df_types['age'] = df_types['age'].astype(int)
        df_types['city'] = df_types['city'].astype('category')
        
        print(f"\n转换后数据类型:")
        print(df_types.dtypes)
        
        self.examples_completed.append("数据清洗")
    
    def data_manipulation(self):
        """数据操作"""
        print("\n🔧 数据操作")
        print("=" * 30)
        
        # 创建示例数据
        df1 = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'department': ['IT', 'HR', 'IT', 'Finance']
        })
        
        df2 = pd.DataFrame({
            'id': [1, 2, 3, 5],
            'salary': [8000, 12000, 15000, 9500],
            'bonus': [800, 1200, 1500, 950]
        })
        
        print("1. 数据合并:")
        print("员工信息表:")
        print(df1)
        print("\n薪资信息表:")
        print(df2)
        
        # 内连接
        df_inner = pd.merge(df1, df2, on='id', how='inner')
        print(f"\n内连接结果:")
        print(df_inner)
        
        # 左连接
        df_left = pd.merge(df1, df2, on='id', how='left')
        print(f"\n左连接结果:")
        print(df_left)
        
        # 2. 分组聚合
        print(f"\n2. 分组聚合:")
        
        # 创建销售数据
        sales_data = pd.DataFrame({
            'product': ['A', 'B', 'A', 'B', 'A', 'B'],
            'region': ['North', 'North', 'South', 'South', 'North', 'South'],
            'sales': [100, 150, 120, 180, 110, 160],
            'quantity': [10, 15, 12, 18, 11, 16]
        })
        
        print("销售数据:")
        print(sales_data)
        
        # 按产品分组
        product_group = sales_data.groupby('product').agg({
            'sales': ['sum', 'mean'],
            'quantity': 'sum'
        })
        print(f"\n按产品分组聚合:")
        print(product_group)
        
        # 按多个字段分组
        multi_group = sales_data.groupby(['product', 'region'])['sales'].sum()
        print(f"\n按产品和地区分组:")
        print(multi_group)
        
        # 3. 数据透视表
        print(f"\n3. 数据透视表:")
        pivot_table = sales_data.pivot_table(
            values='sales',
            index='product',
            columns='region',
            aggfunc='sum',
            fill_value=0
        )
        print(pivot_table)
        
        self.examples_completed.append("数据操作")
    
    def time_series_basics(self):
        """时间序列基础"""
        print("\n📅 时间序列基础")
        print("=" * 30)
        
        # 1. 时间序列创建
        print("1. 时间序列创建:")
        
        # 创建日期范围
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        print(f"日期范围: {dates[:5].tolist()}...")
        
        # 创建时间序列数据
        np.random.seed(42)
        ts_data = pd.Series(
            np.random.randn(30).cumsum() + 100,
            index=dates,
            name='stock_price'
        )
        
        print(f"\n时间序列数据前5个:")
        print(ts_data.head())
        
        # 2. 时间序列索引
        print(f"\n2. 时间序列索引:")
        print(f"按日期索引: {ts_data['2024-01-05']:.2f}")
        print(f"日期范围索引:")
        print(ts_data['2024-01-01':'2024-01-05'])
        
        # 3. 时间序列重采样
        print(f"\n3. 时间序列重采样:")
        
        # 按周重采样
        weekly_data = ts_data.resample('W').mean()
        print(f"按周重采样 (均值):")
        print(weekly_data)
        
        # 4. 移动平均
        print(f"\n4. 移动平均:")
        ts_data_ma = ts_data.rolling(window=7).mean()
        print(f"7日移动平均:")
        print(ts_data_ma.tail())
        
        # 5. 时间序列可视化
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(ts_data.index, ts_data.values, label='原始数据', alpha=0.7)
        plt.plot(ts_data_ma.index, ts_data_ma.values, label='7日移动平均', linewidth=2)
        plt.title('时间序列数据')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.plot(weekly_data.index, weekly_data.values, 'o-', label='周平均')
        plt.title('周重采样数据')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        self.examples_completed.append("时间序列")
    
    def advanced_operations(self):
        """高级操作"""
        print("\n🚀 高级操作")
        print("=" * 30)
        
        # 1. 字符串操作
        print("1. 字符串操作:")
        
        df_str = pd.DataFrame({
            'name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
            'email': ['alice@email.com', 'bob@company.org', 'charlie@test.net']
        })
        
        print("原始数据:")
        print(df_str)
        
        # 字符串分割
        df_str[['first_name', 'last_name']] = df_str['name'].str.split(' ', expand=True)
        print(f"\n分割姓名:")
        print(df_str[['name', 'first_name', 'last_name']])
        
        # 提取邮箱域名
        df_str['domain'] = df_str['email'].str.extract(r'@(.+)')
        print(f"\n提取邮箱域名:")
        print(df_str[['email', 'domain']])
        
        # 2. 条件逻辑
        print(f"\n2. 条件逻辑:")
        
        df_logic = pd.DataFrame({
            'score': [85, 92, 78, 95, 67],
            'age': [22, 25, 23, 24, 21]
        })
        
        # 使用np.where
        df_logic['grade'] = np.where(df_logic['score'] >= 90, 'A',
                                   np.where(df_logic['score'] >= 80, 'B', 'C'))
        
        print("成绩分级:")
        print(df_logic)
        
        # 3. 数据变形
        print(f"\n3. 数据变形:")
        
        # 宽格式转长格式
        df_wide = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'math': [90, 85],
            'english': [88, 92],
            'science': [95, 87]
        })
        
        print("宽格式数据:")
        print(df_wide)
        
        df_long = pd.melt(df_wide, id_vars=['name'], 
                         value_vars=['math', 'english', 'science'],
                         var_name='subject', value_name='score')
        
        print(f"\n长格式数据:")
        print(df_long)
        
        # 长格式转宽格式
        df_wide_back = df_long.pivot(index='name', columns='subject', values='score')
        print(f"\n转回宽格式:")
        print(df_wide_back)
        
        self.examples_completed.append("高级操作")
    
    def run_all_examples(self):
        """运行所有示例"""
        print("🐼 Pandas数据处理完整学习")
        print("=" * 60)
        
        self.series_basics()
        self.dataframe_basics()
        self.data_cleaning()
        self.data_manipulation()
        self.time_series_basics()
        self.advanced_operations()
        
        print(f"\n🎉 Pandas学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. Series基础 - 一维数据结构的创建和操作")
        print("2. DataFrame基础 - 二维数据结构的核心功能")
        print("3. 数据清洗 - 处理缺失值、重复值、数据类型")
        print("4. 数据操作 - 合并、分组、聚合、透视表")
        print("5. 时间序列 - 日期处理、重采样、移动平均")
        print("6. 高级操作 - 字符串处理、条件逻辑、数据变形")

def main():
    """主函数"""
    pandas_tutorial = PandasFundamentals()
    pandas_tutorial.run_all_examples()
    
    print("\n💡 实际应用场景:")
    print("1. 数据分析：销售数据、用户行为分析")
    print("2. 金融分析：股票价格、财务报表分析")
    print("3. 科学研究：实验数据处理和统计分析")
    print("4. 业务报告：数据清洗、汇总、可视化")

if __name__ == "__main__":
    main()
