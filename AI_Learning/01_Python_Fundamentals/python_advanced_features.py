"""
Python高级特性完整实现
装饰器、生成器、上下文管理器等高级概念的详细学习
"""

import time
import functools
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np

class PythonAdvancedFeatures:
    """Python高级特性学习类"""

    def __init__(self):
        self.examples_run = []
        print("🐍 Python高级特性学习系统")
        print("=" * 50)

    def decorator_examples(self):
        """装饰器示例"""
        print("🎯 装饰器学习")
        print("=" * 30)

        # 1. 基础装饰器 - 计时器
        def timer(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f}秒")
                return result
            return wrapper

        # 2. 带参数的装饰器
        def retry(max_attempts=3):
            def decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except Exception as e:
                            if attempt == max_attempts - 1:
                                raise e
                            print(f"第{attempt + 1}次尝试失败: {e}")
                    return None
                return wrapper
            return decorator

        # 示例函数
        @timer
        @retry(max_attempts=2)
        def slow_calculation(n):
            """模拟耗时计算"""
            time.sleep(0.01)
            return sum(i**2 for i in range(n))

        # 运行示例
        print("1. 装饰器示例:")
        result = slow_calculation(100)
        print(f"计算结果: {result}")

        self.examples_run.append("装饰器")

    def generator_examples(self):
        """生成器示例"""
        print("\n🔄 生成器学习")
        print("=" * 30)

        # 1. 基础生成器
        def fibonacci_generator(n):
            """斐波那契生成器"""
            a, b = 0, 1
            count = 0
            while count < n:
                yield a
                a, b = b, a + b
                count += 1

        # 运行示例
        print("1. 斐波那契生成器:")
        fib_gen = fibonacci_generator(10)
        fib_numbers = list(fib_gen)
        print(f"前10个斐波那契数: {fib_numbers}")

        # 2. 内存效率对比
        print("\n2. 内存效率对比:")
        import sys
        list_comp = [x**2 for x in range(1000)]
        gen_exp = (x**2 for x in range(1000))

        print(f"列表推导式内存占用: {sys.getsizeof(list_comp)} 字节")
        print(f"生成器表达式内存占用: {sys.getsizeof(gen_exp)} 字节")

        self.examples_run.append("生成器")

    def context_manager_examples(self):
        """上下文管理器示例"""
        print("\n🔧 上下文管理器学习")
        print("=" * 30)

        # 1. 基础上下文管理器类
        class Timer:
            def __enter__(self):
                self.start_time = time.time()
                print("开始计时...")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                print(f"执行时间: {end_time - self.start_time:.4f}秒")
                return False

        # 2. 使用contextlib的上下文管理器
        @contextmanager
        def database_connection():
            """模拟数据库连接"""
            print("连接数据库...")
            connection = "数据库连接对象"
            try:
                yield connection
            finally:
                print("关闭数据库连接")

        # 运行示例
        print("1. 计时器上下文管理器:")
        with Timer():
            time.sleep(0.01)
            result = sum(i**2 for i in range(1000))

        print("\n2. 数据库连接上下文管理器:")
        with database_connection() as conn:
            print(f"使用连接: {conn}")

        self.examples_run.append("上下文管理器")

    def run_all_examples(self):
        """运行所有示例"""
        self.decorator_examples()
        self.generator_examples()
        self.context_manager_examples()

        print(f"\n🎉 完成所有示例！")
        print(f"已学习的概念: {', '.join(self.examples_run)}")

def main():
    """主函数"""
    features = PythonAdvancedFeatures()
    features.run_all_examples()

if __name__ == "__main__":
    main()