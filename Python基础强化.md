# 🐍 Python基础强化

## 🎯 学习目标
掌握AI开发所需的Python高级特性，为机器学习和深度学习打下坚实基础。

## 📚 核心知识点

### [[装饰器模式]]
**概念**: 在不修改原函数的情况下，为函数添加新功能的设计模式

#### 基础装饰器
```python
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end-start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "完成"
```

#### 带参数装饰器
```python
def retry(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"第{attempt+1}次尝试失败: {e}")
        return wrapper
    return decorator

@retry(max_attempts=5)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise Exception("随机失败")
    return "成功"
```

**应用场景**: 
- 性能监控 → [[模型训练监控]]
- 异常处理 → [[数据处理容错]]
- 缓存机制 → [[计算结果缓存]]

### [[生成器与迭代器]]
**概念**: 内存高效的数据处理方式，特别适合大数据场景

#### 生成器函数
```python
def data_generator(file_path, batch_size=1000):
    """大文件批量读取生成器"""
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  # 处理最后一批数据
            yield batch

# 使用示例
for batch in data_generator('large_dataset.txt'):
    # 处理每批数据，避免内存溢出
    process_batch(batch)
```

#### 生成器表达式
```python
# 内存高效的数据处理
squared_numbers = (x**2 for x in range(1000000))
filtered_data = (x for x in data if x > threshold)

# 链式生成器
def preprocess_pipeline(data):
    # 清洗数据
    cleaned = (clean_text(item) for item in data)
    # 特征提取
    features = (extract_features(item) for item in cleaned)
    # 标准化
    normalized = (normalize(item) for item in features)
    return normalized
```

**应用场景**:
- [[大数据处理]] - 避免内存溢出
- [[数据流水线]] - 高效数据预处理
- [[实时数据处理]] - 流式计算

### [[上下文管理器]]
**概念**: 确保资源的正确获取和释放

#### 类实现方式
```python
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
    
    def __enter__(self):
        print(f"连接到数据库: {self.db_name}")
        self.connection = connect_to_db(self.db_name)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")
        if exc_type:
            print(f"发生异常: {exc_val}")
        return False  # 不抑制异常

# 使用示例
with DatabaseConnection('ml_data') as db:
    data = db.query("SELECT * FROM training_data")
    # 自动处理连接关闭
```

#### contextlib实现
```python
from contextlib import contextmanager
import numpy as np

@contextmanager
def numpy_seed(seed):
    """临时设置numpy随机种子"""
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(old_state)

# 使用示例
with numpy_seed(42):
    # 可重现的随机数生成
    random_data = np.random.randn(1000)
# 自动恢复原始随机状态
```

**应用场景**:
- [[资源管理]] - 文件、数据库连接
- [[状态管理]] - 临时配置、环境变量
- [[异常安全]] - 确保清理操作

### [[元类编程]]
**概念**: 控制类的创建过程，实现高级抽象

#### 基础元类
```python
class ModelMeta(type):
    """机器学习模型元类"""
    def __new__(cls, name, bases, attrs):
        # 自动添加模型验证方法
        if 'validate' not in attrs:
            attrs['validate'] = cls.default_validate
        
        # 自动注册模型
        model_class = super().__new__(cls, name, bases, attrs)
        if name != 'BaseModel':
            ModelRegistry.register(name, model_class)
        
        return model_class
    
    @staticmethod
    def default_validate(self, data):
        """默认验证方法"""
        if data is None or len(data) == 0:
            raise ValueError("数据不能为空")
        return True

class BaseModel(metaclass=ModelMeta):
    """基础模型类"""
    pass

class LinearRegression(BaseModel):
    """线性回归模型 - 自动获得验证和注册功能"""
    def fit(self, X, y):
        self.validate(X)  # 自动添加的验证方法
        # 模型训练逻辑
        pass
```

#### 属性描述符
```python
class ValidatedAttribute:
    """验证属性描述符"""
    def __init__(self, validator=None, default=None):
        self.validator = validator
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name, self.default)
    
    def __set__(self, obj, value):
        if self.validator:
            value = self.validator(value)
        setattr(obj, self.name, value)

# 验证器函数
def positive_number(value):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("必须是正数")
    return value

class MLModel:
    """使用描述符的机器学习模型"""
    learning_rate = ValidatedAttribute(positive_number, 0.01)
    epochs = ValidatedAttribute(lambda x: int(x) if x > 0 else 1, 100)
    
    def __init__(self, learning_rate=None, epochs=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs
```

**应用场景**:
- [[框架开发]] - 自动化代码生成
- [[配置管理]] - 动态类创建
- [[API设计]] - 统一接口规范

### [[函数式编程]]
**概念**: 使用函数作为一等公民的编程范式

#### 高阶函数
```python
from functools import reduce, partial
from operator import add, mul

def compose(*functions):
    """函数组合"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def curry(func):
    """柯里化装饰器"""
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return partial(func, *args, **kwargs)
    return curried

# 数据处理管道
@curry
def filter_data(condition, data):
    return [item for item in data if condition(item)]

@curry
def transform_data(transformer, data):
    return [transformer(item) for item in data]

@curry
def aggregate_data(aggregator, data):
    return reduce(aggregator, data)

# 构建处理管道
process_pipeline = compose(
    aggregate_data(add),
    transform_data(lambda x: x ** 2),
    filter_data(lambda x: x > 0)
)

# 使用管道
result = process_pipeline([-2, -1, 0, 1, 2, 3])  # 结果: 14 (1² + 2² + 3²)
```

#### 函数式数据处理
```python
from itertools import chain, groupby, accumulate
from collections import defaultdict

def functional_data_analysis(data):
    """函数式数据分析"""
    # 数据清洗和转换
    cleaned = map(lambda x: x.strip().lower(), data)
    filtered = filter(lambda x: len(x) > 0, cleaned)
    
    # 分组统计
    grouped = groupby(sorted(filtered))
    counts = {k: len(list(v)) for k, v in grouped}
    
    # 累积计算
    values = list(counts.values())
    cumulative = list(accumulate(values))
    
    return {
        'counts': counts,
        'total': sum(values),
        'cumulative': cumulative
    }

# 函数式特征工程
def feature_engineering_pipeline(data):
    """特征工程管道"""
    return compose(
        # 标准化
        lambda x: [(i - np.mean(x)) / np.std(x) for i in x],
        # 异常值处理
        lambda x: [i for i in x if abs(i) < 3],
        # 数据转换
        lambda x: [float(i) for i in x if str(i).replace('.', '').isdigit()]
    )(data)
```

**应用场景**:
- [[数据处理管道]] - 链式数据转换
- [[特征工程]] - 可组合的特征变换
- [[模型组合]] - 集成学习

## 🔗 知识关联

### 与AI学习的连接
- [[装饰器]] → [[模型训练监控]] → [[实验管理]]
- [[生成器]] → [[大数据处理]] → [[批量训练]]
- [[上下文管理器]] → [[资源管理]] → [[GPU内存管理]]
- [[元类]] → [[框架设计]] → [[模型工厂]]
- [[函数式编程]] → [[数据管道]] → [[特征工程]]

### 实践项目连接
- [[Python性能优化]] - 装饰器 + 生成器
- [[数据处理框架]] - 上下文管理器 + 函数式编程
- [[机器学习库]] - 元类 + 描述符
- [[实验管理系统]] - 装饰器 + 上下文管理器

## 📊 学习进度

### 掌握程度评估
- [ ] 🔴 基础语法 - 能读懂代码
- [ ] 🟡 理解原理 - 能解释概念
- [ ] 🟢 熟练应用 - 能独立实现
- [ ] 🔵 创新使用 - 能设计模式

### 实践检验
- [ ] 实现装饰器库
- [ ] 开发数据生成器
- [ ] 设计上下文管理器
- [ ] 创建元类框架
- [ ] 构建函数式工具

## 🏷️ 标签
`#Python` `#高级特性` `#编程范式` `#代码优化` `#AI基础`

## 📚 相关资源
- [[Python进阶编程]] - 高级特性详解
- [[设计模式Python实现]] - 模式应用
- [[函数式编程指南]] - 范式学习
- [[性能优化技巧]] - 效率提升

---
**导航**: [[AI学习路径图]] | [[数据处理工具]] | [[机器学习算法]]
