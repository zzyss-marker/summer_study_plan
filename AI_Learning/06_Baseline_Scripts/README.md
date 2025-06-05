# 🚀 机器学习和深度学习Baseline脚本集合

这是一套完整的机器学习和深度学习baseline脚本，涵盖表格数据、计算机视觉、自然语言处理和多模态学习等主要领域。每个脚本都是开箱即用的完整解决方案，适合快速原型开发和学习研究。

## 📋 目录结构

```
06_Baseline_Scripts/
├── baseline_manager.py          # 统一管理器
├── README.md                    # 说明文档
├── tabular/                     # 表格数据
│   └── tabular_baseline.py
├── computer_vision/             # 计算机视觉
│   └── cv_baseline.py
├── nlp/                        # 自然语言处理
│   └── nlp_baseline.py
├── multimodal/                 # 多模态学习
│   └── multimodal_baseline.py
└── results/                    # 运行结果保存目录
```

## 🎯 支持的任务类型

### 📊 表格数据 (`tabular_baseline.py`)
- **支持任务**: 分类、回归
- **算法**: RandomForest, GradientBoosting, LogisticRegression, SVM, Ridge, Lasso
- **特色功能**:
  - 自动特征工程和数据预处理
  - 多模型自动比较和选择
  - 完整的模型评估和可视化
  - 支持自定义数据集

### 🖼️ 计算机视觉 (`cv_baseline.py`)
- **支持任务**: 图像分类
- **模型**: ResNet18/50, VGG16, EfficientNet, 自定义CNN
- **特色功能**:
  - 预训练模型微调
  - 数据增强和标准化
  - 训练过程可视化
  - 模型保存和加载

### 📝 自然语言处理 (`nlp_baseline.py`)
- **支持任务**: 文本分类、情感分析
- **模型**: LSTM, Transformer (BERT等)
- **特色功能**:
  - 支持预训练Transformer模型
  - 自动词汇表构建
  - 文本预处理和编码
  - 单文本预测接口

### 🔄 多模态学习 (`multimodal_baseline.py`)
- **支持任务**: 图像+文本分类
- **架构**: ResNet + LSTM + 注意力机制
- **特色功能**:
  - 图像和文本特征融合
  - 注意力机制增强
  - 端到端训练
  - 多模态数据处理

## 🚀 快速开始

### 环境准备

```bash
# 基础依赖
pip install numpy pandas matplotlib seaborn scikit-learn

# 深度学习依赖
pip install torch torchvision

# NLP依赖 (可选)
pip install transformers

# 图像处理依赖
pip install pillow
```

### 使用方法

#### 1. 使用管理器 (推荐)

```bash
# 列出所有可用baseline
python baseline_manager.py --list

# 运行表格数据baseline
python baseline_manager.py --run tabular

# 运行计算机视觉baseline
python baseline_manager.py --run cv --epochs 10 --learning_rate 0.001

# 运行NLP baseline
python baseline_manager.py --run nlp --epochs 5 --batch_size 32

# 运行多模态baseline
python baseline_manager.py --run multimodal --epochs 5

# 交互式模式
python baseline_manager.py --interactive

# 比较历史结果
python baseline_manager.py --compare
```

#### 2. 直接运行脚本

```bash
# 表格数据
cd tabular && python tabular_baseline.py

# 计算机视觉
cd computer_vision && python cv_baseline.py

# 自然语言处理
cd nlp && python nlp_baseline.py

# 多模态学习
cd multimodal && python multimodal_baseline.py
```

## 📖 详细使用说明

### 表格数据Baseline

```python
from tabular.tabular_baseline import TabularBaseline

# 分类任务
classifier = TabularBaseline(task_type='classification')
best_model, results = classifier.run_baseline()

# 回归任务
regressor = TabularBaseline(task_type='regression')
best_model, results = regressor.run_baseline()

# 使用自定义数据
classifier.run_baseline(data_path='your_data.csv', target_column='target')
```

### 计算机视觉Baseline

```python
from computer_vision.cv_baseline import CVBaseline

# 创建baseline
cv_baseline = CVBaseline(num_classes=10, image_size=224)

# 运行训练
model, accuracy = cv_baseline.run_baseline(
    model_type='resnet18',
    epochs=10,
    learning_rate=0.001
)

# 保存模型
cv_baseline.save_model('my_model.pth')
```

### NLP Baseline

```python
from nlp.nlp_baseline import NLPBaseline

# 创建baseline
nlp_baseline = NLPBaseline(num_classes=2, max_length=128)

# 使用LSTM模型
model, accuracy = nlp_baseline.run_baseline(
    model_type='lstm',
    epochs=5,
    learning_rate=0.001
)

# 使用Transformer模型 (需要transformers库)
model, accuracy = nlp_baseline.run_baseline(
    model_type='transformer',
    epochs=3,
    learning_rate=2e-5
)

# 预测单个文本
prediction, confidence = nlp_baseline.predict_text("This is amazing!")
```

### 多模态Baseline

```python
from multimodal.multimodal_baseline import MultimodalBaseline

# 创建baseline
multimodal_baseline = MultimodalBaseline(
    num_classes=2,
    image_size=224,
    max_text_length=50
)

# 运行训练
model, accuracy = multimodal_baseline.run_baseline(
    epochs=10,
    learning_rate=0.001,
    batch_size=16
)
```

## 🎨 自定义和扩展

### 添加自定义数据集

每个baseline都支持自定义数据集，只需替换相应的数据加载方法：

```python
# 表格数据
def load_custom_data(self):
    self.data = pd.read_csv('your_data.csv')

# 图像数据
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # 实现自定义数据集
        pass

# 文本数据
def create_custom_dataset(self):
    self.X_train = your_texts
    self.y_train = your_labels
```

### 添加新模型

```python
# 在相应的baseline类中添加新模型
def create_custom_model(self):
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 定义模型结构
        
        def forward(self, x):
            # 定义前向传播
            return x
    
    return CustomModel()
```

## 📊 结果分析

每次运行都会自动保存结果到 `results/` 目录，包含：

- 训练参数和配置
- 模型性能指标
- 运行时间统计
- 可视化图表

使用管理器可以方便地比较不同运行的结果：

```bash
python baseline_manager.py --compare
```

## 🔧 高级配置

### 模型超参数调优

```python
# 网格搜索示例
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [5, 10, 20]
}

for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
        model, acc = baseline.run_baseline(
            learning_rate=lr,
            batch_size=bs,
            epochs=10
        )
```

### 分布式训练

```python
# 多GPU训练 (PyTorch)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 🎯 最佳实践

1. **数据预处理**: 确保数据质量，处理缺失值和异常值
2. **模型选择**: 从简单模型开始，逐步增加复杂度
3. **超参数调优**: 使用验证集进行超参数搜索
4. **模型评估**: 使用多种评估指标，避免过拟合
5. **结果可视化**: 利用内置的可视化功能分析模型性能

## 🤝 贡献指南

欢迎贡献新的baseline脚本或改进现有功能：

1. Fork 项目
2. 创建特性分支
3. 添加新功能或修复bug
4. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

感谢所有开源项目的贡献者，特别是：
- PyTorch团队
- Scikit-learn团队
- Hugging Face Transformers团队
- 其他相关开源项目

---

**开始您的机器学习之旅吧！** 🚀
