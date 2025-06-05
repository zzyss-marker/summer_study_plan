"""
自然语言处理深度学习Baseline脚本
支持文本分类、情感分析等任务，使用PyTorch和Transformers实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string

# 尝试导入transformers，如果没有安装则使用基础模型
try:
    from transformers import AutoTokenizer, AutoModel, AdamW
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers库未安装，将使用基础LSTM模型")

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if tokenizer is None:
            # 使用简单的词汇表
            self.vocab = self._build_vocab(texts)
            self.vocab_size = len(self.vocab)
    
    def _build_vocab(self, texts):
        """构建词汇表"""
        all_words = []
        for text in texts:
            words = self._simple_tokenize(text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for word, count in word_counts.most_common(10000):  # 限制词汇表大小
            if count >= 2:  # 过滤低频词
                vocab[word] = len(vocab)
        
        return vocab
    
    def _simple_tokenize(self, text):
        """简单分词"""
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        return text.split()
    
    def _encode_text(self, text):
        """编码文本"""
        if self.tokenizer:
            # 使用transformers tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return encoding
        else:
            # 使用简单编码
            words = self._simple_tokenize(text)
            indices = [self.vocab.get(word, 1) for word in words]  # 1是<UNK>
            
            # 截断或填充
            if len(indices) > self.max_length:
                indices = indices[:self.max_length]
            else:
                indices.extend([0] * (self.max_length - len(indices)))  # 0是<PAD>
            
            return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoded = self._encode_text(text)
        
        if self.tokenizer:
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': encoded,
                'label': torch.tensor(label, dtype=torch.long)
            }

class SimpleLSTM(nn.Module):
    """简单LSTM模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2, num_layers=2):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        
        if attention_mask is not None:
            # 处理padding
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        
        return output

class TransformerModel(nn.Module):
    """基于预训练Transformer的模型"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super(TransformerModel, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        
        return logits

class NLPBaseline:
    """NLP Baseline类"""
    
    def __init__(self, task_type='classification', num_classes=2, max_length=128):
        """
        初始化
        task_type: 'classification', 'sentiment'
        num_classes: 类别数量
        max_length: 最大序列长度
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"使用设备: {self.device}")
        print(f"Transformers可用: {TRANSFORMERS_AVAILABLE}")
    
    def create_sample_dataset(self):
        """创建示例数据集"""
        print("创建示例数据集...")
        
        # 创建情感分析示例数据
        positive_texts = [
            "I love this movie, it's amazing!",
            "Great product, highly recommended!",
            "Excellent service and quality.",
            "This is the best thing ever!",
            "Wonderful experience, very satisfied.",
            "Outstanding performance and design.",
            "Perfect solution for my needs.",
            "Incredible value for money.",
            "Fantastic quality and service.",
            "Amazing results, exceeded expectations."
        ] * 50  # 重复以增加数据量
        
        negative_texts = [
            "This movie is terrible and boring.",
            "Poor quality, waste of money.",
            "Disappointing service and product.",
            "Worst experience ever, avoid it.",
            "Completely unsatisfied with this.",
            "Terrible quality and poor design.",
            "Not worth the money at all.",
            "Very disappointed with results.",
            "Poor customer service experience.",
            "Low quality and overpriced."
        ] * 50
        
        # 组合数据
        texts = positive_texts + negative_texts
        labels = [1] * len(positive_texts) + [0] * len(negative_texts)
        
        # 打乱数据
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"数据集创建完成:")
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  测试集: {len(self.X_test)} 样本")
        print(f"  类别数: {self.num_classes}")
    
    def create_model(self, model_type='lstm', model_name='bert-base-uncased'):
        """创建模型"""
        print(f"创建模型: {model_type}")
        
        if model_type == 'transformer' and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TransformerModel(model_name, self.num_classes)
        else:
            # 使用LSTM模型
            if model_type == 'transformer':
                print("Transformers不可用，使用LSTM模型")
            
            # 创建数据集以构建词汇表
            train_dataset = TextDataset(self.X_train, self.y_train, max_length=self.max_length)
            vocab_size = train_dataset.vocab_size
            
            self.model = SimpleLSTM(
                vocab_size=vocab_size,
                embedding_dim=128,
                hidden_dim=128,
                num_classes=self.num_classes
            )
            
            self.vocab = train_dataset.vocab
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    def create_data_loaders(self, batch_size=16):
        """创建数据加载器"""
        train_dataset = TextDataset(
            self.X_train, self.y_train, self.tokenizer, self.max_length
        )
        test_dataset = TextDataset(
            self.X_test, self.y_test, self.tokenizer, self.max_length
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"数据加载器创建完成，batch_size: {batch_size}")
    
    def train_model(self, epochs=5, learning_rate=2e-5):
        """训练模型"""
        print(f"开始训练，epochs: {epochs}")
        
        criterion = nn.CrossEntropyLoss()
        
        if TRANSFORMERS_AVAILABLE and isinstance(self.model, TransformerModel):
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            
            # 验证阶段
            val_loss, val_acc = self.evaluate_model()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    
    def evaluate_model(self):
        """评估模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids)
                
                test_loss += criterion(outputs, labels).item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
    
    def visualize_results(self):
        """可视化训练结果"""
        plt.figure(figsize=(15, 10))
        
        # 训练历史
        plt.subplot(2, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('训练准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # 混淆矩阵
        plt.subplot(2, 3, 3)
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids)
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        plt.tight_layout()
        plt.show()
        
        # 打印分类报告
        print("\n分类报告:")
        class_names = ['Negative', 'Positive'] if self.num_classes == 2 else [f'Class_{i}' for i in range(self.num_classes)]
        print(classification_report(all_targets, all_preds, target_names=class_names))
    
    def predict_text(self, text):
        """预测单个文本"""
        self.model.eval()
        
        # 创建临时数据集
        temp_dataset = TextDataset([text], [0], self.tokenizer, self.max_length)
        temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch in temp_loader:
                input_ids = batch['input_ids'].to(self.device)
                
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids)
                
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                return predicted_class, confidence
    
    def run_baseline(self, model_type='lstm', epochs=5, learning_rate=2e-5, batch_size=16):
        """运行完整的baseline流程"""
        print("🚀 开始自然语言处理深度学习Baseline")
        print("=" * 60)
        
        # 1. 创建数据集
        self.create_sample_dataset()
        
        # 2. 创建模型
        self.create_model(model_type)
        
        # 3. 创建数据加载器
        self.create_data_loaders(batch_size)
        
        # 4. 训练模型
        self.train_model(epochs, learning_rate)
        
        # 5. 评估模型
        final_loss, final_acc = self.evaluate_model()
        
        # 6. 可视化结果
        self.visualize_results()
        
        # 7. 测试预测
        test_texts = [
            "This is absolutely amazing!",
            "I hate this product, it's terrible."
        ]
        
        print(f"\n测试预测:")
        for text in test_texts:
            pred_class, confidence = self.predict_text(text)
            sentiment = "Positive" if pred_class == 1 else "Negative"
            print(f"文本: '{text}'")
            print(f"预测: {sentiment} (置信度: {confidence:.4f})")
            print()
        
        print(f"🎉 Baseline完成！")
        print(f"最终测试准确率: {final_acc:.2f}%")
        
        return self.model, final_acc

def main():
    """主函数 - 演示用法"""
    print("=" * 60)
    print("自然语言处理Baseline示例")
    print("=" * 60)
    
    # 创建baseline实例
    nlp_baseline = NLPBaseline(num_classes=2, max_length=64)
    
    # 运行baseline
    model, accuracy = nlp_baseline.run_baseline(
        model_type='lstm',  # 使用LSTM模型
        epochs=3,  # 少量epochs用于演示
        learning_rate=0.001,
        batch_size=16
    )
    
    print("\n💡 使用说明:")
    print("1. 使用Transformer: run_baseline(model_type='transformer')")
    print("2. 调整训练: run_baseline(epochs=10, learning_rate=2e-5)")
    print("3. 自定义数据: 替换create_sample_dataset方法")
    print("4. 支持的模型: lstm, transformer (需要transformers库)")

if __name__ == "__main__":
    main()
