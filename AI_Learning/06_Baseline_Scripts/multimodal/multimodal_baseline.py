"""
多模态深度学习Baseline脚本
支持图像+文本的多模态任务，如图像描述、视觉问答等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from sklearn.metrics import accuracy_score
import random

class MultimodalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, image_paths, texts, labels, transform=None, max_text_length=50):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.max_text_length = max_text_length
        
        # 构建文本词汇表
        self.vocab = self._build_vocab(texts)
        self.vocab_size = len(self.vocab)
    
    def _build_vocab(self, texts):
        """构建词汇表"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        return vocab
    
    def _encode_text(self, text):
        """编码文本"""
        words = text.lower().split()
        indices = [self.vocab.get(word, 1) for word in words]  # 1是<UNK>
        
        # 添加开始和结束标记
        indices = [2] + indices + [3]  # 2是<START>, 3是<END>
        
        # 截断或填充
        if len(indices) > self.max_text_length:
            indices = indices[:self.max_text_length]
        else:
            indices.extend([0] * (self.max_text_length - len(indices)))  # 0是<PAD>
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        if isinstance(self.image_paths[idx], str):
            image = Image.open(self.image_paths[idx]).convert('RGB')
        else:
            # 如果是numpy数组或tensor
            image = self.image_paths[idx]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # 编码文本
        text_encoded = self._encode_text(self.texts[idx])
        
        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'text': text_encoded,
            'label': label
        }

class MultimodalModel(nn.Module):
    """多模态融合模型"""
    
    def __init__(self, vocab_size, num_classes=2, image_feature_dim=512, text_feature_dim=256):
        super(MultimodalModel, self).__init__()
        
        # 图像编码器 (使用预训练ResNet)
        self.image_encoder = models.resnet18(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, image_feature_dim)
        
        # 文本编码器 (LSTM)
        self.text_embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.text_lstm = nn.LSTM(128, text_feature_dim//2, batch_first=True, bidirectional=True)
        
        # 多模态融合层
        self.fusion_dim = image_feature_dim + text_feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(256)
    
    def forward(self, images, texts):
        # 图像特征提取
        image_features = self.image_encoder(images)  # [batch_size, image_feature_dim]
        
        # 文本特征提取
        text_embedded = self.text_embedding(texts)  # [batch_size, seq_len, 128]
        text_lstm_out, _ = self.text_lstm(text_embedded)  # [batch_size, seq_len, text_feature_dim]
        
        # 使用注意力机制聚合文本特征
        text_attended, _ = self.attention(text_lstm_out, text_lstm_out, text_lstm_out)
        text_attended = self.attention_norm(text_attended + text_lstm_out)
        text_features = text_attended.mean(dim=1)  # [batch_size, text_feature_dim]
        
        # 多模态特征融合
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        # 分类
        output = self.fusion_layer(fused_features)
        
        return output

class MultimodalBaseline:
    """多模态Baseline类"""
    
    def __init__(self, num_classes=2, image_size=224, max_text_length=50):
        """
        初始化
        num_classes: 类别数量
        image_size: 图像大小
        max_text_length: 最大文本长度
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"使用设备: {self.device}")
    
    def create_sample_dataset(self):
        """创建示例多模态数据集"""
        print("创建示例多模态数据集...")
        
        # 生成合成图像数据 (用随机图像代替真实图像)
        np.random.seed(42)
        n_samples = 1000
        
        images = []
        texts = []
        labels = []
        
        # 类别0: 正面情感的图像+文本
        positive_texts = [
            "beautiful sunset over the ocean",
            "happy family enjoying vacation",
            "delicious food at restaurant",
            "amazing mountain landscape view",
            "cute puppy playing in garden",
            "wonderful celebration with friends",
            "peaceful nature scene",
            "exciting adventure in forest",
            "lovely flowers in bloom",
            "joyful children playing together"
        ]
        
        # 类别1: 负面情感的图像+文本
        negative_texts = [
            "dark stormy weather approaching",
            "broken old abandoned building",
            "sad lonely person sitting",
            "polluted dirty city street",
            "dangerous wild animal hunting",
            "destroyed damaged car accident",
            "empty cold winter landscape",
            "scary horror movie scene",
            "sick injured animal suffering",
            "angry crowd protesting loudly"
        ]
        
        # 生成正面样本
        for i in range(n_samples // 2):
            # 生成随机图像 (模拟明亮的正面图像)
            image = np.random.randint(100, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            images.append(image)
            
            # 随机选择正面文本
            text = random.choice(positive_texts)
            texts.append(text)
            
            labels.append(1)  # 正面标签
        
        # 生成负面样本
        for i in range(n_samples // 2):
            # 生成随机图像 (模拟暗淡的负面图像)
            image = np.random.randint(0, 150, (self.image_size, self.image_size, 3), dtype=np.uint8)
            images.append(image)
            
            # 随机选择负面文本
            text = random.choice(negative_texts)
            texts.append(text)
            
            labels.append(0)  # 负面标签
        
        # 打乱数据
        combined = list(zip(images, texts, labels))
        random.shuffle(combined)
        images, texts, labels = zip(*combined)
        
        # 分割数据
        split_idx = int(0.8 * len(images))
        
        self.train_images = images[:split_idx]
        self.train_texts = texts[:split_idx]
        self.train_labels = labels[:split_idx]
        
        self.test_images = images[split_idx:]
        self.test_texts = texts[split_idx:]
        self.test_labels = labels[split_idx:]
        
        print(f"数据集创建完成:")
        print(f"  训练集: {len(self.train_images)} 样本")
        print(f"  测试集: {len(self.test_images)} 样本")
        print(f"  类别数: {self.num_classes}")
    
    def create_data_loaders(self, batch_size=16):
        """创建数据加载器"""
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = MultimodalDataset(
            self.train_images, self.train_texts, self.train_labels,
            transform=transform, max_text_length=self.max_text_length
        )
        
        test_dataset = MultimodalDataset(
            self.test_images, self.test_texts, self.test_labels,
            transform=transform, max_text_length=self.max_text_length
        )
        
        # 保存词汇表大小
        self.vocab_size = train_dataset.vocab_size
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"数据加载器创建完成:")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  批次大小: {batch_size}")
    
    def create_model(self):
        """创建多模态模型"""
        print("创建多模态融合模型...")
        
        self.model = MultimodalModel(
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            image_feature_dim=512,
            text_feature_dim=256
        )
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    def train_model(self, epochs=10, learning_rate=0.001):
        """训练模型"""
        print(f"开始训练，epochs: {epochs}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images, texts)
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
            
            scheduler.step()
            
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
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, texts)
                test_loss += criterion(outputs, labels).item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
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
        
        # 显示一些测试样本
        plt.subplot(2, 3, 3)
        self._show_sample_predictions()
        
        plt.tight_layout()
        plt.show()
    
    def _show_sample_predictions(self):
        """显示样本预测结果"""
        self.model.eval()
        
        # 获取一个batch的数据
        data_iter = iter(self.test_loader)
        batch = next(data_iter)
        
        images = batch['image'].to(self.device)
        texts = batch['text'].to(self.device)
        labels = batch['label'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images, texts)
            _, predicted = torch.max(outputs, 1)
        
        # 显示前4个样本
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        
        for i in range(min(4, len(images))):
            img = images[i].cpu()
            # 反标准化
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            axes[i].imshow(img.permute(1, 2, 0))
            
            # 解码文本 (简化显示)
            text_tokens = texts[i].cpu().numpy()
            text_preview = f"Text tokens: {text_tokens[:5]}..."
            
            sentiment_true = "Positive" if labels[i].item() == 1 else "Negative"
            sentiment_pred = "Positive" if predicted[i].item() == 1 else "Negative"
            
            axes[i].set_title(f'True: {sentiment_true}\nPred: {sentiment_pred}\n{text_preview}')
            axes[i].axis('off')
        
        plt.tight_layout()
    
    def run_baseline(self, epochs=5, learning_rate=0.001, batch_size=16):
        """运行完整的baseline流程"""
        print("🚀 开始多模态深度学习Baseline")
        print("=" * 60)
        
        # 1. 创建数据集
        self.create_sample_dataset()
        
        # 2. 创建数据加载器
        self.create_data_loaders(batch_size)
        
        # 3. 创建模型
        self.create_model()
        
        # 4. 训练模型
        self.train_model(epochs, learning_rate)
        
        # 5. 评估模型
        final_loss, final_acc = self.evaluate_model()
        
        # 6. 可视化结果
        self.visualize_results()
        
        print(f"\n🎉 Baseline完成！")
        print(f"最终测试准确率: {final_acc:.2f}%")
        
        return self.model, final_acc

def main():
    """主函数 - 演示用法"""
    print("=" * 60)
    print("多模态深度学习Baseline示例")
    print("=" * 60)
    
    # 创建baseline实例
    multimodal_baseline = MultimodalBaseline(
        num_classes=2,
        image_size=64,  # 使用较小的图像以便快速训练
        max_text_length=20
    )
    
    # 运行baseline
    model, accuracy = multimodal_baseline.run_baseline(
        epochs=3,  # 少量epochs用于演示
        learning_rate=0.001,
        batch_size=8
    )
    
    print("\n💡 使用说明:")
    print("1. 调整图像大小: MultimodalBaseline(image_size=224)")
    print("2. 调整文本长度: MultimodalBaseline(max_text_length=100)")
    print("3. 自定义数据集: 替换create_sample_dataset方法")
    print("4. 模型架构: 图像ResNet + 文本LSTM + 注意力融合")

if __name__ == "__main__":
    main()
