"""
计算机视觉深度学习Baseline脚本
支持图像分类、目标检测等任务，使用PyTorch实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CVBaseline:
    """计算机视觉Baseline类"""
    
    def __init__(self, task_type='classification', num_classes=10, image_size=224):
        """
        初始化
        task_type: 'classification', 'detection'
        num_classes: 类别数量
        image_size: 图像大小
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"使用设备: {self.device}")
    
    def create_sample_dataset(self):
        """创建示例数据集"""
        print("创建示例数据集...")
        
        # 使用CIFAR-10作为示例
        transform_train = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 下载CIFAR-10数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        self.num_classes = 10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"数据集加载完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        print(f"  类别数: {self.num_classes}")
    
    def create_model(self, model_type='resnet18', pretrained=True):
        """创建模型"""
        print(f"创建模型: {model_type}")
        
        if model_type == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.num_classes)
            
        elif model_type == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
            
        elif model_type == 'simple_cnn':
            self.model = self._create_simple_cnn()
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    def _create_simple_cnn(self):
        """创建简单的CNN模型"""
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                
                # 计算全连接层输入大小
                self.fc1 = nn.Linear(128 * (self.image_size // 8) * (self.image_size // 8), 512)
                self.fc2 = nn.Linear(512, num_classes)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                
                x = x.view(x.size(0), -1)
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return SimpleCNN(self.num_classes)
    
    def train_model(self, epochs=10, learning_rate=0.001):
        """训练模型"""
        print(f"开始训练，epochs: {epochs}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
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
        all_preds = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
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
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 显示一些测试样本
        plt.subplot(2, 3, 4)
        self._show_sample_predictions()
        
        plt.tight_layout()
        plt.show()
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(all_targets, all_preds, 
                                  target_names=self.class_names))
    
    def _show_sample_predictions(self):
        """显示样本预测结果"""
        self.model.eval()
        
        # 获取一个batch的数据
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
        
        # 显示前8个样本
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(8):
            img = images[i].cpu()
            # 反标准化
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(f'真实: {self.class_names[labels[i]]}\n'
                            f'预测: {self.class_names[predicted[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
    
    def save_model(self, path='cv_baseline_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'class_names': self.class_names
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path, model_type='resnet18'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.image_size = checkpoint['image_size']
        self.class_names = checkpoint['class_names']
        
        self.create_model(model_type, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"模型已从 {path} 加载")
    
    def run_baseline(self, model_type='resnet18', epochs=10, learning_rate=0.001):
        """运行完整的baseline流程"""
        print("🚀 开始计算机视觉深度学习Baseline")
        print("=" * 60)
        
        # 1. 创建数据集
        self.create_sample_dataset()
        
        # 2. 创建模型
        self.create_model(model_type)
        
        # 3. 训练模型
        self.train_model(epochs, learning_rate)
        
        # 4. 评估模型
        final_loss, final_acc = self.evaluate_model()
        
        # 5. 可视化结果
        self.visualize_results()
        
        print(f"\n🎉 Baseline完成！")
        print(f"最终测试准确率: {final_acc:.2f}%")
        
        return self.model, final_acc

def main():
    """主函数 - 演示用法"""
    print("=" * 60)
    print("计算机视觉Baseline示例")
    print("=" * 60)
    
    # 创建baseline实例
    cv_baseline = CVBaseline(num_classes=10, image_size=32)  # CIFAR-10使用32x32
    
    # 运行baseline
    model, accuracy = cv_baseline.run_baseline(
        model_type='simple_cnn',  # 使用简单CNN以便快速训练
        epochs=5,  # 少量epochs用于演示
        learning_rate=0.001
    )
    
    # 保存模型
    cv_baseline.save_model('cv_baseline_demo.pth')
    
    print("\n💡 使用说明:")
    print("1. 更换模型: run_baseline(model_type='resnet18')")
    print("2. 调整训练: run_baseline(epochs=20, learning_rate=0.0001)")
    print("3. 自定义数据集: 继承Dataset类并替换create_sample_dataset方法")
    print("4. 支持的模型: resnet18, resnet50, vgg16, efficientnet, simple_cnn")

if __name__ == "__main__":
    main()
