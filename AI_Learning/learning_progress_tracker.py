"""
AI学习进度跟踪器
跟踪每个模块的学习进度和完成情况
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class AILearningTracker:
    """AI学习进度跟踪器"""
    
    def __init__(self):
        self.progress_file = "ai_learning_progress.json"
        self.modules = {
            "01_Python_Fundamentals": {
                "name": "Python基础强化",
                "files": [
                    "python_advanced_features.py",
                    "numpy_fundamentals.py", 
                    "pandas_fundamentals.py"
                ],
                "concepts": [
                    "装饰器和生成器",
                    "NumPy数组操作",
                    "Pandas数据处理",
                    "数据可视化",
                    "性能优化"
                ]
            },
            "02_Math_Foundations": {
                "name": "数学基础",
                "files": [
                    "linear_algebra_fundamentals.py",
                    "probability_statistics.py"
                ],
                "concepts": [
                    "向量和矩阵运算",
                    "特征值分解",
                    "概率分布",
                    "假设检验",
                    "贝叶斯推断"
                ]
            },
            "03_Machine_Learning": {
                "name": "机器学习基础",
                "files": [
                    "supervised_learning.py",
                    "unsupervised_learning.py"
                ],
                "concepts": [
                    "线性回归",
                    "逻辑回归",
                    "决策树",
                    "K-means聚类",
                    "PCA降维"
                ]
            },
            "04_Deep_Learning": {
                "name": "深度学习基础",
                "files": [
                    "neural_networks_from_scratch.py"
                ],
                "concepts": [
                    "激活函数",
                    "感知机",
                    "多层感知机",
                    "反向传播",
                    "梯度下降"
                ]
            },
            "05_Projects": {
                "name": "项目实战",
                "files": [
                    "house_price_prediction.py"
                ],
                "concepts": [
                    "数据探索",
                    "特征工程",
                    "模型比较",
                    "超参数调优",
                    "结果可视化"
                ]
            }
        }
        self.load_progress()
    
    def load_progress(self):
        """加载学习进度"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "modules": {},
                "start_date": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            # 初始化所有模块
            for module_id, module_info in self.modules.items():
                self.progress["modules"][module_id] = {
                    "completed_files": [],
                    "completed_concepts": [],
                    "completion_percentage": 0,
                    "notes": "",
                    "completion_date": None
                }
    
    def save_progress(self):
        """保存学习进度"""
        self.progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)
    
    def mark_file_completed(self, module_id, filename):
        """标记文件完成"""
        if module_id in self.progress["modules"]:
            if filename not in self.progress["modules"][module_id]["completed_files"]:
                self.progress["modules"][module_id]["completed_files"].append(filename)
                self.update_completion_percentage(module_id)
                self.save_progress()
                print(f"✅ 已标记 {module_id}/{filename} 为完成")
            else:
                print(f"📝 {filename} 已经标记为完成")
    
    def mark_concept_completed(self, module_id, concept):
        """标记概念掌握"""
        if module_id in self.progress["modules"]:
            if concept not in self.progress["modules"][module_id]["completed_concepts"]:
                self.progress["modules"][module_id]["completed_concepts"].append(concept)
                self.update_completion_percentage(module_id)
                self.save_progress()
                print(f"🧠 已标记概念 '{concept}' 为掌握")
    
    def update_completion_percentage(self, module_id):
        """更新完成百分比"""
        module_info = self.modules[module_id]
        progress_info = self.progress["modules"][module_id]
        
        total_items = len(module_info["files"]) + len(module_info["concepts"])
        completed_items = len(progress_info["completed_files"]) + len(progress_info["completed_concepts"])
        
        percentage = (completed_items / total_items) * 100
        progress_info["completion_percentage"] = round(percentage, 1)
        
        # 如果100%完成，记录完成日期
        if percentage == 100 and not progress_info["completion_date"]:
            progress_info["completion_date"] = datetime.now().isoformat()
    
    def add_notes(self, module_id, notes):
        """添加学习笔记"""
        if module_id in self.progress["modules"]:
            self.progress["modules"][module_id]["notes"] = notes
            self.save_progress()
            print(f"📝 已添加笔记到 {module_id}")
    
    def show_progress(self):
        """显示学习进度"""
        print("🎓 AI学习进度总览")
        print("=" * 60)
        
        total_completion = 0
        completed_modules = 0
        
        for module_id, module_info in self.modules.items():
            progress_info = self.progress["modules"][module_id]
            completion = progress_info["completion_percentage"]
            total_completion += completion
            
            if completion == 100:
                completed_modules += 1
                status = "✅ 已完成"
            elif completion > 0:
                status = "🔄 进行中"
            else:
                status = "⏳ 未开始"
            
            print(f"\n📚 {module_info['name']} ({module_id})")
            print(f"   进度: {completion}% {status}")
            print(f"   文件: {len(progress_info['completed_files'])}/{len(module_info['files'])}")
            print(f"   概念: {len(progress_info['completed_concepts'])}/{len(module_info['concepts'])}")
            
            if progress_info["completion_date"]:
                completion_date = datetime.fromisoformat(progress_info["completion_date"])
                print(f"   完成日期: {completion_date.strftime('%Y-%m-%d')}")
        
        overall_completion = total_completion / len(self.modules)
        print(f"\n🎯 总体进度: {overall_completion:.1f}%")
        print(f"📊 已完成模块: {completed_modules}/{len(self.modules)}")
        
        # 计算学习天数
        start_date = datetime.fromisoformat(self.progress["start_date"])
        days_learning = (datetime.now() - start_date).days
        print(f"📅 学习天数: {days_learning} 天")
    
    def show_module_details(self, module_id):
        """显示模块详细信息"""
        if module_id not in self.modules:
            print(f"❌ 模块 {module_id} 不存在")
            return
        
        module_info = self.modules[module_id]
        progress_info = self.progress["modules"][module_id]
        
        print(f"\n📚 {module_info['name']} 详细信息")
        print("=" * 50)
        
        print(f"📁 文件列表:")
        for filename in module_info["files"]:
            status = "✅" if filename in progress_info["completed_files"] else "⏳"
            print(f"   {status} {filename}")
        
        print(f"\n🧠 核心概念:")
        for concept in module_info["concepts"]:
            status = "✅" if concept in progress_info["completed_concepts"] else "⏳"
            print(f"   {status} {concept}")
        
        if progress_info["notes"]:
            print(f"\n📝 学习笔记:")
            print(f"   {progress_info['notes']}")
    
    def visualize_progress(self):
        """可视化学习进度"""
        print("\n📊 生成学习进度可视化图表...")
        
        # 准备数据
        module_names = [info["name"] for info in self.modules.values()]
        completions = [self.progress["modules"][mid]["completion_percentage"] 
                      for mid in self.modules.keys()]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 进度条图
        colors = ['green' if c == 100 else 'orange' if c > 0 else 'lightgray' 
                 for c in completions]
        bars = ax1.barh(module_names, completions, color=colors, alpha=0.7)
        ax1.set_xlabel('完成百分比 (%)')
        ax1.set_title('各模块学习进度')
        ax1.set_xlim(0, 100)
        
        # 在条形图上添加百分比标签
        for bar, completion in zip(bars, completions):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{completion}%', ha='left', va='center')
        
        # 饼图显示总体进度
        completed = sum(1 for c in completions if c == 100)
        in_progress = sum(1 for c in completions if 0 < c < 100)
        not_started = sum(1 for c in completions if c == 0)
        
        sizes = [completed, in_progress, not_started]
        labels = ['已完成', '进行中', '未开始']
        colors_pie = ['green', 'orange', 'lightgray']
        
        # 只显示非零的部分
        non_zero_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors_pie) if size > 0]
        if non_zero_sizes:
            sizes_nz, labels_nz, colors_nz = zip(*non_zero_sizes)
            ax2.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%', startangle=90)
        
        ax2.set_title('学习状态分布')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n🎓 AI学习进度跟踪器")
            print("=" * 40)
            print("1. 查看学习进度")
            print("2. 标记文件完成")
            print("3. 标记概念掌握")
            print("4. 查看模块详情")
            print("5. 添加学习笔记")
            print("6. 可视化进度")
            print("0. 退出")
            
            choice = input("\n请选择功能 (0-6): ").strip()
            
            if choice == '1':
                self.show_progress()
            elif choice == '2':
                self.mark_file_interactive()
            elif choice == '3':
                self.mark_concept_interactive()
            elif choice == '4':
                self.show_module_interactive()
            elif choice == '5':
                self.add_notes_interactive()
            elif choice == '6':
                self.visualize_progress()
            elif choice == '0':
                print("👋 继续加油学习！")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def mark_file_interactive(self):
        """交互式标记文件完成"""
        print("\n📁 选择模块:")
        for i, (module_id, module_info) in enumerate(self.modules.items(), 1):
            print(f"{i}. {module_info['name']} ({module_id})")
        
        try:
            choice = int(input("请选择模块编号: ")) - 1
            module_id = list(self.modules.keys())[choice]
            module_info = self.modules[module_id]
            
            print(f"\n📄 选择文件:")
            for i, filename in enumerate(module_info["files"], 1):
                status = "✅" if filename in self.progress["modules"][module_id]["completed_files"] else "⏳"
                print(f"{i}. {status} {filename}")
            
            file_choice = int(input("请选择文件编号: ")) - 1
            filename = module_info["files"][file_choice]
            
            self.mark_file_completed(module_id, filename)
            
        except (ValueError, IndexError):
            print("❌ 无效选择")
    
    def mark_concept_interactive(self):
        """交互式标记概念掌握"""
        print("\n📁 选择模块:")
        for i, (module_id, module_info) in enumerate(self.modules.items(), 1):
            print(f"{i}. {module_info['name']} ({module_id})")
        
        try:
            choice = int(input("请选择模块编号: ")) - 1
            module_id = list(self.modules.keys())[choice]
            module_info = self.modules[module_id]
            
            print(f"\n🧠 选择概念:")
            for i, concept in enumerate(module_info["concepts"], 1):
                status = "✅" if concept in self.progress["modules"][module_id]["completed_concepts"] else "⏳"
                print(f"{i}. {status} {concept}")
            
            concept_choice = int(input("请选择概念编号: ")) - 1
            concept = module_info["concepts"][concept_choice]
            
            self.mark_concept_completed(module_id, concept)
            
        except (ValueError, IndexError):
            print("❌ 无效选择")
    
    def show_module_interactive(self):
        """交互式显示模块详情"""
        print("\n📁 选择模块:")
        for i, (module_id, module_info) in enumerate(self.modules.items(), 1):
            print(f"{i}. {module_info['name']} ({module_id})")
        
        try:
            choice = int(input("请选择模块编号: ")) - 1
            module_id = list(self.modules.keys())[choice]
            self.show_module_details(module_id)
        except (ValueError, IndexError):
            print("❌ 无效选择")
    
    def add_notes_interactive(self):
        """交互式添加笔记"""
        print("\n📁 选择模块:")
        for i, (module_id, module_info) in enumerate(self.modules.items(), 1):
            print(f"{i}. {module_info['name']} ({module_id})")
        
        try:
            choice = int(input("请选择模块编号: ")) - 1
            module_id = list(self.modules.keys())[choice]
            
            current_notes = self.progress["modules"][module_id]["notes"]
            if current_notes:
                print(f"\n当前笔记: {current_notes}")
            
            notes = input("请输入学习笔记: ").strip()
            if notes:
                self.add_notes(module_id, notes)
        except (ValueError, IndexError):
            print("❌ 无效选择")

def main():
    """主函数"""
    tracker = AILearningTracker()
    
    print("🎓 欢迎使用AI学习进度跟踪器！")
    print("这个工具将帮助你跟踪AI学习的每个步骤。")
    
    # 显示当前进度
    tracker.show_progress()
    
    # 启动交互式菜单
    tracker.interactive_menu()

if __name__ == "__main__":
    main()
