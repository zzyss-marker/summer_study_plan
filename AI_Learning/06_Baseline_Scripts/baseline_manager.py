"""
机器学习和深度学习Baseline管理器
统一管理和运行各种类型的baseline脚本
"""

import os
import sys
import importlib.util
import argparse
import json
from datetime import datetime

class BaselineManager:
    """Baseline管理器类"""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.available_baselines = {
            'tabular': {
                'path': 'tabular/tabular_baseline.py',
                'class': 'TabularBaseline',
                'description': '表格数据机器学习 (分类/回归)',
                'requirements': ['pandas', 'scikit-learn', 'matplotlib', 'seaborn']
            },
            'cv': {
                'path': 'computer_vision/cv_baseline.py',
                'class': 'CVBaseline',
                'description': '计算机视觉深度学习 (图像分类)',
                'requirements': ['torch', 'torchvision', 'matplotlib', 'pillow']
            },
            'nlp': {
                'path': 'nlp/nlp_baseline.py',
                'class': 'NLPBaseline',
                'description': '自然语言处理 (文本分类/情感分析)',
                'requirements': ['torch', 'transformers', 'matplotlib']
            },
            'multimodal': {
                'path': 'multimodal/multimodal_baseline.py',
                'class': 'MultimodalBaseline',
                'description': '多模态学习 (图像+文本)',
                'requirements': ['torch', 'torchvision', 'pillow', 'matplotlib']
            }
        }
        
        print("🚀 机器学习和深度学习Baseline管理器")
        print("=" * 60)
    
    def list_baselines(self):
        """列出所有可用的baseline"""
        print("📋 可用的Baseline脚本:")
        print("-" * 40)
        
        for key, info in self.available_baselines.items():
            print(f"🔹 {key}: {info['description']}")
            print(f"   文件: {info['path']}")
            print(f"   依赖: {', '.join(info['requirements'])}")
            print()
    
    def check_requirements(self, baseline_type):
        """检查依赖包是否安装"""
        if baseline_type not in self.available_baselines:
            print(f"❌ 未知的baseline类型: {baseline_type}")
            return False
        
        requirements = self.available_baselines[baseline_type]['requirements']
        missing_packages = []
        
        for package in requirements:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
            print(f"请运行: pip install {' '.join(missing_packages)}")
            return False
        
        print(f"✅ 所有依赖包已安装")
        return True
    
    def load_baseline_class(self, baseline_type):
        """动态加载baseline类"""
        if baseline_type not in self.available_baselines:
            raise ValueError(f"未知的baseline类型: {baseline_type}")
        
        info = self.available_baselines[baseline_type]
        script_path = os.path.join(self.script_dir, info['path'])
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Baseline脚本不存在: {script_path}")
        
        # 动态导入模块
        spec = importlib.util.spec_from_file_location("baseline_module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取类
        baseline_class = getattr(module, info['class'])
        
        return baseline_class
    
    def run_baseline(self, baseline_type, **kwargs):
        """运行指定的baseline"""
        print(f"🎯 运行 {baseline_type} baseline")
        print("=" * 40)
        
        # 检查依赖
        if not self.check_requirements(baseline_type):
            return None
        
        try:
            # 加载baseline类
            BaselineClass = self.load_baseline_class(baseline_type)
            
            # 创建实例
            if baseline_type == 'tabular':
                baseline = BaselineClass(task_type=kwargs.get('task_type', 'auto'))
            elif baseline_type == 'cv':
                baseline = BaselineClass(
                    num_classes=kwargs.get('num_classes', 10),
                    image_size=kwargs.get('image_size', 224)
                )
            elif baseline_type == 'nlp':
                baseline = BaselineClass(
                    num_classes=kwargs.get('num_classes', 2),
                    max_length=kwargs.get('max_length', 128)
                )
            elif baseline_type == 'multimodal':
                baseline = BaselineClass(
                    num_classes=kwargs.get('num_classes', 2),
                    image_size=kwargs.get('image_size', 224),
                    max_text_length=kwargs.get('max_text_length', 50)
                )
            
            # 运行baseline
            start_time = datetime.now()
            
            if baseline_type == 'tabular':
                model, results = baseline.run_baseline(
                    data_path=kwargs.get('data_path'),
                    target_column=kwargs.get('target_column', 'target')
                )
            elif baseline_type == 'cv':
                model, accuracy = baseline.run_baseline(
                    model_type=kwargs.get('model_type', 'resnet18'),
                    epochs=kwargs.get('epochs', 10),
                    learning_rate=kwargs.get('learning_rate', 0.001)
                )
            elif baseline_type == 'nlp':
                model, accuracy = baseline.run_baseline(
                    model_type=kwargs.get('model_type', 'lstm'),
                    epochs=kwargs.get('epochs', 5),
                    learning_rate=kwargs.get('learning_rate', 2e-5),
                    batch_size=kwargs.get('batch_size', 16)
                )
            elif baseline_type == 'multimodal':
                model, accuracy = baseline.run_baseline(
                    epochs=kwargs.get('epochs', 5),
                    learning_rate=kwargs.get('learning_rate', 0.001),
                    batch_size=kwargs.get('batch_size', 16)
                )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # 保存结果
            result_info = {
                'baseline_type': baseline_type,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'parameters': kwargs
            }
            
            if baseline_type == 'tabular':
                result_info['results'] = {name: {
                    'accuracy' if 'accuracy' in result else 'r2': 
                    result.get('accuracy', result.get('r2', 0)),
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                } for name, result in results.items()}
            else:
                result_info['final_accuracy'] = accuracy
            
            self.save_results(result_info)
            
            print(f"\n🎉 Baseline运行完成!")
            print(f"⏱️ 运行时间: {duration}")
            
            return model, result_info
            
        except Exception as e:
            print(f"❌ 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, result_info):
        """保存运行结果"""
        results_dir = os.path.join(self.script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result_info['baseline_type']}_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=2, ensure_ascii=False)
        
        print(f"📄 结果已保存到: {filepath}")
    
    def compare_results(self):
        """比较历史运行结果"""
        results_dir = os.path.join(self.script_dir, 'results')
        
        if not os.path.exists(results_dir):
            print("📁 没有找到历史结果")
            return
        
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        
        if not result_files:
            print("📁 没有找到历史结果文件")
            return
        
        print("📊 历史运行结果:")
        print("-" * 60)
        
        for filename in sorted(result_files):
            filepath = os.path.join(results_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                print(f"🔹 {result['baseline_type']} - {result['start_time'][:19]}")
                print(f"   运行时间: {result['duration_seconds']:.1f}秒")
                
                if 'final_accuracy' in result:
                    print(f"   准确率: {result['final_accuracy']:.2f}%")
                elif 'results' in result:
                    best_model = max(result['results'].items(), 
                                   key=lambda x: list(x[1].values())[0])
                    print(f"   最佳模型: {best_model[0]}")
                    print(f"   最佳得分: {list(best_model[1].values())[0]:.4f}")
                
                print()
                
            except Exception as e:
                print(f"❌ 读取结果文件失败: {filename} - {e}")
    
    def interactive_mode(self):
        """交互式模式"""
        print("🎮 进入交互式模式")
        print("输入 'help' 查看可用命令")
        
        while True:
            try:
                command = input("\n>>> ").strip().lower()
                
                if command == 'help':
                    print("可用命令:")
                    print("  list - 列出所有baseline")
                    print("  run <type> - 运行指定baseline")
                    print("  compare - 比较历史结果")
                    print("  quit - 退出")
                
                elif command == 'list':
                    self.list_baselines()
                
                elif command.startswith('run '):
                    baseline_type = command.split()[1]
                    if baseline_type in self.available_baselines:
                        self.run_baseline(baseline_type)
                    else:
                        print(f"❌ 未知的baseline类型: {baseline_type}")
                        print("可用类型:", list(self.available_baselines.keys()))
                
                elif command == 'compare':
                    self.compare_results()
                
                elif command in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                
                else:
                    print("❌ 未知命令，输入 'help' 查看帮助")
            
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机器学习和深度学习Baseline管理器')
    parser.add_argument('--list', action='store_true', help='列出所有可用的baseline')
    parser.add_argument('--run', type=str, help='运行指定的baseline类型')
    parser.add_argument('--compare', action='store_true', help='比较历史结果')
    parser.add_argument('--interactive', action='store_true', help='进入交互式模式')
    
    # baseline参数
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    
    args = parser.parse_args()
    
    manager = BaselineManager()
    
    if args.list:
        manager.list_baselines()
    elif args.run:
        kwargs = {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_classes': args.num_classes,
            'data_path': args.data_path
        }
        manager.run_baseline(args.run, **kwargs)
    elif args.compare:
        manager.compare_results()
    elif args.interactive:
        manager.interactive_mode()
    else:
        # 默认显示帮助和进入交互模式
        manager.list_baselines()
        print("\n💡 使用说明:")
        print("python baseline_manager.py --run tabular")
        print("python baseline_manager.py --run cv --epochs 10")
        print("python baseline_manager.py --interactive")
        print("\n或者直接运行各个baseline脚本:")
        print("python tabular/tabular_baseline.py")
        print("python computer_vision/cv_baseline.py")
        print("python nlp/nlp_baseline.py")
        print("python multimodal/multimodal_baseline.py")

if __name__ == "__main__":
    main()
