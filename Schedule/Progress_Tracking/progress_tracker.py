"""
暑假学习进度跟踪器
用于记录和可视化学习进度，生成学习报告
"""

import json
import datetime
import os
from collections import defaultdict

class StudyProgressTracker:
    """学习进度跟踪器"""
    
    def __init__(self, data_file="study_progress.json"):
        self.data_file = data_file
        self.data = self.load_data()
        
        # 学习科目配置
        self.subjects = {
            'AI': {
                'name': 'AI机器学习',
                'target_hours': 168  # 8周 * 21小时
            },
            'CTF': {
                'name': 'CTF竞赛',
                'target_hours': 168
            },
            'Math': {
                'name': '高等数学',
                'target_hours': 120  # 8周 * 15小时
            },
            'CS408': {
                'name': '408专业课',
                'target_hours': 48   # 8周 * 6小时
            }
        }
    
    def load_data(self):
        """加载学习数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"daily_records": {}, "weekly_summaries": {}, "achievements": []}
        else:
            return {"daily_records": {}, "weekly_summaries": {}, "achievements": []}
    
    def save_data(self):
        """保存学习数据"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_daily_record(self, date, subject, hours, tasks_completed, notes=""):
        """添加每日学习记录"""
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime.date) else date
        
        if date_str not in self.data["daily_records"]:
            self.data["daily_records"][date_str] = {}
        
        self.data["daily_records"][date_str][subject] = {
            "hours": hours,
            "tasks_completed": tasks_completed,
            "notes": notes,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.save_data()
        print(f"✅ 已记录 {date_str} {self.subjects[subject]['name']} 学习 {hours} 小时")
    
    def add_achievement(self, title, description, category, date=None):
        """添加学习成就"""
        if date is None:
            date = datetime.date.today()
        
        achievement = {
            "title": title,
            "description": description,
            "category": category,
            "date": date.strftime("%Y-%m-%d") if isinstance(date, datetime.date) else date,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.data["achievements"].append(achievement)
        self.save_data()
        print(f"🎉 新成就解锁: {title}")
    
    def calculate_weekly_stats(self, week_start_date):
        """计算周统计数据"""
        week_stats = defaultdict(lambda: {"hours": 0, "tasks": 0})
        
        for i in range(7):
            current_date = week_start_date + datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            if date_str in self.data["daily_records"]:
                for subject, record in self.data["daily_records"][date_str].items():
                    week_stats[subject]["hours"] += record["hours"]
                    week_stats[subject]["tasks"] += len(record["tasks_completed"])
        
        return dict(week_stats)
    
    def generate_weekly_report(self, week_number):
        """生成周报告"""
        print(f"\n📊 第{week_number}周学习报告")
        print("=" * 60)
        
        # 计算本周统计
        week_start = datetime.date(2024, 7, 1) + datetime.timedelta(weeks=week_number-1)
        week_stats = self.calculate_weekly_stats(week_start)
        
        if not week_stats:
            print("本周暂无学习记录")
            return
        
        total_hours = sum(stats["hours"] for stats in week_stats.values())
        total_tasks = sum(stats["tasks"] for stats in week_stats.values())
        
        print(f"📅 时间范围: {week_start} 至 {week_start + datetime.timedelta(days=6)}")
        print(f"⏰ 总学习时间: {total_hours} 小时")
        print(f"✅ 完成任务数: {total_tasks} 个")
        print(f"📈 日均学习时间: {total_hours/7:.1f} 小时")
        
        print("\n📚 各科目详情:")
        for subject, stats in week_stats.items():
            subject_name = self.subjects[subject]['name']
            target_weekly = self.subjects[subject]['target_hours'] / 8  # 8周总计划
            completion_rate = (stats["hours"] / target_weekly) * 100 if target_weekly > 0 else 0
            
            print(f"  {subject_name}:")
            print(f"    学习时间: {stats['hours']} 小时 (目标: {target_weekly:.1f}小时)")
            print(f"    完成度: {completion_rate:.1f}%")
            print(f"    任务数: {stats['tasks']} 个")
        
        # 学习建议
        print(f"\n💡 学习建议:")
        if total_hours < 45:  # 目标每周63小时
            print("  - 学习时间不足，建议增加每日学习时间")
        elif total_hours > 70:
            print("  - 学习强度较高，注意劳逸结合")
        else:
            print("  - 学习节奏良好，继续保持")
    
    def show_progress_summary(self):
        """显示总体进度摘要"""
        print("\n📈 学习进度总览")
        print("=" * 50)
        
        total_hours = defaultdict(float)
        
        for date_records in self.data["daily_records"].values():
            for subject, record in date_records.items():
                total_hours[subject] += record["hours"]
        
        if not total_hours:
            print("暂无学习记录")
            return
        
        print("各科目累计学习时间:")
        for subject, hours in total_hours.items():
            subject_name = self.subjects[subject]['name']
            target_hours = self.subjects[subject]['target_hours']
            completion_rate = (hours / target_hours) * 100
            
            print(f"  {subject_name}: {hours}小时 / {target_hours}小时 ({completion_rate:.1f}%)")
        
        total_study_hours = sum(total_hours.values())
        total_target_hours = sum(config['target_hours'] for config in self.subjects.values())
        overall_completion = (total_study_hours / total_target_hours) * 100
        
        print(f"\n总体进度: {total_study_hours}小时 / {total_target_hours}小时 ({overall_completion:.1f}%)")
    
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n📚 暑假学习进度跟踪器")
            print("=" * 40)
            print("1. 添加今日学习记录")
            print("2. 添加学习成就")
            print("3. 查看进度总览")
            print("4. 生成周报告")
            print("5. 查看所有成就")
            print("0. 退出")
            
            choice = input("\n请选择功能 (0-5): ").strip()
            
            if choice == '1':
                self.add_daily_record_interactive()
            elif choice == '2':
                self.add_achievement_interactive()
            elif choice == '3':
                self.show_progress_summary()
            elif choice == '4':
                week = input("请输入周数 (1-8): ").strip()
                try:
                    self.generate_weekly_report(int(week))
                except ValueError:
                    print("请输入有效的周数")
            elif choice == '5':
                self.show_all_achievements()
            elif choice == '0':
                print("👋 再见！继续加油学习！")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def add_daily_record_interactive(self):
        """交互式添加每日记录"""
        print("\n📝 添加今日学习记录")
        
        # 选择日期
        date_input = input("请输入日期 (YYYY-MM-DD，回车使用今天): ").strip()
        if not date_input:
            date = datetime.date.today()
        else:
            try:
                date = datetime.datetime.strptime(date_input, "%Y-%m-%d").date()
            except ValueError:
                print("❌ 日期格式错误")
                return
        
        # 选择科目
        print("\n可选科目:")
        for i, (key, config) in enumerate(self.subjects.items(), 1):
            print(f"{i}. {config['name']} ({key})")
        
        subject_choice = input("请选择科目编号: ").strip()
        try:
            subject_key = list(self.subjects.keys())[int(subject_choice) - 1]
        except (ValueError, IndexError):
            print("❌ 无效选择")
            return
        
        # 输入学习时间
        try:
            hours = float(input("请输入学习时间 (小时): ").strip())
        except ValueError:
            print("❌ 请输入有效数字")
            return
        
        # 输入完成任务
        tasks_input = input("请输入完成的任务 (用逗号分隔): ").strip()
        tasks = [task.strip() for task in tasks_input.split(',') if task.strip()]
        
        # 输入备注
        notes = input("请输入备注 (可选): ").strip()
        
        # 添加记录
        self.add_daily_record(date, subject_key, hours, tasks, notes)
    
    def add_achievement_interactive(self):
        """交互式添加成就"""
        print("\n🏆 添加学习成就")
        
        title = input("成就标题: ").strip()
        description = input("成就描述: ").strip()
        category = input("成就类别 (AI/CTF/Math/CS408/Other): ").strip()
        
        if title and description:
            self.add_achievement(title, description, category)
        else:
            print("❌ 标题和描述不能为空")
    
    def show_all_achievements(self):
        """显示所有成就"""
        print("\n🏆 所有学习成就")
        print("=" * 50)
        
        if not self.data["achievements"]:
            print("暂无成就记录")
            return
        
        # 按类别分组
        achievements_by_category = defaultdict(list)
        for ach in self.data["achievements"]:
            achievements_by_category[ach["category"]].append(ach)
        
        for category, achievements in achievements_by_category.items():
            print(f"\n📚 {category} ({len(achievements)}个):")
            for ach in sorted(achievements, key=lambda x: x["date"]):
                print(f"  🎉 {ach['date']} - {ach['title']}")
                print(f"      {ach['description']}")

def main():
    """主函数"""
    tracker = StudyProgressTracker()
    
    print("🎯 欢迎使用暑假学习进度跟踪器！")
    print("这个工具将帮助你记录和跟踪学习进度。")
    
    # 启动交互式菜单
    tracker.interactive_menu()

if __name__ == "__main__":
    main()
