"""
计算机科学基础 (CS408)
涵盖数据结构、计算机组成原理、操作系统、计算机网络四门课程
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class ComputerScienceFundamentals:
    """计算机科学基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("💻 计算机科学基础学习系统 (CS408)")
        print("=" * 50)
        print("涵盖：数据结构、组成原理、操作系统、计算机网络")
    
    def data_structures_demo(self):
        """数据结构学习"""
        print("\n📊 数据结构基础")
        print("=" * 40)
        
        # 线性表实现
        print("1. 线性表 - 顺序存储与链式存储")
        
        class ArrayList:
            """顺序表实现"""
            def __init__(self, capacity=10):
                self.data = [None] * capacity
                self.size = 0
                self.capacity = capacity
            
            def insert(self, index, value):
                if self.size >= self.capacity:
                    return False
                for i in range(self.size, index, -1):
                    self.data[i] = self.data[i-1]
                self.data[index] = value
                self.size += 1
                return True
            
            def delete(self, index):
                if index < 0 or index >= self.size:
                    return None
                value = self.data[index]
                for i in range(index, self.size-1):
                    self.data[i] = self.data[i+1]
                self.size -= 1
                return value
        
        class ListNode:
            """链表节点"""
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        
        class LinkedList:
            """单链表实现"""
            def __init__(self):
                self.head = None
                self.size = 0
            
            def insert(self, index, val):
                if index == 0:
                    new_node = ListNode(val, self.head)
                    self.head = new_node
                else:
                    prev = self.head
                    for _ in range(index - 1):
                        if prev is None:
                            return False
                        prev = prev.next
                    new_node = ListNode(val, prev.next)
                    prev.next = new_node
                self.size += 1
                return True
        
        # 演示使用
        arr_list = ArrayList()
        arr_list.insert(0, 1)
        arr_list.insert(1, 2)
        arr_list.insert(2, 3)
        print(f"  顺序表插入后大小: {arr_list.size}")
        
        linked_list = LinkedList()
        linked_list.insert(0, 1)
        linked_list.insert(1, 2)
        print(f"  链表插入后大小: {linked_list.size}")
        
        # 栈和队列
        print(f"\n2. 栈和队列")
        
        class Stack:
            """栈实现"""
            def __init__(self):
                self.items = []
            
            def push(self, item):
                self.items.append(item)
            
            def pop(self):
                return self.items.pop() if self.items else None
            
            def peek(self):
                return self.items[-1] if self.items else None
            
            def is_empty(self):
                return len(self.items) == 0
        
        class Queue:
            """队列实现"""
            def __init__(self):
                self.items = []
            
            def enqueue(self, item):
                self.items.insert(0, item)
            
            def dequeue(self):
                return self.items.pop() if self.items else None
            
            def is_empty(self):
                return len(self.items) == 0
        
        # 演示栈的应用：括号匹配
        def check_parentheses(expression):
            stack = Stack()
            pairs = {'(': ')', '[': ']', '{': '}'}
            
            for char in expression:
                if char in pairs:
                    stack.push(char)
                elif char in pairs.values():
                    if stack.is_empty():
                        return False
                    if pairs[stack.pop()] != char:
                        return False
            
            return stack.is_empty()
        
        test_expr = "((()))"
        print(f"  括号匹配检查 '{test_expr}': {check_parentheses(test_expr)}")
        
        # 树结构
        print(f"\n3. 树结构")
        
        class TreeNode:
            """二叉树节点"""
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        
        class BinaryTree:
            """二叉树实现"""
            def __init__(self):
                self.root = None
            
            def inorder_traversal(self, node):
                """中序遍历"""
                if node:
                    self.inorder_traversal(node.left)
                    print(node.val, end=' ')
                    self.inorder_traversal(node.right)
            
            def preorder_traversal(self, node):
                """前序遍历"""
                if node:
                    print(node.val, end=' ')
                    self.preorder_traversal(node.left)
                    self.preorder_traversal(node.right)
            
            def postorder_traversal(self, node):
                """后序遍历"""
                if node:
                    self.postorder_traversal(node.left)
                    self.postorder_traversal(node.right)
                    print(node.val, end=' ')
        
        # 创建示例二叉树
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        
        tree = BinaryTree()
        tree.root = root
        
        print(f"  前序遍历: ", end='')
        tree.preorder_traversal(root)
        print(f"\n  中序遍历: ", end='')
        tree.inorder_traversal(root)
        print(f"\n  后序遍历: ", end='')
        tree.postorder_traversal(root)
        print()
        
        self.examples_completed.append("数据结构")
    
    def computer_organization_demo(self):
        """计算机组成原理学习"""
        print("\n🔧 计算机组成原理")
        print("=" * 40)
        
        print("1. 数制转换")
        
        def decimal_to_binary(n):
            """十进制转二进制"""
            if n == 0:
                return "0"
            binary = ""
            while n > 0:
                binary = str(n % 2) + binary
                n //= 2
            return binary
        
        def binary_to_decimal(binary):
            """二进制转十进制"""
            decimal = 0
            for i, bit in enumerate(reversed(binary)):
                decimal += int(bit) * (2 ** i)
            return decimal
        
        # 演示数制转换
        test_num = 42
        binary_result = decimal_to_binary(test_num)
        decimal_result = binary_to_decimal(binary_result)
        
        print(f"  十进制 {test_num} → 二进制 {binary_result}")
        print(f"  二进制 {binary_result} → 十进制 {decimal_result}")
        
        print(f"\n2. 补码运算")
        
        def get_complement(binary, bits=8):
            """获取补码"""
            # 确保是指定位数
            binary = binary.zfill(bits)
            
            # 如果是正数，补码就是原码
            if binary[0] == '0':
                return binary
            
            # 如果是负数，先求反码再加1
            inverted = ''.join('1' if bit == '0' else '0' for bit in binary)
            
            # 加1
            carry = 1
            result = ""
            for bit in reversed(inverted):
                sum_bit = int(bit) + carry
                result = str(sum_bit % 2) + result
                carry = sum_bit // 2
            
            return result
        
        print(f"  原码 10000001 的补码: {get_complement('10000001')}")
        
        print(f"\n3. CPU指令执行过程")
        print("  取指 → 译码 → 执行 → 写回")
        print("  • 取指：从内存读取指令到指令寄存器")
        print("  • 译码：分析指令操作码和操作数")
        print("  • 执行：ALU执行运算或控制器执行控制")
        print("  • 写回：将结果写回寄存器或内存")
        
        print(f"\n4. 存储器层次结构")
        storage_hierarchy = [
            ("寄存器", "1ns", "KB级", "最快"),
            ("L1缓存", "2-4ns", "32-64KB", "很快"),
            ("L2缓存", "10-20ns", "256KB-1MB", "快"),
            ("L3缓存", "40-45ns", "8-32MB", "较快"),
            ("主存", "100-300ns", "GB级", "中等"),
            ("硬盘", "5-10ms", "TB级", "慢"),
        ]
        
        print("  存储器类型    访问时间    容量      速度")
        print("  " + "-" * 45)
        for storage, time, capacity, speed in storage_hierarchy:
            print(f"  {storage:<10} {time:<10} {capacity:<8} {speed}")
        
        self.examples_completed.append("计算机组成原理")
    
    def operating_systems_demo(self):
        """操作系统学习"""
        print("\n🖥️ 操作系统原理")
        print("=" * 40)
        
        print("1. 进程调度算法")
        
        class Process:
            """进程类"""
            def __init__(self, pid, arrival_time, burst_time, priority=0):
                self.pid = pid
                self.arrival_time = arrival_time
                self.burst_time = burst_time
                self.priority = priority
                self.waiting_time = 0
                self.turnaround_time = 0
        
        def fcfs_scheduling(processes):
            """先来先服务调度"""
            processes.sort(key=lambda p: p.arrival_time)
            current_time = 0
            
            for process in processes:
                if current_time < process.arrival_time:
                    current_time = process.arrival_time
                
                process.waiting_time = current_time - process.arrival_time
                current_time += process.burst_time
                process.turnaround_time = process.waiting_time + process.burst_time
            
            return processes
        
        def sjf_scheduling(processes):
            """短作业优先调度"""
            processes.sort(key=lambda p: (p.arrival_time, p.burst_time))
            current_time = 0
            completed = []
            remaining = processes.copy()
            
            while remaining:
                # 找到已到达且执行时间最短的进程
                available = [p for p in remaining if p.arrival_time <= current_time]
                if not available:
                    current_time = min(p.arrival_time for p in remaining)
                    continue
                
                process = min(available, key=lambda p: p.burst_time)
                remaining.remove(process)
                
                process.waiting_time = current_time - process.arrival_time
                current_time += process.burst_time
                process.turnaround_time = process.waiting_time + process.burst_time
                completed.append(process)
            
            return completed
        
        # 演示进程调度
        test_processes = [
            Process("P1", 0, 7),
            Process("P2", 2, 4),
            Process("P3", 4, 1),
            Process("P4", 5, 4)
        ]
        
        fcfs_result = fcfs_scheduling([p for p in test_processes])
        print(f"  FCFS调度结果:")
        for p in fcfs_result:
            print(f"    {p.pid}: 等待时间={p.waiting_time}, 周转时间={p.turnaround_time}")
        
        print(f"\n2. 内存管理")
        print("  • 分页管理：将内存分为固定大小的页面")
        print("  • 分段管理：按程序逻辑结构分段")
        print("  • 虚拟内存：页面置换算法（LRU、FIFO、OPT）")
        
        def lru_page_replacement(pages, frame_size):
            """LRU页面置换算法"""
            frames = []
            page_faults = 0
            
            for page in pages:
                if page not in frames:
                    page_faults += 1
                    if len(frames) < frame_size:
                        frames.append(page)
                    else:
                        frames.pop(0)  # 移除最久未使用的页面
                        frames.append(page)
                else:
                    # 将页面移到最后（最近使用）
                    frames.remove(page)
                    frames.append(page)
            
            return page_faults
        
        test_pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        faults = lru_page_replacement(test_pages, 3)
        print(f"  LRU算法页面错误次数: {faults}")
        
        print(f"\n3. 同步与互斥")
        print("  • 临界区问题：多个进程访问共享资源")
        print("  • 信号量：P操作（等待）、V操作（信号）")
        print("  • 管程：高级同步原语")
        print("  • 死锁：必要条件（互斥、占有等待、非抢占、循环等待）")
        
        self.examples_completed.append("操作系统")
    
    def computer_networks_demo(self):
        """计算机网络学习"""
        print("\n🌐 计算机网络原理")
        print("=" * 40)
        
        print("1. OSI七层模型与TCP/IP四层模型")
        
        osi_layers = [
            ("应用层", "HTTP、FTP、SMTP", "为应用程序提供网络服务"),
            ("表示层", "加密、压缩", "数据格式转换"),
            ("会话层", "会话管理", "建立、管理、终止会话"),
            ("传输层", "TCP、UDP", "端到端的数据传输"),
            ("网络层", "IP、ICMP", "路由选择和转发"),
            ("数据链路层", "以太网、PPP", "帧的传输"),
            ("物理层", "电缆、光纤", "比特流传输")
        ]
        
        print("  OSI七层模型:")
        for i, (layer, protocols, function) in enumerate(osi_layers, 1):
            print(f"    {i}. {layer:<8} {protocols:<15} {function}")
        
        print(f"\n2. IP地址与子网划分")
        
        def ip_to_binary(ip):
            """IP地址转二进制"""
            octets = ip.split('.')
            binary = '.'.join(format(int(octet), '08b') for octet in octets)
            return binary
        
        def calculate_subnet(ip, subnet_mask):
            """计算子网"""
            ip_parts = [int(x) for x in ip.split('.')]
            mask_parts = [int(x) for x in subnet_mask.split('.')]
            
            network = [ip_parts[i] & mask_parts[i] for i in range(4)]
            broadcast = [network[i] | (255 - mask_parts[i]) for i in range(4)]
            
            return '.'.join(map(str, network)), '.'.join(map(str, broadcast))
        
        test_ip = "192.168.1.100"
        test_mask = "255.255.255.0"
        network, broadcast = calculate_subnet(test_ip, test_mask)
        
        print(f"  IP地址: {test_ip}")
        print(f"  子网掩码: {test_mask}")
        print(f"  网络地址: {network}")
        print(f"  广播地址: {broadcast}")
        print(f"  二进制表示: {ip_to_binary(test_ip)}")
        
        print(f"\n3. TCP协议特性")
        print("  • 面向连接：三次握手建立连接")
        print("  • 可靠传输：确认应答、超时重传")
        print("  • 流量控制：滑动窗口机制")
        print("  • 拥塞控制：慢启动、拥塞避免")
        
        print(f"\n4. HTTP协议")
        print("  • 请求方法：GET、POST、PUT、DELETE")
        print("  • 状态码：200(成功)、404(未找到)、500(服务器错误)")
        print("  • 无状态：每个请求独立处理")
        print("  • Cookie/Session：维持状态信息")
        
        self.examples_completed.append("计算机网络")
    
    def exam_strategies(self):
        """考试策略与重点"""
        print("\n🎯 CS408考试策略")
        print("=" * 40)
        
        print("1. 各科目分值分布:")
        subjects = [
            ("数据结构", "45分", "重点：树、图、排序、查找"),
            ("计算机组成原理", "45分", "重点：CPU、存储器、指令系统"),
            ("操作系统", "35分", "重点：进程、内存、文件系统"),
            ("计算机网络", "25分", "重点：协议栈、TCP/IP、路由")
        ]
        
        for subject, score, focus in subjects:
            print(f"  {subject:<12} {score:<8} {focus}")
        
        print(f"\n2. 复习重点:")
        print("  数据结构：")
        print("    • 线性表、栈、队列的操作")
        print("    • 二叉树遍历、查找树")
        print("    • 图的遍历算法（DFS、BFS）")
        print("    • 排序算法时间复杂度")
        
        print(f"  计算机组成原理：")
        print("    • 数制转换、补码运算")
        print("    • CPU指令执行过程")
        print("    • 存储器层次结构")
        print("    • 流水线技术")
        
        print(f"  操作系统：")
        print("    • 进程调度算法")
        print("    • 内存管理方式")
        print("    • 页面置换算法")
        print("    • 死锁检测与预防")
        
        print(f"  计算机网络：")
        print("    • OSI模型各层功能")
        print("    • TCP/UDP协议特点")
        print("    • IP地址分类与子网")
        print("    • 路由算法")
        
        print(f"\n3. 答题技巧:")
        print("  • 选择题：排除法、特殊值法")
        print("  • 综合题：分步骤，写清思路")
        print("  • 算法题：先写伪代码，再实现")
        print("  • 时间分配：选择题40分钟，综合题110分钟")
        
        self.examples_completed.append("考试策略")
    
    def run_comprehensive_study(self):
        """运行综合学习"""
        print("💻 计算机科学基础综合学习")
        print("=" * 60)
        
        self.data_structures_demo()
        self.computer_organization_demo()
        self.operating_systems_demo()
        self.computer_networks_demo()
        self.exam_strategies()
        
        print(f"\n🎉 CS408学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📊 知识点掌握情况:")
        print("✅ 数据结构：线性表、树、图、排序查找")
        print("✅ 组成原理：数制、CPU、存储器、指令")
        print("✅ 操作系统：进程、内存、文件、同步")
        print("✅ 计算机网络：协议、IP、TCP、路由")
        
        print(f"\n🎯 备考建议:")
        print("1. 重视基础概念理解")
        print("2. 多做算法编程题")
        print("3. 掌握计算类题目")
        print("4. 关注历年真题")
        print("5. 平衡各科复习时间")

def main():
    """主函数"""
    cs_study = ComputerScienceFundamentals()
    cs_study.run_comprehensive_study()
    
    print("\n💡 推荐学习资源:")
    print("1. 王道考研系列教材")
    print("2. 天勤考研数据结构")
    print("3. 历年真题详解")
    print("4. 在线编程练习平台")

if __name__ == "__main__":
    main()
