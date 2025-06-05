"""
逆向工程基础学习
包含二进制分析、汇编语言、调试技术等
"""

import struct
import binascii
import subprocess
import os
import tempfile

class ReverseEngineeringBasics:
    """逆向工程基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("🔍 逆向工程基础学习系统")
        print("=" * 50)
    
    def binary_analysis_demo(self):
        """二进制文件分析演示"""
        print("🔢 二进制文件分析")
        print("=" * 30)
        
        def analyze_file_header(file_data):
            """分析文件头部信息"""
            if len(file_data) < 4:
                return "文件太小"
            
            # 检查常见文件格式的魔数
            magic_numbers = {
                b'\x7fELF': 'ELF可执行文件',
                b'MZ': 'PE可执行文件',
                b'\x89PNG': 'PNG图片',
                b'\xff\xd8\xff': 'JPEG图片',
                b'PK\x03\x04': 'ZIP压缩文件',
                b'\x50\x4b\x03\x04': 'ZIP压缩文件',
                b'\xca\xfe\xba\xbe': 'Java Class文件',
                b'\xfe\xed\xfa\xce': 'Mach-O可执行文件'
            }
            
            for magic, file_type in magic_numbers.items():
                if file_data.startswith(magic):
                    return file_type
            
            return "未知文件类型"
        
        def hex_dump(data, length=16):
            """十六进制转储"""
            result = []
            for i in range(0, len(data), length):
                chunk = data[i:i+length]
                hex_part = ' '.join(f'{b:02x}' for b in chunk)
                ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
                result.append(f'{i:08x}: {hex_part:<48} |{ascii_part}|')
            return '\n'.join(result)
        
        def extract_strings(data, min_length=4):
            """提取可打印字符串"""
            strings = []
            current_string = ""
            
            for byte in data:
                if 32 <= byte <= 126:  # 可打印ASCII字符
                    current_string += chr(byte)
                else:
                    if len(current_string) >= min_length:
                        strings.append(current_string)
                    current_string = ""
            
            if len(current_string) >= min_length:
                strings.append(current_string)
            
            return strings
        
        # 创建示例二进制数据
        sample_data = bytearray()
        sample_data.extend(b'\x7fELF')  # ELF魔数
        sample_data.extend(b'\x01\x01\x01\x00')  # ELF头部信息
        sample_data.extend(b'Hello World!\x00')  # 字符串
        sample_data.extend(b'\x41\x42\x43\x44')  # 一些数据
        sample_data.extend(b'Secret Flag: flag{reverse_engineering}\x00')
        sample_data.extend(bytes(range(256)))  # 各种字节值
        
        print("1. 文件类型识别:")
        file_type = analyze_file_header(sample_data)
        print(f"文件类型: {file_type}")
        
        print(f"\n2. 十六进制转储 (前128字节):")
        print(hex_dump(sample_data[:128]))
        
        print(f"\n3. 字符串提取:")
        strings = extract_strings(sample_data)
        for i, string in enumerate(strings[:10]):  # 只显示前10个
            print(f"  {i+1}: {string}")
        
        self.examples_completed.append("二进制分析")
    
    def assembly_basics_demo(self):
        """汇编语言基础演示"""
        print("\n⚙️ 汇编语言基础")
        print("=" * 30)
        
        def explain_x86_instructions():
            """解释常见x86指令"""
            instructions = {
                'mov': '数据传送指令',
                'add': '加法指令',
                'sub': '减法指令',
                'mul': '乘法指令',
                'div': '除法指令',
                'cmp': '比较指令',
                'jmp': '无条件跳转',
                'je/jz': '相等/零时跳转',
                'jne/jnz': '不等/非零时跳转',
                'jg/jnle': '大于时跳转',
                'jl/jnge': '小于时跳转',
                'call': '函数调用',
                'ret': '函数返回',
                'push': '压栈',
                'pop': '出栈',
                'nop': '空操作'
            }
            
            print("常见x86汇编指令:")
            for inst, desc in instructions.items():
                print(f"  {inst:<8}: {desc}")
        
        def simulate_simple_program():
            """模拟简单程序执行"""
            print(f"\n🖥️ 模拟程序执行:")
            print("程序功能：计算两个数的和")
            
            # 模拟寄存器
            registers = {'eax': 0, 'ebx': 0, 'ecx': 0, 'edx': 0}
            
            # 模拟汇编代码
            assembly_code = [
                ('mov', 'eax', 10),      # 将10存入eax
                ('mov', 'ebx', 20),      # 将20存入ebx
                ('add', 'eax', 'ebx'),   # eax = eax + ebx
                ('mov', 'ecx', 'eax'),   # 将结果存入ecx
            ]
            
            print("\n汇编代码执行过程:")
            for i, instruction in enumerate(assembly_code):
                if instruction[0] == 'mov':
                    if isinstance(instruction[2], int):
                        registers[instruction[1]] = instruction[2]
                        print(f"{i+1}. mov {instruction[1]}, {instruction[2]} -> {instruction[1]} = {instruction[2]}")
                    else:
                        registers[instruction[1]] = registers[instruction[2]]
                        print(f"{i+1}. mov {instruction[1]}, {instruction[2]} -> {instruction[1]} = {registers[instruction[2]]}")
                
                elif instruction[0] == 'add':
                    registers[instruction[1]] += registers[instruction[2]]
                    print(f"{i+1}. add {instruction[1]}, {instruction[2]} -> {instruction[1]} = {registers[instruction[1]]}")
                
                print(f"   寄存器状态: {registers}")
            
            print(f"\n最终结果: {registers['ecx']}")
        
        def analyze_control_flow():
            """分析控制流"""
            print(f"\n🔄 控制流分析:")
            print("示例：简单的if-else结构")
            
            pseudo_code = """
            if (x > 5):
                result = x * 2
            else:
                result = x + 1
            """
            
            assembly_equivalent = """
            cmp  eax, 5        ; 比较x和5
            jle  else_branch   ; 如果x <= 5，跳转到else分支
            
            ; if分支
            mov  ebx, eax      ; ebx = x
            shl  ebx, 1        ; ebx = ebx * 2 (左移1位)
            jmp  end           ; 跳转到结束
            
            else_branch:
            mov  ebx, eax      ; ebx = x
            add  ebx, 1        ; ebx = ebx + 1
            
            end:
            ; 结果在ebx中
            """
            
            print("伪代码:")
            print(pseudo_code)
            print("对应的汇编代码:")
            print(assembly_equivalent)
        
        explain_x86_instructions()
        simulate_simple_program()
        analyze_control_flow()
        
        self.examples_completed.append("汇编基础")
    
    def debugging_techniques_demo(self):
        """调试技术演示"""
        print("\n🐛 调试技术学习")
        print("=" * 30)
        
        def static_analysis_demo():
            """静态分析演示"""
            print("🔍 静态分析技术:")
            
            # 模拟一个简单的crackme程序
            crackme_code = '''
            def check_password(password):
                if len(password) != 8:
                    return False
                
                # 简单的异或加密检查
                target = [0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x21, 0x21, 0x21]
                for i, char in enumerate(password):
                    if ord(char) ^ 0x20 != target[i]:
                        return False
                
                return True
            '''
            
            print("示例程序代码:")
            print(crackme_code)
            
            print("\n静态分析步骤:")
            print("1. 分析程序逻辑：检查密码长度为8")
            print("2. 发现加密算法：异或运算 (char ^ 0x20)")
            print("3. 逆向计算：target[i] ^ 0x20 = 正确字符")
            
            # 计算正确密码
            target = [0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x21, 0x21, 0x21]
            correct_password = ''.join(chr(byte ^ 0x20) for byte in target)
            print(f"4. 计算结果：正确密码是 '{correct_password}'")
        
        def dynamic_analysis_demo():
            """动态分析演示"""
            print(f"\n🏃 动态分析技术:")
            
            print("动态分析工具和技术:")
            tools = {
                'GDB': 'Linux下的调试器',
                'OllyDbg': 'Windows下的调试器',
                'x64dbg': '现代Windows调试器',
                'IDA Pro': '专业逆向工程工具',
                'Ghidra': 'NSA开源逆向工具',
                'Radare2': '开源逆向框架',
                'strace': 'Linux系统调用跟踪',
                'ltrace': 'Linux库函数调用跟踪'
            }
            
            for tool, desc in tools.items():
                print(f"  {tool:<10}: {desc}")
            
            print(f"\n动态分析步骤:")
            print("1. 设置断点：在关键函数处暂停")
            print("2. 单步执行：逐行分析程序行为")
            print("3. 观察内存：查看变量和数据变化")
            print("4. 修改执行：改变寄存器或内存值")
            print("5. 跟踪调用：记录函数调用序列")
        
        def anti_debugging_demo():
            """反调试技术演示"""
            print(f"\n🛡️ 反调试技术:")
            
            anti_debug_techniques = {
                'IsDebuggerPresent': '检查PEB中的调试标志',
                'CheckRemoteDebuggerPresent': '检查远程调试器',
                'NtQueryInformationProcess': '查询进程调试信息',
                '时间检测': '测量代码执行时间',
                '异常处理': '利用调试器异常处理差异',
                '硬件断点检测': '检查调试寄存器',
                '内存保护': '检测内存访问模式',
                '父进程检测': '检查启动进程'
            }
            
            print("常见反调试技术:")
            for technique, desc in anti_debug_techniques.items():
                print(f"  {technique:<20}: {desc}")
            
            print(f"\n绕过反调试的方法:")
            print("1. 补丁技术：修改反调试代码")
            print("2. Hook技术：拦截API调用")
            print("3. 虚拟机：在隔离环境中分析")
            print("4. 脚本自动化：自动化绕过过程")
        
        static_analysis_demo()
        dynamic_analysis_demo()
        anti_debugging_demo()
        
        self.examples_completed.append("调试技术")
    
    def packing_unpacking_demo(self):
        """加壳脱壳演示"""
        print("\n📦 加壳与脱壳技术")
        print("=" * 30)
        
        def simple_xor_packer():
            """简单异或加壳演示"""
            print("🔐 简单异或加壳演示:")
            
            # 原始"程序"数据
            original_data = b"This is a secret program code!"
            key = 0x42
            
            print(f"原始数据: {original_data}")
            print(f"加密密钥: 0x{key:02x}")
            
            # 加壳（异或加密）
            packed_data = bytes(b ^ key for b in original_data)
            print(f"加壳后数据: {packed_data.hex()}")
            
            # 脱壳（异或解密）
            unpacked_data = bytes(b ^ key for b in packed_data)
            print(f"脱壳后数据: {unpacked_data}")
            
            # 验证
            print(f"脱壳成功: {original_data == unpacked_data}")
        
        def analyze_packer_stub():
            """分析加壳程序的存根代码"""
            print(f"\n🔍 加壳程序结构分析:")
            
            packer_structure = """
            加壳程序典型结构:
            
            ┌─────────────────┐
            │   程序入口点     │ <- 执行从这里开始
            ├─────────────────┤
            │   解壳存根代码   │ <- 负责解密和恢复原程序
            ├─────────────────┤
            │   加密的原程序   │ <- 被加密/压缩的原始代码
            ├─────────────────┤
            │   导入表修复     │ <- 恢复API调用
            └─────────────────┘
            """
            
            print(packer_structure)
            
            print("解壳步骤:")
            print("1. 找到OEP (Original Entry Point)")
            print("2. 分析解壳算法")
            print("3. 让程序自解壳或手动解壳")
            print("4. 转储内存中的原程序")
            print("5. 修复导入表")
            print("6. 重建PE文件")
        
        def common_packers():
            """常见加壳工具介绍"""
            print(f"\n📋 常见加壳工具:")
            
            packers = {
                'UPX': '开源压缩壳，易于脱壳',
                'ASPack': '商业压缩壳',
                'PECompact': '压缩和保护壳',
                'Themida': '强保护壳，反调试功能强',
                'VMProtect': '虚拟化保护',
                'Enigma': '多功能保护工具',
                'Armadillo': '老牌保护工具',
                'ASProtect': '反调试保护'
            }
            
            for packer, desc in packers.items():
                print(f"  {packer:<12}: {desc}")
            
            print(f"\n脱壳工具:")
            unpackers = {
                'OllyDump': 'OllyDbg脱壳插件',
                'ImportREC': '导入表重建工具',
                'PEiD': '加壳检测工具',
                'Detect It Easy': '现代检测工具',
                'Universal Unpacker': '通用脱壳工具'
            }
            
            for tool, desc in unpackers.items():
                print(f"  {tool:<18}: {desc}")
        
        simple_xor_packer()
        analyze_packer_stub()
        common_packers()
        
        self.examples_completed.append("加壳脱壳")
    
    def run_all_demos(self):
        """运行所有逆向工程演示"""
        print("🔍 逆向工程基础完整学习")
        print("=" * 60)
        
        self.binary_analysis_demo()
        self.assembly_basics_demo()
        self.debugging_techniques_demo()
        self.packing_unpacking_demo()
        
        print(f"\n🎉 逆向工程基础学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. 二进制分析 - 文件格式识别和数据提取")
        print("2. 汇编基础 - 理解程序的底层执行")
        print("3. 调试技术 - 静态和动态分析方法")
        print("4. 加壳脱壳 - 程序保护和破解技术")
        
        print(f"\n🎯 CTF逆向技巧:")
        print("1. 先静态分析，了解程序结构")
        print("2. 使用字符串搜索寻找线索")
        print("3. 动态调试验证分析结果")
        print("4. 关注算法逻辑而非具体实现")

def main():
    """主函数"""
    reverse_eng = ReverseEngineeringBasics()
    reverse_eng.run_all_demos()
    
    print("\n💡 进阶学习建议:")
    print("1. 学习更多架构：ARM、MIPS、x64")
    print("2. 深入研究文件格式：PE、ELF、Mach-O")
    print("3. 实践恶意软件分析")
    print("4. 学习现代保护技术：CFG、CET等")

if __name__ == "__main__":
    main()
