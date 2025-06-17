#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTF 逆向工程常用脚本工具集
包含反混淆、算法还原、动态分析等常用工具
"""

import os
import re
import subprocess
import struct
import string
import hashlib
from collections import Counter

class ReverseTools:
    def __init__(self):
        self.printable_chars = string.printable
        
    def extract_strings_advanced(self, file_path, min_length=4, encoding='auto'):
        """高级字符串提取"""
        print(f"[+] Extracting strings from {file_path}")
        
        strings_found = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encodings = ['ascii', 'utf-8', 'utf-16le', 'utf-16be'] if encoding == 'auto' else [encoding]
            
            for enc in encodings:
                try:
                    if enc in ['utf-16le', 'utf-16be']:
                        # Unicode字符串
                        pattern = b'(?:[!-~]\x00){%d,}' % min_length if enc == 'utf-16le' else b'(?:\x00[!-~]){%d,}' % min_length
                        matches = re.findall(pattern, data)
                        for match in matches:
                            try:
                                decoded = match.decode(enc)
                                if len(decoded) >= min_length:
                                    strings_found.append((decoded, enc, data.find(match)))
                            except:
                                continue
                    else:
                        # ASCII/UTF-8字符串
                        pattern = b'[!-~]{%d,}' % min_length
                        matches = re.findall(pattern, data)
                        for match in matches:
                            try:
                                decoded = match.decode(enc)
                                strings_found.append((decoded, enc, data.find(match)))
                            except:
                                continue
                except:
                    continue
            
            # 去重并排序
            unique_strings = list(set(strings_found))
            unique_strings.sort(key=lambda x: x[2])  # 按偏移量排序
            
            print(f"[+] Found {len(unique_strings)} unique strings")
            return unique_strings
            
        except Exception as e:
            print(f"[-] Error extracting strings: {e}")
            return []
    
    def xor_analysis(self, data, key_length_range=(1, 16)):
        """XOR加密分析"""
        print("[+] Performing XOR analysis")
        
        results = []
        
        if isinstance(data, str):
            data = data.encode()
        
        for key_len in range(key_length_range[0], key_length_range[1] + 1):
            # 尝试不同的密钥长度
            best_score = 0
            best_key = None
            best_plaintext = None
            
            # 频率分析找最可能的密钥
            for key_byte in range(256):
                key = bytes([key_byte] * key_len)
                
                # XOR解密
                decrypted = bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))
                
                # 计算可打印字符比例
                printable_count = sum(1 for c in decrypted if chr(c) in self.printable_chars)
                score = printable_count / len(decrypted) if decrypted else 0
                
                if score > best_score:
                    best_score = score
                    best_key = key
                    best_plaintext = decrypted
            
            if best_score > 0.7:  # 阈值
                results.append({
                    'key_length': key_len,
                    'key': best_key,
                    'score': best_score,
                    'plaintext': best_plaintext[:100]  # 前100字节
                })
        
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"[+] Found {len(results)} potential XOR keys")
        for result in results[:5]:  # 显示前5个结果
            print(f"    Key length: {result['key_length']}, Key: {result['key'].hex()}, Score: {result['score']:.2f}")
        
        return results
    
    def caesar_cipher_analysis(self, text):
        """凯撒密码分析"""
        print("[+] Performing Caesar cipher analysis")
        
        results = []
        
        for shift in range(26):
            decrypted = ""
            for char in text.upper():
                if char.isalpha():
                    decrypted += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                else:
                    decrypted += char
            
            # 计算英文字母频率得分
            score = self._calculate_english_score(decrypted)
            results.append({
                'shift': shift,
                'text': decrypted,
                'score': score
            })
        
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"[+] Best Caesar cipher candidates:")
        for result in results[:3]:
            print(f"    Shift {result['shift']}: {result['text'][:50]}... (Score: {result['score']:.2f})")
        
        return results
    
    def _calculate_english_score(self, text):
        """计算英文文本得分"""
        # 英文字母频率
        english_freq = {
            'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7,
            'S': 6.3, 'H': 6.1, 'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8,
            'U': 2.8, 'M': 2.4, 'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0,
            'P': 1.9, 'B': 1.3, 'V': 1.0, 'K': 0.8, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }
        
        if not text:
            return 0
        
        # 计算文本中字母频率
        letter_count = Counter(c for c in text.upper() if c.isalpha())
        total_letters = sum(letter_count.values())
        
        if total_letters == 0:
            return 0
        
        # 计算与英文频率的匹配度
        score = 0
        for letter, expected_freq in english_freq.items():
            actual_freq = (letter_count.get(letter, 0) / total_letters) * 100
            score += min(expected_freq, actual_freq)
        
        return score
    
    def base64_variants_decode(self, encoded_text):
        """Base64变种解码"""
        print("[+] Trying Base64 variants")
        
        import base64
        
        variants = {
            'Standard': lambda x: base64.b64decode(x),
            'URL Safe': lambda x: base64.urlsafe_b64decode(x),
            'Custom (A-Z, a-z, 0-9, +, /)': lambda x: base64.b64decode(x),
            'Custom (A-Z, a-z, 0-9, -, _)': lambda x: base64.urlsafe_b64decode(x),
        }
        
        results = {}
        
        for variant_name, decode_func in variants.items():
            try:
                decoded = decode_func(encoded_text)
                # 尝试解码为文本
                try:
                    text = decoded.decode('utf-8')
                    results[variant_name] = text
                except:
                    results[variant_name] = decoded.hex()
            except Exception as e:
                results[variant_name] = f"Failed: {e}"
        
        return results
    
    def detect_packing(self, file_path):
        """检测文件是否被加壳"""
        print(f"[+] Detecting packing for {file_path}")
        
        indicators = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 检查熵值
            entropy = self._calculate_entropy(data)
            if entropy > 7.5:
                indicators.append(f"High entropy: {entropy:.2f}")
            
            # 检查已知加壳器签名
            packer_signatures = {
                b'UPX!': 'UPX Packer',
                b'FSG!': 'FSG Packer',
                b'PECompact': 'PECompact',
                b'ASPack': 'ASPack',
                b'Themida': 'Themida',
                b'VMProtect': 'VMProtect'
            }
            
            for signature, packer_name in packer_signatures.items():
                if signature in data:
                    indicators.append(f"Packer signature found: {packer_name}")
            
            # 检查导入表异常
            if b'GetProcAddress' in data and b'LoadLibrary' in data:
                if data.count(b'GetProcAddress') < 5:  # 导入函数过少
                    indicators.append("Suspicious import table")
            
            # 检查节区特征
            if b'.text' in data and b'.data' in data:
                text_pos = data.find(b'.text')
                data_pos = data.find(b'.data')
                if abs(text_pos - data_pos) < 100:  # 节区过于接近
                    indicators.append("Suspicious section layout")
            
        except Exception as e:
            print(f"[-] Error detecting packing: {e}")
            return []
        
        if indicators:
            print("[+] Packing indicators found:")
            for indicator in indicators:
                print(f"    - {indicator}")
        else:
            print("[-] No obvious packing detected")
        
        return indicators
    
    def _calculate_entropy(self, data):
        """计算数据熵值"""
        if not data:
            return 0
        
        # 计算字节频率
        byte_counts = Counter(data)
        data_len = len(data)
        
        # 计算熵
        entropy = 0
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def disassemble_shellcode(self, shellcode_hex):
        """反汇编shellcode"""
        print("[+] Disassembling shellcode")
        
        try:
            # 将十六进制转换为字节
            shellcode_bytes = bytes.fromhex(shellcode_hex.replace(' ', ''))
            
            # 保存到临时文件
            temp_file = '/tmp/shellcode.bin'
            with open(temp_file, 'wb') as f:
                f.write(shellcode_bytes)
            
            # 使用objdump反汇编
            result = subprocess.run([
                'objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', temp_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("[+] Disassembly:")
                print(result.stdout)
                return result.stdout
            else:
                print("[-] Disassembly failed")
                return None
                
        except Exception as e:
            print(f"[-] Error disassembling shellcode: {e}")
            return None
        finally:
            # 清理临时文件
            if os.path.exists('/tmp/shellcode.bin'):
                os.remove('/tmp/shellcode.bin')
    
    def analyze_pe_imports(self, file_path):
        """分析PE文件导入表"""
        print(f"[+] Analyzing PE imports for {file_path}")
        
        try:
            # 使用objdump分析导入表
            result = subprocess.run([
                'objdump', '-p', file_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                imports = []
                lines = result.stdout.split('\n')
                
                for line in lines:
                    if 'DLL Name:' in line:
                        dll_name = line.split('DLL Name:')[1].strip()
                        imports.append({'dll': dll_name, 'functions': []})
                    elif line.strip() and imports and not line.startswith('\t'):
                        # 函数名
                        func_name = line.strip()
                        if func_name and not func_name.startswith('DLL'):
                            imports[-1]['functions'].append(func_name)
                
                print(f"[+] Found {len(imports)} imported DLLs")
                for imp in imports:
                    print(f"    {imp['dll']}: {len(imp['functions'])} functions")
                
                return imports
            else:
                print("[-] Failed to analyze imports")
                return []
                
        except Exception as e:
            print(f"[-] Error analyzing PE imports: {e}")
            return []
    
    def deobfuscate_simple(self, obfuscated_code):
        """简单反混淆"""
        print("[+] Attempting simple deobfuscation")
        
        deobfuscated = obfuscated_code
        
        # 移除多余空格
        deobfuscated = re.sub(r'\s+', ' ', deobfuscated)
        
        # 替换常见混淆
        replacements = {
            r'eval\s*\(\s*': 'eval(',
            r'document\s*\[\s*["\']write["\']\s*\]': 'document.write',
            r'window\s*\[\s*["\']eval["\']\s*\]': 'eval',
            r'String\s*\.\s*fromCharCode': 'String.fromCharCode',
        }
        
        for pattern, replacement in replacements.items():
            deobfuscated = re.sub(pattern, replacement, deobfuscated, flags=re.IGNORECASE)
        
        # 解码十六进制字符串
        hex_pattern = r'\\x([0-9a-fA-F]{2})'
        def hex_replace(match):
            return chr(int(match.group(1), 16))
        
        deobfuscated = re.sub(hex_pattern, hex_replace, deobfuscated)
        
        # 解码Unicode字符串
        unicode_pattern = r'\\u([0-9a-fA-F]{4})'
        def unicode_replace(match):
            return chr(int(match.group(1), 16))
        
        deobfuscated = re.sub(unicode_pattern, unicode_replace, deobfuscated)
        
        return deobfuscated
    
    def find_crypto_constants(self, file_path):
        """查找加密常量"""
        print(f"[+] Searching for crypto constants in {file_path}")
        
        crypto_constants = {
            # MD5
            b'\x67\x45\x23\x01': 'MD5 Initial Hash Value 1',
            b'\xef\xcd\xab\x89': 'MD5 Initial Hash Value 2',
            b'\x98\xba\xdc\xfe': 'MD5 Initial Hash Value 3',
            b'\x10\x32\x54\x76': 'MD5 Initial Hash Value 4',
            
            # SHA-1
            b'\x67\x45\x23\x01': 'SHA-1 Initial Hash Value 1',
            b'\xef\xcd\xab\x89': 'SHA-1 Initial Hash Value 2',
            
            # AES S-Box
            b'\x63\x7c\x77\x7b': 'AES S-Box Start',
            
            # DES S-Box
            b'\x0e\x04\x0d\x01': 'DES S-Box Pattern',
        }
        
        found_constants = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            for constant, description in crypto_constants.items():
                positions = []
                start = 0
                while True:
                    pos = data.find(constant, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                if positions:
                    found_constants.append({
                        'constant': constant.hex(),
                        'description': description,
                        'positions': positions
                    })
            
            if found_constants:
                print("[+] Crypto constants found:")
                for const in found_constants:
                    print(f"    {const['description']}: {const['constant']} at {const['positions']}")
            else:
                print("[-] No known crypto constants found")
                
        except Exception as e:
            print(f"[-] Error searching for crypto constants: {e}")
        
        return found_constants

def main():
    """主函数 - 演示用法"""
    tools = ReverseTools()
    
    print("CTF 逆向工程工具集")
    print("=" * 50)
    
    # 示例用法
    print("\n1. 字符串提取示例:")
    print("strings = tools.extract_strings_advanced('binary_file')")
    
    print("\n2. XOR分析示例:")
    print("results = tools.xor_analysis(encrypted_data)")
    
    print("\n3. 凯撒密码分析示例:")
    print("results = tools.caesar_cipher_analysis('KHOOR ZRUOG')")
    
    print("\n4. 加壳检测示例:")
    print("indicators = tools.detect_packing('suspicious.exe')")
    
    print("\n5. PE导入表分析示例:")
    print("imports = tools.analyze_pe_imports('program.exe')")
    
    print("\n6. 反混淆示例:")
    print("clean_code = tools.deobfuscate_simple(obfuscated_js)")
    
    print("\n7. 加密常量查找示例:")
    print("constants = tools.find_crypto_constants('crypto_program')")

if __name__ == "__main__":
    main()
