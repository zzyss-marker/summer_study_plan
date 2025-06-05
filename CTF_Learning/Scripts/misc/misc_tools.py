#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTF Misc杂项常用脚本工具集
包含隐写术、编码解码、文件分析等常用工具
"""

import os
import re
import zipfile
import tarfile
import gzip
import base64
import binascii
import hashlib
import itertools
import string
from PIL import Image
import numpy as np
import subprocess

class MiscTools:
    def __init__(self):
        self.common_passwords = [
            "123456", "password", "admin", "root", "guest", "test",
            "qwerty", "123123", "abc123", "password123", "admin123"
        ]
    
    def extract_strings(self, file_path, min_length=4):
        """提取文件中的字符串"""
        print(f"[+] Extracting strings from {file_path}")
        
        strings = []
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # 提取ASCII字符串
            ascii_strings = re.findall(b'[!-~]{%d,}' % min_length, data)
            strings.extend([s.decode('ascii', errors='ignore') for s in ascii_strings])
            
            # 提取Unicode字符串
            unicode_strings = re.findall(b'(?:[!-~]\x00){%d,}' % min_length, data)
            strings.extend([s.decode('utf-16le', errors='ignore') for s in unicode_strings])
            
        except Exception as e:
            print(f"[-] Error extracting strings: {e}")
        
        return strings
    
    def analyze_file_header(self, file_path):
        """分析文件头"""
        print(f"[+] Analyzing file header for {file_path}")
        
        file_signatures = {
            b'\x89PNG\r\n\x1a\n': 'PNG Image',
            b'\xff\xd8\xff': 'JPEG Image',
            b'GIF87a': 'GIF Image (87a)',
            b'GIF89a': 'GIF Image (89a)',
            b'BM': 'BMP Image',
            b'PK\x03\x04': 'ZIP Archive',
            b'PK\x05\x06': 'ZIP Archive (empty)',
            b'PK\x07\x08': 'ZIP Archive (spanned)',
            b'\x1f\x8b\x08': 'GZIP Archive',
            b'Rar!\x1a\x07\x00': 'RAR Archive (v1.5)',
            b'Rar!\x1a\x07\x01\x00': 'RAR Archive (v5.0)',
            b'\x7fELF': 'ELF Executable',
            b'MZ': 'PE Executable',
            b'\xca\xfe\xba\xbe': 'Java Class File',
            b'%PDF': 'PDF Document',
            b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1': 'Microsoft Office Document',
            b'RIFF': 'RIFF Container (AVI/WAV)',
            b'\x00\x00\x00\x18ftypmp4': 'MP4 Video',
            b'\x00\x00\x00\x20ftypM4A': 'M4A Audio'
        }
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
            
            for signature, file_type in file_signatures.items():
                if header.startswith(signature):
                    print(f"[+] File type detected: {file_type}")
                    return file_type
            
            print("[-] Unknown file type")
            return "Unknown"
            
        except Exception as e:
            print(f"[-] Error analyzing file header: {e}")
            return None
    
    def extract_lsb_image(self, image_path, output_path=None):
        """提取图片LSB隐写数据"""
        print(f"[+] Extracting LSB data from {image_path}")
        
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # 提取每个像素的最低位
            lsb_data = []
            
            if len(img_array.shape) == 3:  # 彩色图片
                for channel in range(img_array.shape[2]):
                    channel_data = img_array[:, :, channel]
                    lsb_bits = channel_data & 1
                    lsb_data.extend(lsb_bits.flatten())
            else:  # 灰度图片
                lsb_bits = img_array & 1
                lsb_data.extend(lsb_bits.flatten())
            
            # 将位转换为字节
            lsb_bytes = []
            for i in range(0, len(lsb_data), 8):
                if i + 7 < len(lsb_data):
                    byte_bits = lsb_data[i:i+8]
                    byte_value = 0
                    for j, bit in enumerate(byte_bits):
                        byte_value |= (bit << j)
                    lsb_bytes.append(byte_value)
            
            # 保存提取的数据
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(bytes(lsb_bytes))
                print(f"[+] LSB data saved to {output_path}")
            
            return bytes(lsb_bytes)
            
        except Exception as e:
            print(f"[-] Error extracting LSB data: {e}")
            return None
    
    def crack_zip_password(self, zip_path, wordlist=None):
        """破解ZIP密码"""
        print(f"[+] Cracking ZIP password for {zip_path}")
        
        if wordlist is None:
            wordlist = self.common_passwords
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for password in wordlist:
                    try:
                        zf.extractall(pwd=password.encode())
                        print(f"[+] Password found: {password}")
                        return password
                    except:
                        continue
                
                print("[-] Password not found in wordlist")
                return None
                
        except Exception as e:
            print(f"[-] Error cracking ZIP password: {e}")
            return None
    
    def generate_wordlist(self, base_words, rules=None):
        """生成密码字典"""
        print("[+] Generating wordlist")
        
        if rules is None:
            rules = [
                lambda x: x,  # 原始
                lambda x: x.upper(),  # 大写
                lambda x: x.lower(),  # 小写
                lambda x: x.capitalize(),  # 首字母大写
                lambda x: x + '123',  # 添加数字
                lambda x: x + '!',  # 添加符号
                lambda x: '123' + x,  # 前缀数字
                lambda x: x[::-1],  # 反转
            ]
        
        wordlist = set()
        
        for word in base_words:
            for rule in rules:
                try:
                    new_word = rule(word)
                    wordlist.add(new_word)
                except:
                    continue
        
        # 添加常见组合
        for word in base_words:
            for year in ['2020', '2021', '2022', '2023', '2024']:
                wordlist.add(word + year)
                wordlist.add(year + word)
        
        return list(wordlist)
    
    def decode_multiple_base64(self, encoded_text, max_iterations=10):
        """多重Base64解码"""
        print("[+] Attempting multiple Base64 decoding")
        
        current = encoded_text
        iterations = 0
        
        while iterations < max_iterations:
            try:
                # 尝试解码
                decoded = base64.b64decode(current)
                
                # 检查是否为可打印字符
                try:
                    decoded_str = decoded.decode('utf-8')
                    if all(c in string.printable for c in decoded_str):
                        current = decoded_str
                        iterations += 1
                        print(f"[+] Iteration {iterations}: {current}")
                    else:
                        break
                except:
                    break
            except:
                break
        
        return current
    
    def extract_hidden_files(self, file_path, output_dir=None):
        """提取隐藏文件"""
        print(f"[+] Extracting hidden files from {file_path}")
        
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        
        extracted_files = []
        
        try:
            # 使用binwalk提取
            result = subprocess.run(['binwalk', '-e', file_path], 
                                  capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode == 0:
                print("[+] Binwalk extraction completed")
                
                # 查找提取的文件
                extract_dir = os.path.join(output_dir, f"_{os.path.basename(file_path)}.extracted")
                if os.path.exists(extract_dir):
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            extracted_files.append(os.path.join(root, file))
            
        except FileNotFoundError:
            print("[-] Binwalk not found, trying manual extraction")
            
            # 手动查找文件签名
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 查找ZIP文件
            zip_signature = b'PK\x03\x04'
            zip_positions = [m.start() for m in re.finditer(re.escape(zip_signature), data)]
            
            for i, pos in enumerate(zip_positions):
                zip_data = data[pos:]
                zip_file = os.path.join(output_dir, f"extracted_{i}.zip")
                
                with open(zip_file, 'wb') as f:
                    f.write(zip_data)
                
                extracted_files.append(zip_file)
                print(f"[+] Extracted ZIP file: {zip_file}")
        
        return extracted_files
    
    def qr_code_decode(self, image_path):
        """解码二维码"""
        print(f"[+] Decoding QR code from {image_path}")
        
        try:
            # 使用zxing解码
            result = subprocess.run(['zxing', image_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                decoded_text = result.stdout.strip()
                print(f"[+] QR code decoded: {decoded_text}")
                return decoded_text
            else:
                print("[-] Failed to decode QR code")
                return None
                
        except FileNotFoundError:
            print("[-] zxing not found")
            
            try:
                # 尝试使用pyzbar
                from pyzbar import pyzbar
                
                img = Image.open(image_path)
                codes = pyzbar.decode(img)
                
                if codes:
                    decoded_text = codes[0].data.decode('utf-8')
                    print(f"[+] QR code decoded: {decoded_text}")
                    return decoded_text
                else:
                    print("[-] No QR code found")
                    return None
                    
            except ImportError:
                print("[-] pyzbar not installed")
                return None
    
    def frequency_analysis_text(self, text):
        """文本频率分析"""
        print("[+] Performing frequency analysis")
        
        # 字符频率
        char_freq = {}
        for char in text.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # 排序
        sorted_freq = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 英文字母频率（参考）
        english_freq = [
            ('e', 12.7), ('t', 9.1), ('a', 8.2), ('o', 7.5), ('i', 7.0),
            ('n', 6.7), ('s', 6.3), ('h', 6.1), ('r', 6.0), ('d', 4.3)
        ]
        
        print("[+] Character frequency analysis:")
        for char, count in sorted_freq[:10]:
            percentage = (count / len([c for c in text if c.isalpha()])) * 100
            print(f"    {char}: {count} ({percentage:.1f}%)")
        
        return sorted_freq
    
    def hex_dump(self, file_path, length=256):
        """十六进制转储"""
        print(f"[+] Hex dump of {file_path} (first {length} bytes)")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read(length)
            
            for i in range(0, len(data), 16):
                hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
                ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+16])
                print(f"{i:08x}: {hex_part:<48} |{ascii_part}|")
                
        except Exception as e:
            print(f"[-] Error creating hex dump: {e}")
    
    def find_flag_patterns(self, text):
        """查找flag模式"""
        print("[+] Searching for flag patterns")
        
        flag_patterns = [
            r'flag\{[^}]+\}',
            r'FLAG\{[^}]+\}',
            r'ctf\{[^}]+\}',
            r'CTF\{[^}]+\}',
            r'[a-zA-Z0-9_]+\{[^}]+\}',
            r'[0-9a-f]{32}',  # MD5
            r'[0-9a-f]{40}',  # SHA1
            r'[0-9a-f]{64}',  # SHA256
        ]
        
        found_flags = []
        
        for pattern in flag_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                found_flags.append(match)
                print(f"[+] Potential flag found: {match}")
        
        return found_flags
    
    def steganography_detect(self, image_path):
        """隐写术检测"""
        print(f"[+] Detecting steganography in {image_path}")
        
        detections = []
        
        try:
            # 检查文件大小异常
            file_size = os.path.getsize(image_path)
            img = Image.open(image_path)
            expected_size = img.width * img.height * len(img.getbands())
            
            if file_size > expected_size * 1.5:
                detections.append("File size larger than expected")
            
            # 检查EXIF数据
            if hasattr(img, '_getexif') and img._getexif():
                detections.append("EXIF data present")
            
            # LSB分析
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                for channel in range(img_array.shape[2]):
                    lsb_plane = img_array[:, :, channel] & 1
                    if np.std(lsb_plane) > 0.4:  # 高方差可能表示隐写
                        detections.append(f"High variance in LSB plane (channel {channel})")
            
        except Exception as e:
            print(f"[-] Error in steganography detection: {e}")
        
        if detections:
            print("[+] Potential steganography detected:")
            for detection in detections:
                print(f"    - {detection}")
        else:
            print("[-] No obvious steganography detected")
        
        return detections

def main():
    """主函数 - 演示用法"""
    tools = MiscTools()
    
    print("CTF Misc工具集")
    print("=" * 50)
    
    # 示例用法
    print("\n1. 文件分析示例:")
    print("file_type = tools.analyze_file_header('suspicious_file.bin')")
    print("strings = tools.extract_strings('suspicious_file.bin')")
    
    print("\n2. 隐写术分析示例:")
    print("lsb_data = tools.extract_lsb_image('image.png', 'extracted.bin')")
    print("detections = tools.steganography_detect('image.png')")
    
    print("\n3. 密码破解示例:")
    print("password = tools.crack_zip_password('encrypted.zip')")
    print("wordlist = tools.generate_wordlist(['admin', 'test'])")
    
    print("\n4. 编码解码示例:")
    print("decoded = tools.decode_multiple_base64('SGVsbG8gV29ybGQ=')")
    print("qr_text = tools.qr_code_decode('qrcode.png')")
    
    print("\n5. 数据提取示例:")
    print("extracted = tools.extract_hidden_files('container.bin')")
    print("flags = tools.find_flag_patterns(text_data)")

if __name__ == "__main__":
    main()
