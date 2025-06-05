"""
密码学基础学习
包含经典密码、现代密码学算法的实现和破解
"""

import string
import random
import hashlib
from collections import Counter
import matplotlib.pyplot as plt

class CryptographyFundamentals:
    """密码学基础学习类"""
    
    def __init__(self):
        self.examples_completed = []
        print("🔐 密码学基础学习系统")
        print("=" * 50)
    
    def caesar_cipher_demo(self):
        """凯撒密码演示"""
        print("🏛️ 凯撒密码学习")
        print("=" * 30)
        
        def caesar_encrypt(text, shift):
            """凯撒密码加密"""
            result = ""
            for char in text:
                if char.isalpha():
                    ascii_offset = 65 if char.isupper() else 97
                    shifted = (ord(char) - ascii_offset + shift) % 26
                    result += chr(shifted + ascii_offset)
                else:
                    result += char
            return result
        
        def caesar_decrypt(text, shift):
            """凯撒密码解密"""
            return caesar_encrypt(text, -shift)
        
        def caesar_brute_force(ciphertext):
            """凯撒密码暴力破解"""
            print("🔨 暴力破解结果:")
            results = []
            for shift in range(26):
                decrypted = caesar_decrypt(ciphertext, shift)
                results.append((shift, decrypted))
                print(f"偏移量 {shift:2d}: {decrypted}")
            return results
        
        # 演示加密解密
        plaintext = "Hello World! This is a secret message."
        shift = 13  # ROT13
        
        print(f"原文: {plaintext}")
        encrypted = caesar_encrypt(plaintext, shift)
        print(f"加密 (偏移{shift}): {encrypted}")
        
        decrypted = caesar_decrypt(encrypted, shift)
        print(f"解密: {decrypted}")
        
        print(f"\n🔍 破解未知偏移量的密文:")
        unknown_cipher = "Wklv lv d vhfuhw phvvdjh!"
        print(f"密文: {unknown_cipher}")
        caesar_brute_force(unknown_cipher)
        
        self.examples_completed.append("凯撒密码")
    
    def frequency_analysis_demo(self):
        """频率分析演示"""
        print("\n📊 频率分析学习")
        print("=" * 30)
        
        # 英文字母频率表
        english_freq = {
            'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7,
            'S': 6.3, 'H': 6.1, 'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8,
            'U': 2.8, 'M': 2.4, 'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0,
            'P': 1.9, 'B': 1.3, 'V': 1.0, 'K': 0.8, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }
        
        def analyze_frequency(text):
            """分析文本字母频率"""
            # 只统计字母
            letters_only = ''.join(c.upper() for c in text if c.isalpha())
            total_letters = len(letters_only)
            
            if total_letters == 0:
                return {}
            
            # 计算频率
            freq_count = Counter(letters_only)
            freq_percent = {letter: (count / total_letters) * 100 
                          for letter, count in freq_count.items()}
            
            return freq_percent
        
        def plot_frequency_comparison(text_freq, title="频率分析"):
            """绘制频率对比图"""
            letters = list(string.ascii_uppercase)
            text_freqs = [text_freq.get(letter, 0) for letter in letters]
            english_freqs = [english_freq.get(letter, 0) for letter in letters]
            
            plt.figure(figsize=(15, 6))
            
            x = range(len(letters))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], text_freqs, width, 
                   label='密文频率', alpha=0.7, color='red')
            plt.bar([i + width/2 for i in x], english_freqs, width, 
                   label='英文标准频率', alpha=0.7, color='blue')
            
            plt.xlabel('字母')
            plt.ylabel('频率 (%)')
            plt.title(title)
            plt.xticks(x, letters)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # 演示频率分析
        sample_text = """
        The quick brown fox jumps over the lazy dog. This pangram contains 
        every letter of the alphabet at least once. It is commonly used for 
        testing typewriters and computer keyboards, and in other applications 
        requiring all letters of the alphabet.
        """
        
        print("1. 英文文本频率分析:")
        print(f"样本文本: {sample_text[:100]}...")
        
        freq_result = analyze_frequency(sample_text)
        print("\n字母频率统计:")
        for letter in sorted(freq_result.keys()):
            print(f"{letter}: {freq_result[letter]:.2f}%")
        
        # 绘制频率对比图
        plot_frequency_comparison(freq_result, "英文文本频率分析")
        
        # 演示密文频率分析
        print("\n2. 密文频率分析:")
        # 使用凯撒密码加密的文本
        cipher_text = "WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ"
        print(f"密文: {cipher_text}")
        
        cipher_freq = analyze_frequency(cipher_text)
        plot_frequency_comparison(cipher_freq, "密文频率分析")
        
        # 找出最高频字母
        if cipher_freq:
            most_frequent = max(cipher_freq.items(), key=lambda x: x[1])
            print(f"最高频字母: {most_frequent[0]} ({most_frequent[1]:.2f}%)")
            print(f"如果假设它对应英文中的 'E'，则偏移量可能是: {(ord(most_frequent[0]) - ord('E')) % 26}")
        
        self.examples_completed.append("频率分析")
    
    def vigenere_cipher_demo(self):
        """维吉尼亚密码演示"""
        print("\n🔑 维吉尼亚密码学习")
        print("=" * 30)
        
        def vigenere_encrypt(plaintext, key):
            """维吉尼亚密码加密"""
            result = ""
            key = key.upper()
            key_index = 0
            
            for char in plaintext:
                if char.isalpha():
                    # 获取密钥字符的偏移量
                    key_char = key[key_index % len(key)]
                    shift = ord(key_char) - ord('A')
                    
                    # 加密
                    if char.isupper():
                        encrypted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
                    else:
                        encrypted_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                    
                    result += encrypted_char
                    key_index += 1
                else:
                    result += char
            
            return result
        
        def vigenere_decrypt(ciphertext, key):
            """维吉尼亚密码解密"""
            result = ""
            key = key.upper()
            key_index = 0
            
            for char in ciphertext:
                if char.isalpha():
                    # 获取密钥字符的偏移量
                    key_char = key[key_index % len(key)]
                    shift = ord(key_char) - ord('A')
                    
                    # 解密
                    if char.isupper():
                        decrypted_char = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                    else:
                        decrypted_char = chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
                    
                    result += decrypted_char
                    key_index += 1
                else:
                    result += char
            
            return result
        
        def kasiski_examination(ciphertext):
            """卡西斯基检验 - 寻找重复片段"""
            ciphertext = ''.join(c.upper() for c in ciphertext if c.isalpha())
            repeats = {}
            
            # 寻找长度为3-5的重复片段
            for length in range(3, 6):
                for i in range(len(ciphertext) - length + 1):
                    substring = ciphertext[i:i + length]
                    positions = []
                    
                    # 寻找所有出现位置
                    for j in range(i + length, len(ciphertext) - length + 1):
                        if ciphertext[j:j + length] == substring:
                            positions.append(j)
                    
                    if positions:
                        positions.insert(0, i)
                        repeats[substring] = positions
            
            return repeats
        
        # 演示维吉尼亚密码
        plaintext = "HELLO WORLD THIS IS A SECRET MESSAGE"
        key = "KEY"
        
        print(f"原文: {plaintext}")
        print(f"密钥: {key}")
        
        encrypted = vigenere_encrypt(plaintext, key)
        print(f"加密: {encrypted}")
        
        decrypted = vigenere_decrypt(encrypted, key)
        print(f"解密: {decrypted}")
        
        # 演示卡西斯基检验
        print(f"\n🔍 卡西斯基检验:")
        long_cipher = vigenere_encrypt("ATTACKATDAWN" * 5, "LEMON")
        print(f"长密文: {long_cipher}")
        
        repeats = kasiski_examination(long_cipher)
        print("发现的重复片段:")
        for pattern, positions in repeats.items():
            if len(positions) > 1:
                distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                print(f"  {pattern}: 位置 {positions}, 距离 {distances}")
        
        self.examples_completed.append("维吉尼亚密码")
    
    def hash_functions_demo(self):
        """哈希函数演示"""
        print("\n#️⃣ 哈希函数学习")
        print("=" * 30)
        
        def demonstrate_hash_properties():
            """演示哈希函数的性质"""
            test_strings = [
                "Hello World",
                "Hello World!",  # 微小变化
                "hello world",   # 大小写变化
                "A" * 1000,      # 长字符串
                ""               # 空字符串
            ]
            
            print("哈希函数性质演示:")
            print("输入 -> MD5 -> SHA256")
            print("-" * 80)
            
            for text in test_strings:
                md5_hash = hashlib.md5(text.encode()).hexdigest()
                sha256_hash = hashlib.sha256(text.encode()).hexdigest()
                
                display_text = text if len(text) <= 20 else text[:17] + "..."
                print(f"{display_text:20} -> {md5_hash} -> {sha256_hash[:16]}...")
        
        def hash_collision_demo():
            """哈希碰撞演示（生日攻击原理）"""
            print(f"\n🎂 生日攻击原理演示:")
            print("在23个人中，有两人生日相同的概率约为50%")
            
            # 简化的哈希碰撞模拟（使用短哈希）
            def short_hash(text, length=4):
                """生成短哈希用于演示碰撞"""
                full_hash = hashlib.md5(text.encode()).hexdigest()
                return full_hash[:length]
            
            print(f"\n使用{4}位哈希进行碰撞测试:")
            seen_hashes = {}
            attempts = 0
            
            while True:
                attempts += 1
                random_string = ''.join(random.choices(string.ascii_letters, k=10))
                hash_value = short_hash(random_string)
                
                if hash_value in seen_hashes:
                    print(f"🎯 发现碰撞!")
                    print(f"  尝试次数: {attempts}")
                    print(f"  哈希值: {hash_value}")
                    print(f"  字符串1: {seen_hashes[hash_value]}")
                    print(f"  字符串2: {random_string}")
                    break
                
                seen_hashes[hash_value] = random_string
                
                if attempts > 10000:  # 防止无限循环
                    print("未在10000次尝试内找到碰撞")
                    break
        
        def password_hashing_demo():
            """密码哈希演示"""
            print(f"\n🔒 密码哈希最佳实践:")
            
            import os
            
            def hash_password_simple(password):
                """简单哈希（不安全）"""
                return hashlib.md5(password.encode()).hexdigest()
            
            def hash_password_with_salt(password, salt=None):
                """加盐哈希（更安全）"""
                if salt is None:
                    salt = os.urandom(16).hex()
                
                salted_password = salt + password
                hash_value = hashlib.sha256(salted_password.encode()).hexdigest()
                return f"{salt}:{hash_value}"
            
            password = "mypassword123"
            
            print(f"原始密码: {password}")
            print(f"简单MD5: {hash_password_simple(password)}")
            print(f"加盐SHA256: {hash_password_with_salt(password)}")
            print(f"再次加盐: {hash_password_with_salt(password)}")
            print("注意：每次加盐结果都不同，但都能验证同一密码")
        
        demonstrate_hash_properties()
        hash_collision_demo()
        password_hashing_demo()
        
        self.examples_completed.append("哈希函数")
    
    def run_all_demos(self):
        """运行所有密码学演示"""
        print("🔐 密码学基础完整学习")
        print("=" * 60)
        
        self.caesar_cipher_demo()
        self.frequency_analysis_demo()
        self.vigenere_cipher_demo()
        self.hash_functions_demo()
        
        print(f"\n🎉 密码学基础学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. 凯撒密码 - 最简单的替换密码，易被暴力破解")
        print("2. 频率分析 - 破解单表替换密码的经典方法")
        print("3. 维吉尼亚密码 - 多表替换密码，更难破解")
        print("4. 哈希函数 - 单向函数，用于完整性验证")
        
        print(f"\n🎯 CTF密码学技巧:")
        print("1. 识别密码类型：观察密文特征")
        print("2. 尝试常见密码：凯撒、维吉尼亚、Base64")
        print("3. 频率分析：统计字符出现频率")
        print("4. 在线工具：CyberChef、dcode.fr")

def main():
    """主函数"""
    crypto = CryptographyFundamentals()
    crypto.run_all_demos()
    
    print("\n💡 进阶学习建议:")
    print("1. 学习现代密码学：AES、RSA、ECC")
    print("2. 研究密码学攻击：差分分析、线性分析")
    print("3. 实践密码学工具：OpenSSL、GnuPG")
    print("4. 关注密码学竞赛和CTF题目")

if __name__ == "__main__":
    main()
