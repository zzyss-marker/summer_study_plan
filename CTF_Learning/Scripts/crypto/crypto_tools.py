#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTF 密码学常用脚本工具集
包含RSA攻击、古典密码、编码解码等常用工具
"""

import base64
import hashlib
import itertools
import math
import random
import string
from Crypto.Util.number import *
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, DES
import gmpy2

class CryptoTools:
    def __init__(self):
        self.alphabet = string.ascii_lowercase
        
    def gcd(self, a, b):
        """计算最大公约数"""
        while b:
            a, b = b, a % b
        return a
    
    def extended_gcd(self, a, b):
        """扩展欧几里得算法"""
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = self.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    def mod_inverse(self, a, m):
        """计算模逆元"""
        gcd, x, y = self.extended_gcd(a, m)
        if gcd != 1:
            return None
        return (x % m + m) % m
    
    def chinese_remainder_theorem(self, remainders, moduli):
        """中国剩余定理"""
        total = 0
        prod = 1
        for m in moduli:
            prod *= m
        
        for r, m in zip(remainders, moduli):
            p = prod // m
            total += r * self.mod_inverse(p, m) * p
        
        return total % prod
    
    def factor_pollard_rho(self, n):
        """Pollard's rho算法分解质因数"""
        if n % 2 == 0:
            return 2
        
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        
        while d == 1:
            x = (pow(x, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            d = self.gcd(abs(x - y), n)
            
            if d == n:
                return self.factor_pollard_rho(n)
        
        return d
    
    def rsa_attack_small_e(self, c, e, n):
        """RSA小指数攻击"""
        print(f"[+] Attempting small exponent attack (e={e})")
        
        # 直接开e次方根
        m = gmpy2.iroot(c, e)[0]
        if pow(m, e) == c:
            return int(m)
        
        # 多个接收者攻击（需要多个密文）
        return None
    
    def rsa_attack_common_modulus(self, c1, c2, e1, e2, n):
        """RSA共模攻击"""
        print(f"[+] Attempting common modulus attack")
        
        if self.gcd(e1, e2) != 1:
            return None
        
        gcd, s, t = self.extended_gcd(e1, e2)
        
        if s < 0:
            s = -s
            c1 = self.mod_inverse(c1, n)
        if t < 0:
            t = -t
            c2 = self.mod_inverse(c2, n)
        
        m = (pow(c1, s, n) * pow(c2, t, n)) % n
        return m
    
    def rsa_attack_wiener(self, e, n):
        """Wiener攻击（低解密指数）"""
        print(f"[+] Attempting Wiener attack")
        
        def continued_fractions(e, n):
            fractions = []
            while n:
                fractions.append(e // n)
                e, n = n, e % n
            return fractions
        
        def convergents(fractions):
            convergents = []
            for i in range(len(fractions)):
                if i == 0:
                    convergents.append((fractions[i], 1))
                elif i == 1:
                    convergents.append((fractions[i] * fractions[i-1] + 1, fractions[i]))
                else:
                    num = fractions[i] * convergents[i-1][0] + convergents[i-2][0]
                    den = fractions[i] * convergents[i-1][1] + convergents[i-2][1]
                    convergents.append((num, den))
            return convergents
        
        fractions = continued_fractions(e, n)
        convs = convergents(fractions)
        
        for k, d in convs:
            if k == 0:
                continue
            
            phi = (e * d - 1) // k
            s = n - phi + 1
            discriminant = s * s - 4 * n
            
            if discriminant >= 0:
                sqrt_discriminant = int(math.sqrt(discriminant))
                if sqrt_discriminant * sqrt_discriminant == discriminant:
                    p = (s + sqrt_discriminant) // 2
                    q = (s - sqrt_discriminant) // 2
                    if p * q == n:
                        return d
        
        return None
    
    def rsa_factor_fermat(self, n):
        """费马分解法"""
        print(f"[+] Attempting Fermat factorization")
        
        a = gmpy2.isqrt(n) + 1
        b2 = a * a - n
        
        while not gmpy2.is_square(b2):
            a += 1
            b2 = a * a - n
        
        b = gmpy2.isqrt(b2)
        p = a + b
        q = a - b
        
        if p * q == n:
            return int(p), int(q)
        return None, None
    
    def caesar_cipher_decode(self, ciphertext, shift=None):
        """凯撒密码解码"""
        results = []
        
        if shift is not None:
            # 指定位移量
            result = ""
            for char in ciphertext.lower():
                if char in self.alphabet:
                    result += self.alphabet[(self.alphabet.index(char) - shift) % 26]
                else:
                    result += char
            return result
        else:
            # 尝试所有可能的位移量
            for shift in range(26):
                result = ""
                for char in ciphertext.lower():
                    if char in self.alphabet:
                        result += self.alphabet[(self.alphabet.index(char) - shift) % 26]
                    else:
                        result += char
                results.append(f"Shift {shift}: {result}")
            return results
    
    def vigenere_cipher_decode(self, ciphertext, key):
        """维吉尼亚密码解码"""
        result = ""
        key = key.lower()
        key_index = 0
        
        for char in ciphertext.lower():
            if char in self.alphabet:
                shift = self.alphabet.index(key[key_index % len(key)])
                result += self.alphabet[(self.alphabet.index(char) - shift) % 26]
                key_index += 1
            else:
                result += char
        
        return result
    
    def rail_fence_decode(self, ciphertext, rails):
        """栅栏密码解码"""
        fence = [[None for _ in range(len(ciphertext))] for _ in range(rails)]
        
        # 标记栅栏位置
        rail = 0
        direction = 1
        for i in range(len(ciphertext)):
            fence[rail][i] = True
            rail += direction
            if rail == rails - 1 or rail == 0:
                direction = -direction
        
        # 填充密文
        index = 0
        for r in range(rails):
            for c in range(len(ciphertext)):
                if fence[r][c]:
                    fence[r][c] = ciphertext[index]
                    index += 1
        
        # 读取明文
        result = ""
        rail = 0
        direction = 1
        for i in range(len(ciphertext)):
            result += fence[rail][i]
            rail += direction
            if rail == rails - 1 or rail == 0:
                direction = -direction
        
        return result
    
    def base64_variants_decode(self, encoded_text):
        """Base64变种解码"""
        results = {}
        
        # 标准Base64
        try:
            results["Standard Base64"] = base64.b64decode(encoded_text).decode('utf-8', errors='ignore')
        except:
            results["Standard Base64"] = "Decode failed"
        
        # Base32
        try:
            results["Base32"] = base64.b32decode(encoded_text).decode('utf-8', errors='ignore')
        except:
            results["Base32"] = "Decode failed"
        
        # Base16 (Hex)
        try:
            results["Base16"] = base64.b16decode(encoded_text).decode('utf-8', errors='ignore')
        except:
            results["Base16"] = "Decode failed"
        
        # URL Safe Base64
        try:
            results["URL Safe Base64"] = base64.urlsafe_b64decode(encoded_text).decode('utf-8', errors='ignore')
        except:
            results["URL Safe Base64"] = "Decode failed"
        
        return results
    
    def frequency_analysis(self, text):
        """频率分析"""
        text = text.lower()
        freq = {}
        total = 0
        
        for char in text:
            if char in self.alphabet:
                freq[char] = freq.get(char, 0) + 1
                total += 1
        
        # 计算频率百分比
        for char in freq:
            freq[char] = (freq[char] / total) * 100
        
        # 按频率排序
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_freq
    
    def hash_functions(self, text):
        """常用哈希函数"""
        text_bytes = text.encode('utf-8')
        
        hashes = {
            "MD5": hashlib.md5(text_bytes).hexdigest(),
            "SHA1": hashlib.sha1(text_bytes).hexdigest(),
            "SHA224": hashlib.sha224(text_bytes).hexdigest(),
            "SHA256": hashlib.sha256(text_bytes).hexdigest(),
            "SHA384": hashlib.sha384(text_bytes).hexdigest(),
            "SHA512": hashlib.sha512(text_bytes).hexdigest()
        }
        
        return hashes
    
    def morse_code_decode(self, morse_text):
        """摩斯密码解码"""
        morse_dict = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
            '...--': '3', '....-': '4', '.....': '5', '-....': '6',
            '--...': '7', '---..': '8', '----.': '9'
        }
        
        words = morse_text.split('  ')  # 双空格分隔单词
        result = []
        
        for word in words:
            letters = word.split(' ')  # 单空格分隔字母
            decoded_word = ""
            for letter in letters:
                if letter in morse_dict:
                    decoded_word += morse_dict[letter]
                else:
                    decoded_word += '?'
            result.append(decoded_word)
        
        return ' '.join(result)
    
    def binary_decode(self, binary_text):
        """二进制解码"""
        # 移除空格
        binary_text = binary_text.replace(' ', '')
        
        # 确保长度是8的倍数
        if len(binary_text) % 8 != 0:
            return "Invalid binary length"
        
        result = ""
        for i in range(0, len(binary_text), 8):
            byte = binary_text[i:i+8]
            try:
                char = chr(int(byte, 2))
                result += char
            except:
                result += '?'
        
        return result
    
    def hex_decode(self, hex_text):
        """十六进制解码"""
        hex_text = hex_text.replace(' ', '').replace('0x', '')
        
        try:
            result = bytes.fromhex(hex_text).decode('utf-8', errors='ignore')
            return result
        except:
            return "Invalid hex string"
    
    def rot13_decode(self, text):
        """ROT13解码"""
        result = ""
        for char in text:
            if char.isalpha():
                if char.islower():
                    result += chr((ord(char) - ord('a') + 13) % 26 + ord('a'))
                else:
                    result += chr((ord(char) - ord('A') + 13) % 26 + ord('A'))
            else:
                result += char
        return result
    
    def atbash_decode(self, text):
        """埃特巴什密码解码"""
        result = ""
        for char in text.lower():
            if char in self.alphabet:
                result += self.alphabet[25 - self.alphabet.index(char)]
            else:
                result += char
        return result
    
    def generate_rsa_keys(self, bits=1024):
        """生成RSA密钥对"""
        p = getPrime(bits // 2)
        q = getPrime(bits // 2)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        d = self.mod_inverse(e, phi)
        
        return {
            'p': p, 'q': q, 'n': n, 'e': e, 'd': d, 'phi': phi
        }

def main():
    """主函数 - 演示用法"""
    tools = CryptoTools()
    
    print("CTF 密码学工具集")
    print("=" * 50)
    
    # 演示各种功能
    print("\n1. 凯撒密码解码:")
    caesar_result = tools.caesar_cipher_decode("KHOOR", 3)
    print(f"Result: {caesar_result}")
    
    print("\n2. Base64变种解码:")
    base64_results = tools.base64_variants_decode("SGVsbG8gV29ybGQ=")
    for variant, result in base64_results.items():
        print(f"{variant}: {result}")
    
    print("\n3. 摩斯密码解码:")
    morse_result = tools.morse_code_decode(".... . .-.. .-.. ---  .-- --- .-. .-.. -..")
    print(f"Result: {morse_result}")
    
    print("\n4. 频率分析:")
    freq_result = tools.frequency_analysis("hello world")
    print(f"Frequency: {freq_result}")
    
    print("\n5. 哈希函数:")
    hash_results = tools.hash_functions("hello")
    for hash_type, hash_value in hash_results.items():
        print(f"{hash_type}: {hash_value}")
    
    print("\n6. RSA密钥生成:")
    rsa_keys = tools.generate_rsa_keys(512)  # 小密钥用于演示
    print(f"Generated RSA keys: n={rsa_keys['n']}, e={rsa_keys['e']}")

if __name__ == "__main__":
    main()
