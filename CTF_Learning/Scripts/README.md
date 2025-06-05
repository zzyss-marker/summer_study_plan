# 🛠️ CTF常用脚本工具集

这是一套完整的CTF竞赛常用脚本工具集，涵盖Web安全、密码学、Pwn、Misc杂项和逆向工程五大方向。每个脚本都是实用的工具函数集合，可以直接在CTF比赛中使用。

## 📁 目录结构

```
Scripts/
├── README.md                    # 使用说明文档
├── web/
│   └── web_tools.py            # Web安全工具集
├── crypto/
│   └── crypto_tools.py         # 密码学工具集
├── pwn/
│   └── pwn_tools.py            # Pwn工具集
├── misc/
│   └── misc_tools.py           # Misc杂项工具集
└── reverse/
    └── reverse_tools.py        # 逆向工程工具集
```

## 🌐 Web安全工具集 (`web/web_tools.py`)

### 主要功能
- **SQL注入测试**: 自动化SQL注入检测和利用
- **XSS测试**: 跨站脚本攻击检测
- **目录爆破**: 网站目录和文件发现
- **文件上传测试**: 文件上传漏洞检测
- **LFI测试**: 本地文件包含漏洞检测
- **命令注入测试**: 命令执行漏洞检测

### 使用示例
```python
from web.web_tools import WebTools

tools = WebTools()

# SQL注入测试
vulnerabilities = tools.sql_injection_test('http://target.com/login.php', 'username')

# XSS测试
xss_results = tools.xss_test('http://target.com/search.php', 'query')

# 目录爆破
directories = tools.directory_bruteforce('http://target.com')

# 文件上传测试
upload_results = tools.file_upload_test('http://target.com/upload.php')
```

### 核心特性
- 支持多种SQL注入类型（联合注入、盲注、报错注入）
- 内置WAF绕过技术
- 多线程目录爆破
- 自动化漏洞检测和验证

## 🔒 密码学工具集 (`crypto/crypto_tools.py`)

### 主要功能
- **RSA攻击**: 小指数攻击、共模攻击、Wiener攻击
- **古典密码**: 凯撒密码、维吉尼亚密码、栅栏密码
- **编码解码**: Base64变种、摩斯密码、二进制、十六进制
- **数论算法**: 扩展欧几里得、中国剩余定理、质因数分解
- **哈希函数**: MD5、SHA系列哈希计算

### 使用示例
```python
from crypto.crypto_tools import CryptoTools

tools = CryptoTools()

# RSA小指数攻击
plaintext = tools.rsa_attack_small_e(ciphertext, e=3, n=modulus)

# 凯撒密码解码
results = tools.caesar_cipher_decode("KHOOR ZRUOG")

# Base64变种解码
decoded = tools.base64_variants_decode("SGVsbG8gV29ybGQ=")

# 摩斯密码解码
morse_text = tools.morse_code_decode(".... . .-.. .-.. ---")

# 频率分析
frequency = tools.frequency_analysis("encrypted text")
```

### 核心特性
- 完整的RSA攻击技术实现
- 支持多种古典密码自动破解
- 内置英文频率分析
- 数学工具函数集合

## 💥 Pwn工具集 (`pwn/pwn_tools.py`)

### 主要功能
- **栈溢出利用**: ret2text、ret2libc、ret2shellcode
- **ROP链构造**: 自动化ROP gadget查找和链构造
- **Shellcode生成**: 多架构shellcode生成
- **保护机制检测**: ASLR、NX、Canary等保护检测
- **格式化字符串**: 任意地址读写利用

### 使用示例
```python
from pwn.pwn_tools import PwnTools

tools = PwnTools()

# 查找栈溢出偏移
offset = tools.find_offset('./vulnerable_binary')

# 检查保护机制
protections = tools.check_protections('./binary')

# 生成ret2libc exploit
payload = tools.ret2libc_exploit('./binary', offset)

# 查找ROP gadgets
gadgets = tools.find_gadgets('./binary')

# 生成shellcode
shellcode = tools.generate_shellcode('amd64', 'execve')
```

### 核心特性
- 基于pwntools的高级封装
- 自动化exploit生成
- 支持多种攻击技术
- 内置保护机制绕过

## 🎲 Misc杂项工具集 (`misc/misc_tools.py`)

### 主要功能
- **隐写术分析**: LSB隐写检测和提取
- **文件分析**: 文件头分析、字符串提取
- **压缩包破解**: ZIP密码暴力破解
- **编码识别**: 多重编码自动识别和解码
- **二维码解析**: QR码识别和解码
- **取证分析**: 文件恢复、隐藏文件提取

### 使用示例
```python
from misc.misc_tools import MiscTools

tools = MiscTools()

# 文件类型识别
file_type = tools.analyze_file_header('suspicious_file.bin')

# LSB隐写提取
lsb_data = tools.extract_lsb_image('image.png', 'extracted.bin')

# ZIP密码破解
password = tools.crack_zip_password('encrypted.zip')

# 多重Base64解码
decoded = tools.decode_multiple_base64('encoded_text')

# 二维码解码
qr_content = tools.qr_code_decode('qrcode.png')

# 隐写术检测
detections = tools.steganography_detect('image.png')
```

### 核心特性
- 全面的隐写术检测技术
- 自动化文件分析
- 智能编码识别
- 综合取证工具

## 🔄 逆向工程工具集 (`reverse/reverse_tools.py`)

### 主要功能
- **字符串分析**: 高级字符串提取和分析
- **加密分析**: XOR分析、凯撒密码分析
- **加壳检测**: 文件加壳和混淆检测
- **反汇编**: Shellcode反汇编
- **PE分析**: PE文件导入表分析
- **反混淆**: 简单代码反混淆

### 使用示例
```python
from reverse.reverse_tools import ReverseTools

tools = ReverseTools()

# 高级字符串提取
strings = tools.extract_strings_advanced('binary_file')

# XOR加密分析
xor_results = tools.xor_analysis(encrypted_data)

# 加壳检测
packing_indicators = tools.detect_packing('suspicious.exe')

# PE导入表分析
imports = tools.analyze_pe_imports('program.exe')

# 反混淆
clean_code = tools.deobfuscate_simple(obfuscated_code)

# 加密常量查找
crypto_constants = tools.find_crypto_constants('crypto_program')
```

### 核心特性
- 智能加密算法识别
- 自动化反混淆技术
- 全面的文件分析
- 加密常量数据库

## 🚀 快速开始

### 环境要求
```bash
# Python依赖
pip install requests pycryptodome pillow numpy pwntools gmpy2

# 系统工具（可选）
sudo apt-get install binwalk zxing-tools objdump
```

### 基本使用
```python
# 导入所需工具
from web.web_tools import WebTools
from crypto.crypto_tools import CryptoTools
from pwn.pwn_tools import PwnTools
from misc.misc_tools import MiscTools
from reverse.reverse_tools import ReverseTools

# 创建工具实例
web_tools = WebTools()
crypto_tools = CryptoTools()
pwn_tools = PwnTools()
misc_tools = MiscTools()
reverse_tools = ReverseTools()

# 使用相应的方法
result = web_tools.sql_injection_test(url, param)
```

## 📝 使用技巧

### 1. Web安全测试
```python
# 完整的Web漏洞扫描流程
tools = WebTools()

# 1. 目录发现
directories = tools.directory_bruteforce('http://target.com')

# 2. 参数测试
for endpoint in directories:
    sql_vulns = tools.sql_injection_test(endpoint, 'id')
    xss_vulns = tools.xss_test(endpoint, 'search')
    lfi_vulns = tools.lfi_test(endpoint, 'file')

# 3. 文件上传测试
upload_results = tools.file_upload_test('http://target.com/upload.php')
```

### 2. 密码学分析
```python
# RSA攻击流程
tools = CryptoTools()

# 1. 尝试小指数攻击
plaintext = tools.rsa_attack_small_e(c, e, n)

# 2. 尝试共模攻击
if not plaintext:
    plaintext = tools.rsa_attack_common_modulus(c1, c2, e1, e2, n)

# 3. 尝试Wiener攻击
if not plaintext:
    d = tools.rsa_attack_wiener(e, n)
    if d:
        plaintext = pow(c, d, n)
```

### 3. Pwn利用开发
```python
# 栈溢出利用流程
tools = PwnTools()

# 1. 检查保护机制
protections = tools.check_protections('./binary')

# 2. 查找偏移量
offset = tools.find_offset('./binary')

# 3. 根据保护机制选择攻击方式
if not protections['NX']:
    payload = tools.ret2shellcode_exploit('./binary', offset)
elif 'system' in binary_functions:
    payload = tools.ret2libc_exploit('./binary', offset)
else:
    payload = tools.ret2text_exploit('./binary', 'backdoor', offset)
```

### 4. Misc综合分析
```python
# 文件分析流程
tools = MiscTools()

# 1. 文件类型识别
file_type = tools.analyze_file_header('mystery_file')

# 2. 字符串提取
strings = tools.extract_strings('mystery_file')

# 3. 隐藏文件提取
extracted = tools.extract_hidden_files('mystery_file')

# 4. 如果是图片，检查隐写
if 'image' in file_type.lower():
    lsb_data = tools.extract_lsb_image('mystery_file')
    detections = tools.steganography_detect('mystery_file')
```

## 💡 高级用法

### 自定义Payload生成
```python
# Web工具自定义payload
web_tools = WebTools()
custom_payloads = web_tools.generate_payloads('xss', {'target': 'specific_app'})

# 密码学工具自定义攻击
crypto_tools = CryptoTools()
custom_key = crypto_tools.generate_rsa_keys(2048)
```

### 批量处理
```python
# 批量文件分析
import os
misc_tools = MiscTools()

for filename in os.listdir('./samples'):
    file_path = os.path.join('./samples', filename)
    file_type = misc_tools.analyze_file_header(file_path)
    strings = misc_tools.extract_strings(file_path)
    print(f"{filename}: {file_type}, {len(strings)} strings")
```

### 结果导出
```python
# 将结果保存为JSON
import json

results = {
    'sql_injection': web_tools.sql_injection_test(url, param),
    'xss': web_tools.xss_test(url, param),
    'directories': web_tools.directory_bruteforce(url)
}

with open('scan_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## 🔧 扩展开发

### 添加新功能
```python
# 扩展Web工具
class ExtendedWebTools(WebTools):
    def custom_attack(self, url, param):
        # 自定义攻击逻辑
        pass

# 扩展密码学工具
class ExtendedCryptoTools(CryptoTools):
    def custom_cipher_attack(self, ciphertext):
        # 自定义密码攻击
        pass
```

### 集成外部工具
```python
# 集成sqlmap
def run_sqlmap(url, param):
    import subprocess
    cmd = f"sqlmap -u '{url}' --data '{param}=test' --batch"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout
```

## 📚 学习建议

1. **熟悉工具**: 先了解每个工具的基本功能和使用方法
2. **实践应用**: 在CTF练习中实际使用这些脚本
3. **源码学习**: 阅读脚本源码，理解实现原理
4. **自定义扩展**: 根据需要添加新功能或优化现有功能
5. **工具组合**: 学会组合使用多个工具解决复杂问题

## ⚠️ 注意事项

1. **合法使用**: 仅在授权的环境中使用这些工具
2. **安全防护**: 在隔离环境中测试，避免影响生产系统
3. **依赖管理**: 确保安装所有必要的依赖包
4. **版本兼容**: 注意Python版本和库版本的兼容性
5. **错误处理**: 使用时注意异常处理和错误信息

这套脚本工具集将大大提高您在CTF竞赛中的效率，祝您在比赛中取得好成绩！🏆
