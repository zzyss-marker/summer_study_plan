# 🔐 CTF竞赛技能学习计划

## 🎯 学习目标
1. 掌握CTF五大方向的基础知识和技能
2. 能够独立解决初级到中级CTF题目
3. 具备参加CTF比赛的实战能力
4. 建立完整的安全知识体系

## 🗂️ CTF五大方向详解

### 🌐 Web安全 (第1-2周重点)
**基础较好，重点提升**

#### 核心知识点
- **注入攻击**: SQL注入、NoSQL注入、命令注入、代码注入
- **XSS攻击**: 反射型、存储型、DOM型XSS
- **CSRF攻击**: 跨站请求伪造原理和防护
- **文件上传**: 绕过技巧、文件包含漏洞
- **反序列化**: PHP、Python、Java反序列化漏洞
- **逻辑漏洞**: 越权、条件竞争、业务逻辑缺陷

#### 学习路径
1. **第1周**: 基础漏洞原理 + DVWA实验
2. **第2周**: 进阶技巧 + CTF题目练习

#### 实践环境
- DVWA (Damn Vulnerable Web Application)
- WebGoat
- Pikachu靶场
- BurpSuite工具使用

### 💥 Pwn (第3-4周重点，从零开始)
**全新领域，需要系统学习**

#### 前置知识
- **汇编语言**: x86/x64汇编基础
- **C语言**: 指针、内存管理、函数调用
- **操作系统**: 进程内存布局、系统调用
- **调试工具**: GDB、IDA、Ghidra

#### 核心技术
1. **栈溢出**: 
   - 栈结构理解
   - 返回地址覆盖
   - Shellcode编写
   - NX绕过技术

2. **堆溢出**:
   - 堆管理机制
   - Use After Free
   - Double Free
   - 堆风水技术

3. **格式化字符串**:
   - printf族函数漏洞
   - 任意地址读写
   - 栈地址泄露

4. **ROP技术**:
   - Return-Oriented Programming
   - Gadget查找和利用
   - ASLR绕过

#### 学习路径
1. **第3周**: 汇编+C语言+调试工具
2. **第4周**: 栈溢出基础+简单题目

#### 实践环境
- Ubuntu 18.04 (关闭保护机制)
- pwntools工具包
- IDA Pro / Ghidra
- GDB + pwndbg插件

### 🔒 Crypto密码学 (第1-2周，从零开始)
**全新领域，重点理论学习**

#### 基础密码学
1. **古典密码**:
   - 凯撒密码、维吉尼亚密码
   - 栅栏密码、摩斯密码
   - 频率分析方法

2. **现代密码学**:
   - 对称加密: AES、DES
   - 非对称加密: RSA、ECC
   - 哈希函数: MD5、SHA系列
   - 数字签名和认证

3. **CTF常见攻击**:
   - RSA小指数攻击
   - 维纳攻击
   - 共模攻击
   - 随机数预测

#### 学习路径
1. **第1周**: 密码学基础理论
2. **第2周**: CTF密码题技巧

#### 实践工具
- SageMath (数学计算)
- Python cryptography库
- OpenSSL工具
- 在线密码分析工具

### 🔍 Reverse逆向工程 (第3-4周，从零开始)
**全新领域，需要耐心学习**

#### 基础技能
1. **静态分析**:
   - IDA Pro使用技巧
   - 汇编代码阅读
   - 函数识别和分析
   - 字符串和常量分析

2. **动态分析**:
   - 调试器使用 (x64dbg, OllyDbg)
   - 断点设置和单步调试
   - 内存和寄存器监控
   - API监控

3. **反混淆技术**:
   - 花指令识别
   - 控制流平坦化
   - 字符串加密
   - 虚拟机保护

#### 常见题型
- 算法逆向
- 密码验证逆向
- 协议逆向
- 恶意软件分析

#### 学习路径
1. **第3周**: 工具使用+简单逆向
2. **第4周**: 算法逆向+CTF题目

#### 实践环境
- Windows 10 + IDA Pro
- Linux + Ghidra
- x64dbg调试器
- Python脚本辅助

### 🎲 Misc杂项 (贯穿整个学习过程)
**综合技能，随时练习**

#### 主要方向
1. **隐写术**:
   - 图片隐写 (LSB、DCT)
   - 音频隐写
   - 文档隐写
   - 网络隐写

2. **取证分析**:
   - 磁盘镜像分析
   - 内存取证
   - 网络流量分析
   - 日志分析

3. **编码解码**:
   - Base64、URL编码
   - 十六进制、二进制
   - 自定义编码识别

4. **其他技能**:
   - 压缩包破解
   - 二维码分析
   - 区块链安全
   - 人工智能安全

## 📅 8周详细学习计划

### 第1-2周：Web + Crypto基础
- **上午**: Web安全理论学习
- **下午**: DVWA靶场练习
- **晚上**: 密码学基础理论

### 第3-4周：Pwn + Reverse入门
- **上午**: 汇编语言学习
- **下午**: Pwn基础实践
- **晚上**: 逆向工程入门

### 第5-6周：综合实战
- **上午**: CTF题目练习
- **下午**: 弱项强化训练
- **晚上**: Misc技能学习

### 第7-8周：比赛模拟
- **上午**: 模拟CTF比赛
- **下午**: 题目复盘总结
- **晚上**: 工具脚本编写

## 🛠️ 环境搭建

### 虚拟机环境
1. **Kali Linux**: 集成安全工具
2. **Ubuntu 18.04**: Pwn练习环境
3. **Windows 10**: 逆向分析环境

### 必备工具
- **Web**: BurpSuite, OWASP ZAP, Sqlmap
- **Pwn**: pwntools, GDB, IDA Pro
- **Crypto**: SageMath, Python, OpenSSL
- **Reverse**: IDA Pro, Ghidra, x64dbg
- **Misc**: Wireshark, Volatility, Steghide

## 📚 学习资源

### 在线平台
- **练习平台**: CTFHub, BugKu, 攻防世界
- **比赛平台**: CTFtime, XCTF
- **学习网站**: CTF Wiki, i春秋

### 推荐书籍
- 《Web安全深度剖析》
- 《0day安全：软件漏洞分析技术》
- 《密码编码学与网络安全》
- 《恶意代码分析实战》

### 视频教程
- 安全牛CTF课程
- i春秋在线课程
- YouTube安全频道

## 🎯 学习成果检验

### 每周目标
- **第1-2周**: 解决20道Web题 + 10道Crypto题
- **第3-4周**: 解决10道Pwn题 + 10道Reverse题
- **第5-6周**: 解决各方向题目各15道
- **第7-8周**: 参加2次线上CTF比赛

### 技能认证
- 完成DVWA所有关卡
- 独立分析一个恶意软件样本
- 编写自动化利用脚本
- 参加CTF比赛并获得积分

## 💡 学习建议

1. **循序渐进**: 从简单题目开始，逐步提高难度
2. **动手实践**: 理论学习必须结合实际操作
3. **记录总结**: 建立个人知识库和工具库
4. **交流学习**: 加入CTF学习群，互相讨论
5. **持续练习**: 每天至少解决1-2道题目
6. **关注前沿**: 跟踪最新漏洞和攻击技术
