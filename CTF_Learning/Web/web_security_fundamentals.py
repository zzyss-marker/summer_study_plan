"""
Web安全基础学习
包含SQL注入、XSS、CSRF等常见Web漏洞的演示和防护
"""

import sqlite3
import os
import hashlib
import re
import html

class WebSecurityFundamentals:
    """Web安全基础学习类"""
    
    def __init__(self):
        self.db_name = "web_security_demo.db"
        self.setup_database()
        self.examples_completed = []
        print("🔒 Web安全基础学习系统")
        print("=" * 50)
    
    def setup_database(self):
        """设置演示数据库"""
        if os.path.exists(self.db_name):
            os.remove(self.db_name)
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                profile TEXT
            )
        ''')
        
        # 创建敏感信息表
        cursor.execute('''
            CREATE TABLE secrets (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                access_level TEXT DEFAULT 'admin'
            )
        ''')
        
        # 插入测试数据
        users_data = [
            (1, 'admin', self.hash_password('admin123'), 'admin@example.com', 'admin', '管理员账户'),
            (2, 'user1', self.hash_password('password1'), 'user1@example.com', 'user', '普通用户'),
            (3, 'guest', self.hash_password('guest123'), 'guest@example.com', 'guest', '访客账户'),
            (4, 'flag_user', self.hash_password('flag{web_security_demo}'), 'flag@example.com', 'user', 'CTF Flag用户')
        ]
        
        secrets_data = [
            (1, '系统密钥', 'SECRET_KEY_12345', 'admin'),
            (2, '数据库密码', 'db_password_xyz', 'admin'),
            (3, 'Flag', 'flag{sql_injection_success}', 'admin')
        ]
        
        cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)', users_data)
        cursor.executemany('INSERT INTO secrets VALUES (?, ?, ?, ?)', secrets_data)
        
        conn.commit()
        conn.close()
        print("✅ Web安全演示数据库创建完成")
    
    def hash_password(self, password):
        """简单的密码哈希（演示用）"""
        return hashlib.md5(password.encode()).hexdigest()
    
    def sql_injection_demo(self):
        """SQL注入漏洞演示"""
        print("\n💉 SQL注入漏洞演示")
        print("=" * 30)
        
        def vulnerable_login(username, password):
            """存在SQL注入漏洞的登录函数"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # 危险：直接拼接用户输入
            query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{self.hash_password(password)}'"
            print(f"🔍 执行的SQL查询: {query}")
            
            try:
                cursor.execute(query)
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    print(f"✅ 登录成功! 用户: {result[1]}, 角色: {result[4]}")
                    return True, result
                else:
                    print("❌ 登录失败!")
                    return False, None
            except sqlite3.Error as e:
                print(f"💥 SQL错误: {e}")
                conn.close()
                return False, None
        
        def safe_login(username, password):
            """安全的登录函数"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # 安全：使用参数化查询
            query = "SELECT * FROM users WHERE username = ? AND password = ?"
            print(f"🔒 安全查询: {query}")
            
            try:
                cursor.execute(query, (username, self.hash_password(password)))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    print(f"✅ 安全登录成功! 用户: {result[1]}")
                    return True, result
                else:
                    print("❌ 登录失败!")
                    return False, None
            except sqlite3.Error as e:
                print(f"💥 SQL错误: {e}")
                conn.close()
                return False, None
        
        # 演示正常登录
        print("1. 正常登录测试:")
        vulnerable_login("admin", "admin123")
        
        print("\n2. SQL注入攻击测试:")
        # 经典SQL注入 - 绕过密码验证
        print("攻击载荷: admin' OR '1'='1' --")
        vulnerable_login("admin' OR '1'='1' --", "任意密码")
        
        print("\n3. UNION注入 - 获取敏感信息:")
        print("攻击载荷: ' UNION SELECT id, title, content, access_level, 'hacked' FROM secrets --")
        vulnerable_login("' UNION SELECT id, title, content, access_level, 'hacked' FROM secrets --", "")
        
        print("\n4. 安全防护演示:")
        safe_login("admin' OR '1'='1' --", "任意密码")
        
        self.examples_completed.append("SQL注入")
    
    def xss_demo(self):
        """XSS跨站脚本攻击演示"""
        print("\n🎭 XSS跨站脚本攻击演示")
        print("=" * 30)
        
        def vulnerable_profile_update(user_id, profile_content):
            """存在XSS漏洞的用户资料更新"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # 危险：直接存储用户输入
            cursor.execute("UPDATE users SET profile = ? WHERE id = ?", (profile_content, user_id))
            conn.commit()
            conn.close()
            
            print(f"📝 用户资料已更新: {profile_content}")
            return profile_content
        
        def vulnerable_profile_display(user_id):
            """存在XSS漏洞的用户资料显示"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute("SELECT profile FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                profile = result[0]
                # 危险：直接输出用户内容，未进行HTML转义
                print(f"🖥️ 显示用户资料: {profile}")
                return profile
            return None
        
        def safe_profile_display(user_id):
            """安全的用户资料显示"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute("SELECT profile FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                profile = result[0]
                # 安全：HTML转义
                safe_profile = html.escape(profile)
                print(f"🔒 安全显示用户资料: {safe_profile}")
                return safe_profile
            return None
        
        # 演示XSS攻击
        print("1. 正常用户资料:")
        vulnerable_profile_update(2, "我是一个普通用户")
        vulnerable_profile_display(2)
        
        print("\n2. XSS攻击载荷:")
        xss_payload = "<script>alert('XSS攻击成功！')</script>"
        print(f"攻击载荷: {xss_payload}")
        vulnerable_profile_update(2, xss_payload)
        vulnerable_profile_display(2)
        
        print("\n3. 反射型XSS:")
        reflected_xss = "<img src=x onerror=alert('反射型XSS')>"
        print(f"反射型XSS载荷: {reflected_xss}")
        print(f"如果直接输出到页面: {reflected_xss}")
        
        print("\n4. 安全防护 - HTML转义:")
        safe_profile_display(2)
        
        self.examples_completed.append("XSS攻击")
    
    def input_validation_demo(self):
        """输入验证演示"""
        print("\n✅ 输入验证和过滤演示")
        print("=" * 30)
        
        def validate_email(email):
            """邮箱格式验证"""
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return re.match(pattern, email) is not None
        
        def validate_username(username):
            """用户名验证"""
            # 只允许字母、数字和下划线，长度3-20
            pattern = r'^[a-zA-Z0-9_]{3,20}$'
            return re.match(pattern, username) is not None
        
        def sanitize_input(user_input):
            """输入清理"""
            # 移除潜在的危险字符
            dangerous_chars = ['<', '>', '"', "'", '&', 'script', 'javascript']
            sanitized = user_input
            
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            
            return sanitized.strip()
        
        # 测试输入验证
        test_cases = [
            ("admin@example.com", "邮箱"),
            ("invalid-email", "邮箱"),
            ("valid_user123", "用户名"),
            ("invalid user!", "用户名"),
            ("<script>alert('xss')</script>", "危险输入"),
            ("正常的用户输入", "正常输入")
        ]
        
        print("输入验证测试:")
        for test_input, input_type in test_cases:
            print(f"\n测试输入: {test_input} ({input_type})")
            
            if input_type == "邮箱":
                result = validate_email(test_input)
                print(f"邮箱验证: {'✅ 有效' if result else '❌ 无效'}")
            
            elif input_type == "用户名":
                result = validate_username(test_input)
                print(f"用户名验证: {'✅ 有效' if result else '❌ 无效'}")
            
            else:
                sanitized = sanitize_input(test_input)
                print(f"输入清理: {test_input} → {sanitized}")
        
        self.examples_completed.append("输入验证")
    
    def security_headers_demo(self):
        """安全头部演示"""
        print("\n🛡️ 安全头部演示")
        print("=" * 30)
        
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        print("推荐的安全头部:")
        for header, value in security_headers.items():
            print(f"  {header}: {value}")
        
        print("\n安全头部作用:")
        print("• X-Content-Type-Options: 防止MIME类型嗅探攻击")
        print("• X-Frame-Options: 防止点击劫持攻击")
        print("• X-XSS-Protection: 启用浏览器XSS过滤器")
        print("• Strict-Transport-Security: 强制使用HTTPS")
        print("• Content-Security-Policy: 控制资源加载策略")
        print("• Referrer-Policy: 控制Referer头部信息")
        
        self.examples_completed.append("安全头部")
    
    def run_all_demos(self):
        """运行所有Web安全演示"""
        print("🔒 Web安全基础完整学习")
        print("=" * 60)
        
        self.sql_injection_demo()
        self.xss_demo()
        self.input_validation_demo()
        self.security_headers_demo()
        
        print(f"\n🎉 Web安全基础学习完成！")
        print(f"完成的模块: {', '.join(self.examples_completed)}")
        
        print(f"\n📚 学习总结:")
        print("1. SQL注入 - 参数化查询是最佳防护")
        print("2. XSS攻击 - 输入验证和输出编码")
        print("3. 输入验证 - 白名单验证优于黑名单")
        print("4. 安全头部 - 多层防护机制")
        
        print(f"\n🎯 CTF实战技巧:")
        print("1. 寻找输入点和注入点")
        print("2. 尝试各种绕过技巧")
        print("3. 利用错误信息收集信息")
        print("4. 组合多种攻击技术")

def main():
    """主函数"""
    web_security = WebSecurityFundamentals()
    web_security.run_all_demos()
    
    print("\n💡 进阶学习建议:")
    print("1. 学习更多注入类型：NoSQL注入、LDAP注入")
    print("2. 深入研究XSS绕过技术")
    print("3. 学习CSRF、SSRF等其他Web漏洞")
    print("4. 实践Web渗透测试工具")

if __name__ == "__main__":
    main()
