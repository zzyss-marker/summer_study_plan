"""
SQL注入漏洞学习演示
包含常见的SQL注入类型和防护方法
"""

import sqlite3
import hashlib
import os
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 创建数据库和表
def init_database():
    """初始化数据库"""
    conn = sqlite3.connect('demo_app.db')
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT,
            role TEXT DEFAULT 'user'
        )
    ''')
    
    # 插入测试数据
    test_users = [
        ('admin', 'admin123', 'admin@test.com', 'admin'),
        ('user1', 'password1', 'user1@test.com', 'user'),
        ('flag_user', 'flag{sql_injection_demo}', 'flag@test.com', 'user')
    ]
    
    for username, password, email, role in test_users:
        hashed_password = hashlib.md5(password.encode()).hexdigest()
        try:
            cursor.execute(
                "INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                (username, hashed_password, email, role)
            )
        except sqlite3.IntegrityError:
            pass  # 用户已存在
    
    conn.commit()
    conn.close()

# 漏洞示例：经典SQL注入
@app.route('/vulnerable_login', methods=['GET', 'POST'])
def vulnerable_login():
    """存在SQL注入漏洞的登录页面"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        # 危险：直接拼接SQL语句
        conn = sqlite3.connect('demo_app.db')
        cursor = conn.cursor()
        
        # 这里存在SQL注入漏洞！
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashlib.md5(password.encode()).hexdigest()}'"
        
        print(f"执行的SQL查询: {query}")  # 调试信息
        
        try:
            cursor.execute(query)
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return f"登录成功！欢迎 {user[1]}，你的角色是 {user[4]}"
            else:
                return "登录失败：用户名或密码错误"
        except Exception as e:
            conn.close()
            return f"数据库错误: {str(e)}"
    
    return '''
    <h2>漏洞登录页面 (存在SQL注入)</h2>
    <form method="post">
        用户名: <input type="text" name="username" placeholder="试试: admin' OR '1'='1' --"><br><br>
        密码: <input type="password" name="password" placeholder="任意密码"><br><br>
        <input type="submit" value="登录">
    </form>
    <h3>SQL注入测试用例:</h3>
    <ul>
        <li>用户名: <code>admin' OR '1'='1' --</code></li>
        <li>用户名: <code>' UNION SELECT 1,username,password,email,role FROM users --</code></li>
    </ul>
    '''

# 安全示例：参数化查询
@app.route('/secure_login', methods=['GET', 'POST'])
def secure_login():
    """安全的登录页面（使用参数化查询）"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        conn = sqlite3.connect('demo_app.db')
        cursor = conn.cursor()
        
        # 安全：使用参数化查询
        query = "SELECT * FROM users WHERE username = ? AND password = ?"
        hashed_password = hashlib.md5(password.encode()).hexdigest()
        
        cursor.execute(query, (username, hashed_password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return f"登录成功！欢迎 {user[1]}"
        else:
            return "登录失败：用户名或密码错误"
    
    return '''
    <h2>安全登录页面 (防SQL注入)</h2>
    <form method="post">
        用户名: <input type="text" name="username"><br><br>
        密码: <input type="password" name="password"><br><br>
        <input type="submit" value="登录">
    </form>
    <p>这个页面使用了参数化查询，可以防止SQL注入攻击。</p>
    '''

@app.route('/')
def index():
    """主页"""
    return '''
    <h1>SQL注入学习演示</h1>
    <h2>演示页面:</h2>
    <ul>
        <li><a href="/vulnerable_login">漏洞登录页面</a> - 存在SQL注入</li>
        <li><a href="/secure_login">安全登录页面</a> - 防护示例</li>
    </ul>
    
    <h2>学习目标:</h2>
    <ol>
        <li>理解SQL注入的原理和危害</li>
        <li>掌握SQL注入的检测方法</li>
        <li>学会防护措施和安全编码</li>
    </ol>
    
    <p><strong>警告：仅用于学习目的，请勿在生产环境中使用！</strong></p>
    '''

if __name__ == '__main__':
    print("🔐 SQL注入学习演示")
    print("=" * 50)
    
    # 初始化数据库
    init_database()
    print("✅ 数据库初始化完成")
    
    print("🌐 启动Web服务器...")
    print("📝 访问 http://127.0.0.1:5000 开始学习")
    print("💡 学习重点：")
    print("   - 理解SQL注入原理")
    print("   - 掌握注入检测技术")
    print("   - 学会安全防护方法")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
