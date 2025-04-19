# 例【8-1】
import requests
# 设置 API 端点和 API 密钥
url = 'https://api.deepseek.com/v1/resource'  # 请将 'resource' 替换为实际的 API 端点
api_key = 'YOUR_API_KEY'  # 将 'YOUR_API_KEY' 替换为实际的 API 密钥
# 设置请求头，包括授权信息
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}
# 设置请求数据
data = {
    'prompt': '你好，世界',
    'max_tokens': 100
}
# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)
# 检查响应状态码并处理响应数据
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f'请求失败，状态码：{response.status_code}')


# 例【8-2】
import requests
import json
# 设置 API 密钥和请求 URL
api_key = 'YOUR_API_KEY'
url = 'https://api.deepseek.com/chat/completions'
# 定义请求头
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}
# 定义请求体
data = {
    'model': 'deepseek-chat',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the weather like today?'}
    ]
}
# 发送 POST 请求
response = requests.post(url, headers=headers, data=json.dumps(data))
# 处理响应
if response.status_code == 200:
    completion = response.json()
    print(completion['choices'][0]['message']['content'])
else:
    print(f'Request failed with status code {response.status_code}')


# 例【8-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_security.py
-------------------------
本示例展示如何在基于 Deepseek-R1 大模型 API 的服务中实现 API 权限控制与安全性优化。
主要功能包括：
    1. 用户登录和 API Key 生成；
    2. 请求认证装饰器：对每个 API 调用进行权限验证；
    3. 速率限制器：限制单个 API Key 的访问频率，防止滥用；
    4. 日志记录：记录所有请求与异常信息；
    5. 模拟 Deepseek-R1 API 调用：返回模拟回复。
通过多重安全防护措施，确保 API 服务仅供授权用户使用，避免恶意请求和暴力攻击。
"""
import os
import time
import random
import string
import json
import threading
import functools
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, abort
import logging
# ===============================
# 1. 全局配置与日志设置
# ===============================
# 日志设置：记录请求信息和错误日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("API_Security")
# API 配置参数
API_KEY_LENGTH = 32                # API Key 长度
TOKEN_EXPIRY_SECONDS = 3600        # API Key 过期时间（1小时）
RATE_LIMIT_COUNT = 10              # 单个 API Key 每分钟最大请求次数
RATE_LIMIT_INTERVAL = 60           # 速率限制时间间隔（秒）
# 全局存储 API Key 信息，格式：{api_key: {"user": username, "expiry": timestamp}}
api_keys: Dict[str, Dict[str, Any]] = {}
# 全局存储速率限制信息，格式：{api_key: {"count": int, "reset_time": timestamp}}
rate_limits: Dict[str, Dict[str, Any]] = {}
rate_limits_lock = threading.Lock()
# ===============================
# 2. API Key 生成与管理函数
# ===============================
def generate_api_key(length: int = API_KEY_LENGTH) -> str:
    """
    生成随机 API Key，由字母和数字组成
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))
def create_api_key(user: str) -> str:
    """
    创建 API Key 并保存到全局存储中，同时设置过期时间
    """
    key = generate_api_key()
    expiry_time = time.time() + TOKEN_EXPIRY_SECONDS
    api_keys[key] = {"user": user, "expiry": expiry_time}
    logger.info(f"创建 API Key: {key} 用户: {user} 过期时间: {expiry_time}")
    return key
def validate_api_key(key: str) -> bool:
    """
    验证 API Key 是否存在且未过期
    """
    if key in api_keys:
        expiry = api_keys[key]["expiry"]
        if time.time() < expiry:
            return True
        else:
            logger.warning(f"API Key {key} 已过期。")
            del api_keys[key]
    return False
# ===============================
# 3. 速率限制器实现
# ===============================
def check_rate_limit(api_key: str) -> bool:
    """
    检查 API Key 的速率限制：
    每个 API Key 每分钟最多允许 RATE_LIMIT_COUNT 次请求
    """
    with rate_limits_lock:
        now = time.time()
        if api_key not in rate_limits:
            rate_limits[api_key] = {"count": 1, "reset_time": now + RATE_LIMIT_INTERVAL}
            return True
        limit_info = rate_limits[api_key]
        if now > limit_info["reset_time"]:
            # 重置计数
            rate_limits[api_key] = {"count": 1, "reset_time": now + RATE_LIMIT_INTERVAL}
            return True
        elif limit_info["count"] < RATE_LIMIT_COUNT:
            rate_limits[api_key]["count"] += 1
            return True
        else:
            logger.warning(f"API Key {api_key} 速率限制触发。")
            return False
# ===============================
# 4. 权限验证装饰器
# ===============================
def require_api_key(func):
    """
    装饰器：要求请求必须包含有效的 API Key
    检查 HTTP 头部 Authorization: Bearer <API_KEY>
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.error("缺少 Bearer Token。")
            return jsonify({"error": "缺少 Bearer Token"}), 401
        api_key = auth_header.split(" ")[1]
        if not validate_api_key(api_key):
            logger.error("无效或过期的 API Key。")
            return jsonify({"error": "无效或过期的 API Key"}), 403
        if not check_rate_limit(api_key):
            logger.error("请求速率超限。")
            return jsonify({"error": "请求速率超限"}), 429
        return func(*args, **kwargs)
    return wrapper
# ===============================
# 5. 模拟 Deepseek-R1 API 调用函数
# ===============================
def deepseek_r1_simulation(prompt: str) -> str:
    """
    模拟调用 Deepseek-R1 大模型 API 生成回复。
    实际中可使用 requests.post 调用官方 API。
    这里简单模拟返回固定格式的回复，延时 500ms。
    """
    time.sleep(0.5)
    response = f"Deepseek-R1 回复：根据提示 '{prompt}' 生成的回复内容。"
    return response
# ===============================
# 6. Flask API 服务实现
# ===============================
app = Flask(__name__)
@app.route('/login', methods=['POST'])
def login():
    """
    用户登录接口：
    接收 JSON 请求 {"username": "用户名称", "password": "密码"}
    （此处密码验证模拟，实际应接入认证系统）
    返回生成的 API Key
    """
    data = request.get_json(force=True)
    username = data.get("username", "")
    password = data.get("password", "")
    # 简单模拟用户名和密码验证
    if username == "" or password == "":
        return jsonify({"error": "用户名或密码不能为空"}), 400
    if password != "secret":  # 模拟固定密码
        return jsonify({"error": "密码错误"}), 403
    key = create_api_key(username)
    return jsonify({"api_key": key, "expiry": TOKEN_EXPIRY_SECONDS})
@app.route('/deepseek', methods=['POST'])
@require_api_key
def deepseek():
    """
    Deepseek-R1 API 接口：
    接收 JSON 请求 {"prompt": "提示文本"}
    返回 Deepseek-R1 模型生成的回复
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if prompt == "":
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    # 模拟调用 Deepseek-R1 API
    reply = deepseek_r1_simulation(prompt)
    logger.info(f"生成回复：{reply}")
    return jsonify({"reply": reply})
@app.route('/admin/logs', methods=['GET'])
@require_api_key
def view_logs():
    """
    管理员日志查看接口：
    返回当前系统运行日志（模拟返回固定日志内容）
    实际中应读取日志文件或日志系统数据
    """
    logs = [
        "2024-03-01 10:00:00 INFO 用户登录成功。",
        "2024-03-01 10:05:00 INFO 调用 Deepseek-R1 API 成功。",
        "2024-03-01 10:10:00 WARNING API Key 请求速率超限。"
    ]
    return jsonify({"logs": logs})
@app.route('/', methods=['GET'])
def index():
    return "Deepseek-R1 API 服务已启动，请使用 /login 进行登录获取 API Key，然后使用 /deepseek 进行调用。"
# ===============================
# 7. 安全性优化示例：HTTPS 配置、错误处理、访问日志记录等
# ===============================
@app.before_request
def log_request_info():
    """
    请求前日志记录，记录请求 IP、路径和请求方法
    """
    logger.info(f"请求：{request.remote_addr} {request.method} {request.path}")
@app.errorhandler(404)
def not_found_error(error):
    """
    404 错误处理
    """
    logger.error("404 错误：请求的资源不存在。")
    return jsonify({"error": "资源不存在"}), 404
@app.errorhandler(500)
def internal_error(error):
    """
    500 错误处理
    """
    logger.error("500 错误：服务器内部错误。")
    return jsonify({"error": "服务器内部错误"}), 500
# ===============================
# 8. 模拟访问统计与缓存（简易实现）
# ===============================
access_counts: Dict[str, int] = {}
access_lock = threading.Lock()
def update_access_count(api_key: str):
    """
    更新访问计数，记录每个 API Key 的访问次数
    """
    with access_lock:
        if api_key in access_counts:
            access_counts[api_key] += 1
        else:
            access_counts[api_key] = 1
@app.after_request
def after_request(response):
    """
    请求后处理：记录每个请求的 API Key 及访问计数（若存在）
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.split(" ")[1]
        update_access_count(api_key)
    return response
@app.route('/admin/access', methods=['GET'])
@require_api_key
def get_access_counts():
    """
    管理员接口：返回所有 API Key 的访问计数
    """
    return jsonify(access_counts)
# ===============================
# 9. 主函数：启动 Flask 服务
# ===============================
def main():
    # 可选：加载 HTTPS 证书，实现 HTTPS 加密（此处使用自签名证书示例）
    cert_file = "server.crt"
    key_file = "server.key"
    if os.path.exists(cert_file) and os.path.exists(key_file):
        context = (cert_file, key_file)
        logger.info("检测到 HTTPS 证书，启用 HTTPS 模式。")
    else:
        context = None
        logger.info("未检测到 HTTPS 证书，启用 HTTP 模式。")
    
    logger.info("启动 Deepseek-R1 API 服务……")
    # 启动 Flask 服务，监听所有网卡，端口 8000
    app.run(host="0.0.0.0", port=8000, ssl_context=context)
if __name__ == "__main__":
    main()


# 例【8-4】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_app.py
-------------------------
基于 Python 的 Deepseek‑R1 简单应用示例
本示例演示如何调用 Deepseek‑R1 大模型 API 进行在线文本生成与多轮对话。
功能包括：
    1. 用户登录并获取 API Key；
    2. 构造请求参数，调用 Deepseek‑R1 API 生成回复（模拟实现）；
    3. 对话缓存管理，实现多轮对话历史记录保存；
    4. 在线命令行交互界面与 Flask Web 服务接口；
    5. 错误处理、日志记录及请求频率限制等安全控制。
所有代码均附详细中文注释，确保实际运行无误。
"""
import os
import sys
import time
import random
import string
import json
import threading
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from flask import Flask, request, jsonify
# ------------------------------
# 1. 全局配置与日志设置
# ------------------------------
API_KEY_LENGTH = 32                   # API Key 长度
TOKEN_EXPIRY_SECONDS = 3600           # API Key 有效期（秒）
RATE_LIMIT_MAX = 5                    # 每个 API Key 每分钟最大请求次数
RATE_LIMIT_INTERVAL = 60              # 速率限制时间间隔，单位秒
# 日志打印函数
def log_info(message: str):
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}")
def log_error(message: str):
    print(f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}")
# ------------------------------
# 2. API Key 管理与速率限制
# ------------------------------
# 全局存储 API Key 信息：{api_key: {"user": 用户名, "expiry": 到期时间}}
api_keys: Dict[str, Dict[str, any]] = {}
# 全局存储速率限制信息：{api_key: {"count": 当前计数, "reset_time": 重置时间}}
rate_limits: Dict[str, Dict[str, any]] = {}
rate_lock = threading.Lock()
def generate_api_key(length: int = API_KEY_LENGTH) -> str:
    """生成随机 API Key"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))
def create_api_key(user: str) -> str:
    """创建 API Key 并保存到全局存储中，同时设置有效期"""
    key = generate_api_key()
    expiry = time.time() + TOKEN_EXPIRY_SECONDS
    api_keys[key] = {"user": user, "expiry": expiry}
    log_info(f"为用户 {user} 创建 API Key: {key}，有效期 {TOKEN_EXPIRY_SECONDS} 秒")
    return key
def validate_api_key(key: str) -> bool:
    """验证 API Key 是否存在且未过期"""
    if key in api_keys:
        if time.time() < api_keys[key]["expiry"]:
            return True
        else:
            log_error(f"API Key {key} 已过期。")
            del api_keys[key]
    return False
def check_rate_limit(key: str) -> bool:
    """检查 API Key 的请求频率是否超过限制"""
    with rate_lock:
        current_time = time.time()
        if key not in rate_limits:
            rate_limits[key] = {"count": 1, "reset_time": current_time + RATE_LIMIT_INTERVAL}
            return True
        entry = rate_limits[key]
        if current_time > entry["reset_time"]:
            # 重置计数
            rate_limits[key] = {"count": 1, "reset_time": current_time + RATE_LIMIT_INTERVAL}
            return True
        elif entry["count"] < RATE_LIMIT_MAX:
            entry["count"] += 1
            return True
        else:
            log_error(f"API Key {key} 请求频率超限。")
            return False
# ------------------------------
# 3. 权限验证装饰器
# ------------------------------
def require_api_key(func):
    """装饰器：要求请求中包含有效的 API Key"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "缺少 Bearer Token"}), 401
        key = auth.split(" ")[1]
        if not validate_api_key(key):
            return jsonify({"error": "无效或过期的 API Key"}), 403
        if not check_rate_limit(key):
            return jsonify({"error": "请求频率超限"}), 429
        return func(*args, **kwargs)
    return wrapper
# ------------------------------
# 4. 模拟 Deepseek-R1 API 调用
# ------------------------------
def deepseek_r1_api_call(prompt: str) -> str:
    """
    模拟 Deepseek-R1 API 调用函数：
    实际中可使用 requests.post 调用官方 API，此处模拟返回生成回复
    延时 0.5 秒以模拟网络与计算延迟
    """
    time.sleep(0.5)
    simulated_reply = f"Deepseek-R1 回复：根据提示 '{prompt}' 生成的回复内容。"
    return simulated_reply
# ------------------------------
# 5. 缓存机制：保存对话与请求结果
# ------------------------------
class SimpleCache:
    """
    简单内存缓存，保存键值对，并支持超时删除
    """
    def __init__(self, max_size: int = 200, expiry: int = 300):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.expiry = expiry
        self.lock = threading.Lock()
    
    def set(self, key: str, value: any):
        with self.lock:
            self.cache[key] = (value, time.time())
            if len(self.cache) > self.max_size:
                # 删除最旧的项
                oldest_key = min(self.cache.items(), key=lambda item: item[1][1])[0]
                del self.cache[oldest_key]
    
    def get(self, key: str) -> Optional[any]:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.expiry:
                    return value
                else:
                    del self.cache[key]
        return None
global_cache = SimpleCache()
# ------------------------------
# 6. 对话管理器：保存多轮对话上下文
# ------------------------------
class ConversationManager:
    """
    对话管理器，管理多轮对话缓存与历史记录
    """
    def __init__(self, cache: SimpleCache):
        self.cache = cache
    
    def get_context(self, session_id: str) -> str:
        context = self.cache.get(session_id)
        if context is None:
            context = ""
        return context
    
    def update_context(self, session_id: str, user_input: str, bot_reply: str):
        context = self.get_context(session_id)
        new_context = context + f"用户: {user_input}\n回复: {bot_reply}\n"
        self.cache.set(session_id, new_context)
    
    def clear_context(self, session_id: str):
        self.cache.set(session_id, "")
conversation_manager = ConversationManager(global_cache)
# ------------------------------
# 7. Python 应用：交互式命令行与 Flask API 示例
# ------------------------------
def interactive_chat():
    """
    交互式命令行聊天应用
    模拟多轮对话，调用 Deepseek-R1 API，并使用缓存保存上下文
    """
    session_id = input("请输入会话 ID：").strip() or "default_session"
    conversation_manager.clear_context(session_id)
    print("开始对话，输入 'exit' 退出。")
    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() == "exit":
            print("对话结束。")
            break
        # 构造上下文：获取历史对话
        context = conversation_manager.get_context(session_id)
        prompt = context + f"用户: {user_input}\n回复: "
        # 检查缓存是否命中
        cached_reply = global_cache.get(prompt)
        if cached_reply:
            reply = cached_reply
            print("[缓存命中] ", reply)
        else:
            # 调用 Deepseek-R1 API 模拟
            reply = deepseek_r1_api_call(prompt)
            global_cache.set(prompt, reply)
            print("Deepseek-R1 回复: ", reply)
        # 更新对话上下文
        conversation_manager.update_context(session_id, user_input, reply)
def flask_app_run():
    """
    Flask 应用：提供在线多轮对话接口
    包含 /login 获取 API Key、/chat 进行对话、/history 查看历史记录接口
    """
    app = Flask(__name__)
    
    @app.route('/login', methods=['POST'])
    def login():
        """
        登录接口：接收 {"username": "xxx", "password": "xxx"}，验证后返回 API Key
        密码统一为 "secret"，仅作示例
        """
        data = request.get_json(force=True)
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400
        if password != "secret":
            return jsonify({"error": "密码错误"}), 403
        key = create_api_key(username)
        return jsonify({"api_key": key, "expiry": TOKEN_EXPIRY_SECONDS})
    
    @app.route('/chat', methods=['POST'])
    @require_api_key
    def chat():
        """
        对话接口：接收 {"session_id": "会话ID", "user_input": "用户输入"}，
        返回 Deepseek-R1 模型生成的回复，同时更新对话历史缓存
        """
        data = request.get_json(force=True)
        session_id = data.get("session_id", "default_session")
        user_input = data.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "缺少 'user_input' 参数"}), 400
        # 构造上下文
        context = conversation_manager.get_context(session_id)
        prompt = context + f"用户: {user_input}\n回复: "
        # 检查缓存
        cached_reply = global_cache.get(prompt)
        if cached_reply:
            reply = cached_reply
        else:
            reply = deepseek_r1_api_call(prompt)
            global_cache.set(prompt, reply)
        conversation_manager.update_context(session_id, user_input, reply)
        return jsonify({"reply": reply})
    
    @app.route('/history', methods=['GET'])
    @require_api_key
    def history():
        """
        查看对话历史接口：根据 query 参数 session_id 返回对话历史
        """
        session_id = request.args.get("session_id", "default_session")
        context = conversation_manager.get_context(session_id)
        return jsonify({"history": context})
    
    @app.route('/', methods=['GET'])
    def index():
        return "Deepseek-R1 简单应用示例在线服务已启动，请使用 /login 进行登录。"
    
    # 启动 Flask 服务，支持 HTTPS 如有证书，否则使用 HTTP
    cert_file = "server.crt"
    key_file = "server.key"
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_context = (cert_file, key_file)
    else:
        ssl_context = None
    app.run(host="0.0.0.0", port=8000, ssl_context=ssl_context)
def main():
    """
    主函数：根据命令行参数选择运行交互式聊天或 Flask API 服务
    """
    import argparse
    parser = argparse.ArgumentParser(description="基于 Python 的 Deepseek-R1 简单应用示例")
    parser.add_argument("--mode", type=str, default="cli", help="运行模式：cli 或 flask")
    args = parser.parse_args()
    if args.mode == "cli":
        interactive_chat()
    elif args.mode == "flask":
        flask_app_run()
    else:
        print("无效的模式，请选择 cli 或 flask")
if __name__ == "__main__":
    main()


# 例【8-5】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_r1_local_deploy.py
---------------------------
本示例展示 Deepseek-R1 模型的本地化部署流程。
流程主要包括：
    1. 环境准备：依赖安装与系统配置（假设系统为 Ubuntu 20.04，已安装 NVIDIA 驱动、CUDA、Docker、nvidia-docker2）
    2. 模型文件获取：从官方或开源仓库下载 Deepseek-R1 预训练权重与配置文件
    3. Docker 镜像构建：编写 Dockerfile 构建支持 GPU 的运行环境，并将模型代码与权重复制进去
    4. docker-compose 部署：通过 docker-compose 文件映射端口、挂载数据卷，配置 GPU 资源使用
    5. API 服务实现：利用 Flask 实现推理接口，加载 Deepseek-R1 模型并提供在线推理
    6. 测试与验证：通过 curl 或浏览器测试 API 接口，查看生成结果
"""
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 模型与依赖说明

# 本示例基于 Deepseek-R1 模型的 API，本地化部署主要依赖 Docker 环境，
# 采用 NVIDIA 官方 CUDA 镜像构建基础环境，利用 transformers 库加载模型，
# 同时使用 Flask 提供在线推理接口。
# 模型权重与配置文件应提前下载至本地指定目录，例如 ./deepseek_r1_model/
# 实际文件可通过 Deepseek 开源仓库或官方平台获取，此处假设模型文件已经满足 transformers 格式
MODEL_DIR = "./deepseek_r1_model"  # 模型权重存放目录
MODEL_NAME = "deepseek-r1"         # 模型名称

# 2. Dockerfile 示例（保存为 Dockerfile 文件）

DOCKERFILE_CONTENT = r'''
# 使用 NVIDIA CUDA 基础镜像（Ubuntu 20.04 + CUDA 11.7）
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
# 设置工作目录
WORKDIR /workspace
# 避免交互提示
ENV DEBIAN_FRONTEND=noninteractive
# 安装系统依赖及 Python3.8
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3-venv git wget curl unzip vim \
    && rm -rf /var/lib/apt/lists/*
# 升级 pip 并安装依赖库：transformers、torch、flask 等
RUN python3.8 -m pip install --upgrade pip && \
    pip install flask torch torchvision transformers==4.28.0 numpy
# 复制本地模型文件和代码到容器中
COPY . /workspace
# 暴露端口：8000 用于 API 服务
EXPOSE 8000
# 设置启动命令：运行 API 服务
CMD ["python3.8", "deepseek_r1_api_service.py"]
'''
# 将 Dockerfile 内容写入文件
with open("Dockerfile", "w", encoding="utf-8") as f:
    f.write(DOCKERFILE_CONTENT)
print("Dockerfile 已生成。")

# 3. docker-compose.yml 示例（保存为 docker-compose.yml 文件）

DOCKER_COMPOSE_CONTENT = r'''
version: '3.8'
services:
  deepseek_r1:
    build: .
    container_name: deepseek_r1_api
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    ports:
      - "8000:8000"
    volumes:
      - ./deepseek_r1_model:/workspace/deepseek_r1_model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
'''
with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.write(DOCKER_COMPOSE_CONTENT)
print("docker-compose.yml 已生成。")

# 4. Deepseek-R1 API 服务实现（deepseek_r1_api_service.py）

API_SERVICE_CODE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_r1_api_service.py
---------------------------
基于 Flask 的 Deepseek-R1 模型 API 服务示例。
加载本地模型权重，提供 /predict 接口进行文本生成推理。
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
app = Flask(__name__)
# 模型文件目录与名称
MODEL_DIR = "./deepseek_r1_model"
MODEL_NAME = "deepseek-r1"
# 加载 tokenizer 与模型
def load_model(model_dir: str):
    """
    加载 Deepseek-R1 模型与 tokenizer
    模型文件需满足 transformers from_pretrained 格式
    """
    print("开始加载模型……")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    elapsed = time.time() - start_time
    print(f"模型加载完成，耗时 {elapsed:.2f} 秒。")
    return tokenizer, model, device
tokenizer, model, device = load_model(MODEL_DIR)
@app.route('/predict', methods=['POST'])
def predict():
    """
    /predict 接口：
    接收 JSON 请求 {"prompt": "提示文本", "max_length": 整数}
    返回模型生成的文本回复
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if prompt == "":
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    max_length = data.get("max_length", 50)
    # 对 prompt 进行编码
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # 使用模型生成文本，采用自动混合精度（可选）
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"reply": generated_text})
@app.route('/', methods=['GET'])
def index():
    return "Deepseek-R1 模型 API 服务已启动，请使用 /predict 接口进行调用。"
if __name__ == "__main__":
    # 启动 Flask 服务，使用 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000)
'''
# 将 API 服务代码写入文件 deepseek_r1_api_service.py
with open("deepseek_r1_api_service.py", "w", encoding="utf-8") as f:
    f.write(API_SERVICE_CODE)
print("deepseek_r1_api_service.py 已生成。")

# 5. 说明：本地部署流程总结

DEPLOYMENT_SUMMARY = """
本地部署流程如下：
1. 环境准备：确保系统已安装 NVIDIA 驱动、CUDA、Docker 与 nvidia-docker2。
2. 模型文件获取：将 Deepseek-R1 模型权重及配置文件下载到 ./deepseek_r1_model 目录。
3. Dockerfile 与 docker-compose.yml 文件已生成，通过 docker-compose 构建镜像：
       docker-compose up --build
4. 部署完成后，容器内会启动 deepseek_r1_api_service.py，服务监听 8000 端口。
5. 可通过 /predict 接口发送 POST 请求，传入提示文本，获得模型生成回复。
"""
print(DEPLOYMENT_SUMMARY)

# 6. 测试脚本（本地调用 API 服务，模拟请求）

def test_api():
    """
    测试 API 接口：发送请求到 /predict，打印返回结果
    """
    url = "http://localhost:8000/predict"
    prompt = "介绍一下深度学习的发展历程。"
    payload = {"prompt": prompt, "max_length": 60}
    headers = {"Content-Type": "application/json"}
    print("发送请求到 /predict 接口……")
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("API 返回结果：")
            print(data["reply"])
        else:
            print("API 请求失败，状态码：", response.status_code)
            print(response.text)
    except Exception as e:
        print("请求异常：", str(e))
if __name__ == "__main__":
    # 本模块可作为脚本测试 API 服务（需先启动 docker-compose 服务）
    # test_api() 仅用于本地测试，实际部署通过 docker-compose 进行
    pass
