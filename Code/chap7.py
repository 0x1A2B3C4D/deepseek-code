# 例【7-1】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conversation_cache.py
----------------------
本示例展示了缓存机制在多轮对话和长文本生成中的应用与影响。
通过构建一个对话管理器，利用内存缓存保存历史对话上下文和部分生成结果，
减少重复的 Deepseek-R1 API 调用，从而降低响应延迟并提升对话连贯性。
使用 Flask 提供在线对话接口，模拟多轮对话场景。
"""

import os
import time
import random
import threading
import functools
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim

# ===============================
# 1. 模拟 Deepseek-R1 API 调用函数
# ===============================

def deepseek_r1_api_call(prompt: str) -> str:
    """
    模拟调用 Deepseek-R1 API 生成文本。
    实际中应调用官方 API，此处模拟返回处理结果。
    模拟过程中增加延时，模拟网络通信及计算延迟。
    """
    time.sleep(0.5)  # 模拟 500ms 的延时
    # 模拟返回结果：简单返回原文本并附加提示
    return f"Deepseek-R1 模型回复：{prompt} ... [生成内容]"

# ===============================
# 2. 缓存模块实现
# ===============================

class ConversationCache:
    """
    对话缓存模块：基于内存的简单缓存
    保存每个对话会话中的历史对话文本及生成结果
    支持设置过期时间和最大缓存条数，采用简单的 LRU 策略
    """
    def __init__(self, max_items: int = 100, expiry_seconds: int = 300):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()
        self.max_items = max_items
        self.expiry_seconds = expiry_seconds

    def _cleanup(self):
        """
        清理过期缓存和超出最大缓存条数的记录
        """
        current_time = time.time()
        keys_to_delete = []
        with self.lock:
            for key, (value, timestamp) in self.cache.items():
                if current_time - timestamp > self.expiry_seconds:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del self.cache[key]
            # 如果缓存超过最大条数，删除最早的记录
            if len(self.cache) > self.max_items:
                sorted_keys = sorted(self.cache.items(), key=lambda item: item[1][1])
                for key, _ in sorted_keys[:len(self.cache) - self.max_items]:
                    del self.cache[key]

    def set(self, key: str, value: Any):
        """
        将值保存到缓存中，并记录当前时间戳
        """
        with self.lock:
            self.cache[key] = (value, time.time())
        self._cleanup()

    def get(self, key: str) -> Optional[Any]:
        """
        从缓存中获取值，如果存在且未过期则返回，否则返回 None
        """
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp <= self.expiry_seconds:
                    return value
                else:
                    del self.cache[key]
        return None

# ===============================
# 3. 对话管理器实现：多轮对话与缓存结合
# ===============================

class DialogManager:
    """
    对话管理器：管理多轮对话的上下文及缓存
    每个对话使用唯一会话 ID 保存对话历史，支持缓存上一次的回复结果，
    并在生成长文本时使用缓存内容以提高响应速度。
    """
    def __init__(self, cache: ConversationCache):
        self.cache = cache

    def get_session_key(self, session_id: str) -> str:
        """
        根据会话 ID 生成缓存键
        """
        return f"dialog_session:{session_id}"

    def append_turn(self, session_id: str, user_input: str, bot_response: str):
        """
        将对话轮次记录到缓存中
        """
        key = self.get_session_key(session_id)
        history = self.cache.get(key)
        if history is None:
            history = []
        history.append({"user": user_input, "bot": bot_response})
        self.cache.set(key, history)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取会话历史记录
        """
        key = self.get_session_key(session_id)
        history = self.cache.get(key)
        if history is None:
            history = []
        return history

    def clear_history(self, session_id: str):
        """
        清除会话历史记录
        """
        key = self.get_session_key(session_id)
        self.cache.set(key, [])

    def generate_response(self, session_id: str, user_input: str) -> str:
        """
        生成对话回复：
        首先从缓存中获取历史对话记录，
        结合历史和当前输入构造上下文，
        调用 Deepseek-R1 API（模拟）生成回复，
        并缓存回复结果。
        """
        history = self.get_history(session_id)
        # 构造上下文：简单拼接所有轮次的对话内容
        context = ""
        for turn in history:
            context += f"用户：{turn['user']}\n回复：{turn['bot']}\n"
        context += f"用户：{user_input}\n回复："
        # 检查缓存是否已有相同上下文的回复
        cached_reply = self.cache.get(context)
        if cached_reply is not None:
            print("[缓存命中] 返回缓存回复")
            return cached_reply
        # 调用 Deepseek-R1 API 模拟生成回复
        response = deepseek_r1_api_call(context)
        # 将生成结果缓存
        self.cache.set(context, response)
        # 同时将本轮对话加入会话历史中
        self.append_turn(session_id, user_input, response)
        return response

# ===============================
# 4. Flask API 服务实现：多轮对话接口
# ===============================

app = Flask(__name__)

# 初始化全局缓存和对话管理器
global_cache = ConversationCache(max_items=500, expiry_seconds=600)
dialog_manager = DialogManager(global_cache)

@app.route('/dialog', methods=['POST'])
def dialog():
    """
    多轮对话接口：
    接收 JSON 请求 {"session_id": "会话唯一标识", "user_input": "用户输入文本"}
    返回生成的对话回复
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default_session")
    user_input = data.get("user_input", "")
    if user_input == "":
        return jsonify({"error": "缺少 'user_input' 参数"}), 400
    response = dialog_manager.generate_response(session_id, user_input)
    return jsonify({"reply": response})

@app.route('/dialog/history', methods=['GET'])
def dialog_history():
    """
    对话历史接口：
    根据 session_id 返回当前对话历史记录
    """
    session_id = request.args.get("session_id", "default_session")
    history = dialog_manager.get_history(session_id)
    return jsonify({"history": history})

@app.route('/', methods=['GET'])
def index():
    return "多轮对话推理服务已启动，请使用 /dialog 接口进行对话。"

# ===============================
# 5. 模拟长文本生成任务（利用缓存机制）
# ===============================

def generate_long_text(initial_prompt: str, max_turns: int = 5) -> str:
    """
    模拟长文本生成任务：
    利用缓存机制，分多轮生成长文本，每轮生成结果与历史拼接，
    返回完整的长文本。
    """
    session_id = "long_text_session"
    dialog_manager.clear_history(session_id)
    current_text = initial_prompt
    full_text = initial_prompt
    for turn in range(max_turns):
        # 模拟用户继续生成长文本请求：这里以当前生成的文本作为用户输入
        user_input = current_text + f" [续写{turn+1}]"
        reply = dialog_manager.generate_response(session_id, user_input)
        full_text += "\n" + reply
        current_text = reply
        print(f"[长文本生成] 轮次 {turn+1}: {reply}")
    return full_text

@app.route('/long_text', methods=['POST'])
def long_text():
    """
    长文本生成接口：
    接收 JSON 请求 {"initial_prompt": "初始文本"}
    返回完整生成的长文本
    """
    data = request.get_json(force=True)
    initial_prompt = data.get("initial_prompt", "")
    if initial_prompt == "":
        return jsonify({"error": "缺少 'initial_prompt' 参数"}), 400
    generated_text = generate_long_text(initial_prompt, max_turns=5)
    return jsonify({"long_text": generated_text})

# ===============================
# 6. 主函数：启动 Flask 服务
# ===============================

def main():
    print("启动多轮对话与长文本生成在线推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()