# 例【10-1】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multimodal_completion.py
------------------------
基于 Deepseek-R1 模型 API 的多模态对话系统补全策略示例。
接收用户输入的文本与图像描述（或图像 URL），融合多模态信息后构造提示，
调用模拟的 Deepseek-R1 API 生成回复，并通过 Flask 在线返回生成结果。
"""

import time
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_r1_api(prompt: str) -> str:
    """
    模拟调用 Deepseek-R1 模型 API，根据输入提示生成回复文本。
    延时 0.5 秒以模拟网络和计算延时。
    参数:
        prompt: 融合文本和图像信息构造的提示文本
    返回:
        模拟生成的回复内容。
    """
    time.sleep(0.5)  # 模拟延时
    return f"Deepseek-R1 回复：根据提示 '{prompt}' 生成的回复内容。"

@app.route('/multimodal', methods=['POST'])
def multimodal_completion():
    """
    /multimodal 接口：
    接收 JSON 请求，格式为：
    {
        "text": "用户文本输入",
        "image": "图像描述或URL"
    }
    将文本与图像信息进行融合，构造提示文本，并调用模拟的 Deepseek-R1 API 生成对话回复，
    最后以 JSON 格式返回生成结果。
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    image = data.get("image", "")
    if not text and not image:
        return jsonify({"error": "至少需提供文本或图像信息"}), 400
    # 构造多模态提示文本
    prompt = ""
    if text:
        prompt += f"文本信息：{text}\n"
    if image:
        prompt += f"图像信息：{image}\n"
    prompt += "请根据以上信息补全对话回复。"
    # 调用 Deepseek-R1 API 模拟函数生成回复
    reply = simulate_deepseek_r1_api(prompt)
    return jsonify({"reply": reply})

@app.route('/', methods=['GET'])
def index():
    """
    根路由返回服务说明信息
    """
    return "多模态对话补全服务已启动，请使用 /multimodal 接口调用。"

if __name__ == "__main__":
    # 启动 Flask 服务，监听 0.0.0.0 的 8000 端口
    app.run(host="0.0.0.0", port=8000)


# 例【10-2】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
legal_summary.py
---------------------------
本示例展示基于 Deepseek-R1 API 模拟的自动摘要生成器，用于生成法律文书摘要。
接收长文本输入，通过调用模拟的 Deepseek-R1 API 生成摘要，并返回生成结果。
"""

import time
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_summary(document: str) -> str:
    """
    模拟 Deepseek-R1 模型 API 调用，生成输入文档的摘要。
    延时 0.5 秒模拟网络与计算延时，根据输入文本关键字返回固定摘要示例。
    参数:
        document: 长文本输入，例如法律文书全文
    返回:
        摘要文本
    """
    time.sleep(0.5)
    # 模拟摘要生成：根据文档中是否包含“合同”、“诉讼”等关键词返回不同摘要
    if "合同" in document:
        return ("摘要：该合同主要涉及双方权利义务、违约责任及争议解决机制，"
                "对合同履行风险进行了详细约定。")
    elif "诉讼" in document:
        return ("摘要：该法律文书主要论述了诉讼案件的事实、证据及法律适用问题，"
                "提出了针对性的法律意见。")
    else:
        return ("摘要：本文档内容涉及复杂法律条款，重点说明了相关法律适用及合同履行注意事项。")

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    /summarize 接口：
    接收 JSON 请求 {"document": "长文本内容"}，
    调用 simulate_deepseek_summary() 生成摘要，并返回 JSON 格式的摘要文本。
    """
    data = request.get_json(force=True)
    document = data.get("document", "")
    if not document:
        return jsonify({"error": "缺少 'document' 参数"}), 400
    summary = simulate_deepseek_summary(document)
    return jsonify({"summary": summary})

@app.route('/', methods=['GET'])
def index():
    """
    根路由返回服务说明信息
    """
    return "法律文书自动摘要生成服务已启动，请使用 /summarize 接口提交文档。"

if __name__ == "__main__":
    # 启动 Flask 服务，监听 0.0.0.0 的 8000 端口
    app.run(host="0.0.0.0", port=8000)


# 例【10-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function_callback_demo.py
--------------------------
基于 Deepseek-R1 模型函数回调机制的交互式开发示例。
本示例模拟函数调用，当提示中包含特定函数调用标识时，
触发预定义函数的执行并返回结果，再将该结果反馈给用户。
"""

import time
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# 模拟的预定义函数：计算两个数字的和
def calculate_sum(a: float, b: float) -> float:
    """计算两个数字的和"""
    return a + b

# 模拟 Deepseek-R1 API 的函数调用解析
def simulate_deepseek_function_call(prompt: str) -> dict:
    """
    模拟解析 Deepseek-R1 模型返回的函数调用指令
    如果提示中包含 "CALL:calculate_sum" 则解析参数，并返回函数调用结果。
    否则返回普通文本回复。
    """
    if "CALL:calculate_sum" in prompt:
        try:
            # 示例格式：CALL:calculate_sum({"a": 3, "b": 5})
            start = prompt.find("CALL:calculate_sum")
            json_str = prompt[start + len("CALL:calculate_sum"):].strip()
            # 尝试解析JSON字符串
            params = json.loads(json_str)
            result = calculate_sum(params["a"], params["b"])
            return {"function_called": "calculate_sum", "result": result}
        except Exception as e:
            return {"error": f"函数调用解析错误：{str(e)}"}
    else:
        # 普通回复模拟
        return {"reply": f"Deepseek-R1 回复：{prompt} 的自动回复内容。"}

@app.route('/function_call', methods=['POST'])
def function_call():
    """
    /function_call 接口：
    接收 JSON 请求 {"prompt": "输入提示文本"}
    调用 simulate_deepseek_function_call() 模拟函数回调机制，
    返回函数调用结果或普通回复。
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    response = simulate_deepseek_function_call(prompt)
    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    return "Deepseek-R1 函数回调示例服务已启动，请使用 /function_call 接口进行调用。"

if __name__ == "__main__":
    # 启动 Flask 服务，监听 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000)


# 例【10-4】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本示例演示如何利用硬盘缓存机制在大规模推理任务中存储与复用推理结果，
避免重复计算。程序通过对输入提示进行哈希计算生成缓存文件名，
如果缓存存在则直接读取返回，否则调用Deepseek-R1 API模拟函数生成结果，
并将结果写入硬盘缓存后返回。此机制适用于长文本生成和多轮对话场景。
"""

import os
import time
import hashlib
import json

# 缓存目录配置
CACHE_DIR = "./cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def hash_prompt(prompt: str) -> str:
    """对提示文本进行MD5哈希处理，生成缓存文件名"""
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()

def simulate_deepseek_r1_api(prompt: str) -> str:
    """
    模拟Deepseek-R1 API调用，延时0.5秒后返回生成的回复文本
    参数:
        prompt: 输入提示文本
    返回:
        模拟生成的回复文本
    """
    time.sleep(0.5)
    return f"Deepseek-R1回复：根据提示'{prompt}'生成的回复内容。"

def get_inference_result(prompt: str) -> str:
    """
    利用硬盘缓存获取推理结果：
    如果缓存存在，直接读取返回，否则调用API生成结果并保存到缓存中。
    """
    file_hash = hash_prompt(prompt)
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}.json")
    # 检查缓存是否存在
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print("[缓存命中] 返回缓存结果。")
            return data.get("reply", "")
    # 缓存不存在，调用API生成结果
    reply = simulate_deepseek_r1_api(prompt)
    # 保存结果到缓存文件
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "reply": reply, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)
    print("[缓存更新] 结果已保存到缓存。")
    return reply
if __name__ == "__main__":
    # 测试示例
    test_prompt = "请介绍一下深度学习在图像识别中的应用。"
    print("输入提示：", test_prompt)
    result1 = get_inference_result(test_prompt)
    print("推理结果：", result1)
    
    # 再次调用相同提示，验证缓存命中
    result2 = get_inference_result(test_prompt)
    print("重复调用结果：", result2)