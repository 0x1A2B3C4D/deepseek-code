# 例【9-1】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_evaluation.py
------------------------
本示例演示如何通过 Deepseek-R1 模型 API 模拟复杂方程求解与逻辑推理能力评估。
接收包含数学方程或逻辑问题的提示，返回模型生成的详细解答。
"""

import time
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_r1(prompt: str) -> str:
    """
    模拟 Deepseek-R1 模型 API 调用，根据输入提示生成解答文本。
    为了演示复杂方程求解与逻辑推理，针对不同类型提示返回固定示例结果。
    延时 0.5 秒模拟网络和计算延时。
    """
    time.sleep(0.5)
    # 如果提示中包含 "求解" 或 "方程"，返回数学方程求解示例；否则返回逻辑推理示例
    if "求解" in prompt or "方程" in prompt:
        # 模拟复杂方程求解结果
        return ("复杂方程求解结果：经过多轮迭代计算，方程的近似解为 x ≈ 3.142，"
                "中间计算过程包括数值逼近、误差修正等步骤。")
    else:
        # 模拟逻辑推理结果
        return ("逻辑推理结果：根据条件推断，假设 A > B 且 B > C，则 A > C；"
                "由此推理得到结论符合预期逻辑。")

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    /evaluate 接口：
    接收 JSON 请求 {"prompt": "输入提示文本"}，
    调用模拟 Deepseek-R1 API 生成解答，并返回生成结果。
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    result = simulate_deepseek_r1(prompt)
    return jsonify({"reply": result})

@app.route('/', methods=['GET'])
def index():
    return "Deepseek-R1 复杂方程求解与逻辑推理评估服务已启动，请使用 /evaluate 接口进行调用。"

if __name__ == "__main__":
    # 启动 Flask 服务，监听 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000)


# 例【9-2】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_code_completion.py
----------------------------
本示例演示基于 Deepseek-R1 API 的代码补全与常用算法自动生成实践。
通过调用模拟的 Deepseek-R1 API 函数，对输入的代码补全提示或算法描述进行推理，
生成完整代码。采用 Flask 构建在线 API 服务，便于交互调用。
"""

import time
from flask import Flask, request, jsonify

app = Flask(__name__)

def deepseek_r1_api_call(prompt: str) -> str:
    """
    模拟调用 Deepseek-R1 大模型 API 生成代码补全或算法实现结果。
    参数:
        prompt: 用户输入的不完整代码片段或算法描述
    返回:
        模拟生成的完整代码文本。
    模拟过程中延时 0.5 秒以模拟实际网络和计算延迟。
    """
    time.sleep(0.5)  # 模拟网络延时
    # 根据提示文本判断生成内容
    if "排序" in prompt:
        # 返回一个冒泡排序算法示例
        return ("def bubble_sort(arr):\n"
                "    n = len(arr)\n"
                "    for i in range(n):\n"
                "        for j in range(0, n-i-1):\n"
                "            if arr[j] > arr[j+1]:\n"
                "                arr[j], arr[j+1] = arr[j+1], arr[j]\n"
                "    return arr")
    elif "代码补全" in prompt:
        # 返回一个代码补全示例
        return ("# 代码补全示例\n"
                "def example_function(x):\n"
                "    # 根据 x 计算结果\n"
                "    result = x * 2\n"
                "    return result")
    else:
        # 默认返回简单代码模板
        return ("# 自动生成代码模板\n"
                "def auto_generated():\n"
                "    print('Deepseek-R1 自动生成代码')\n")

@app.route('/code_completion', methods=['POST'])
def code_completion():
    """
    /code_completion 接口：
    接收 JSON 请求 {"prompt": "代码补全提示文本"}，
    返回生成的完整代码文本。
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    completion = deepseek_r1_api_call(prompt)
    return jsonify({"completion": completion})

@app.route('/algorithm_generation', methods=['POST'])
def algorithm_generation():
    """
    /algorithm_generation 接口：
    接收 JSON 请求 {"description": "算法描述"}，
    返回自动生成的算法代码实现。
    """
    data = request.get_json(force=True)
    description = data.get("description", "")
    if not description:
        return jsonify({"error": "缺少 'description' 参数"}), 400
    algorithm_code = deepseek_r1_api_call(description)
    return jsonify({"algorithm_code": algorithm_code})

@app.route('/', methods=['GET'])
def index():
    """
    根路由，返回服务说明信息。
    """
    return "Deepseek-R1 代码补全与常用算法生成服务已启动，请使用 /code_completion 或 /algorithm_generation 接口调用。"

if __name__ == "__main__":
    # 启动 Flask 服务，监听 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000)


# 例【9-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_code_analysis.py
---------------------------
基于 Deepseek-R1 模型 API（模拟调用）实现深度代码分析与 Bug 检测。
接收用户提交的代码片段及问题提示，返回模型生成的分析报告和 Bug 检测结果。
"""

import time
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_analysis(prompt: str) -> str:
    """
    模拟调用 Deepseek-R1 API 进行深度代码分析与 Bug 检测。
    根据输入的提示文本判断返回不同的分析结果，延时 0.5 秒模拟网络延迟。
    参数:
        prompt: 包含代码片段和问题提示的文本
    返回:
        分析报告文本，描述代码潜在 Bug 及优化建议。
    """
    time.sleep(0.5)  # 模拟网络和计算延时
    if "bug" in prompt.lower() or "错误" in prompt:
        return ("经过深度代码分析，检测到代码中存在潜在的空指针引用错误和边界检查不足问题。"
                "建议在变量赋值前进行空值判断，并增加数组索引合法性检查，确保程序稳定运行。")
    elif "排序" in prompt or "算法" in prompt:
        return ("代码分析显示，排序算法实现中存在不稳定排序问题，"
                "建议采用归并排序或快速排序替换现有实现，以提高算法效率和稳定性。")
    else:
        return ("代码整体结构合理，但建议进一步优化变量命名和代码注释，"
                "以提升代码可读性与维护性。")

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    /analyze 接口：
    接收 JSON 请求 {"prompt": "代码片段及问题提示"}，
    返回 Deepseek-R1 模型生成的代码分析与 Bug 检测报告。
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    analysis_result = simulate_deepseek_analysis(prompt)
    return jsonify({"analysis": analysis_result})

@app.route('/', methods=['GET'])
def index():
    """
    根路由返回服务说明信息
    """
    return "Deepseek-R1 代码分析与 Bug 检测服务已启动，请使用 /analyze 接口调用。"

if __name__ == "__main__":
    # 启动 Flask 服务，监听 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000)