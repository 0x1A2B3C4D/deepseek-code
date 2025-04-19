# 例【11-1】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_generation_feedback.py
----------------------------
示例展示业务逻辑到代码生成的映射、DSL与代码模板结合使用，以及生成质量评估与模型反馈机制。
基于Deepseek-R1 API（模拟调用），接收业务逻辑描述与DSL信息，生成代码并自动评估质量，反馈优化建议。
"""

import time, json
from flask import Flask, request, jsonify

app = Flask(__name__)

def deepseek_r1_generate(logic: str, dsl: str) -> str:
    """
    模拟调用Deepseek-R1 API生成代码。
    根据输入的业务逻辑描述和DSL信息生成代码模板，
    返回生成的完整代码文本。
    """
    time.sleep(0.5)  # 模拟延时
    code = (f"def generated_function():\n"
            f"    # 实现业务逻辑：{logic}\n"
            f"    # DSL指令：{dsl}\n"
            f"    result = '执行结果'\n"
            f"    return result")
    return code

def evaluate_code_quality(code: str) -> int:
    """
    简单评估生成代码质量。
    依据代码中是否包含'def'、'return'及行数多于3行进行评分，
    返回质量得分（最高3分）。
    """
    score = 0
    if "def" in code: score += 1
    if "return" in code: score += 1
    if len(code.splitlines()) > 3: score += 1
    return score

def model_feedback(score: int) -> str:
    """
    根据生成代码的质量得分提供反馈建议。
    得分低于2分则提示代码质量较低，需重新优化逻辑描述；否则反馈生成代码质量良好。
    """
    if score < 2:
        return "生成代码质量较低，建议重新优化业务逻辑描述。"
    return "生成代码质量良好。"

@app.route('/generate_code', methods=['POST'])
def generate_code():
    """
    /generate_code 接口：
    接收JSON请求，格式为 {"logic": "业务逻辑描述", "dsl": "领域特定语言描述"}。
    调用模拟的Deepseek-R1 API生成代码，评估生成质量，并返回代码及反馈信息。
    """
    data = request.get_json(force=True)
    logic = data.get("logic", "")
    dsl = data.get("dsl", "")
    if not logic:
        return jsonify({"error": "缺少业务逻辑描述"}), 400
    generated_code = deepseek_r1_generate(logic, dsl)
    quality_score = evaluate_code_quality(generated_code)
    feedback = model_feedback(quality_score)
    response = {"generated_code": generated_code, "quality_score": quality_score, "feedback": feedback}
    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    return "业务逻辑到代码生成与质量反馈服务已启动，请使用 /generate_code 接口调用。"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# 例【11-2】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_api_doc.py
--------------------------
基于代码注释的API文档自动生成示例。
该服务接收包含代码及注释的文本，通过模拟Deepseek-R1 API生成
结构化的API文档，并以JSON格式返回生成结果。
"""

import time
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_r1_api(code_text: str) -> dict:
    """
    模拟调用Deepseek-R1 API对代码注释进行解析，
    根据代码中的注释生成API文档内容。
    参数：
        code_text：包含代码及注释的字符串。
    返回：
        一个字典，包含函数名称、参数说明、返回值描述及示例代码。
    延时0.5秒模拟实际调用延时。
    """
    time.sleep(0.5)
    # 简单模拟：提取注释关键字生成文档
    doc = {}
    if "def" in code_text:
        # 模拟提取函数名和描述
        doc["function_name"] = "generated_function"
        doc["description"] = "自动生成的函数，用于实现业务逻辑。"
        doc["parameters"] = [
            {"name": "param1", "type": "int", "description": "第一个参数"},
            {"name": "param2", "type": "int", "description": "第二个参数"}
        ]
        doc["return"] = {"type": "int", "description": "返回两个参数的和"}
        doc["example"] = "generated_function(1, 2)  # 返回 3"
    else:
        doc["error"] = "未检测到函数定义。"
    return doc

@app.route('/generate_api_doc', methods=['POST'])
def generate_api_doc():
    """
    /generate_api_doc 接口：
    接收JSON格式请求，要求包含字段"code"，其值为包含代码注释的文本。
    调用simulate_deepseek_r1_api()模拟API文档生成，返回标准JSON格式文档描述。
    """
    data = request.get_json(force=True)
    code_text = data.get("code", "")
    if not code_text:
        return jsonify({"error": "缺少 'code' 参数"}), 400
    # 调用Deepseek-R1 API模拟函数生成API文档
    api_doc = simulate_deepseek_r1_api(code_text)
    return jsonify(api_doc)

@app.route('/', methods=['GET'])
def index():
    return "基于代码注释的API文档自动生成服务已启动，请使用 /generate_api_doc 接口提交代码。"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# 例【11-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_response(prompt: str) -> str:
    """
    模拟调用Deepseek-R1 API生成响应内容，用于交互示例。
    延时0.3秒以模拟实际网络和计算延时，返回与输入提示相关的模拟结果。
    参数:
        prompt: 用户提供的输入提示文本
    返回:
        模拟生成的响应字符串
    """
    time.sleep(0.3)
    return f"模拟响应：针对 '{prompt}' 生成的示例结果。"

@app.route('/doc', methods=['GET'])
def api_doc():
    """
    /doc 接口：
    返回标准化的API文档，内容包括各接口的路径、请求方法、功能描述、参数说明及示例请求。
    文档以JSON格式输出，便于开发者快速了解接口定义及交互方式，同时支持交互测试功能。
    """
    doc = {
        "endpoints": [
            {
                "path": "/login",
                "method": "POST",
                "description": "用户登录接口，接收用户名与密码，返回API Key及有效期。",
                "parameters": {
                    "username": "字符串，用户名称",
                    "password": "字符串，用户密码（固定为 'secret'）"
                },
                "example_request": {"username": "testuser", "password": "secret"}
            },
            {
                "path": "/predict",
                "method": "POST",
                "description": "预测接口，接收 'prompt' 参数，返回Deepseek-R1模型生成的回复。",
                "parameters": {"prompt": "字符串，输入提示文本"},
                "example_request": {"prompt": "介绍深度学习的基本原理。"}
            }
        ],
        "interactive": "访问 /doc_test 接口可进行交互式示例测试。"
    }
    return jsonify(doc)

@app.route('/doc_test', methods=['POST'])
def api_doc_test():
    """
    /doc_test 接口：
    接收JSON格式的请求，其中包含 'prompt' 字段。
    调用simulate_deepseek_response()模拟Deepseek-R1 API生成交互式示例响应，
    并返回测试结果，帮助开发者验证接口调用与文档描述的一致性。
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 'prompt' 参数"}), 400
    result = simulate_deepseek_response(prompt)
    return jsonify({"test_response": result})

@app.route('/', methods=['GET'])
def index():
    """
    根路由返回服务说明信息。
    提示访问 /doc 查看标准化API文档或 /doc_test 进行交互式测试。
    """
    return "API文档与交互示例服务已启动，请访问 /doc 查看文档，或 /doc_test 进行交互测试。"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# 例【11-4】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_language_conversion.py
----------------------------
基于Deepseek-R1大模型API模拟实现跨语言代码转换的在线服务示例。
该服务提供一个API接口，接收源代码、源语言和目标语言，
返回转换后的代码文本。示例中以Python转Java为例，模拟转换过程。
"""

import time
from flask import Flask, request, jsonify

app = Flask(__name__)

def simulate_deepseek_conversion(source_code: str, source_lang: str, target_lang: str) -> str:
    """
    模拟Deepseek-R1 API进行跨语言代码转换。
    参数：
        source_code: 源代码文本
        source_lang: 源代码语言，如"Python"
        target_lang: 目标代码语言，如"Java"
    模拟过程延时0.5秒，返回转换后的代码示例文本。
    """
    time.sleep(0.5)  # 模拟网络与计算延时
    # 简单模拟转换逻辑，根据目标语言返回固定模板示例
    if source_lang.lower() == "python" and target_lang.lower() == "java":
        converted = (
            "public class ConvertedCode {\n"
            "    public static int add(int a, int b) {\n"
            "        // 原始Python代码逻辑转换为Java实现\n"
            "        return a + b;\n"
            "    }\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(add(3, 5));\n"
            "    }\n"
            "}"
        )
    else:
        converted = f"无法转换: 源语言{source_lang}到目标语言{target_lang}的转换暂未支持。"
    return converted

@app.route('/convert', methods=['POST'])
def convert():
    """
    /convert 接口：
    接收JSON格式请求，要求包含字段：
      - source_code: 源代码文本
      - source_language: 源代码语言（例如"Python"）
      - target_language: 目标代码语言（例如"Java"）
    调用simulate_deepseek_conversion()模拟代码转换，返回转换后的代码文本。
    """
    data = request.get_json(force=True)
    source_code = data.get("source_code", "")
    source_lang = data.get("source_language", "")
    target_lang = data.get("target_language", "")
    if not source_code or not source_lang or not target_lang:
        return jsonify({"error": "缺少必要参数（source_code, source_language, target_language）"}), 400
    result = simulate_deepseek_conversion(source_code, source_lang, target_lang)
    return jsonify({"converted_code": result})

@app.route('/', methods=['GET'])
def index():
    """
    根路由返回服务说明信息
    """
    return "跨语言代码转换服务已启动，请使用 /convert 接口提交转换请求。"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# 例【11-5】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

# Deepseek-R1 API端点地址
API_URL = "https://api.deepseek.com/v1/deepseek-reasoner"
# 配置有效的API密钥
API_KEY = "your_api_key_here"

def analyze_static_code(code_text: str) -> dict:
    """
    调用Deepseek-R1 API进行静态代码错误检测
    参数:
      code_text: 待检测的源代码文本（字符串）
    返回:
      JSON格式的错误检测结果字典，包含检测到的错误、警告和改进建议等信息
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    # 构造请求payload，任务类型设为"static_error_detection"
    payload = {
        "task": "static_error_detection",
        "code": code_text
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        # 如果响应成功，返回解析后的JSON数据
        if response.status_code == 200:
            return response.json()
        else:
            # 输出错误状态码和响应文本
            print("API请求失败，状态码:", response.status_code)
            print("错误信息:", response.text)
            return {}
    except Exception as e:
        print("请求异常:", str(e))
        return {}

if __name__ == "__main__":
    # 示例待检测代码（包含潜在错误，如缺少return语句等）
    sample_code = '''
def foo(x):
    return x + 1

def bar(y):
    # 这里可能缺少return，导致函数隐性返回None
    result = foo(y)
    print("Result:", result)
'''
    # 调用Deepseek-R1 API进行静态代码错误检测
    detection_result = analyze_static_code(sample_code)
    # 将检测结果以格式化JSON字符串形式输出
    if detection_result:
        print("静态代码错误检测结果:")
        print(json.dumps(detection_result, indent=2, ensure_ascii=False))
    else:
        print("未能获取检测结果。")


# 例【11-6】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time

# Deepseek-R1 API真实端点（请根据实际情况配置）
API_URL = "https://api.deepseek.com/v1/deepseek-reasoner"
# 有效的API密钥（请替换为实际密钥）
API_KEY = "your_actual_api_key"

def detect_and_repair_exception(code_snippet: str, error_log: str) -> dict:
    """
    调用Deepseek-R1 API实现运行时异常自动识别与修复。
    参数：
        code_snippet: 包含异常的代码文本
        error_log: 运行时异常日志信息
    返回：
        一个字典，包含修复后的代码和优化建议，例如：
        {"repaired_code": "修复后的代码", "suggestions": "优化建议"}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    # 构造请求负载，任务类型设为runtime_exception_repair
    payload = {
        "task": "runtime_exception_repair",
        "code": code_snippet,
        "error_log": error_log
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print("API请求失败，状态码：", response.status_code)
            print("响应内容：", response.text)
            return {}
    except Exception as e:
        print("请求异常：", str(e))
        return {}

if __name__ == "__main__":
    # 示例代码片段，假设存在除零错误
    sample_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print("Result:", result)
"""
    # 示例错误日志
    sample_error_log = "ZeroDivisionError: division by zero in function divide at line 3"
    
    print("正在检测并修复代码异常...")
    result = detect_and_repair_exception(sample_code, sample_error_log)
    
    if result:
        print("自动修复结果：")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("未能获取修复结果。")


# 例【11-7】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time

# Deepseek-R1 API真实端点（请根据官方文档确认实际地址）
API_URL = "https://api.deepseek.com/v1/deepseek-api"
# 有效的API密钥，请替换为真实的密钥
API_KEY = "your_actual_api_key_here"

def detect_security_vulnerabilities(code_text: str) -> dict:
    """
    调用Deepseek-R1 API进行代码安全漏洞自动检测。
    参数：
        code_text: 待检测的源代码文本（字符串）
    返回：
        一个字典，包含检测到的安全漏洞、错误描述和修复建议等信息。
    该函数使用真实的Deepseek-R1 API接口，不进行模拟调用。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    # 构造请求payload，任务类型为"security_vulnerability_detection"
    payload = {
        "task": "security_vulnerability_detection",
        "code": code_text
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print("API请求失败，状态码：", response.status_code)
            print("响应内容：", response.text)
            return {}
    except Exception as e:
        print("请求异常：", str(e))
        return {}

if __name__ == "__main__":
    # 示例代码：存在潜在安全漏洞，如未对输入数据进行验证，可能导致SQL注入或缓冲区溢出问题
    sample_code = """
def execute_query(query):
    # 注意：未对查询语句进行参数化处理，存在SQL注入风险
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    return results
"""
    print("正在检测代码安全漏洞...")
    detection_result = detect_security_vulnerabilities(sample_code)
    if detection_result:
        print("检测结果：")
        print(json.dumps(detection_result, indent=2, ensure_ascii=False))
    else:
      print("未能获取检测结果。")
