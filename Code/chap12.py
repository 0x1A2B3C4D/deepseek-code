# 例【12-1】
# 基于NVIDIA CUDA基础镜像，确保GPU支持
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖和Python环境
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip git wget curl && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装Python依赖
COPY requirements.txt /app/requirements.txt
RUN python3.8 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码和模型文件到镜像内
COPY . /app

# 暴露API服务端口
EXPOSE 8000

# 设置容器启动命令，运行Deepseek-R1 API服务
CMD ["python3.8", "deepseek_r1_api_service.py"]


# 例【12-2】
# Kubernetes Deployment 配置：部署Deepseek-R1模型服务
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek-r1-deployment
  labels:
    app: deepseek-r1
spec:
  replicas: 3  # 初始部署3个副本，确保高可用性
  selector:
    matchLabels:
      app: deepseek-r1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: deepseek-r1
    spec:
      containers:
      - name: deepseek-r1
        image: your_dockerhub_username/deepseek-r1:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1  # 分配1个GPU资源
          requests:
            cpu: "500m"
            memory: "1Gi"
---
# Kubernetes Service 配置：暴露Deepseek-R1模型服务
apiVersion: v1
kind: Service
metadata:
  name: deepseek-r1-service
spec:
  type: LoadBalancer  # 使用负载均衡器分发流量
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: deepseek-r1
---
# Horizontal Pod Autoscaler 配置：实现自动扩缩容
apiVersion: autoscaling/v2beta2
kind: HorizontalPodAutoscaler
metadata:
  name: deepseek-r1-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepseek-r1-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50


# 例【12-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Deepseek-R1 API配置（用于查询语义解析）
DSR1_API_URL = "https://api.deepseek.com/v1/create-chat-completion"
DSR1_API_KEY = "your_deepseek_r1_api_key"

# Deepseek-V3 API配置（用于查询扩展与补全）
DSV3_API_URL = "https://api.deepseek.com/v1/create-completion"
DSV3_API_KEY = "your_deepseek_v3_api_key"

def call_deepseek_r1(query: str) -> str:
    """
    调用Deepseek-R1 API解析搜索查询语义，返回结构化查询描述。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DSR1_API_KEY}"
    }
    payload = {
        "prompt": f"解析以下搜索查询的核心意图：{query}",
        "max_tokens": 50
    }
    response = requests.post(DSR1_API_URL, headers=headers, json=payload, timeout=10)
    if response.status_code == 200:
        data = response.json()
        # 假设返回的文本在data["choices"][0]["text"]
        return data.get("choices", [{}])[0].get("text", "").strip()
    else:
        return query  # 若失败则返回原查询

def call_deepseek_v3(expanded_query: str) -> str:
    """
    调用Deepseek-V3 API进行查询扩展与补全，返回优化后的查询文本。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DSV3_API_KEY}"
    }
    payload = {
        "prompt": f"扩展并优化以下搜索查询：{expanded_query}",
        "max_tokens": 60
    }
    response = requests.post(DSV3_API_URL, headers=headers, json=payload, timeout=10)
    if response.status_code == 200:
        data = response.json()
        return data.get("choices", [{}])[0].get("text", "").strip()
    else:
        return expanded_query

@app.route('/optimize_search', methods=['POST'])
def optimize_search():
    """
    /optimize_search接口：
    接收JSON请求 {"query": "用户原始搜索查询"}，
    先调用Deepseek-R1 API解析查询语义，再调用Deepseek-V3 API扩展查询，
    返回优化后的查询字符串。
    """
    data = request.get_json(force=True)
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "缺少'query'参数"}), 400
    # 调用Deepseek-R1进行查询解析
    parsed_query = call_deepseek_r1(query)
    # 调用Deepseek-V3进行查询扩展
    optimized_query = call_deepseek_v3(parsed_query)
    return jsonify({"optimized_query": optimized_query})

@app.route('/', methods=['GET'])
def index():
    return "自然语言处理驱动的搜索算法优化服务已启动，请使用 /optimize_search 接口调用。"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


# 例【12-4】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import requests
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.errors import KafkaError


# 全局配置

KAFKA_BROKER = "localhost:9092"  # Kafka Broker地址
INPUT_TOPIC = "deepseek_input"   # 输入主题名称
OUTPUT_TOPIC = "deepseek_output" # 输出主题名称

# Deepseek-R1 API配置（请替换为所需要的的API密钥和端点）
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/deepseek-api"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_actual_api_key_here")


# 日志配置

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("KafkaDeepseekIntegration")


# Deepseek-R1 API调用函数

def call_deepseek_api(prompt: str) -> dict:
    """
    调用Deepseek-R1 API进行推理任务。
    参数：
      prompt: 输入提示文本（字符串）
    返回：
      API返回的JSON数据，包含生成的文本或错误信息。
    采用真实HTTP请求调用Deepseek-R1 API端点，请确保API_KEY与API_URL正确配置。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "task": "text_generation",
        "prompt": prompt,
        "max_tokens": 100
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Deepseek-R1 API调用成功。")
            return response.json()
        else:
            logger.error(f"Deepseek-R1 API调用失败，状态码：{response.status_code}")
            logger.error(f"响应内容：{response.text}")
            return {"error": f"API调用失败，状态码：{response.status_code}"}
    except requests.RequestException as e:
        logger.exception("Deepseek-R1 API调用异常：")
        return {"error": str(e)}


# Kafka消费与生产类

class KafkaDeepseekProcessor:
    """
    KafkaDeepseekProcessor封装了Kafka消费者与生产者的初始化及数据处理流程。
    从输入主题中消费消息，将消息中的文本提示发送给Deepseek-R1 API，
    再将生成结果发布到输出主题中。实现高并发数据流处理和实时推理。
    """
    def __init__(self, broker: str, input_topic: str, output_topic: str):
        self.broker = broker
        self.input_topic = input_topic
        self.output_topic = output_topic
        # 初始化Kafka消费者
        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=[self.broker],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='deepseek_group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        # 初始化Kafka生产者
        self.producer = KafkaProducer(
            bootstrap_servers=[self.broker],
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )
    
    def process_messages(self):
        """
        主处理流程：从Kafka输入主题消费消息，对每条消息调用Deepseek-R1 API，
        获取推理结果后构造输出消息，发布至输出主题。
        """
        logger.info("开始处理Kafka消息...")
        for message in self.consumer:
            try:
                logger.info(f"接收到消息，offset: {message.offset}")
                data = message.value
                prompt = data.get("prompt", "")
                if not prompt:
                    logger.warning("消息中未包含'prompt'字段，跳过处理。")
                    continue
                # 调用Deepseek-R1 API进行推理
                api_result = call_deepseek_api(prompt)
                # 构造输出消息，包含原始提示、API结果和时间戳
                output_message = {
                    "original_prompt": prompt,
                    "api_result": api_result,
                    "timestamp": int(time.time())
                }
                self.producer.send(self.output_topic, output_message)
                self.producer.flush()
                logger.info(f"消息处理完毕，已发送至主题 {self.output_topic}。")
            except Exception as e:
                logger.exception("处理消息时出现异常：")
                continue


# 主函数入口

def main():
    logger.info("初始化Kafka与Deepseek-R1集成处理器...")
    processor = KafkaDeepseekProcessor(KAFKA_BROKER, INPUT_TOPIC, OUTPUT_TOPIC)
    try:
        processor.process_messages()
    except KeyboardInterrupt:
        logger.info("检测到键盘中断，正在关闭处理器...")
    finally:
        processor.consumer.close()
        processor.producer.close()
        logger.info("Kafka连接已关闭。")

if __name__ == "__main__":
    main()


# 例【12-5】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Deepseek-R1 API配置，请确保替换为有效的API密钥和真实端点
API_URL = "https://api.deepseek.com/v1/deepseek-api"
API_KEY = "your_actual_api_key_here"

def call_deepseek_api(prompt: str) -> dict:
    """
    调用Deepseek-R1 API对输入的提示文本进行推理，
    根据用户行为数据生成针对性的广告投放策略。
    参数:
      prompt: 拼接后的业务逻辑提示文本
    返回:
      Deepseek-R1 API返回的JSON响应数据。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "task": "ad_strategy_optimization",
        "prompt": prompt,
        "max_tokens": 100
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        print("API调用失败，状态码：", response.status_code)
        print("错误信息：", response.text)
        return {"error": f"API调用失败，状态码：{response.status_code}"}

@app.route('/optimize_ad', methods=['POST'])
def optimize_ad():
    """
    /optimize_ad接口：
    接收JSON请求，格式为：
    {
      "user_id": "用户ID",
      "click_history": "点击记录描述",
      "browsing_time": "浏览时长（秒）",
      "search_queries": "搜索关键词列表"
    }
    解析用户行为数据后，构造提示文本，调用Deepseek-R1 API生成广告投放策略，
    并返回生成的策略方案。
    """
    data = request.get_json(force=True)
    user_id = data.get("user_id", "")
    click_history = data.get("click_history", "")
    browsing_time = data.get("browsing_time", "")
    search_queries = data.get("search_queries", "")
    
    if not user_id or not click_history or not browsing_time or not search_queries:
        return jsonify({"error": "缺少必要的用户行为数据字段"}), 400
    
    # 构造提示文本，融合用户行为数据以生成优化策略
    prompt = (
        f"用户ID：{user_id}\n"
        f"点击记录：{click_history}\n"
        f"浏览时长：{browsing_time}秒\n"
        f"搜索关键词：{search_queries}\n"
        "请根据以上用户行为数据生成一份针对性广告投放策略，包括广告展示频次、展示位置、投放时间等优化建议。"
    )
    
    # 调用Deepseek-R1 API获取广告策略
    result = call_deepseek_api(prompt)
    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    return "广告投放策略优化服务已启动，请使用 /optimize_ad 接口提交用户行为数据。"

if __name__ == "__main__":
    # 启动Flask服务，监听0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000)


# 例【12-6】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import requests
from flask import Flask, request, jsonify


# 全局配置与日志设置

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/deepseek-api"  # Deepseek-R1 API真实端点
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_actual_api_key_here")  # 从环境变量中读取API密钥

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ABTestAdEvaluation")

app = Flask(__name__)


# Deepseek-R1 API客户端封装

class DeepseekAPIClient:
    """
    封装Deepseek-R1 API调用功能，实现业务逻辑到推理结果的映射。
    """
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def call_api(self, task: str, prompt: str, max_tokens: int = 100) -> dict:
        """
        调用Deepseek-R1 API，传入任务类型、提示文本及生成token数限制。
        返回API返回的JSON数据。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "task": task,
            "prompt": prompt,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"API调用成功，任务类型：{task}")
                return response.json()
            else:
                logger.error(f"API调用失败，状态码：{response.status_code}")
                logger.error(f"响应内容：{response.text}")
                return {"error": f"API调用失败，状态码：{response.status_code}"}
        except requests.RequestException as e:
            logger.exception("API调用异常：")
            return {"error": str(e)}

# 创建DeepseekAPIClient实例
deepseek_client = DeepseekAPIClient(DEEPSEEK_API_URL, API_KEY)


# A/B测试与广告效果评估接口

@app.route('/ab_test', methods=['POST'])
def ab_test():
    """
    /ab_test 接口：
    接收JSON请求，格式如下：
    {
      "campaign_id": "广告活动ID",
      "variant_a": {"ad_content": "广告A内容", "target_url": "http://example.com/A"},
      "variant_b": {"ad_content": "广告B内容", "target_url": "http://example.com/B"},
      "metrics": {"impressions": 10000, "clicks_a": 150, "clicks_b": 200}
    }
    根据用户提交的A/B测试数据构造提示文本，
    调用Deepseek-R1 API生成效果对比报告及优化建议，
    返回生成的报告结果。
    """
    data = request.get_json(force=True)
    campaign_id = data.get("campaign_id", "")
    variant_a = data.get("variant_a", {})
    variant_b = data.get("variant_b", {})
    metrics = data.get("metrics", {})

    if not campaign_id or not variant_a or not variant_b or not metrics:
        return jsonify({"error": "缺少必要的A/B测试数据字段"}), 400

    prompt = (
        f"广告活动ID: {campaign_id}\n"
        f"广告A内容: {variant_a.get('ad_content', '')}\n"
        f"广告A目标URL: {variant_a.get('target_url', '')}\n"
        f"广告B内容: {variant_b.get('ad_content', '')}\n"
        f"广告B目标URL: {variant_b.get('target_url', '')}\n"
        f"展示次数: {metrics.get('impressions', '')}, 点击次数A: {metrics.get('clicks_a', '')}, 点击次数B: {metrics.get('clicks_b', '')}\n"
        "请基于以上数据进行A/B测试分析，比较两种广告的效果，并提出优化建议。"
    )
    result = deepseek_client.call_api(task="ab_testing_evaluation", prompt=prompt, max_tokens=150)
    return jsonify(result)

@app.route('/ad_evaluation', methods=['POST'])
def ad_evaluation():
    """
    /ad_evaluation 接口：
    接收JSON请求，格式如下：
    {
      "ad_id": "广告ID",
      "impressions": 5000,
      "clicks": 300,
      "conversions": 50,
      "spend": 200.0
    }
    构造提示文本后，调用Deepseek-R1 API对广告效果进行实时评估，
    包括点击率、转化率、ROI等指标的计算和分析，并生成改进建议，
    返回评估报告。
    """
    data = request.get_json(force=True)
    ad_id = data.get("ad_id", "")
    impressions = data.get("impressions", 0)
    clicks = data.get("clicks", 0)
    conversions = data.get("conversions", 0)
    spend = data.get("spend", 0.0)

    if not ad_id:
        return jsonify({"error": "缺少'ad_id'参数"}), 400

    prompt = (
        f"广告ID: {ad_id}\n"
        f"展示次数: {impressions}\n"
        f"点击次数: {clicks}\n"
        f"转化次数: {conversions}\n"
        f"广告支出: {spend}元\n"
        "请根据以上数据计算广告点击率、转化率和ROI，并给出优化建议。"
    )
    result = deepseek_client.call_api(task="ad_effect_evaluation", prompt=prompt, max_tokens=150)
    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    """
    根路由返回服务说明信息
    """
    return "A/B测试与广告效果实时评估服务已启动，请使用 /ab_test 或 /ad_evaluation 接口调用。"

def additional_logging():
    """
    此函数用于演示额外日志记录功能，可扩展系统监控和调试功能。
    """
    logger.info("额外日志记录：系统运行状态正常。")
    logger.info("正在监控API调用性能和响应时间。")
    time.sleep(0.1)
    logger.info("日志记录完毕。")

for i in range(5):
    additional_logging()

# 主函数入口

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)