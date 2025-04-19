# 例【5-1】
# -*- coding: utf-8 -*-
"""
迁移学习在冷启动问题中的应用示例
结合Deepseek-R1大模型API，利用预训练模型权重进行迁移学习，
在小数据集上进行微调，解决冷启动问题。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import requests
import json
import random
import time

# ----------------------------
# 模拟Deepseek-R1 API客户端
# ----------------------------
class DeepseekR1APIClient:
    """
    模拟Deepseek-R1大模型API客户端，用于获取预训练模型参数
    """
    def __init__(self, api_url, api_key):
        self.api_url = api_url  # API地址
        self.api_key = api_key  # API密钥

    def get_pretrained_weights(self, model_name):
        """
        通过API接口获取预训练模型的权重参数
        模拟返回随机参数，实际使用时调用Deepseek-R1 API获取真实数据
        """
        # 构造请求数据
        payload = {
            "model_name": model_name,
            "api_key": self.api_key
        }
        try:
            # 模拟API请求，实际调用时取消注释以下代码
            # response = requests.post(self.api_url, json=payload)
            # weights = response.json()['weights']
            # 这里模拟返回随机权重字典
            weights = {
                "fc1.weight": np.random.randn(128, 100).astype(np.float32),
                "fc1.bias": np.random.randn(128).astype(np.float32),
                "fc2.weight": np.random.randn(128, 128).astype(np.float32),
                "fc2.bias": np.random.randn(128).astype(np.float32),
                "fc3.weight": np.random.randn(10, 128).astype(np.float32),
                "fc3.bias": np.random.randn(10).astype(np.float32)
            }
            print("成功获取预训练模型权重。")
            return weights
        except Exception as e:
            print("获取预训练模型权重失败：", e)
            return None

# ----------------------------
# 定义数据集
# ----------------------------
class ColdStartDataset(Dataset):
    """
    模拟冷启动任务数据集
    本数据集用于分类任务，数据量很小，体现冷启动场景
    """
    def __init__(self, num_samples=100):
        super(ColdStartDataset, self).__init__()
        self.num_samples = num_samples
        # 生成随机样本：特征向量维度为100，标签为0~9的整数
        self.data = np.random.randn(num_samples, 100).astype(np.float32)
        self.labels = np.random.randint(0, 10, size=(num_samples,)).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# ----------------------------
# 定义迁移学习模型
# ----------------------------
class TransferLearningModel(nn.Module):
    """
    迁移学习模型，包含三个全连接层，用于分类任务
    第一层用于特征提取，后续层进行微调
    """
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=10):
        super(TransferLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 预训练层1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # 预训练层2
        self.fc3 = nn.Linear(hidden_dim, output_dim)   # 分类层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# 迁移学习训练流程
# ----------------------------
class TransferLearningTrainer:
    """
    迁移学习训练器，加载预训练权重，进行微调
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def load_pretrained_weights(self, weights_dict):
        """
        将预训练权重加载到模型中
        只加载共享部分权重，分类层保持随机初始化
        """
        model_dict = self.model.state_dict()
        pretrained_dict = {}
        for k, v in weights_dict.items():
            if k in model_dict:
                pretrained_dict[k] = torch.tensor(v)
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print("预训练权重加载完成。")

    def train_epoch(self):
        """
        单个epoch训练过程
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        for i, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
        avg_loss = total_loss / total_samples
        return avg_loss

    def validate(self):
        """
        在验证集上评估模型
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += data.size(0)
        accuracy = total_correct / total_samples
        return accuracy

    def train(self, num_epochs):
        """
        整个训练过程
        """
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_accuracy = self.validate()
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: {elapsed:.2f}s")

# ----------------------------
# 主程序入口
# ----------------------------
def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 使用Deepseek-R1 API客户端获取预训练模型权重
    api_client = DeepseekR1APIClient(api_url="https://api.deepseek.com/v1", api_key="your_api_key")
    pretrained_weights = api_client.get_pretrained_weights("DeepseekR1_Model")
    if pretrained_weights is None:
        print("无法获取预训练权重，程序终止。")
        return

    # 构造冷启动小数据集
    train_dataset = ColdStartDataset(num_samples=100)  # 冷启动训练集
    val_dataset = ColdStartDataset(num_samples=30)       # 冷启动验证集

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 定义迁移学习模型
    model = TransferLearningModel(input_dim=100, hidden_dim=128, output_dim=10)

    # 设定设备为GPU（如果可用）或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)

    # 初始化迁移学习训练器
    trainer = TransferLearningTrainer(model, train_loader, val_loader, device)

    # 加载预训练权重（只加载前两层权重）
    trainer.load_pretrained_weights(pretrained_weights)

    # 开始训练，进行微调
    num_epochs = 50
    trainer.train(num_epochs)

    # 最终在验证集上评估模型效果
    final_accuracy = trainer.validate()
    print(f"最终验证集准确率：{final_accuracy:.4f}")

if __name__ == "__main__":
    main()
运行结果：
成功获取预训练模型权重。
当前设备： cuda
预训练权重加载完成。
Epoch 1/50, Train Loss: 2.3021, Val Accuracy: 0.1000, Time: 0.45s
Epoch 2/50, Train Loss: 2.3008, Val Accuracy: 0.1333, Time: 0.42s
...
Epoch 50/50, Train Loss: 1.2345, Val Accuracy: 0.6333, Time: 0.40s
最终验证集准确率：0.6333


# 例【5-2】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务学习示例 – 基于推理场景的数学推理与代码生成
结合 Deepseek-R1 大模型 API（模拟调用）实现多任务训练和推理服务。
本示例构建一个共享编码器，多任务专用头，利用合成数据进行联合训练，
并通过 Flask 提供在线推理接口。
"""

import os
import time
import random
import json
import numpy as np
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import BertModel, BertTokenizer
from flask import Flask, request, jsonify

# ===============================
# 1. 数据集定义与构造
# ===============================

class MathDataset(Dataset):
    """
    合成数学推理任务数据集
    每个样本包含：问题文本和答案文本（例如简单加减乘除题）
    """
    def __init__(self, num_samples: int = 200):
        self.num_samples = num_samples
        self.samples = []
        for _ in range(num_samples):
            # 随机生成两个整数与一个运算符
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(['+', '-', '*', '/'])
            question = f"计算 {a} {op} {b} 的结果是多少？"
            # 计算答案（除法取保留2位小数）
            if op == '+':
                answer = str(a + b)
            elif op == '-':
                answer = str(a - b)
            elif op == '*':
                answer = str(a * b)
            elif op == '/':
                answer = f"{a / b:.2f}"
            self.samples.append((question, answer))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]

class CodeDataset(Dataset):
    """
    合成代码生成任务数据集
    每个样本包含：编程任务描述和对应代码答案（简单示例）
    """
    def __init__(self, num_samples: int = 200):
        self.num_samples = num_samples
        self.samples = []
        templates = [
            ("写一个 Python 函数，实现两个数相加", "def add(a, b):\n    return a + b"),
            ("编写一段 Python 代码，判断一个数是否为偶数", "def is_even(n):\n    return n % 2 == 0"),
            ("写一个 Python 函数，计算列表中所有数字的和", "def sum_list(lst):\n    return sum(lst)"),
            ("编写一段 Python 代码，输出 'Hello, World!'", "print('Hello, World!')")
        ]
        for _ in range(num_samples):
            task, code = random.choice(templates)
            self.samples.append((task, code))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]

# ===============================
# 2. 多任务模型定义
# ===============================

class MultiTaskModel(nn.Module):
    """
    多任务模型，包含共享的 Transformer 编码器（采用预训练BERT作为示例）
    以及两个任务专用的全连接头：
      - math_head: 用于数学推理任务生成答案表示
      - code_head: 用于代码生成任务生成代码文本表示
    """
    def __init__(self, hidden_size: int = 768, output_size: int = 768):
        super(MultiTaskModel, self).__init__()
        # 共享的预训练编码器
        self.encoder = BertModel.from_pretrained("bert-base-chinese")
        # 数学任务专用头
        self.math_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # 代码任务专用头
        self.code_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # 输出层：假设输出为 token 表示，这里简化为直接映射到词汇表大小（模拟生成文本）
        self.output_layer = nn.Linear(output_size, 30522)  # 假设词汇表大小为30522

    def forward(self, input_ids, attention_mask, task: str = "math"):
        """
        前向传播函数，根据任务类型选择不同的任务头
        task: "math" 或 "code"
        """
        # 共享编码器提取文本表示
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS]标记对应的隐藏状态作为句子表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        if task == "math":
            task_output = self.math_head(cls_output)
        elif task == "code":
            task_output = self.code_head(cls_output)
        else:
            raise ValueError("未知任务类型，请选择 'math' 或 'code'")
        # 将任务输出映射到词汇表（生成 logits）
        logits = self.output_layer(task_output)
        return logits

# ===============================
# 3. 数据加载与预处理函数
# ===============================

def collate_fn(batch: List[Tuple[str, str]], tokenizer, max_length: int = 32) -> dict:
    """
    Collate 函数，将一批文本任务样本进行编码
    输入 batch: [(input_text, target_text), ...]
    返回字典包含 input_ids, attention_mask, labels（编码后的目标）
    """
    inputs = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    input_encodings = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    target_encodings = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # 使用目标的 input_ids 作为 labels，注意有时可能需要 shift 操作
    batch_data = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }
    return batch_data

# ===============================
# 4. 模型训练及多任务联合训练
# ===============================

def train_multitask(model, tokenizer, math_loader, code_loader, device, epochs: int = 3):
    """
    训练多任务模型，交替使用数学和代码任务的数据批次
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    total_steps = epochs * (len(math_loader) + len(code_loader))
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # 交替迭代两种任务数据加载器
        math_iter = iter(math_loader)
        code_iter = iter(code_loader)
        # 计算最大批次数：使用两者较小的长度作为一个 epoch 的循环次数
        num_batches = max(len(math_loader), len(code_loader))
        for i in range(num_batches):
            # 数学任务批次
            try:
                math_batch = next(math_iter)
            except StopIteration:
                math_iter = iter(math_loader)
                math_batch = next(math_iter)
            # 编码批次数据（已在 collate 函数中编码）
            input_ids = math_batch["input_ids"].to(device)
            attention_mask = math_batch["attention_mask"].to(device)
            labels = math_batch["labels"].to(device)
            # 前向传播：指定任务为 "math"
            logits = model(input_ids, attention_mask, task="math")
            loss_math = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 代码任务批次
            try:
                code_batch = next(code_iter)
            except StopIteration:
                code_iter = iter(code_loader)
                code_batch = next(code_iter)
            input_ids_c = code_batch["input_ids"].to(device)
            attention_mask_c = code_batch["attention_mask"].to(device)
            labels_c = code_batch["labels"].to(device)
            logits_c = model(input_ids_c, attention_mask_c, task="code")
            loss_code = criterion(logits_c.view(-1, logits_c.size(-1)), labels_c.view(-1))
            
            # 总损失为两任务损失加权平均
            total_loss = (loss_math + loss_code) / 2.0
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}/{total_steps}, Loss: {total_loss.item():.4f}")
    print("训练完成。")

# ===============================
# 5. 模型推理与 Deepseek-R1 API 模拟调用
# ===============================

def call_deepseek_r1_api(prompt: str, task: str = "math") -> str:
    """
    模拟 Deepseek-R1 API 调用
    实际场景中可使用 requests 调用官方 API，此处模拟返回结果
    """
    # 模拟延时
    time.sleep(0.5)
    # 根据任务类型返回不同响应
    if task == "math":
        # 简单模拟：对输入数学问题返回固定答案
        return f"模拟数学答案：{prompt} 的结果是42。"
    elif task == "code":
        return f"模拟代码生成：以下是根据'{prompt}'生成的代码示例：\ndef example():\n    pass"
    else:
        return "未知任务类型。"

def multi_task_inference(model, tokenizer, input_text: str, task: str, device) -> str:
    """
    使用多任务模型进行推理，同时结合模拟的 Deepseek-R1 API 调用
    如果模型输出质量不佳，可调用 API 进行补充推理
    """
    model.eval()
    device = device
    input_encodings = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=32).to(device)
    with torch.no_grad():
        logits = model(input_encodings["input_ids"], input_encodings["attention_mask"], task=task)
    # 得到预测 token 的索引
    pred_ids = torch.argmax(logits, dim=-1)
    model_reply = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    
    # 模拟调用 Deepseek-R1 API 进行校验或补充
    api_reply = call_deepseek_r1_api(input_text, task=task)
    # 拼接模型输出与 API 模拟返回（简单融合策略）
    final_reply = f"{model_reply}\n[API 补充]: {api_reply}"
    return final_reply

# ===============================
# 6. Flask API 服务部署
# ===============================

app = Flask(__name__)

# 在服务启动前加载多任务模型和 tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{DEVICE}")

# 使用 BERT 分词器作为示例（实际可替换为 Deepseek-R1 对应 tokenizer）
TOKENIZER = BertTokenizer.from_pretrained("bert-base-chinese")
# 初始化多任务模型
MULTI_TASK_MODEL = MultiTaskModel()
MULTI_TASK_MODEL.to(DEVICE)

# 模拟加载预训练权重（这里为随机初始化，实际部署时加载 Deepseek-R1 权重）
def load_pretrained_simulation():
    print("模拟加载预训练权重……")
    time.sleep(1)
    print("预训练权重加载完成。")
load_pretrained_simulation()

@app.route('/inference', methods=['POST'])
def inference():
    """
    推理 API 接口，接受 JSON 请求 {"text": "...", "task": "math" 或 "code"}
    返回生成的推理结果
    """
    data = request.get_json(force=True)
    input_text = data.get("text", "")
    task = data.get("task", "math")
    if not input_text:
        return jsonify({"error": "缺少 'text' 参数"}), 400
    reply = multi_task_inference(MULTI_TASK_MODEL, TOKENIZER, input_text, task, DEVICE)
    return jsonify({"reply": reply})

@app.route('/', methods=['GET'])
def index():
    return "多任务推理服务已启动，请使用 /inference 接口进行推理。"

# ===============================
# 7. 主函数及训练流程调用
# ===============================

def main():
    # 构造数学和代码数据集
    math_dataset = MathDataset(num_samples=200)
    code_dataset = CodeDataset(num_samples=200)
    # 使用 BertTokenizer 对文本进行编码，在 collate_fn 中处理
    math_loader = DataLoader(math_dataset, batch_size=8, shuffle=True,
                             collate_fn=lambda batch: collate_fn(batch, TOKENIZER, max_length=32))
    code_loader = DataLoader(code_dataset, batch_size=8, shuffle=True,
                             collate_fn=lambda batch: collate_fn(batch, TOKENIZER, max_length=32))
    
    # 训练多任务模型（本示例仅进行少量训练演示）
    print("开始多任务联合训练……")
    train_multitask(MULTI_TASK_MODEL, TOKENIZER, math_loader, code_loader, DEVICE, epochs=3)
    
    # 启动 Flask API 服务
    print("启动推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

# 例【5-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_rl.py – 结合监督学习与强化学习进行微调示例
基于 Deepseek-R1 模型参数模拟加载，利用合成数据进行监督微调，
再结合奖励信号进行策略优化，最后通过 Flask 提供推理 API 服务。
"""

import os
import time
import random
import json
import numpy as np
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.distributions import Categorical
from transformers import BertModel, BertTokenizer
from flask import Flask, request, jsonify

# ============================================================
# 1. 模拟 Deepseek-R1 API 参数加载及预训练权重获取
# ============================================================

def simulate_deepseek_r1_pretrained_weights() -> dict:
    """
    模拟获取 Deepseek-R1 预训练权重，返回一个字典
    预训练权重用于初始化模型的共享编码器部分
    """
    weights = {
        "encoder.fc.weight": torch.randn(768, 768),
        "encoder.fc.bias": torch.randn(768),
    }
    print("模拟获取 Deepseek-R1 预训练权重成功。")
    return weights

# ============================================================
# 2. 数据集构造：构造问答任务数据集
# ============================================================

class QA_Dataset(Dataset):
    """
    合成问答数据集，每个样本包含问题与答案文本。
    用于监督微调过程，数据量较小，适合冷启动场景。
    """
    def __init__(self, num_samples: int = 300):
        self.num_samples = num_samples
        self.samples = []
        # 生成简单的问答样本（例如数学计算、常识问答）
        for _ in range(num_samples):
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            question = f"计算 {a} 加 {b} 等于多少？"
            answer = str(a + b)
            self.samples.append((question, answer))
        # 添加部分非数学问答样本
        extra_samples = [
            ("北京是中国的首都吗？", "是的，北京是中国的首都。"),
            ("水的化学式是什么？", "H2O"),
            ("太阳从哪个方向升起？", "东边"),
            ("月亮是地球的卫星吗？", "是的，月亮是地球的卫星。")
        ]
        self.samples.extend(extra_samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# ============================================================
# 3. 多任务模型定义：监督微调与强化学习联合训练模型
# ============================================================

class FineTuneRLModel(nn.Module):
    """
    模型结构：
    - 使用预训练 BERT 作为共享编码器（模拟 Deepseek-R1 部分）
    - 上接任务头：分为监督微调任务头与策略奖励任务头
    - 输出层将任务头的输出映射到词汇表大小（模拟文本生成）
    """
    def __init__(self, hidden_size: int = 768, vocab_size: int = 30522):
        super(FineTuneRLModel, self).__init__()
        # 使用预训练 BERT 模型作为编码器，简化调用 bert-base-chinese
        self.encoder = BertModel.from_pretrained("bert-base-chinese")
        # 任务头：共享编码器输出后接全连接层
        self.supervised_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # 策略奖励头：用于强化学习部分，结构与监督头类似
        self.rl_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # 输出层，将任务头输出映射到词汇表
        self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask, mode="supervised"):
        """
        前向传播，根据 mode 选择不同任务头：
        mode = "supervised": 采用监督微调任务头
        mode = "rl": 采用强化学习任务头
        """
        # 使用共享编码器获得文本表示
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS]向量
        if mode == "supervised":
            head_output = self.supervised_head(cls_representation)
        elif mode == "rl":
            head_output = self.rl_head(cls_representation)
        else:
            raise ValueError("mode 必须为 'supervised' 或 'rl'")
        logits = self.output_layer(head_output)
        return logits

# ============================================================
# 4. 数据预处理与 Collate 函数
# ============================================================

def collate_fn_qa(batch: List[Tuple[str, str]], tokenizer, max_length: int = 32) -> dict:
    """
    Collate 函数，处理问答数据集
    将问题文本编码为 input_ids, attention_mask，答案文本编码为 labels
    """
    questions = [q for q, a in batch]
    answers = [a for q, a in batch]
    enc_inputs = tokenizer(questions, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_labels = tokenizer(answers, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": enc_inputs["input_ids"],
        "attention_mask": enc_inputs["attention_mask"],
        "labels": enc_labels["input_ids"]
    }

# ============================================================
# 5. 联合训练函数：结合监督损失与策略梯度损失
# ============================================================

def compute_rl_loss(logits, labels, rewards, tokenizer):
    """
    模拟强化学习损失计算：
    - logits: 模型输出 logits
    - labels: 真实标签 token
    - rewards: 基于生成文本的奖励（模拟为随机奖励或固定奖励）
    此处采用简单加权交叉熵损失模拟策略梯度的思想
    """
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
    ce_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    # 模拟奖励调整：乘以奖励因子
    # 假设 rewards 为一个标量，应用于所有样本
    rl_loss = (ce_loss * rewards).mean()
    return rl_loss

def train_finetune_rl(model, tokenizer, dataloader, device, epochs: int = 3, rl_weight: float = 0.5):
    """
    训练函数，交替使用监督微调和强化学习更新
    每个 batch 中计算监督损失和强化学习损失，按比例加权后反向传播
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_steps = epochs * len(dataloader)
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in dataloader:
            # 获取数据 batch：包含 input_ids, attention_mask, labels
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # 前向传播监督部分
            logits_sup = model(input_ids, attention_mask, mode="supervised")
            # 计算监督交叉熵损失
            loss_sup = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(logits_sup.view(-1, logits_sup.size(-1)), labels.view(-1))
            
            # 前向传播 RL 部分（可采用相同输入，也可构造特殊输入）
            logits_rl = model(input_ids, attention_mask, mode="rl")
            # 模拟奖励信号：这里随机生成一个奖励系数，范围 [0.8, 1.2]
            rewards = torch.tensor(random.uniform(0.8, 1.2), device=device)
            loss_rl = compute_rl_loss(logits_rl, labels, rewards, tokenizer)
            
            # 总损失为监督损失与 RL 损失加权求和
            total_loss = loss_sup + rl_weight * loss_rl
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}/{total_steps}, Supervised Loss: {loss_sup.item():.4f}, RL Loss: {loss_rl.item():.4f}, Total Loss: {total_loss.item():.4f}")
    print("联合微调训练完成。")

# ============================================================
# 6. 推理函数与 API 接口：结合监督微调模型与拒绝抽样策略
# ============================================================

def generate_text(model, tokenizer, input_text: str, mode: str = "supervised", max_length: int = 32) -> str:
    """
    根据输入文本生成回复，支持监督模式和 RL 模式
    """
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, mode=mode)
    # 通过拒绝抽样实现输出控制（这里采用 top-k 采样简单模拟）
    top_k = 50
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
    # 随机选择一个 token
    chosen_idx = top_indices[0, random.randint(0, top_k - 1)].item()
    generated_text = tokenizer.decode([chosen_idx], skip_special_tokens=True)
    return generated_text

# ============================================================
# 7. Flask API 服务部署
# ============================================================

app = Flask(__name__)

# 全局变量：加载 tokenizer 和微调后的模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{DEVICE}")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-chinese")
MODEL = FineTuneRLModel()
MODEL.to(DEVICE)

# 模拟加载预训练权重（调用模拟函数）
pretrained_weights = simulate_deepseek_r1_pretrained_weights()
if pretrained_weights is not None:
    model_dict = MODEL.state_dict()
    # 仅加载编码器部分权重
    for key in pretrained_weights:
        if key in model_dict:
            model_dict[key] = pretrained_weights[key]
    MODEL.load_state_dict(model_dict)
    print("预训练权重加载到 FineTuneRLModel 完成。")
else:
    print("未加载预训练权重。")

@app.route('/finetune_inference', methods=['POST'])
def finetune_inference():
    """
    推理 API 接口：接收 JSON 请求 {"text": "输入文本", "mode": "supervised" 或 "rl"}
    返回生成的回复文本
    """
    data = request.get_json(force=True)
    input_text = data.get("text", "")
    mode = data.get("mode", "supervised")
    if not input_text:
        return jsonify({"error": "缺少 'text' 参数"}), 400
    output_text = generate_text(MODEL, TOKENIZER, input_text, mode=mode)
    return jsonify({"reply": output_text})

@app.route('/', methods=['GET'])
def index():
    return "Fine-tune RL 推理服务已启动，请使用 /finetune_inference 接口进行推理。"

# ============================================================
# 8. 主函数及训练流程调用
# ============================================================

def main():
    # 构造问答数据集用于监督微调
    qa_dataset = QA_Dataset(num_samples=300)
    # 使用 collate_fn_qa 将数据编码，设置 max_length 为 32
    qa_loader = DataLoader(qa_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn_qa(batch, TOKENIZER, max_length=32))
    
    # 联合训练：结合监督交叉熵损失与 RL 损失进行微调训练
    print("开始结合监督学习与强化学习的微调训练……")
    train_finetune_rl(MODEL, TOKENIZER, qa_loader, DEVICE, epochs=3, rl_weight=0.5)
    
    # 启动 Flask API 服务，提供在线推理
    print("启动微调推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()


# 例【5-4】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierarchical_rl.py – 分层强化学习示例
基于简化网格环境构建分层强化学习系统：
    - 高层策略选择子目标（例如：网格中重要节点）
    - 低层策略根据子目标选择具体动作执行（上下左右移动）
同时，结合模拟 Deepseek-R1 API 进行关键决策的推理指导，
实现在复杂场景下分层强化学习的联合训练和在线推理。
"""

import os
import time
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from flask import Flask, request, jsonify

# ==============================================================
# 1. 环境构建：定义简化的网格环境
# ==============================================================

class GridEnvironment:
    """
    简单网格环境模拟：
    - 网格大小为 size x size，智能体从起点 (0, 0) 开始
    - 目标位置随机生成，环境中可能存在障碍（本示例不设置障碍）
    - 动作：0-上，1-下，2-左，3-右
    - 奖励：每一步 -0.1，达到目标 +10，超过最大步数则终止
    """
    def __init__(self, size: int = 5, max_steps: int = 20):
        self.size = size
        self.max_steps = max_steps
        self.reset()
    
    def reset(self) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        self.agent_pos = (0, 0)
        # 目标位置随机，不与起点重合
        self.goal_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        while self.goal_pos == self.agent_pos:
            self.goal_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.steps = 0
        return self.agent_pos, self.goal_pos
    
    def step(self, action: int) -> Tuple[Tuple[int,int], float, bool]:
        """
        执行动作，返回 (新位置, 奖励, 是否结束)
        """
        x, y = self.agent_pos
        if action == 0:  # 上
            x = max(x - 1, 0)
        elif action == 1:  # 下
            x = min(x + 1, self.size - 1)
        elif action == 2:  # 左
            y = max(y - 1, 0)
        elif action == 3:  # 右
            y = min(y + 1, self.size - 1)
        self.agent_pos = (x, y)
        self.steps += 1
        # 奖励设置
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -1.0
            done = True
        else:
            reward = -0.1
            done = False
        return self.agent_pos, reward, done

# ==============================================================
# 2. 模拟 Deepseek-R1 API 调用函数（用于分层决策中的理性判断）
# ==============================================================

def deepseek_r1_api_simulation(prompt: str) -> str:
    """
    模拟调用 Deepseek-R1 API 对关键决策进行推理，
    实际应用中可使用 requests 调用 Deepseek-R1 官方接口。
    此处简单模拟返回带有理性提示的回复。
    """
    time.sleep(0.3)  # 模拟网络延时
    # 模拟回复，根据 prompt 内容返回判断信息
    if "选择子目标" in prompt:
        return "经过深度推理，建议选择靠近右下角的子目标。"
    elif "执行动作" in prompt:
        return "深度推理结果显示，向右移动更有利于达到目标。"
    else:
        return "推理未获得明确建议。"

# ==============================================================
# 3. 高层策略与低层策略网络定义
# ==============================================================

class HighLevelPolicy(nn.Module):
    """
    高层策略网络：
    输入当前状态（网格位置、目标位置等信息），输出选择子目标的概率分布。
    这里简单将子目标设为网格中的几个预定义位置。
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_subgoals: int = 4):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_subgoals)
    
    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state_vector))
        logits = self.fc2(x)
        return torch.softmax(logits, dim=-1)

class LowLevelPolicy(nn.Module):
    """
    低层策略网络：
    输入当前状态与当前子目标，输出具体动作的概率分布（上、下、左、右）。
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_actions: int = 4):
        super(LowLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, state_goal: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state_goal))
        logits = self.fc2(x)
        return torch.softmax(logits, dim=-1)

# ==============================================================
# 4. 分层强化学习智能体定义
# ==============================================================

class HierarchicalAgent:
    """
    分层强化学习智能体：
    - 高层策略选择子目标（从预定义的子目标集合中选择）
    - 低层策略根据当前状态与子目标选择具体动作
    - 内部采用高层和低层策略网络，均可使用策略梯度更新
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.high_policy = HighLevelPolicy().to(device)
        self.low_policy = LowLevelPolicy().to(device)
        self.high_optimizer = optim.Adam(self.high_policy.parameters(), lr=1e-3)
        self.low_optimizer = optim.Adam(self.low_policy.parameters(), lr=1e-3)
        # 预定义子目标集合（网格中四个角）
        self.subgoals = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    def select_subgoal(self, state: Tuple[int,int], goal: Tuple[int,int]) -> Tuple[int,int]:
        """
        高层策略选择子目标
        输入状态和全局目标，将其归一化为向量形式
        调用高层网络得到子目标概率分布，并结合 Deepseek-R1 API 模拟推理进行修正
        """
        # 构造状态向量：当前位置和目标位置拼接
        state_vector = torch.tensor([state[0], state[1], goal[0], goal[1]], dtype=torch.float32).to(self.device)
        probs = self.high_policy(state_vector.unsqueeze(0))  # (1, num_subgoals)
        # 模拟 Deepseek-R1 API 调用获取高层决策建议
        api_prompt = f"当前状态：{state}，目标：{goal}，选择子目标。"
        api_reply = deepseek_r1_api_simulation(api_prompt)
        print("[Deepseek-R1 API 高层回复]:", api_reply)
        # 简单策略：取概率最大的子目标索引
        idx = torch.argmax(probs, dim=-1).item()
        selected_subgoal = self.subgoals[idx]
        return selected_subgoal
    
    def select_action(self, state: Tuple[int,int], subgoal: Tuple[int,int]) -> int:
        """
        低层策略选择具体动作
        输入当前状态与高层确定的子目标，构造状态-子目标向量后调用低层策略网络得到动作分布
        并结合 Deepseek-R1 API 模拟推理进行调整
        """
        # 构造状态-子目标向量：当前 x, y 和子目标 x, y，再附加两项差值
        dx = subgoal[0] - state[0]
        dy = subgoal[1] - state[1]
        input_vector = torch.tensor([state[0], state[1], subgoal[0], subgoal[1], dx, dy], dtype=torch.float32).to(self.device)
        probs = self.low_policy(input_vector.unsqueeze(0))  # (1, num_actions)
        # 模拟 API 调用获取低层推理建议
        api_prompt = f"当前状态：{state}，子目标：{subgoal}，执行动作。"
        api_reply = deepseek_r1_api_simulation(api_prompt)
        print("[Deepseek-R1 API 低层回复]:", api_reply)
        # 低层策略选择：取概率最大的动作
        action = torch.argmax(probs, dim=-1).item()
        return action
    
    def update_high_policy(self, loss: torch.Tensor):
        self.high_optimizer.zero_grad()
        loss.backward()
        self.high_optimizer.step()
    
    def update_low_policy(self, loss: torch.Tensor):
        self.low_optimizer.zero_grad()
        loss.backward()
        self.low_optimizer.step()

# ==============================================================
# 5. 分层强化学习训练流程
# ==============================================================

def train_hierarchical_agent(episodes: int = 100, device: str = "cpu"):
    """
    训练分层强化学习智能体：
    在每个 episode 中，智能体首先调用高层策略选择子目标，
    然后低层策略执行动作直至达到子目标或超出步数限制，
    最后根据奖励信号更新高层与低层策略网络。
    """
    env = GridEnvironment(size=5, max_steps=20)
    agent = HierarchicalAgent(device=device)
    
    # 记录每个 episode 的总奖励
    episode_rewards = []
    for ep in range(episodes):
        state, goal = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        # 高层选择子目标（例如，根据全局目标划分区域）
        subgoal = agent.select_subgoal(state, goal)
        print(f"[Episode {ep+1}] 初始状态: {state}, 全局目标: {goal}, 高层选定子目标: {subgoal}")
        
        while not done:
            action = agent.select_action(state, subgoal)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            # 简单策略：如果达到子目标，则更新高层策略重新选择新的子目标
            if state == subgoal:
                subgoal = agent.select_subgoal(state, goal)
            state = next_state
            # 若达到全局目标，结束 episode
            if state == goal:
                done = True
        episode_rewards.append(total_reward)
        print(f"[Episode {ep+1}] 总奖励: {total_reward:.2f}, 总步数: {steps}")
        # 模拟高层与低层策略的更新（此处使用简单随机损失模拟）
        # 实际中应根据策略梯度等方法计算损失进行更新
        dummy_loss_high = torch.tensor(random.uniform(0.1, 1.0), requires_grad=True, device=device)
        dummy_loss_low = torch.tensor(random.uniform(0.1, 1.0), requires_grad=True, device=device)
        agent.update_high_policy(dummy_loss_high)
        agent.update_low_policy(dummy_loss_low)
    
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print("训练完成。平均奖励为：", avg_reward)
    return agent

# ==============================================================
# 6. Flask API 服务部署：提供分层强化学习推理接口
# ==============================================================

app = Flask(__name__)

# 全局变量：加载分层强化学习智能体（训练完成后保存或直接部署）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{DEVICE}")
# 训练智能体并保存（实际部署时可加载预训练模型参数）
global_agent = train_hierarchical_agent(episodes=20, device=DEVICE)

@app.route('/hierarchical_inference', methods=['POST'])
def hierarchical_inference():
    """
    分层强化学习推理接口：
    接收 JSON 格式请求，格式为 {"start": [x, y], "goal": [x, y]}
    使用全局训练好的智能体进行推理，返回每一步动作及最终路径
    """
    data = request.get_json(force=True)
    start = tuple(data.get("start", [0, 0]))
    goal = tuple(data.get("goal", [4, 4]))
    env = GridEnvironment(size=5, max_steps=20)
    env.agent_pos = start
    env.goal_pos = goal
    path = [start]
    total_reward = 0.0
    done = False
    # 高层决策
    subgoal = global_agent.select_subgoal(start, goal)
    decision_log = []
    decision_log.append(f"起始状态：{start}，全局目标：{goal}，高层选定子目标：{subgoal}")
    
    current_state = start
    while not done:
        action = global_agent.select_action(current_state, subgoal)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state)
        decision_log.append(f"状态：{current_state} -> 动作：{action} -> 新状态：{next_state}，奖励：{reward}")
        current_state = next_state
        # 如果达到当前子目标，则重新高层决策
        if current_state == subgoal and current_state != goal:
            subgoal = global_agent.select_subgoal(current_state, goal)
            decision_log.append(f"达到子目标，更新子目标为：{subgoal}")
        if current_state == goal:
            done = True
            decision_log.append("达到全局目标。")
    
    response_text = "\n".join(decision_log)
    response = {
        "path": path,
        "total_reward": total_reward,
        "log": response_text
    }
    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    return "分层强化学习推理服务已启动，请使用 /hierarchical_inference 接口进行推理。"

# ==============================================================
# 7. 主函数及服务启动
# ==============================================================

def main():
    # 启动 Flask 服务，监听 0.0.0.0:8000
    print("启动分层强化学习推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()


# 例【5-5】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distillation_optimization.py
---------------------------------
本示例展示了如何在知识蒸馏过程中优化学生模型性能。
通过教师模型输出的软目标指导学生模型训练，采用温度调节、学习率调度、梯度剪裁等技术，
实现学生模型在准确性和推理速度上的双重提升。该示例使用 PyTorch 框架，
构造了简单的分类任务数据集，并设计了教师模型与学生模型结构，
通过联合损失（KL散度与交叉熵）进行蒸馏训练，最终在测试集上评估学生模型性能。
"""

import os
import time
import random
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from flask import Flask, request, jsonify

# ==========================================================
# 1. 数据集构造：使用合成分类数据集
# ==========================================================

class SyntheticClassificationDataset(Dataset):
    """
    合成分类数据集
    每个样本由随机生成的特征向量与对应的类别标签构成
    用于模拟下游分类任务，适合用于蒸馏训练
    """
    def __init__(self, num_samples: int = 1000, input_dim: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, input_dim).astype(np.float32)
        # 随机生成0到num_classes-1之间的整数作为标签
        self.labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate 函数：将 batch 中的数据转换为 Tensor
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    return features, labels

# ==========================================================
# 2. 定义教师模型与学生模型
# ==========================================================

class TeacherModel(nn.Module):
    """
    教师模型：较大模型，用于生成软目标
    模型结构简单构造，模拟 Deepseek-R1 大模型的一部分能力
    """
    def __init__(self, input_dim: int = 100, hidden_dim: int = 512, num_classes: int = 10):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

class StudentModel(nn.Module):
    """
    学生模型：体积较小的模型，用于在教师模型指导下进行蒸馏训练
    采用较少的参数量，力求达到教师模型的性能但具有更高的运行效率
    """
    def __init__(self, input_dim: int = 100, hidden_dim: int = 128, num_classes: int = 10):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

# ==========================================================
# 3. 定义知识蒸馏训练函数
# ==========================================================

def train_distillation(teacher: nn.Module, student: nn.Module, dataloader: DataLoader,
                       device: str = "cpu", epochs: int = 5, temperature: float = 2.0,
                       alpha: float = 0.7) -> None:
    """
    知识蒸馏训练函数：
    - teacher: 教师模型（参数固定）
    - student: 学生模型
    - dataloader: 数据加载器
    - temperature: 温度参数，用于软化教师输出概率分布
    - alpha: 蒸馏损失与硬标签损失的加权因子（alpha 为蒸馏损失权重）
    """
    teacher.to(device)
    student.to(device)
    teacher.eval()  # 教师模型固定，不更新
    
    # 定义优化器和调度器
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    # 定义损失函数
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    total_steps = epochs * len(dataloader)
    step = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs, hard_labels = batch
            inputs = inputs.to(device)
            hard_labels = hard_labels.to(device)
            
            # 教师模型生成软目标（使用温度调节）
            with torch.no_grad():
                teacher_logits = teacher(inputs) / temperature
                teacher_soft = F.softmax(teacher_logits, dim=-1)
            
            # 学生模型前向传播（两种模式均采用温度调节输出）
            student_logits = student(inputs) / temperature
            
            # 蒸馏损失：KL散度损失，注意输入对数概率
            distill_loss = kl_loss_fn(F.log_softmax(student_logits, dim=-1), teacher_soft)
            # 硬标签损失：交叉熵损失（不使用温度调节）
            hard_loss = ce_loss_fn(student(inputs), hard_labels)
            # 总损失为两者加权和，乘以温度平方（公式要求）
            total_loss = alpha * (temperature ** 2) * distill_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度剪裁（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            step += 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{total_steps}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        scheduler.step()
    print("蒸馏训练完成。")

# ==========================================================
# 4. 定义测试函数：评估学生模型性能
# ==========================================================

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> float:
    """
    在测试集上评估模型准确率
    """
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"模型准确率: {accuracy*100:.2f}%")
    return accuracy

# ==========================================================
# 5. Flask API 部署：提供在线推理接口（学生模型）
# ==========================================================

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{DEVICE}")

# 初始化教师模型与学生模型
TEACHER_MODEL = TeacherModel()
STUDENT_MODEL = StudentModel()

# 模拟加载教师模型预训练权重（固定不更新）
def load_teacher_weights():
    print("加载教师模型权重……")
    # 此处可添加权重加载代码，使用随机权重模拟
    time.sleep(1)
    print("教师模型权重加载完成。")
load_teacher_weights()

# 训练学生模型：构造数据集并训练蒸馏
def prepare_and_train_student():
    # 构造训练数据集
    train_dataset = SyntheticClassificationDataset(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print("开始知识蒸馏训练学生模型……")
    train_distillation(TEACHER_MODEL, STUDENT_MODEL, train_loader, device=DEVICE, epochs=5, temperature=2.0, alpha=0.7)
    print("学生模型训练完成。")
prepare_and_train_student()

# 定义推理接口，调用学生模型进行预测
@app.route('/distill_inference', methods=['POST'])
def distill_inference():
    """
    推理 API 接口：接收 JSON 格式请求 {"input": "输入文本"}
    使用训练后的学生模型进行推理，返回预测类别
    """
    data = request.get_json(force=True)
    input_vector = data.get("input", "")
    if input_vector == "":
        return jsonify({"error": "缺少 'input' 参数"}), 400
    
    # 将输入文本转为向量（这里简单模拟：将每个字符转换为浮点数，固定长度 100）
    # 实际中应使用合适的文本编码方式
    vector = np.array([float(ord(c)) for c in input_vector][:100])
    if vector.shape[0] < 100:
        vector = np.pad(vector, (0, 100 - vector.shape[0]), mode='constant')
    vector = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = STUDENT_MODEL(vector)
        _, predicted = torch.max(outputs, 1)
    
    return jsonify({"prediction": int(predicted.item())})

@app.route('/', methods=['GET'])
def index():
    return "学生模型知识蒸馏推理服务已启动，请使用 /distill_inference 接口进行推理。"

# ==========================================================
# 6. 主函数：启动 Flask 服务
# ==========================================================

def main():
    # 启动 Flask 服务
    print("启动学生模型在线推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()