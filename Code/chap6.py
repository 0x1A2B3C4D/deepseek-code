# 例【6-1】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mixture_of_experts.py
---------------------
本示例演示如何构建一个混合专家模型（Mixture-of-Experts, MoE），
利用多个专家网络和门控网络实现并行化计算与扩展能力优化。
模型采用 PyTorch 实现，结合 Deepseek-R1 模型 API 模拟（调用模拟函数），
展示专家模型在不同任务下的协同作用和高效推理能力。
示例包括数据构造、专家模型、门控网络、混合专家模型定义、
并行计算实现、训练流程、在线推理接口及详细注释。
"""

import os
import time
import random
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
from flask import Flask, request, jsonify


# 1. 数据集定义：构造简单的分类任务数据集


class SimpleClassificationDataset(Dataset):
    """
    简单分类数据集
    每个样本包含 100 维特征及对应 0-9 的标签，模拟分类任务数据
    """
    def __init__(self, num_samples: int = 1000, input_dim: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, input_dim).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def simple_collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate 函数，将列表样本转换为 Tensor
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)
    return features_tensor, labels_tensor


# 2. 模型定义


# 2.1 定义单个专家模型：用于处理输入数据
class ExpertModel(nn.Module):
    """
    单个专家网络模型
    结构为简单全连接网络，输出 10 维向量表示分类 logits
    """
    def __init__(self, input_dim: int = 100, hidden_dim: int = 128, num_classes: int = 10):
        super(ExpertModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

# 2.2 定义门控网络：根据输入动态选择专家的权重
class GatingNetwork(nn.Module):
    """
    门控网络，用于计算每个专家的权重分布
    输入为特征向量，输出为专家数量个数的概率分布
    """
    def __init__(self, input_dim: int = 100, num_experts: int = 4):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出 softmax 概率
        gate_logits = self.fc(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        return gate_probs

# 2.3 定义混合专家模型：结合多个专家与门控网络，实现并行化计算
class MixtureOfExperts(nn.Module):
    """
    混合专家模型：
    - 包含多个专家模型（并行执行）
    - 包含一个门控网络，用于计算每个专家的权重
    - 最终输出为各专家输出加权求和后的结果
    """
    def __init__(self, input_dim: int = 100, hidden_dim: int = 128, num_classes: int = 10, num_experts: int = 4):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        # 构造专家网络列表
        self.experts = nn.ModuleList([ExpertModel(input_dim, hidden_dim, num_classes) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算混合专家模型输出：
        - 通过门控网络计算专家权重
        - 并行调用各个专家网络计算 logits
        - 对专家输出进行加权求和
        """
        # 计算门控概率（形状：[batch_size, num_experts]）
        gate_probs = self.gating_network(x)  # (B, E)
        
        # 并行计算所有专家输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)  # (B, num_classes)
            expert_outputs.append(output.unsqueeze(2))  # (B, num_classes, 1)
        # 拼接专家输出 (B, num_classes, num_experts)
        experts_concat = torch.cat(expert_outputs, dim=2)
        
        # 将门控概率扩展维度以进行加权 (B, 1, num_experts)
        gate_probs_expanded = gate_probs.unsqueeze(1)
        
        # 加权求和各专家输出 (B, num_classes, num_experts) * (B, 1, num_experts)
        weighted_output = experts_concat * gate_probs_expanded
        output = torch.sum(weighted_output, dim=2)  # (B, num_classes)
        return output


# 3. 蒸馏过程：利用教师模型指导学生模型训练（加入混合专家并行）


class TeacherModelForDistillation(nn.Module):
    """
    模拟教师模型：较大模型，输出软标签
    此处结构较深，参数较多，模拟 Deepseek-R1 部分能力
    """
    def __init__(self, input_dim: int = 100, hidden_dim: int = 256, num_classes: int = 10):
        super(TeacherModelForDistillation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.out(x)
        return logits

def train_student_with_distillation(teacher: nn.Module, student: MixtureOfExperts,
                                    dataloader: DataLoader, device: str = "cpu",
                                    epochs: int = 5, temperature: float = 2.0, alpha: float = 0.7) -> None:
    """
    利用知识蒸馏训练学生模型：
    - 教师模型输出软标签（经过温度调节）
    - 学生模型为混合专家模型，输出加权结果
    - 损失由蒸馏损失（KL 散度）与硬标签损失（交叉熵）组成
    """
    teacher.to(device)
    student.to(device)
    teacher.eval()  # 教师模型固定参数
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    total_steps = epochs * len(dataloader)
    step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, hard_labels in dataloader:
            inputs = inputs.to(device)
            hard_labels = hard_labels.to(device)
            
            # 教师模型输出软标签，温度调节
            with torch.no_grad():
                teacher_logits = teacher(inputs) / temperature
                teacher_soft = F.softmax(teacher_logits, dim=-1)
            
            # 学生模型输出（混合专家模型）
            student_logits = student(inputs) / temperature
            
            # 计算蒸馏损失：KL散度损失
            distill_loss = kl_loss_fn(F.log_softmax(student_logits, dim=-1), teacher_soft)
            # 计算硬标签损失：交叉熵损失（不使用温度调节）
            student_logits_hard = student(inputs)
            hard_loss = ce_loss_fn(student_logits_hard, hard_labels)
            
            total_loss = alpha * (temperature ** 2) * distill_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            step += 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{total_steps}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        scheduler.step()
    print("学生模型蒸馏训练完成。")


# 4. 并行化测试：利用多线程并行计算专家输出


def parallel_expert_inference(student: MixtureOfExperts, input_tensor: torch.Tensor, num_workers: int = 4) -> torch.Tensor:
    """
    利用并行化技术计算混合专家模型中各专家的输出，并返回加权求和结果
    使用 concurrent.futures.ThreadPoolExecutor 实现并行调用
    """
    # 定义一个函数用于计算单个专家的输出
    def compute_expert_output(expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return expert(x)
    
    # 获取门控权重
    with torch.no_grad():
        gate_probs = student.gating_network(input_tensor)  # (B, num_experts)
    
    expert_outputs = []
    # 并行计算每个专家的输出
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_expert_output, expert, input_tensor) for expert in student.experts]
        for future in concurrent.futures.as_completed(futures):
            expert_outputs.append(future.result().unsqueeze(2))  # (B, num_classes, 1)
    
    # 按照专家顺序堆叠输出
    experts_concat = torch.cat(expert_outputs, dim=2)  # (B, num_classes, num_experts)
    gate_probs_expanded = gate_probs.unsqueeze(1)        # (B, 1, num_experts)
    weighted_sum = torch.sum(experts_concat * gate_probs_expanded, dim=2)  # (B, num_classes)
    return weighted_sum

def test_parallel_inference(student: MixtureOfExperts, device: str = "cpu"):
    """
    测试函数：生成随机输入，利用并行化函数计算混合专家模型输出，
    并打印输出结果，用于验证并行计算效果
    """
    student.to(device)
    # 生成一个随机输入样本（batch_size=4, input_dim=100）
    sample_input = torch.randn(4, 100).to(device)
    # 使用并行推理计算专家加权输出
    output_parallel = parallel_expert_inference(student, sample_input)
    # 同时直接调用混合专家模型前向传播作为对比
    output_direct = student(sample_input)
    print("并行化专家输出结果：", output_parallel)
    print("直接模型输出结果：", output_direct)


# 5. Flask API 服务：提供混合专家模型在线推理接口


app = Flask(__name__)

# 全局变量：加载教师模型与经过蒸馏训练的学生混合专家模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{DEVICE}")

# 初始化教师模型与学生模型
TEACHER_MODEL = TeacherModelForDistillation()
STUDENT_MODEL = MixtureOfExperts()
TEACHER_MODEL.to(DEVICE)
STUDENT_MODEL.to(DEVICE)

# 模拟加载教师模型权重（随机权重模拟）
def load_teacher_model_weights():
    print("加载教师模型权重（模拟）……")
    time.sleep(1)
    print("教师模型权重加载完成。")
load_teacher_model_weights()

# 训练学生模型：构造数据集并执行知识蒸馏训练
def prepare_and_train_student():
    train_dataset = SyntheticClassificationDataset(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=simple_collate_fn)
    print("开始学生模型知识蒸馏训练……")
    train_distillation(TEACHER_MODEL, STUDENT_MODEL, train_loader, device=DEVICE, epochs=5, temperature=2.0, alpha=0.7)
    print("学生模型蒸馏训练完成。")
prepare_and_train_student()

# 定义推理 API 接口，使用混合专家模型进行在线推理
@app.route('/moe_inference', methods=['POST'])
def moe_inference():
    """
    混合专家模型推理接口：
    接收 JSON 请求 {"input": [数值列表]}，返回模型预测类别
    """
    data = request.get_json(force=True)
    input_data = data.get("input", [])
    if not input_data:
        return jsonify({"error": "缺少 'input' 参数"}), 400
    # 确保输入为 100 维
    if len(input_data) < 100:
        input_data.extend([0.0] * (100 - len(input_data)))
    elif len(input_data) > 100:
        input_data = input_data[:100]
    input_tensor = torch.tensor([input_data], dtype=torch.float32).to(DEVICE)
    # 使用并行专家推理计算输出
    with torch.no_grad():
        logits = parallel_expert_inference(STUDENT_MODEL, input_tensor)
        _, predicted = torch.max(logits, 1)
    return jsonify({"prediction": int(predicted.item())})

@app.route('/', methods=['GET'])
def index():
    return "混合专家模型推理服务已启动，请使用 /moe_inference 接口进行推理。"


# 6. 主函数：启动 Flask 服务及测试并行推理


def main():
    # 测试并行专家推理函数
    print("测试并行化专家推理……")
    test_parallel_inference(STUDENT_MODEL, device=DEVICE)
    # 启动 Flask 服务
    print("启动混合专家模型在线推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()


# 例【6-2】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mixed_precision_training.py
---------------------------
本示例展示了混合精度训练与基于 FP16/模拟 FP8 内存计算的实现方法，
旨在利用 PyTorch 自动混合精度 (AMP) 和梯度缩放 (GradScaler) 技术，
在保持模型性能的前提下降低内存占用和加速训练过程。
此外，部分代码模拟了 FP8 的量化计算思路，用于展示新一代低精度计算的优势。
该示例以合成分类任务为例，定义简单的多层全连接网络，
使用自动混合精度进行训练，并提供在线推理接口（Flask 部署）。
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


# 1. 数据集定义与预处理


class SyntheticDataset(Dataset):
    """
    合成分类数据集：
    每个样本为随机生成的 100 维特征向量，
    标签为 0-9 之间的整数，用于分类任务。
    """
    def __init__(self, num_samples: int = 1000, input_dim: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, input_dim).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate 函数：将列表形式的样本转换为 Tensor
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)
    return features_tensor, labels_tensor


# 2. 模型定义：构造一个简单的多层全连接网络


class ClassificationModel(nn.Module):
    """
    分类模型：采用多层全连接网络实现
    模型结构：
      - 输入层：100 维
      - 隐藏层1：256 维，ReLU 激活
      - 隐藏层2：128 维，ReLU 激活
      - 输出层：10 维（对应 10 个分类）
    """
    def __init__(self, input_dim: int = 100, hidden_dim1: int = 256, hidden_dim2: int = 128, num_classes: int = 10):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits


# 3. 模拟 FP8 量化函数（简单模拟，实际环境中需硬件支持）


def simulate_fp8_quantization(tensor: torch.Tensor) -> torch.Tensor:
    """
    模拟 FP8 量化操作：
    本函数将输入 tensor 先转换为 FP16，再进行简单量化模拟为 FP8 表示（不是真正的 FP8）
    实际应用中 FP8 需要专门硬件支持，本函数仅作为示例说明思想。
    """
    # 先转换为 FP16（半精度）
    tensor_fp16 = tensor.half()
    # 模拟量化：将 FP16 数值缩放、四舍五入后再缩放回去（假设模拟 FP8 精度较低）
    scale = 16.0  # 模拟量化因子
    tensor_scaled = tensor_fp16 * scale
    tensor_rounded = torch.round(tensor_scaled)
    tensor_fp8_simulated = tensor_rounded / scale
    return tensor_fp8_simulated


# 4. 混合精度训练实现：利用 torch.cuda.amp 进行 FP16 训练


def train_model_mixed_precision(model: nn.Module, dataloader: DataLoader, device: str = "cpu",
                                epochs: int = 5) -> None:
    """
    使用混合精度训练模型：
    - 利用 torch.cuda.amp.autocast 实现自动 FP16 计算
    - 使用 GradScaler 进行梯度缩放，防止数值不稳定
    - 同时展示如何在部分计算中模拟 FP8 量化
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    ce_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    
    total_steps = epochs * len(dataloader)
    step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 使用自动混合精度进行前向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(inputs)
                loss = ce_loss_fn(logits, labels)
                # 模拟部分计算采用 FP8 量化（仅用于展示，不影响整体梯度计算）
                logits_fp8 = simulate_fp8_quantization(logits)
                # 计算一个附加损失，衡量 FP8 与原始 FP16 之间的差异（作为正则项）
                quant_loss = F.mse_loss(logits, logits_fp8)
                total_loss = loss + 0.1 * quant_loss
            
            # 使用梯度缩放反向传播
            scaler.scale(total_loss).backward()
            # 梯度剪裁
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item()
            step += 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{total_steps}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        scheduler.step()
    print("混合精度训练完成。")


# 5. 模型评估函数：计算测试集准确率


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> float:
    """
    在测试集上评估模型准确率
    """
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"测试模型准确率: {accuracy*100:.2f}%")
    return accuracy


# 6. Flask API 服务部署：提供在线推理接口


app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{DEVICE}")

# 加载模型与数据：使用混合精度训练后的模型用于在线推理
MODEL = ClassificationModel()
MODEL.to(DEVICE)


# 7. 训练流程与模型保存


def prepare_and_train_model():
    """
    构造数据集，进行混合精度训练，并保存模型参数
    """
    dataset = SyntheticDataset(num_samples=1000, input_dim=100, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print("开始混合精度训练……")
    train_model_mixed_precision(MODEL, dataloader, device=DEVICE, epochs=5)
    print("训练结束。")
    # 保存模型参数
    torch.save(MODEL.state_dict(), "student_model_fp16.pth")
    print("模型参数已保存至 student_model_fp16.pth。")

prepare_and_train_model()


# 8. 定义推理 API 接口，调用训练后的模型进行预测


@app.route('/inference', methods=['POST'])
def inference():
    """
    在线推理 API 接口：
    接收 JSON 格式请求 {"input": [数值列表]}，返回模型预测的分类标签
    """
    data = request.get_json(force=True)
    input_data = data.get("input", [])
    if not input_data:
        return jsonify({"error": "缺少 'input' 参数"}), 400
    # 确保输入为 100 维
    if len(input_data) < 100:
        input_data.extend([0.0] * (100 - len(input_data)))
    elif len(input_data) > 100:
        input_data = input_data[:100]
    input_tensor = torch.tensor([input_data], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return jsonify({"prediction": int(predicted.item())})

@app.route('/', methods=['GET'])
def index():
    return "混合精度训练与 FP8/FP16 内存计算推理服务已启动，请使用 /inference 接口进行推理。"


# 9. 主函数：启动 Flask 服务


def main():
    print("启动在线推理服务……")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()


# 例【6-3】
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nvlink_optimization.py
----------------------
本示例演示如何利用 NVLink 带宽优化技术对 Deepseek‑R1 模型进行分布式训练加速，
通过重叠梯度通信与计算、参数分片、混合精度训练等技术降低通信延迟，提升整体训练效率。
使用 PyTorch 分布式训练模块模拟多 GPU 环境，采用梯度钩子实现通信与计算的重叠，
并提供基于 Flask 的推理接口，展示在线推理效果。
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
import torch.distributed as dist
import torch.multiprocessing as mp
from flask import Flask, request, jsonify

# ===============================
# 1. 分布式环境初始化函数
# ===============================
def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    初始化分布式训练环境
    使用 NCCL 后端，适用于多 GPU 通信（NVLink 依赖 NCCL 优化）
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"[Rank {rank}] 分布式环境初始化完成。")

def cleanup_distributed():
    """
    清理分布式环境
    """
    dist.destroy_process_group()

# ===============================
# 2. 数据集定义：使用合成分类数据集
# ===============================
class SyntheticDataset(Dataset):
    """
    合成分类数据集：
    每个样本为随机生成的 100 维特征向量，
    标签为 0-9 之间的整数，用于分类任务。
    """
    def __init__(self, num_samples: int = 1000, input_dim: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, input_dim).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate 函数：将 batch 转换为 Tensor 格式
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return torch.tensor(features), torch.tensor(labels)

# ===============================
# 3. 模型定义：构造简单分类模型
# ===============================
class ClassificationModel(nn.Module):
    """
    简单分类模型：
    - 输入层：100 维
    - 隐藏层1：256 维，ReLU 激活
    - 隐藏层2：128 维，ReLU 激活
    - 输出层：10 维（分类）
    """
    def __init__(self, input_dim: int = 100, hidden_dim1: int = 256, hidden_dim2: int = 128, num_classes: int = 10):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

# ===============================
# 4. 混合精度训练与 NVLink 带宽优化模拟
# ===============================
def simulate_nvlink_optimization(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    模拟 NVLink 带宽优化：
    在多 GPU 环境中，使用 NCCL 实现高效的 all-reduce 操作，
    此处用 torch.distributed.all_reduce 模拟通信延迟降低。
    """
    # 复制输入，用于通信模拟
    tensor = inputs.clone()
    # 开始计时
    start_time = time.time()
    # 使用分布式 all_reduce 进行梯度同步（模拟通信优化）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_time = time.time()
    # 打印通信时间
    print(f"[NVLink 模拟] all_reduce 通信耗时：{(end_time - start_time) * 1000:.2f} 毫秒")
    return tensor

def train_epoch_distributed(model: nn.Module, dataloader: DataLoader, device: str, rank: int) -> float:
    """
    分布式训练单个 epoch：
    模拟在多 GPU 环境下进行混合精度训练，并在反向传播过程中调用 NVLink 带宽优化函数
    """
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    epoch_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            outputs = model(inputs)
            loss = ce_loss_fn(outputs, labels)
        # 模拟 NVLink 优化，调用 all_reduce 优化通信
        _ = simulate_nvlink_optimization(model, inputs)
        scaler.scale(loss).backward()
        # 模拟梯度通信优化（实际在 DDP 中自动执行）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"[Rank {rank}] Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss

def distributed_training(rank: int, world_size: int):
    """
    分布式训练入口：
    初始化分布式环境，构造数据集、模型和 DataLoader，
    执行混合精度训练，并输出每个 epoch 的损失
    """
    setup_distributed(rank, world_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 构造数据集与 DataLoader
    dataset = SyntheticDataset(num_samples=1000, input_dim=100, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # 初始化模型
    model = ClassificationModel().to(device)
    # 使用 DistributedDataParallel 包装模型（注意：本示例未使用 DDP 以便自定义 all_reduce 模拟）
    num_epochs = 3
    for epoch in range(num_epochs):
        avg_loss = train_epoch_distributed(model, dataloader, device, rank)
        print(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    # 保存模型仅由 rank==0 执行
    if rank == 0:
        torch.save(model.state_dict(), "distributed_model.pth")
        print("模型参数已保存至 distributed_model.pth。")
    cleanup_distributed()


# 5. Flask API 部署：提供在线推理接口，加载分布式训练后的模型

app = Flask(__name__)
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{GLOBAL_DEVICE}")
# 初始化全局模型（结构与训练模型一致）
GLOBAL_MODEL = ClassificationModel()
GLOBAL_MODEL.to(GLOBAL_DEVICE)
# 加载保存的模型参数（假设由分布式训练后 rank 0 保存）
if os.path.exists("distributed_model.pth"):
    GLOBAL_MODEL.load_state_dict(torch.load("distributed_model.pth", map_location=GLOBAL_DEVICE))
    print("全局模型参数加载完成。")
else:
    print("未找到训练好的模型参数，使用随机初始化。")

@app.route('/inference', methods=['POST'])
def inference():
    """
    在线推理 API 接口：
    接收 JSON 格式请求 {"input": [数值列表]}，
    返回模型预测的分类标签
    """
    data = request.get_json(force=True)
    input_data = data.get("input", [])
    if not input_data:
        return jsonify({"error": "缺少 'input' 参数"}), 400
    if len(input_data) < 100:
        input_data.extend([0.0] * (100 - len(input_data)))
    elif len(input_data) > 100:
        input_data = input_data[:100]
    input_tensor = torch.tensor([input_data], dtype=torch.float32).to(GLOBAL_DEVICE)
    with torch.no_grad():
        outputs = GLOBAL_MODEL(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return jsonify({"prediction": int(predicted.item())})

@app.route('/', methods=['GET'])
def index():
    return "混合精度训练与 NVLink 带宽优化推理服务已启动，请使用 /inference 接口进行推理。"


# 6. 主函数：启动 Flask 服务或分布式训练模式


def main():
    import argparse
    parser = argparse.ArgumentParser(description="混合精度训练与 NVLink 带宽优化示例")
    parser.add_argument("--mode", type=str, default="train", help="模式：train 或 inference")
    parser.add_argument("--rank", type=int, default=0, help="进程 rank（仅训练模式有效）")
    parser.add_argument("--world_size", type=int, default=1, help="总进程数（仅训练模式有效）")
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.world_size > 1:
            # 多进程分布式训练：使用 mp.spawn 启动多个进程
            mp.spawn(distributed_training, args=(args.world_size,), nprocs=args.world_size, join=True)
        else:
            # 单机训练模式
            distributed_training(args.rank, args.world_size)
    else:
        # 推理模式：启动 Flask API 服务
        print("启动在线推理服务……")
        app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()