# 例【2-1】
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 为了让 Matplotlib 正常显示中文，设置中文字体（根据自己系统配置字体）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 如系统无 SimHei，可换成其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False

# 检查是否支持 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义简单的卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输出 (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输出 (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)   # 输出 (64, 7, 7)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 实例化网络、损失函数和优化器
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义学习率衰减策略，每 5 个 epoch 衰减为原来的 0.5 倍
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 用于记录训练过程数据
num_epochs = 20
train_losses = []
test_accuracies = []
lr_history = []

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()       # 清空梯度
        outputs = model(inputs)     # 前向传播
        loss = criterion(outputs, targets)
        loss.backward()             # 反向传播
        optimizer.step()            # 参数更新
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    test_acc = correct / total
    test_accuracies.append(test_acc)
    
    print(f"第 {epoch+1} 个 epoch：训练损失 = {avg_loss:.4f}, 测试准确率 = {test_acc:.4f}, 当前学习率 = {current_lr:.6f}")

# 绘制训练过程曲线
plt.figure(figsize=(10, 12))

# 绘制训练损失曲线
plt.subplot(3, 1, 1)
plt.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-', color='blue')
plt.title("训练损失曲线", fontsize=14)
plt.xlabel("迭代周期 (Epoch)", fontsize=12)
plt.ylabel("平均损失", fontsize=12)
plt.grid(True)

# 绘制测试准确率曲线
plt.subplot(3, 1, 2)
plt.plot(range(1, num_epochs+1), test_accuracies, marker='s', linestyle='-', color='green')
plt.title("测试准确率曲线", fontsize=14)
plt.xlabel("迭代周期 (Epoch)", fontsize=12)
plt.ylabel("准确率", fontsize=12)
plt.grid(True)

# 绘制学习率变化曲线
plt.subplot(3, 1, 3)
plt.plot(range(1, num_epochs+1), lr_history, marker='^', linestyle='-', color='red')
plt.title("学习率变化曲线", fontsize=14)
plt.xlabel("迭代周期 (Epoch)", fontsize=12)
plt.ylabel("学习率", fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()


# 例【2-2】
import torch

# 创建两个张量
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 基本运算
add_result = tensor_a + tensor_b          # 张量加法
mul_result = tensor_a * tensor_b          # 元素乘法
matmul_result = torch.matmul(tensor_a, tensor_b)  # 矩阵乘法


# 例【2-3】
import torch

# 定义可求导张量
x = torch.tensor(2.0, requires_grad=True)

# 定义简单的函数
y = x ** 3 + 2 * x ** 2 + 5

# 反向传播，计算梯度
y.backward()


# 例【2-4】
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # 输入层到隐藏层
        self.fc2 = nn.Linear(8, 3)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))     # ReLU 激活函数
        x = self.fc2(x)             # 输出层
        return x

# 模型实例化
model = SimpleNet()
input_data = torch.randn(1, 4)     # 随机输入数据
output = model(input_data)

print("模型输出：", output)


# 例【2-5】
# 使用 nn.Sequential 构建模型
sequential_model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

input_data = torch.randn(1, 4)
output = sequential_model(input_data)

print("Sequential模型输出：", output)


# 例【2-6】
import torch
import torch.nn as nn

# 定义简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        if x.mean() > 0:  # 动态控制流
            x = torch.relu(self.fc1(x))
        else:
            x = torch.sigmoid(self.fc1(x))
        return self.fc2(x)

# 检测GPU并迁移模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# 随机输入数据
input_data = torch.randn(3, 4).to(device)
output = model(input_data)

print("模型输出：", output)


# 例【2-7】
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义Actor-Critic模型
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

# 环境模拟（简化版）
class SimpleEnv:
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        reward = np.random.randn() + action
        self.state += 1
        done = self.state >= 10
        return np.array([self.state], dtype=np.float32), reward, done

# 参数设置
state_dim = 1
action_dim = 2
env = SimpleEnv()
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练过程
rewards_history = []
for episode in range(10):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        policy, value = model(state_tensor)
        action = np.random.choice(action_dim, p=policy.detach().numpy())
        next_state, reward, done = env.step(action)

        # 计算优势函数
        _, next_value = model(torch.FloatTensor(next_state))
        advantage = reward + (0.99 * next_value.item() * (1 - int(done))) - value.item()

        # 损失计算与反向传播
        actor_loss = -torch.log(policy[action]) * advantage
        critic_loss = advantage ** 2
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)

# 输出训练结果
print(rewards_history)


# 例【2-8】
import numpy as np
import matplotlib.pyplot as plt

# 为了让 Matplotlib 正常显示中文，设置中文字体（系统中需要有 SimHei 字体）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 如果没有该字体，可以更换为其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False

# 参数设置
epsilon_start = 1.0    # 初始 epsilon
epsilon_final = 0.01   # 最终 epsilon
decay_rate = 500       # 衰减速率（数值越大，衰减越慢）
num_steps = 1000       # 总步数

# 记录每一步的 epsilon 值
epsilon_values = []

for step in range(num_steps):
    # 指数衰减策略
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * step / decay_rate)
    epsilon_values.append(epsilon)

# 打印部分关键步数的 epsilon 值
print("部分步数对应的 epsilon 值：")
for step in [0, 100, 300, 500, 700, 900]:
    print(f"步数 {step}：epsilon = {epsilon_values[step]:.4f}")

# 绘制 epsilon 动态调整曲线
plt.figure(figsize=(8, 6))
plt.plot(range(num_steps), epsilon_values, color='blue', marker='o', markersize=3, label="探索率 ε")
plt.xlabel("步数 (Step)", fontsize=14)
plt.ylabel("探索率 (ε)", fontsize=14)
plt.title("Epsilon 参数动态调整策略", fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 例【2-9】
import random
from collections import deque, namedtuple
import numpy as np

# 定义 transition 数据结构，用于存储 (状态, 动作, 奖励, 下一个状态, 终止标志)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        :param capacity: 缓冲区的最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存储一个 transition 到缓冲区中
        :param state: 当前状态
        :param action: 采取的动作
        :param reward: 收到的奖励
        :param next_state: 下一个状态
        :param done: 是否终止
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        随机采样一个批次的数据
        :param batch_size: 批次大小
        :return: 包含批次数据的 Transition 元组，每个字段均为 batch_size 长度的 tuple
        """
        transitions = random.sample(self.buffer, batch_size)
        # 利用 * 运算符实现转置，使得每个字段分别聚集到一起
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        """
        返回当前缓冲区中存储的 transition 数量
        """
        return len(self.buffer)

if __name__ == '__main__':
    # 创建一个经验回放缓冲区，容量为100
    replay_buffer = ReplayBuffer(capacity=100)
    
    # 模拟交互数据的生成（例如：状态为 numpy 数组，动作为整数，奖励为浮点数）
    for i in range(150):  # 超出容量限制，确保旧数据被覆盖
        state = np.array([i, i+1])
        action = i % 4
        reward = float(i)
        next_state = state + 1
        done = (i % 10 == 0)
        replay_buffer.push(state, action, reward, next_state, done)
    
    print("当前经验回放缓冲区中的样本数量:", len(replay_buffer))
    
    # 从缓冲区中采样一个批次数据
    batch_size = 8
    sample_batch = replay_buffer.sample(batch_size)
    
    # 打印采样的批次数据
    print("采样的批次数据：")
    for idx in range(batch_size):
        print(f"样本 {idx+1}:")
        print("状态:       ", sample_batch.state[idx])
        print("动作:       ", sample_batch.action[idx])
        print("奖励:       ", sample_batch.reward[idx])
        print("下一个状态: ", sample_batch.next_state[idx])
        print("终止标志:   ", sample_batch.done[idx])
        print("-" * 30)


# 例【2-10】
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# 设置随机种子，方便结果复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 检查是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 定义 Transition 数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.buffer)

# 定义 Dueling DQN 网络结构
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # 特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # 状态价值分支
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 动作优势分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # 合并得到 Q 值：去均值操作有助于稳定训练
        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_value

# 超参数设置
env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 创建在线网络和目标网络（采用 Dueling DQN 结构）
online_net = DuelingDQN(state_dim, action_dim).to(device)
target_net = DuelingDQN(state_dim, action_dim).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练参数
num_episodes = 300
batch_size = 64
gamma = 0.99
replay_buffer = ReplayBuffer(capacity=10000)
target_update_freq = 10  # 每隔 10 个 episode 更新一次目标网络
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 300  # 衰减步数

def get_epsilon(episode):
    # 指数衰减策略
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)
    return epsilon

episode_rewards = []

for episode in range(1, num_episodes+1):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        epsilon = get_epsilon(episode)
        # ε-贪婪策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = online_net(state_tensor)
                action = q_values.argmax().item()
        
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # 存储交互数据到经验回放缓冲区
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        
        # 当缓冲区内样本足够时，从中采样一个批次进行训练
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            # 转换为 tensor
            state_batch      = torch.FloatTensor(batch.state).to(device)
            action_batch     = torch.LongTensor(batch.action).unsqueeze(1).to(device)
            reward_batch     = torch.FloatTensor(batch.reward).to(device)
            next_state_batch = torch.FloatTensor(batch.next_state).to(device)
            done_batch       = torch.FloatTensor(batch.done).to(device)
            
            # 计算当前 Q 值
            q_values = online_net(state_batch)
            current_q = q_values.gather(1, action_batch).squeeze(1)
            
            # Double DQN 部分：在线网络选择动作，目标网络计算对应 Q 值
            with torch.no_grad():
                next_q_online = online_net(next_state_batch)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)  # 在线网络选取动作
                next_q_target = target_net(next_state_batch)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
                # 若终止则 target 为 reward
                target_q = reward_batch + gamma * next_q * (1 - done_batch)
            
            loss = criterion(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    episode_rewards.append(episode_reward)
    
    # 每隔一定 episode 更新目标网络参数
    if episode % target_update_freq == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode: {episode:3d}, 平均奖励: {avg_reward:.2f}, ε: {epsilon:.3f}")