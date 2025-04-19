# 例【4-1】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from collections import deque

# 创建一个简单的策略网络，PPO与TRPO都可以共享这个网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# 定义PPO算法的类
class PPO:
    def __init__(self, env, policy_network, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.env = env
        self.policy_network = policy_network
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, num_episodes):
        all_rewards = []
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            values = []
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            total_reward = 0
            
            while not done:
                state = state.unsqueeze(0)
                dist = self.policy_network(state)
                m = Categorical(dist)
                action = m.sample()
                log_prob = m.log_prob(action)
                value = dist[0, action]  # 简化处理，实际应使用 value 网络
                
                next_state, reward, done, _ = self.env.step(action.item())
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(value)
                
                state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward
            
            all_rewards.append(total_reward)
            returns = self.compute_returns(rewards, dones, values, value)
            advantages = self.compute_advantages(rewards, dones, values, value)
            self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
        
        return all_rewards

# 创建环境
env = gym.make('CartPole-v1')
env.seed(42)

# 初始化策略网络和 PPO
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)
ppo = PPO(env, policy_network)

# 训练PPO
num_episodes = 100
rewards = ppo.train(num_episodes)

# 输出训练过程的总奖励
print("训练完成，最终奖励：", rewards[-1])

# 进行评估
def evaluate(env, policy_network, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(env, policy_network)
print(f"平均评估奖励: {avg_reward}")


# 例【4-2】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from collections import deque
import threading
import time
import requests

# 模拟一个简单的Deepseek API接口请求类，用于分布式获取大模型的结果
class DeepseekAPIClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def get_model_output(self, input_data):
        # 模拟发送请求并获取模型输出
        response = requests.post(self.api_url, json={"input": input_data, "key": self.api_key})
        return response.json()

# 策略网络模型定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# 训练过程管理类
class DistributedRLTrainer:
    def __init__(self, env, policy_network, num_workers=4, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.env = env
        self.policy_network = policy_network
        self.num_workers = num_workers
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)
        self.global_rewards = deque(maxlen=100)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def worker_train(self, worker_id):
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        values = []
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            state = state.unsqueeze(0)
            dist = self.policy_network(state)
            m = Categorical(dist)
            action = m.sample()
            log_prob = m.log_prob(action)
            value = dist[0, action]  # 简化处理，实际应使用 value 网络

            next_state, reward, done, _ = self.env.step(action.item())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(value)

            state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

        returns = self.compute_returns(rewards, dones, values, value)
        advantages = self.compute_advantages(rewards, dones, values, value)
        self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

        self.global_rewards.append(total_reward)
        if worker_id == 0 and len(self.global_rewards) % 10 == 0:
            print(f"Worker {worker_id} - Episode {len(self.global_rewards)}, Reward: {total_reward}")

    def train(self, num_episodes):
        threads = []
        for worker_id in range(self.num_workers):
            thread = threading.Thread(target=self.worker_train, args=(worker_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"训练完成，总奖励：{np.mean(self.global_rewards)}")
        return np.mean(self.global_rewards)

# 创建环境
env = gym.make('CartPole-v1')
env.seed(42)

# 初始化DeepseekAPIClient，用于模拟调用大模型接口
api_client = DeepseekAPIClient(api_url="https://api.deepseek.com/v1", api_key="your_api_key")

# 初始化分布式强化学习训练器
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)
trainer = DistributedRLTrainer(env, policy_network)

# 开始训练
num_episodes = 100
trainer.train(num_episodes)

# 模型评估
def evaluate(env, policy_network, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(env, policy_network)
print(f"平均评估奖励: {avg_reward}")


# 例【4-3】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import random
from collections import deque
import time
import requests

# 模拟一个简单的Deepseek API接口请求类，用于分布式获取大模型的结果
class DeepseekAPIClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def get_model_output(self, input_data):
        # 模拟发送请求并获取模型输出
        response = requests.post(self.api_url, json={"input": input_data, "key": self.api_key})
        return response.json()

# 策略网络模型定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# DQN算法中的经验回放类
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 策略优化的PPO（Proximal Policy Optimization）类
class PPO:
    def __init__(self, env, policy_network, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10, memory_capacity=10000):
        self.env = env
        self.policy_network = policy_network
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)
        self.memory = ExperienceReplay(memory_capacity)
        self.global_rewards = deque(maxlen=100)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def worker_train(self, worker_id):
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        values = []
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            state = state.unsqueeze(0)
            dist = self.policy_network(state)
            m = Categorical(dist)
            action = m.sample()
            log_prob = m.log_prob(action)
            value = dist[0, action]  # 简化处理，实际应使用 value 网络

            next_state, reward, done, _ = self.env.step(action.item())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(value)

            state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

        returns = self.compute_returns(rewards, dones, values, value)
        advantages = self.compute_advantages(rewards, dones, values, value)
        self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

        self.global_rewards.append(total_reward)
        if worker_id == 0 and len(self.global_rewards) % 10 == 0:
            print(f"Worker {worker_id} - Episode {len(self.global_rewards)}, Reward: {total_reward}")

    def train(self, num_episodes):
        threads = []
        for worker_id in range(self.num_workers):
            thread = threading.Thread(target=self.worker_train, args=(worker_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"训练完成，总奖励：{np.mean(self.global_rewards)}")
        return np.mean(self.global_rewards)

# 创建环境
env = gym.make('CartPole-v1')
env.seed(42)

# 初始化DeepseekAPIClient，用于模拟调用大模型接口
api_client = DeepseekAPIClient(api_url="https://api.deepseek.com/v1", api_key="your_api_key")

# 初始化分布式强化学习训练器
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)
trainer = PPO(env, policy_network)

# 开始训练
num_episodes = 100
trainer.train(num_episodes)

# 模型评估
def evaluate(env, policy_network, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(env, policy_network)
print(f"平均评估奖励: {avg_reward}")


# 例【4-4】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import requests

# 模拟一个简单的Deepseek API接口请求类，用于分布式获取大模型的结果
class DeepseekAPIClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def get_model_output(self, input_data):
        # 模拟发送请求并获取模型输出
        response = requests.post(self.api_url, json={"input": input_data, "key": self.api_key})
        return response.json()

# 策略网络模型定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# 自定义奖励生成模型
class RewardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, state_action):
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward

# 奖励建模策略类
class RewardBasedLearning:
    def __init__(self, env, policy_network, reward_model, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.env = env
        self.policy_network = policy_network
        self.reward_model = reward_model
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(list(policy_network.parameters()) + list(reward_model.parameters()), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            reward_predictions = self.reward_model(torch.cat([states, actions.unsqueeze(1)], dim=1))
            reward_loss = torch.mean((returns - reward_predictions) ** 2)

            total_loss = loss + reward_loss  # Combine policy loss and reward model loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            values = []
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            total_reward = 0

            while not done:
                state = state.unsqueeze(0)
                dist = self.policy_network(state)
                m = Categorical(dist)
                action = m.sample()
                log_prob = m.log_prob(action)
                value = dist[0, action]  # Simplified, should be from value network

                next_state, reward, done, _ = self.env.step(action.item())
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(value)

                state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward

            # Calculate rewards and advantages
            returns = self.compute_returns(rewards, dones, values, value)
            advantages = self.compute_advantages(rewards, dones, values, value)

            # Update policy and reward model
            self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

            print(f"Episode {episode}, Total Reward: {total_reward}")

# 创建环境
env = gym.make('CartPole-v1')
env.seed(42)

# 初始化DeepseekAPIClient，用于模拟调用大模型接口
api_client = DeepseekAPIClient(api_url="https://api.deepseek.com/v1", api_key="your_api_key")

# 初始化策略网络和奖励生成模型
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)
reward_model = RewardModel(input_dim + 1, 1)  # state + action as input

# 初始化奖励建模策略
trainer = RewardBasedLearning(env, policy_network, reward_model)

# 开始训练
num_episodes = 100
trainer.train(num_episodes)

# 模型评估
def evaluate(env, policy_network, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(env, policy_network)
print(f"平均评估奖励: {avg_reward}")


# 例【4-5】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import requests

# 模拟一个简单的Deepseek API接口请求类，用于分布式获取大模型的结果
class DeepseekAPIClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def get_model_output(self, input_data):
        # 模拟发送请求并获取模型输出
        response = requests.post(self.api_url, json={"input": input_data, "key": self.api_key})
        return response.json()

# 策略网络模型定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# 自适应奖励模型：动态生成奖励
class AdaptiveRewardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdaptiveRewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, state_action):
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)  # 输出动态计算的奖励
        return reward

# 自适应奖励学习
class AdaptiveRewardLearning:
    def __init__(self, env, policy_network, reward_model, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.env = env
        self.policy_network = policy_network
        self.reward_model = reward_model
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(list(policy_network.parameters()) + list(reward_model.parameters()), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            # 自适应奖励模型的损失函数
            reward_predictions = self.reward_model(torch.cat([states, actions.unsqueeze(1)], dim=1))
            reward_loss = torch.mean((returns - reward_predictions) ** 2)

            total_loss = loss + reward_loss  # 合并策略损失和奖励损失

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            values = []
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            total_reward = 0

            while not done:
                state = state.unsqueeze(0)
                dist = self.policy_network(state)
                m = Categorical(dist)
                action = m.sample()
                log_prob = m.log_prob(action)
                value = dist[0, action]  # 简化处理，实际应使用 value 网络

                next_state, reward, done, _ = self.env.step(action.item())
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(value)

                state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward

            # 计算奖励和优势
            returns = self.compute_returns(rewards, dones, values, value)
            advantages = self.compute_advantages(rewards, dones, values, value)

            # 更新策略和奖励模型
            self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

            print(f"Episode {episode}, Total Reward: {total_reward}")

# 创建环境
env = gym.make('CartPole-v1')
env.seed(42)

# 初始化DeepseekAPIClient，用于模拟调用大模型接口
api_client = DeepseekAPIClient(api_url="https://api.deepseek.com/v1", api_key="your_api_key")

# 初始化策略网络和奖励生成模型
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)
reward_model = AdaptiveRewardModel(input_dim + 1, 1)  # state + action as input

# 初始化自适应奖励学习策略
trainer = AdaptiveRewardLearning(env, policy_network, reward_model)

# 开始训练
num_episodes = 100
trainer.train(num_episodes)

# 模型评估
def evaluate(env, policy_network, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(env, policy_network)
print(f"平均评估奖励: {avg_reward}")


# 例【4-6】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import requests

# 模拟一个简单的Deepseek API接口请求类，用于分布式获取大模型的结果
class DeepseekAPIClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def get_model_output(self, input_data):
        # 模拟发送请求并获取模型输出
        response = requests.post(self.api_url, json={"input": input_data, "key": self.api_key})
        return response.json()

# 策略网络模型定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# 奖励塑形模型，用于在稀疏奖励环境中引入中间奖励
class RewardShapingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardShapingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, state_action):
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)  # 输出增强的奖励
        return reward

# 奖励学习类，结合奖励塑形进行训练
class RewardLearning:
    def __init__(self, env, policy_network, reward_model, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.env = env
        self.policy_network = policy_network
        self.reward_model = reward_model
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(list(policy_network.parameters()) + list(reward_model.parameters()), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            # 使用奖励塑形模型来调整奖励信号
            reward_predictions = self.reward_model(torch.cat([states, actions.unsqueeze(1)], dim=1))
            reward_loss = torch.mean((returns - reward_predictions) ** 2)

            total_loss = loss + reward_loss  # 合并策略损失和奖励塑形损失

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            values = []
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            total_reward = 0

            while not done:
                state = state.unsqueeze(0)
                dist = self.policy_network(state)
                m = Categorical(dist)
                action = m.sample()
                log_prob = m.log_prob(action)
                value = dist[0, action]  # 简化处理，实际应使用 value 网络

                next_state, reward, done, _ = self.env.step(action.item())
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(value)

                state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward

            # 计算奖励和优势
            returns = self.compute_returns(rewards, dones, values, value)
            advantages = self.compute_advantages(rewards, dones, values, value)

            # 更新策略和奖励模型
            self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

            print(f"Episode {episode}, Total Reward: {total_reward}")

# 创建环境
env = gym.make('CartPole-v1')
env.seed(42)

# 初始化DeepseekAPIClient，用于模拟调用大模型接口
api_client = DeepseekAPIClient(api_url="https://api.deepseek.com/v1", api_key="your_api_key")

# 初始化策略网络和奖励塑形模型
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)
reward_model = RewardShapingModel(input_dim + 1, 1)  # state + action as input

# 初始化奖励学习策略
trainer = RewardLearning(env, policy_network, reward_model)

# 开始训练
num_episodes = 100
trainer.train(num_episodes)

# 模型评估
def evaluate(env, policy_network, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(env, policy_network)
print(f"平均评估奖励: {avg_reward}")


# 例【4-7】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import random
import time
import concurrent.futures

# 策略网络模型定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# 经验回放类
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 多任务并行训练类
class MultiTaskRL:
    def __init__(self, envs, policy_network, gamma=0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.envs = envs  # 多任务环境
        self.policy_network = policy_network
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_task(self, env):
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        values = []
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            state = state.unsqueeze(0)
            dist = self.policy_network(state)
            m = Categorical(dist)
            action = m.sample()
            log_prob = m.log_prob(action)
            value = dist[0, action]  # Simplified, should be from value network

            next_state, reward, done, _ = env.step(action.item())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(value)

            state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

        returns = self.compute_returns(rewards, dones, values, value)
        advantages = self.compute_advantages(rewards, dones, values, value)

        self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

        return total_reward

    def train(self, num_episodes):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for episode in range(num_episodes):
                results = executor.map(self.train_task, self.envs)
                total_rewards = list(results)

                avg_reward = np.mean(total_rewards)
                print(f"Episode {episode}, Average Reward: {avg_reward}")

# 创建多个环境用于并行训练
envs = [gym.make('CartPole-v1') for _ in range(4)]  # 多任务环境
for env in envs:
    env.seed(42)

# 初始化策略网络
input_dim = envs[0].observation_space.shape[0]
output_dim = envs[0].action_space.n
policy_network = PolicyNetwork(input_dim, output_dim)

# 初始化多任务训练器
trainer = MultiTaskRL(envs, policy_network)

# 开始训练
num_episodes = 100
trainer.train(num_episodes)

# 模型评估
def evaluate(envs, policy_network, num_episodes=10):
    total_rewards = []
    for env in envs:
        total_reward = 0
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        while not done:
            state = state.unsqueeze(0)
            dist = policy_network(state)
            action = dist.argmax(dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

avg_reward = evaluate(envs, policy_network)
print(f"平均评估奖励: {avg_reward}")