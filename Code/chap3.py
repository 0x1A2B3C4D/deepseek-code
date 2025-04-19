# 例【3-1】
# -*- coding: utf-8 -*-
import numpy as np
from gensim.models import Word2Vec
from glove import Corpus, Glove

# 构造简单的中文语料库，每个列表元素表示一句话（已分词）
sentences = [
    ["我", "喜欢", "学习", "深度", "学习"],
    ["深度", "学习", "非常", "有趣"],
    ["我", "喜欢", "用", "Python", "编程"],
    ["编程", "和", "学习", "都", "很", "重要"],
    ["深度", "学习", "和", "机器", "学习", "都是", "热门", "方向"],
    ["我", "热爱", "编程", "与", "学习"]
]

#########################################
# 1. Word2Vec 实现
#########################################
# 使用 gensim 训练 Word2Vec 模型（采用 Skip-gram 模型）
w2v_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=1, sg=1)

# 输出词“学习”的词向量
print("【Word2Vec】词“学习”的词向量：")
print(w2v_model.wv["学习"])

# 输出与“学习”最相似的前5个词
print("\n【Word2Vec】与“学习”最相似的前5个词：")
similar_words_w2v = w2v_model.wv.most_similar("学习", topn=5)
for word, score in similar_words_w2v:
    print(f"{word}: {score:.4f}")

#########################################
# 2. GloVe 实现
#########################################
# 构造语料库（Corpus），并统计共现矩阵（窗口大小设为2）
corpus = Corpus()
corpus.fit(sentences, window=2)

# 使用 glove-python 训练 GloVe 模型
glove_model = Glove(no_components=50, learning_rate=0.05)
glove_model.fit(corpus.matrix, epochs=30, no_threads=1, verbose=False)
glove_model.add_dictionary(corpus.dictionary)

# 输出词“学习”的词向量
print("\n【GloVe】词“学习”的词向量：")
vec_learning = glove_model.word_vectors[glove_model.dictionary["学习"]]
print(vec_learning)

# 定义函数计算余弦相似度，并找出与指定词最相似的词
def most_similar_glove(word, topn=5):
    if word not in glove_model.dictionary:
        return []
    idx = glove_model.dictionary[word]
    query_vec = glove_model.word_vectors[idx]
    sims = {}
    for other_word, other_idx in glove_model.dictionary.items():
        if other_word == word:
            continue
        vec = glove_model.word_vectors[other_idx]
        cosine = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        sims[other_word] = cosine
    sims_sorted = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return sims_sorted[:topn]

# 输出与“学习”最相似的前5个词（GloVe）
print("\n【GloVe】与“学习”最相似的前5个词：")
similar_words_glove = most_similar_glove("学习", topn=5)
for word, score in similar_words_glove:
    print(f"{word}: {score:.4f}")


# 例【3-2】
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 生成正弦波时间序列数据
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps)

# 2. 设置超参数
input_size = 1        # 每个时间步的输入维度
hidden_size = 32      # 隐藏层维度
num_layers = 1        # RNN 层数
output_size = 1       # 输出维度（预测下一个值）
seq_length = 20       # 序列长度
num_epochs = 100      # 训练轮数
learning_rate = 0.01  # 学习率

# 3. 构造滑动窗口序列数据
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

x, y = create_sequences(data, seq_length)
# 调整数据形状，x 的形状为 (样本数, 序列长度, 输入维度)
x = x.reshape(-1, seq_length, 1)
y = y.reshape(-1, 1)

# 转换为 torch tensor
x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y)

# 4. 定义 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        # 使用 batch_first=True 表示输入数据的形状为 (batch, seq_length, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_length, input_size)
        out, h = self.rnn(x)
        # 取最后一个时间步的输出，送入全连接层预测
        out = self.fc(out[:, -1, :])
        return out

model = SimpleRNN(input_size, hidden_size, num_layers, output_size)

# 5. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# 7. 模型预测
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).detach().numpy()

# 输出部分预测结果与实际值对比
print("\n部分预测结果：")
for i in range(5):
    actual = y[i][0]
    pred = predicted[i][0]
    print(f"实际值: {actual:.4f}, 预测值: {pred:.4f}")