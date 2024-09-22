
'''
    查看CUDA的版本号
'''
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda) # 查看CUDA的版本号

'''
    关于MultiheadAttention 张量问题的测试
'''

import torch
import torch.nn as nn

## 假设嵌入维度为512，头数为8
embed_dim = 512
num_heads = 8
dropout = 0.1

## 初始化多头注意力层
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

## 创建一些模拟的输入数据
query = torch.randn(10, 32, embed_dim)  # (batch_size, sequence_length, embed_dim)
key = torch.randn(10, 48, embed_dim)    # (batch_size, sequence_length, embed_dim)
value = torch.randn(10, 48, embed_dim)  # (batch_size, sequence_length, embed_dim)

## 计算多头注意力
output, attn_output_weights = multihead_attn(query, key, value)

print(output.shape)  # 应该输出 (10, 32, 512)，与query的shape一致

'''
    关于图神经网络聚合
'''

import torch
from torch_geometric.nn import MessagePassing

class SimpleGNN(MessagePassing):
    def __init__(self):
        super(SimpleGNN, self).__init__(aggr='add')  # 使用加法聚合

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  # 直接使用邻居节点的特征作为消息

    def update(self, aggr_out):
        return aggr_out.relu()  # 更新节点特征

# 示例使用
x = torch.rand((4, 3))  # 4 个节点，每个节点 3 个特征
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]], dtype=torch.long)

gnn = SimpleGNN()
output = gnn(x, edge_index)
print("更新后的节点特征：")
print(output)

'''
    关于Linear输入输出维度的问题
    没有强制要求,linear输入为2维矩阵,多维度也可
'''
nn_liuchao = nn.Linear(64, 32)
input_liuchao = torch.torch.randn(2, 2, 64)
output_liuchao = nn_liuchao(input_liuchao)
print(output_liuchao.shape)