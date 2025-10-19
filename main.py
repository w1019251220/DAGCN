import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from scipy.io import loadmat
import random

############################################
# 1. 数据加载与图构建相关函数
############################################

def load_data(data):
    """
    加载图、特征和标签
    返回值：图的列表（同构图和不同关系图）、特征矩阵、标签数组
    """
    prefix = 'data/'
    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A  # numpy.array类型，形状为[N, nfeat]
        # 加载预处理好的邻接列表（字典格式：key为节点id，value为邻居列表）
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    elif data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    return [homo, relation1, relation2, relation3], feat_data, labels


def build_adj_matrix(adj_list, num_nodes):
    """
    根据邻接列表（字典）构建图的邻接矩阵（密集表示）
    :param adj_list: dict，{node: [neighbor1, neighbor2, ...]}
    :param num_nodes: 节点总数
    :return: torch.FloatTensor类型的邻接矩阵，形状为 (num_nodes, num_nodes)
    """
    adj = np.zeros((num_nodes, num_nodes))
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            # 注意：这里假设节点索引在0~num_nodes-1之间
            adj[node, neighbor] = 1
    return torch.FloatTensor(adj)


############################################
# 2. GAT模型相关代码
############################################

class GraphAttentionLayer(nn.Module):
    """
    单层图注意力网络
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param dropout: dropout概率
        :param alpha: LeakyReLU参数
        :param concat: 是否拼接（True: 中间层使用拼接，False: 输出层直接输出）
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 线性变换权重
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力参数
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: 输入节点特征矩阵，形状 (N, in_features)
        :param adj: 邻接矩阵，形状 (N, N)
        :return: 节点新的特征表示，形状 (N, out_features)
        """
        Wh = torch.mm(h, self.W)  # (N, out_features)
        N = Wh.size()[0]

        # 计算注意力机制的所有组合：先将Wh复制扩展，再连接计算注意力分数
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1),
                             Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # 计算注意力系数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 将不相邻节点的注意力分数设为负无穷大（这里假设adj中0表示无边，1表示有边）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """GAT模型：基于多头注意力机制的图神经网络"""
        super(GAT, self).__init__()
        self.dropout = dropout

        # 多头图注意力层（中间层：拼接各头的输出）
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 输出层：使用一个头，直接输出类别概率分布（不拼接）
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # dropout处理输入
        x = F.dropout(x, self.dropout, training=self.training)
        # 拼接各个头的输出
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


############################################
# 3. 实时检测模块相关函数
############################################

def get_related_nodes(new_node, full_adj_list):
    """
    根据新节点（评论）在同构图中提取相关节点集合：包括自身及其直接邻居
    :param new_node: 新节点的索引（整数）
    :param full_adj_list: 同构图的邻接列表（字典格式）
    :return: 节点集合（set）
    """
    related = set()
    related.add(new_node)
    if new_node in full_adj_list:
        neighbors = full_adj_list[new_node]
        for nb in neighbors:
            related.add(nb)
    return related


def extract_local_subgraph(new_node, full_adj_list, features):
    """
    根据新节点抽取局部子图，并提取子图对应的特征矩阵和新的邻接矩阵
    :param new_node: 新节点索引
    :param full_adj_list: 同构图邻接列表
    :param features: 全图节点特征矩阵 (Tensor, shape: [N, nfeat])
    :return: sub_adj: 子图的邻接矩阵 (Tensor, shape: [n_sub, n_sub])
             sub_features: 子图的特征矩阵 (Tensor, shape: [n_sub, nfeat])
             mapping: 原图节点到子图节点的映射字典
    """
    related_nodes = list(get_related_nodes(new_node, full_adj_list))
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(related_nodes)}
    sub_features = features[torch.LongTensor(related_nodes)]
    num_sub = len(related_nodes)
    # 构造子图的邻接矩阵（密集表示）
    sub_adj = np.zeros((num_sub, num_sub))
    for old_node in related_nodes:
        if old_node in full_adj_list:
            for nb in full_adj_list[old_node]:
                if nb in mapping:  # 只保留在子图中的边
                    sub_adj[mapping[old_node], mapping[nb]] = 1
    sub_adj = torch.FloatTensor(sub_adj)
    return sub_adj, sub_features, mapping


# 简单的缓存机制，记录节点的最新嵌入表示
node_cache = {}  # 全局字典，key为节点id，value为嵌入向量


def update_node_cache(mapping, embeddings):
    """
    更新缓存：将子图中计算出的节点嵌入保存到全局缓存中
    :param mapping: 原图节点到子图节点索引的映射字典
    :param embeddings: 子图中节点嵌入表示，形状 (n_sub, embed_dim)
    """
    for old_node, sub_idx in mapping.items():
        node_cache[old_node] = embeddings[sub_idx].detach()  # detach避免梯度追踪


def real_time_inference(new_node, full_adj_list, features, model):
    """
    针对新节点（新评论）实现实时检测：
      1. 提取新节点相关的局部子图及其特征
      2. 使用GAT模型进行推理，输出预测概率
      3. 更新缓存中的节点表示
    :param new_node: 新节点索引（评论）
    :param full_adj_list: 同构图邻接列表
    :param features: 全图特征矩阵 (Tensor)
    :param model: 已训练好的GAT模型
    :return: 预测结果（log_softmax后的概率分布）
    """
    model.eval()
    sub_adj, sub_features, mapping = extract_local_subgraph(new_node, full_adj_list, features)
    # 进行前向传播得到局部子图中所有节点的嵌入
    with torch.no_grad():
        sub_output = model(sub_features, sub_adj)
    # 更新缓存
    update_node_cache(mapping, sub_output)
    # 返回新节点对应的预测结果（利用映射获得子图中对应位置）
    new_node_idx = mapping[new_node]
    prediction = sub_output[new_node_idx]
    return prediction


############################################
# 4. 对抗训练模块相关函数
############################################

def generate_adversarial_sample(features, epsilon):
    """
    生成对抗样本：在原始特征上添加小幅随机噪声
    :param features: 原始特征矩阵 (Tensor)
    :param epsilon: 扰动幅度（例如 0.01）
    :return: 加噪声后的特征矩阵
    """
    noise = epsilon * torch.randn_like(features)
    adv_features = features + noise
    return adv_features


def compute_loss(predictions, labels, mask=None):
    """
    计算交叉熵损失。mask用于仅计算部分节点的损失（例如训练集）
    :param predictions: 模型输出（log_softmax后的Tensor）
    :param labels: 标签 (Tensor)
    :param mask: Bool类型Tensor，指示参与损失计算的节点
    :return: 损失标量
    """
    if mask is not None:
        loss = F.nll_loss(predictions[mask], labels[mask])
    else:
        loss = F.nll_loss(predictions, labels)
    return loss


def adversarial_training_step(features, adj, labels, model, optimizer, alpha, beta, epsilon, mask=None):
    """
    对抗训练单步：计算原始样本损失、生成对抗样本并计算对抗损失，然后反向传播更新模型参数
    :param features: 原始特征矩阵 (Tensor)
    :param adj: 邻接矩阵 (Tensor)
    :param labels: 标签 (Tensor)
    :param model: GAT模型
    :param optimizer: 优化器
    :param alpha: 原始样本损失权重
    :param beta: 对抗样本损失权重
    :param epsilon: 对抗扰动幅度
    :param mask: 可选，训练时的节点掩码
    :return: 总损失
    """
    model.train()
    optimizer.zero_grad()

    # 正常样本损失
    predictions = model(features, adj)
    normal_loss = compute_loss(predictions, labels, mask)

    # 生成对抗样本（在特征上添加扰动）
    adv_features = generate_adversarial_sample(features, epsilon)
    adv_predictions = model(adv_features, adj)
    adv_loss = compute_loss(adv_predictions, labels, mask)

    total_loss = alpha * normal_loss + beta * adv_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


############################################
# 5. 主函数：加载数据、训练模型及实时检测示例
############################################

def main():
    # 选择数据集，例如'yelp'或'amazon'
    dataset = 'yelp'
    graphs, feat_data, labels_np = load_data(dataset)

    # 这里使用同构图（homo）构建邻接矩阵用于GAT模型
    homo_adj_list = graphs[0]
    num_nodes = feat_data.shape[0]
    adj = build_adj_matrix(homo_adj_list, num_nodes)  # 邻接矩阵，形状 (num_nodes, num_nodes)

    # 将特征和标签转换为torch张量
    features = torch.FloatTensor(feat_data)  # shape: [N, nfeat]
    labels = torch.LongTensor(labels_np)  # shape: [N]

    # 定义模型超参数
    nfeat = features.shape[1]
    nhid = 8
    nclass = int(labels.max().item() + 1)  # 假设标签从0开始
    dropout = 0.6
    alpha = 0.2
    nheads = 8
    learning_rate = 0.005
    weight_decay = 5e-4
    epochs = 200
    epsilon = 0.01  # 对抗扰动幅度
    adv_alpha = 1.0  # 正常损失权重
    adv_beta = 0.5  # 对抗损失权重

    # 如果需要可以定义训练集、验证集、测试集的mask；这里为了简单，全部参与训练
    mask = None

    # 初始化GAT模型和优化器
    model = GAT(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout, alpha=alpha, nheads=nheads)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 对抗训练的训练循环
    print("开始对抗训练...")
    for epoch in range(epochs):
        loss = adversarial_training_step(features, adj, labels, model, optimizer, adv_alpha, adv_beta, epsilon, mask)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # 模型训练结束后，演示实时检测模块：假设有新评论对应的节点索引new_node
    # 为演示，这里随机选择一个节点作为新到评论
    new_node = random.randint(0, num_nodes - 1)
    prediction = real_time_inference(new_node, homo_adj_list, features, model)
    # prediction为log_softmax后的输出，可使用torch.exp转换为概率分布
    pred_prob = torch.exp(prediction)
    pred_class = torch.argmax(pred_prob).item()
    print(f"新节点（评论）索引 {new_node} 的预测类别：{pred_class}, 概率分布：{pred_prob.cpu().numpy()}")


if __name__ == '__main__':
    main()
