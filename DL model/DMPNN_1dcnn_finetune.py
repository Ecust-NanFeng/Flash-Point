import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import json
import os
from tqdm import tqdm
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import copy
from collections import Counter

warnings.filterwarnings('ignore')


# ===================== 分子特征提取 =====================
class MolecularFeatures:
    """分子特征提取类"""

    @staticmethod
    def atom_features(atom):
        """原子特征"""
        return np.array([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass() * 0.01,
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            int(atom.IsInRingSize(8))
        ], dtype=np.float32)

    @staticmethod
    def bond_features(bond):
        """键特征"""
        if bond is None:
            return np.zeros(7, dtype=np.float32)

        bond_type = bond.GetBondType()
        return np.array([
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo())
        ], dtype=np.float32)

    @staticmethod
    def smiles_to_graph(smiles):
        """将SMILES转换为图数据"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(MolecularFeatures.atom_features(atom))
        atom_features = torch.FloatTensor(atom_features)

        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
            bond_feat = MolecularFeatures.bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])

        if len(edge_indices) == 0:
            edge_indices = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((0, 7), dtype=torch.float)
        else:
            edge_indices = torch.LongTensor(edge_indices).t()
            edge_features = torch.FloatTensor(edge_features)

        graph_data = Data(
            x=atom_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
            num_nodes=atom_features.size(0)
        )

        return graph_data


# ===================== 1D-CNN描述符编码器 =====================
class CNN1DDescriptorEncoder(nn.Module):
    """1D-CNN描述符编码器"""

    def __init__(self, descriptor_dim, output_dim,
                 filters1=64, filters2=128, kernel_size=7,
                 pool_size=2, dropout=0.1):
        super(CNN1DDescriptorEncoder, self).__init__()

        self.descriptor_dim = descriptor_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=filters1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(filters1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)

        self.conv2 = nn.Conv1d(
            in_channels=filters1,
            out_channels=filters2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(filters2)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size)

        pooled_dim = descriptor_dim // (pool_size ** 2)
        flattened_dim = filters2 * pooled_dim

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


# ===================== 多头注意力机制 =====================
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        if d_model % num_heads != 0:
            d_model = ((d_model + num_heads - 1) // num_heads) * num_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(attention_output)
        output = self.layer_norm(output + residual)

        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)

        return output


# ===================== 交叉注意力融合模块 =====================
class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, graph_dim, descriptor_dim, d_model, num_heads, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        if d_model % num_heads != 0:
            d_model = ((d_model + num_heads - 1) // num_heads) * num_heads

        self.d_model = d_model

        self.graph_proj = nn.Linear(graph_dim, d_model)
        self.descriptor_proj = nn.Linear(descriptor_dim, d_model)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.fusion_weights = nn.Parameter(torch.ones(2))

        self.fusion_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_repr, descriptor_repr):
        batch_size = graph_repr.size(0)

        graph_feat = self.graph_proj(graph_repr)
        desc_feat = self.descriptor_proj(descriptor_repr)

        combined_features = torch.stack([graph_feat, desc_feat], dim=1)

        attended_features = self.cross_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features
        )

        normalized_weights = F.softmax(self.fusion_weights, dim=0)
        fused_repr = torch.sum(attended_features * normalized_weights.view(1, 2, 1), dim=1)

        output = self.fusion_ffn(fused_repr)
        output = self.layer_norm(output + fused_repr)
        output = self.dropout(output)

        return output


# ===================== D-MPNN模型实现 =====================
class DMPNNConv(MessagePassing):
    """D-MPNN消息传递层"""

    def __init__(self, hidden_size, edge_dim):
        super(DMPNNConv, self).__init__(aggr='add')
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim

        self.message_net = nn.Sequential(
            nn.Linear(hidden_size + edge_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_net(message_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class DMPNN(nn.Module):
    """D-MPNN模型"""

    def __init__(self, atom_dim=15, bond_dim=7, descriptor_dim=34, hidden_size=300, depth=3,
                 ffn_hidden_size=300, dropout=0.1, output_size=1, use_descriptors=True,
                 num_attention_heads=8, use_cross_attention=True,
                 cnn_filters1=64, cnn_filters2=128, cnn_kernel_size=7):
        super(DMPNN, self).__init__()

        self.hidden_size = hidden_size
        self.depth = depth
        self.use_descriptors = use_descriptors
        self.descriptor_dim = descriptor_dim
        self.use_cross_attention = use_cross_attention
        self.num_attention_heads = num_attention_heads

        self.atom_encoder = nn.Linear(atom_dim, hidden_size)

        if self.use_descriptors and descriptor_dim > 0:
            self.descriptor_encoder = CNN1DDescriptorEncoder(
                descriptor_dim=descriptor_dim,
                output_dim=hidden_size // 2,
                filters1=cnn_filters1,
                filters2=cnn_filters2,
                kernel_size=cnn_kernel_size,
                dropout=dropout
            )

            if self.use_cross_attention:
                attention_dim = hidden_size
                if attention_dim % num_attention_heads != 0:
                    attention_dim = ((
                                             attention_dim + num_attention_heads - 1) // num_attention_heads) * num_attention_heads

                self.attention_fusion = CrossAttentionFusion(
                    graph_dim=hidden_size,
                    descriptor_dim=hidden_size // 2,
                    d_model=attention_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout
                )
                fusion_dim = attention_dim
            else:
                fusion_dim = hidden_size + hidden_size // 2
                self.attention_fusion = None
        else:
            self.descriptor_encoder = None
            self.attention_fusion = None
            fusion_dim = hidden_size

        self.conv_layers = nn.ModuleList([
            DMPNNConv(hidden_size, bond_dim) for _ in range(depth)
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, output_size)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)
        x = self.activation(x)

        for conv_layer in self.conv_layers:
            x_new = conv_layer(x, edge_index, edge_attr)
            x = x + x_new
            x = self.activation(x)
            x = self.dropout(x)

        from torch_geometric.nn import global_add_pool
        graph_repr = global_add_pool(x, batch)

        if (self.use_descriptors and
                self.descriptor_encoder is not None and
                hasattr(data, 'mol_descriptors') and
                data.mol_descriptors is not None):

            mol_desc = data.mol_descriptors
            batch_size = graph_repr.size(0)

            if mol_desc.numel() == batch_size * self.descriptor_dim:
                mol_desc = mol_desc.view(batch_size, self.descriptor_dim)
            else:
                if mol_desc.dim() == 1 and mol_desc.size(0) == self.descriptor_dim:
                    mol_desc = mol_desc.unsqueeze(0).repeat(batch_size, 1)
                else:
                    expected_size = batch_size * self.descriptor_dim
                    if mol_desc.numel() > expected_size:
                        mol_desc = mol_desc[:expected_size]
                    else:
                        padding = expected_size - mol_desc.numel()
                        mol_desc = torch.cat([mol_desc, torch.zeros(padding, device=mol_desc.device)])
                    mol_desc = mol_desc.view(batch_size, self.descriptor_dim)

            descriptor_repr = self.descriptor_encoder(mol_desc)

            if self.use_cross_attention and self.attention_fusion is not None:
                combined_repr = self.attention_fusion(graph_repr, descriptor_repr)
            else:
                combined_repr = torch.cat([graph_repr, descriptor_repr], dim=-1)
        else:
            combined_repr = graph_repr

        output = self.ffn(combined_repr)

        return output


# ===================== 微调数据集类 =====================
class FineTuneDataset:
    """微调数据集处理类 - 支持基于type的分组划分"""

    def __init__(self, csv_path, smiles_col='smiles', target_col='FP (K)',
                 type_col='type', descriptor_cols=None, use_descriptors=True,
                 pretrain_descriptor_scaler=None):
        """初始化微调数据集"""
        self.data = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.type_col = type_col
        self.descriptor_cols = descriptor_cols
        self.use_descriptors = use_descriptors
        self.pretrain_descriptor_scaler = pretrain_descriptor_scaler

        # 检查type列是否存在
        if type_col not in self.data.columns:
            raise ValueError(f"列'{type_col}'不存在，请确保数据已经过增强并包含type列")

        # 删除缺失值
        self.data = self.data.dropna(subset=[smiles_col, target_col, type_col])

        if (self.use_descriptors and
                self.descriptor_cols is not None and
                len(self.descriptor_cols) > 0):

            if hasattr(self.descriptor_cols, 'tolist'):
                self.descriptor_cols = self.descriptor_cols.tolist()

            missing_cols = [col for col in self.descriptor_cols if col not in self.data.columns]
            if missing_cols:
                print(f"警告: 以下描述符列不存在: {missing_cols}")
                self.descriptor_cols = [col for col in self.descriptor_cols if col in self.data.columns]

            if len(self.descriptor_cols) > 0:
                self.data = self.data.dropna(subset=self.descriptor_cols)
                print(f"使用 {len(self.descriptor_cols)} 个描述符")
            else:
                print("警告: 没有有效的描述符列")
                self.use_descriptors = False
        else:
            self.use_descriptors = False

        print(f"加载了 {len(self.data)} 条微调数据")
        print(f"唯一分子数（type数量）: {self.data[type_col].nunique()}")

        self.graphs = []
        self.targets = []
        self.descriptor_data = []
        self.types = []  # 存储每个样本的type

        print("转换SMILES为图数据...")
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            graph = MolecularFeatures.smiles_to_graph(row[smiles_col])
            if graph is not None:
                self.graphs.append(graph)
                self.targets.append(row[target_col])
                self.types.append(row[type_col])

                if self.use_descriptors and len(self.descriptor_cols) > 0:
                    descriptors = row[self.descriptor_cols].values.astype(np.float32)
                    self.descriptor_data.append(descriptors)
                    graph.mol_descriptors = torch.FloatTensor(descriptors)
                else:
                    graph.mol_descriptors = None

        print(f"成功转换 {len(self.graphs)} 条数据")

        # 创建type到索引的映射
        self.types = np.array(self.types)
        self.type_to_indices = {}
        for idx, mol_type in enumerate(self.types):
            if mol_type not in self.type_to_indices:
                self.type_to_indices[mol_type] = []
            self.type_to_indices[mol_type].append(idx)

        print(f"总共有 {len(self.type_to_indices)} 个唯一分子（type）")

        # 验证每个type的样本数量
        type_counts = [len(indices) for indices in self.type_to_indices.values()]
        print(
            f"每个分子的SMILES变体数量统计: min={min(type_counts)}, max={max(type_counts)}, mean={np.mean(type_counts):.1f}")

        if len(self.targets) > 0:
            self.targets = torch.FloatTensor(self.targets)
            self.target_scaler = StandardScaler()
            self.targets = torch.FloatTensor(
                self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
            )
        else:
            self.targets = torch.FloatTensor()
            self.target_scaler = StandardScaler()

        # 标准化分子描述符
        if self.use_descriptors and len(self.descriptor_data) > 0:
            descriptor_array = np.array(self.descriptor_data)

            if self.pretrain_descriptor_scaler is not None:
                print("使用预训练的描述符标准化器")
                self.descriptor_scaler = self.pretrain_descriptor_scaler
                scaled_descriptors = self.descriptor_scaler.transform(descriptor_array)
            else:
                print("创建新的描述符标准化器")
                self.descriptor_scaler = StandardScaler()
                scaled_descriptors = self.descriptor_scaler.fit_transform(descriptor_array)

            for i, scaled_desc in enumerate(scaled_descriptors):
                self.graphs[i].mol_descriptors = torch.FloatTensor(scaled_desc)

            self.descriptor_dim = descriptor_array.shape[1]
            print(f"分子描述符维度: {self.descriptor_dim}")
        else:
            self.descriptor_scaler = None
            self.descriptor_dim = 0
            self.use_descriptors = False

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]

    def get_splits(self, train_idx, test_idx):
        """根据索引获取数据划分"""
        train_graphs = [self.graphs[i] for i in train_idx]
        train_targets = self.targets[train_idx]

        test_graphs = [self.graphs[i] for i in test_idx]
        test_targets = self.targets[test_idx]

        return (train_graphs, train_targets), (test_graphs, test_targets)
    def applicability_domain(self):

        graphs = self.graphs
        targets = self.targets

        return (graphs, targets)

    def get_grouped_splits_train_test(self, test_size=0.2, random_state=42):
        """基于type进行分组的训练/测试集划分"""
        unique_types = np.array(list(self.type_to_indices.keys()))

        # 对type进行划分
        from sklearn.model_selection import train_test_split
        train_types, test_types = train_test_split(
            unique_types, test_size=test_size, random_state=random_state
        )

        # 将type映射回样本索引
        train_indices = []
        for mol_type in train_types:
            train_indices.extend(self.type_to_indices[mol_type])

        test_indices = []
        for mol_type in test_types:
            test_indices.extend(self.type_to_indices[mol_type])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # 验证划分
        train_types_set = set(self.types[train_indices])
        test_types_set = set(self.types[test_indices])
        assert len(train_types_set & test_types_set) == 0, "训练集和测试集存在相同type！"

        print(f"\n基于type的数据划分:")
        print(f"  训练集: {len(train_types)} types ({len(train_indices)} 样本)")
        print(f"  测试集: {len(test_types)} types ({len(test_indices)} 样本)")

        return train_indices, test_indices

    def get_grouped_kfold_splits(self, n_splits=5, random_state=42):
        """基于type进行分组的K折交叉验证划分"""
        unique_types = np.array(list(self.type_to_indices.keys()))
        print(f"\n基于type进行分组K折划分，唯一分子数: {len(unique_types)}")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []

        for fold_idx, (train_val_types_idx, test_types_idx) in enumerate(kfold.split(unique_types)):
            train_val_types = unique_types[train_val_types_idx]
            test_types = unique_types[test_types_idx]

            # 在训练验证type中进一步划分验证集
            n_val_types = len(train_val_types) // (n_splits - 1)
            val_types = train_val_types[:n_val_types]
            train_types = train_val_types[n_val_types:]

            # 将type映射回实际的样本索引
            train_indices = []
            for mol_type in train_types:
                train_indices.extend(self.type_to_indices[mol_type])

            val_indices = []
            for mol_type in val_types:
                val_indices.extend(self.type_to_indices[mol_type])

            test_indices = []
            for mol_type in test_types:
                test_indices.extend(self.type_to_indices[mol_type])

            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)

            # 验证划分
            train_types_set = set(self.types[train_indices])
            val_types_set = set(self.types[val_indices])
            test_types_set = set(self.types[test_indices])

            assert len(train_types_set & val_types_set) == 0, "训练集和验证集存在相同type！"
            assert len(train_types_set & test_types_set) == 0, "训练集和测试集存在相同type！"
            assert len(val_types_set & test_types_set) == 0, "验证集和测试集存在相同type！"

            print(f"  折 {fold_idx + 1}: "
                  f"训练type={len(train_types_set)} ({len(train_indices)}样本), "
                  f"验证type={len(val_types_set)} ({len(val_indices)}样本), "
                  f"测试type={len(test_types_set)} ({len(test_indices)}样本)")

            splits.append({
                'train_idx': train_indices,
                'val_idx': val_indices,
                'test_idx': test_indices,
                'train_types': train_types_set,
                'val_types': val_types_set,
                'test_types': test_types_set
            })

        return splits


# ===================== 超参数搜索空间 =====================
class HyperparameterSpace:
    """超参数搜索空间定义"""

    @staticmethod
    def get_safe_tunable_space():
        """✅ 获取可以安全调节的超参数空间（推荐用于微调）"""
        return {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64, 128],
            'dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
        }

    @staticmethod
    def get_compact_safe_space():
        """✅ 获取精简的安全超参数空间（快速测试）"""
        return {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
            'batch_size': [16, 32],
            'dropout': [0.0,0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15]
        }

    @staticmethod
    def get_partial_tunable_space():
        """⚠️ 获取部分可调节的超参数空间"""
        return {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
            'batch_size': [16, 32, 64, 128],
            'dropout': [0.0, 0.05, 0.1, 0.15, 0.2],
            'ffn_hidden_size': [128, 256, 300, 512]
        }

    @staticmethod
    def get_compact_partial_space():
        """⚠️ 获取精简的部分可调节空间"""
        return {
            'learning_rate': [5e-5, 1e-4, 5e-4],
            'batch_size': [32, 64],
            'dropout': [0.0, 0.05, 0.1],
            'ffn_hidden_size': [256, 300]
        }


# ===================== 微调训练器 =====================
class FineTuneTrainer:
    """微调训练器"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            output = self.model(batch)
            loss = criterion(output.squeeze(), batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, dataloader, criterion, scaler=None, return_details=False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        all_indices = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = criterion(output.squeeze(), batch.y)

                total_loss += loss.item()

                batch_predictions = output.squeeze().cpu().numpy()
                batch_targets = batch.y.cpu().numpy()

                # 确保是1维数组
                if batch_predictions.ndim == 0:
                    batch_predictions = batch_predictions.reshape(-1)
                if batch_targets.ndim == 0:
                    batch_targets = batch_targets.reshape(-1)

                predictions.extend(batch_predictions.tolist())
                targets.extend(batch_targets.tolist())

                # 记录批次索引
                batch_size = len(batch_predictions)
                start_idx = batch_idx * dataloader.batch_size
                all_indices.extend(range(start_idx, start_idx + batch_size))

        predictions = np.array(predictions)
        targets = np.array(targets)

        # 保存标准化后的值
        predictions_scaled = predictions.copy()
        targets_scaled = targets.copy()

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        result = {
            'loss': total_loss / len(dataloader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }

        if return_details:
            result.update({
                'predictions': predictions,
                'targets': targets,
                'predictions_scaled': predictions_scaled,
                'targets_scaled': targets_scaled,
                'indices': all_indices
            })

        return result

    def train_with_early_stopping(self, train_loader, val_loader, optimizer, criterion,
                                  epochs=100, patience=15, scaler=None, verbose=False):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        train_history = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics = self.evaluate(val_loader, criterion, scaler)
            val_loss = val_metrics['loss']

            train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2']
            })

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val RMSE: {val_metrics['rmse']:.4f}, "
                      f"R²: {val_metrics['r2']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  早停在第 {epoch + 1} epoch")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return train_history, val_metrics


# ===================== 超参数优化器 =====================
class HyperparameterOptimizer:
    """超参数优化器 - 智能处理可调节和固定的超参数"""

    def __init__(self, pretrain_data, finetune_dataset, device='cuda'):
        self.pretrain_data = pretrain_data
        self.finetune_dataset = finetune_dataset
        self.device = device

        # 从预训练模型加载基础配置
        self.pretrain_hyperparams = pretrain_data['hyperparams']
        self.pretrain_descriptor_scaler = pretrain_data.get('descriptor_scaler', None)
        self.descriptor_cols = pretrain_data.get('descriptor_cols', None)
        self.use_descriptors = pretrain_data.get('use_descriptors', True)
        self.use_cross_attention = pretrain_data.get('use_cross_attention', False)
        self.descriptor_dim = pretrain_data.get('descriptor_dim', 0)

        # ❌ 这些架构参数从预训练模型继承，不能改变
        self.fixed_architecture_params = {
            'hidden_size': self.pretrain_hyperparams['hidden_size'],
            'depth': self.pretrain_hyperparams['depth'],
            'num_attention_heads': self.pretrain_hyperparams.get('num_attention_heads', 8),
            'cnn_filters1': self.pretrain_hyperparams.get('cnn_filters1', 64),
            'cnn_filters2': self.pretrain_hyperparams.get('cnn_filters2', 128),
            'cnn_kernel_size': self.pretrain_hyperparams.get('cnn_kernel_size', 7)
        }

        print("\n" + "=" * 60)
        print("固定的架构参数（来自预训练模型，不可调节）:")
        print("=" * 60)
        for k, v in self.fixed_architecture_params.items():
            print(f"  {k}: {v}")
        print("=" * 60)

    def create_model(self, tunable_params, load_pretrained_weights=True,
                     freeze_encoder=False, allow_partial_load=True):
        """创建模型"""
        # 合并固定参数和可调节参数
        full_params = self.fixed_architecture_params.copy()

        # ✅ 添加可安全调节的参数
        full_params['dropout'] = tunable_params.get('dropout', 0.1)

        # ⚠️ FFN层可以调节，但会重新初始化
        full_params['ffn_hidden_size'] = tunable_params.get(
            'ffn_hidden_size',
            self.pretrain_hyperparams['ffn_hidden_size']
        )

        # 创建模型
        model = DMPNN(
            descriptor_dim=self.descriptor_dim,
            hidden_size=full_params['hidden_size'],
            depth=full_params['depth'],
            ffn_hidden_size=full_params['ffn_hidden_size'],
            dropout=full_params['dropout'],
            use_descriptors=self.use_descriptors,
            num_attention_heads=full_params['num_attention_heads'],
            use_cross_attention=self.use_cross_attention,
            cnn_filters1=full_params['cnn_filters1'],
            cnn_filters2=full_params['cnn_filters2'],
            cnn_kernel_size=full_params['cnn_kernel_size']
        )

        # 加载预训练权重
        if load_pretrained_weights:
            pretrained_state = self.pretrain_data['model_state_dict']
            model_state = model.state_dict()

            # 检查FFN参数是否改变
            ffn_changed = full_params['ffn_hidden_size'] != self.pretrain_hyperparams['ffn_hidden_size']

            print("模型内 descriptor_dim:", model.descriptor_dim)
            print("当前数据 descriptor_dim:", self.finetune_dataset.descriptor_dim)
            print("FFN 第一层输入:", model.ffn[0].in_features)

            # 智能加载权重
            loaded_keys = []
            skipped_keys = []

            for k, v in pretrained_state.items():
                # 如果FFN参数改变了，跳过FFN层
                if ffn_changed and 'ffn' in k:
                    skipped_keys.append(k)
                    continue

                # 只加载形状匹配的参数
                if k in model_state and v.size() == model_state[k].size():
                    model_state[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)

            model.load_state_dict(model_state)

            print(f"\n权重加载信息:")
            print(f"  ✅ 成功加载: {len(loaded_keys)}/{len(pretrained_state)} 个参数")
            if skipped_keys:
                if ffn_changed:
                    print(f"  ⚠️  FFN层参数改变，重新初始化FFN层")
                print(f"  ⏭️  跳过: {len(skipped_keys)} 个参数")

        # 冻结编码器
        if freeze_encoder:
            frozen_count = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += 1
                if any(x in name for x in ['atom_encoder', 'conv_layers', 'descriptor_encoder']):
                    param.requires_grad = False
                    frozen_count += 1
            print(f"\n参数冻结信息:")
            print(f"  ❄️  冻结: {frozen_count}/{total_params} 个参数")
            print(f"  🔥 可训练: {total_params - frozen_count}/{total_params} 个参数")

        return model, full_params

    def create_data_loaders(self, train_data, val_data, batch_size):
        """创建数据加载器"""
        train_graphs, train_targets = train_data
        val_graphs, val_targets = val_data

        train_data_list = []
        for graph, target in zip(train_graphs, train_targets):
            graph.y = target
            train_data_list.append(graph)

        val_data_list = []
        for graph, target in zip(val_graphs, val_targets):
            graph.y = target
            val_data_list.append(graph)

        def custom_collate_fn(batch):
            from torch_geometric.data import Batch
            mol_descriptors = []
            for data in batch:
                if hasattr(data, 'mol_descriptors'):
                    mol_descriptors.append(data.mol_descriptors)
                    temp_desc = data.mol_descriptors
                    delattr(data, 'mol_descriptors')
                else:
                    mol_descriptors.append(None)

            batched_data = Batch.from_data_list(batch)

            if any(desc is not None for desc in mol_descriptors):
                valid_descriptors = [desc for desc in mol_descriptors if desc is not None]
                if valid_descriptors:
                    batched_descriptors = torch.stack(valid_descriptors, dim=0)
                    batched_data.mol_descriptors = batched_descriptors

            for data, desc in zip(batch, mol_descriptors):
                if desc is not None:
                    data.mol_descriptors = desc

            return batched_data

        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True,
                                  collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False,
                                collate_fn=custom_collate_fn)

        return train_loader, val_loader

    def evaluate_hyperparams(self, tunable_params, train_data, val_data,
                             epochs=50, patience=10, freeze_encoder=False, verbose=False):
        """评估一组可调节的超参数"""
        # 创建模型
        model, full_params = self.create_model(
            tunable_params,
            load_pretrained_weights=True,
            freeze_encoder=freeze_encoder,
            allow_partial_load=True
        )

        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(
            train_data, val_data, tunable_params['batch_size']
        )

        # 训练
        trainer = FineTuneTrainer(model, self.device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tunable_params['learning_rate']
        )
        criterion = nn.MSELoss()

        _, val_metrics = trainer.train_with_early_stopping(
            train_loader, val_loader, optimizer, criterion,
            epochs=epochs, patience=patience,
            scaler=self.finetune_dataset.target_scaler,
            verbose=verbose
        )

        return val_metrics, full_params

    def grid_search(self, param_space, train_data, val_data,
                    epochs=50, patience=10, freeze_encoder=False,
                    max_trials=None, metric='rmse'):
        """网格搜索超参数"""
        # 生成所有超参数组合
        keys = list(param_space.keys())
        values = [param_space[k] for k in keys]
        all_combinations = list(product(*values))

        # 限制尝试次数
        if max_trials is not None and len(all_combinations) > max_trials:
            import random
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_trials]

        print(f"\n开始网格搜索，共 {len(all_combinations)} 组超参数...")

        best_score = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        best_tunable_params = None
        best_full_params = None
        best_metrics = None
        all_results = []

        for i, combination in enumerate(all_combinations, 1):
            tunable_params = dict(zip(keys, combination))

            print(f"\n[{i}/{len(all_combinations)}] 测试超参数:")
            for k, v in tunable_params.items():
                print(f"  {k}: {v}")

            try:
                metrics, full_params = self.evaluate_hyperparams(
                    tunable_params, train_data, val_data,
                    epochs=epochs, patience=patience,
                    freeze_encoder=freeze_encoder,
                    verbose=False
                )

                score = metrics[metric]

                print(f"  结果: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

                all_results.append({
                    'tunable_params': tunable_params.copy(),
                    'full_params': full_params.copy(),
                    'metrics': metrics
                })

                # 更新最佳结果
                is_better = (score < best_score) if metric in ['rmse', 'mae'] else (score > best_score)
                if is_better:
                    best_score = score
                    best_tunable_params = tunable_params.copy()
                    best_full_params = full_params.copy()
                    best_metrics = metrics
                    print(f"  ★ 新的最佳结果!")

            except Exception as e:
                print(f"  ❌ 错误: {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"网格搜索完成!")
        print(f"{'=' * 60}")
        print(f"最佳可调节超参数:")
        for k, v in best_tunable_params.items():
            print(f"  {k}: {v}")
        print(
            f"\n最佳结果: RMSE={best_metrics['rmse']:.4f}, MAE={best_metrics['mae']:.4f}, R²={best_metrics['r2']:.4f}")

        return best_tunable_params, best_metrics, all_results, best_full_params

    def random_search(self, param_space, train_data, val_data,
                      n_trials=20, epochs=50, patience=10,
                      freeze_encoder=False, metric='rmse'):
        """随机搜索超参数"""
        import random

        print(f"\n开始随机搜索，共 {n_trials} 次尝试...")

        best_score = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        best_tunable_params = None
        best_full_params = None
        best_metrics = None
        all_results = []

        for i in range(n_trials):
            # 随机选择超参数
            tunable_params = {k: random.choice(v) for k, v in param_space.items()}

            print(f"\n[{i + 1}/{n_trials}] 测试超参数:")
            for k, v in tunable_params.items():
                print(f"  {k}: {v}")

            try:
                metrics, full_params = self.evaluate_hyperparams(
                    tunable_params, train_data, val_data,
                    epochs=epochs, patience=patience,
                    freeze_encoder=freeze_encoder,
                    verbose=False
                )

                score = metrics[metric]

                print(f"  结果: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

                all_results.append({
                    'tunable_params': tunable_params.copy(),
                    'full_params': full_params.copy(),
                    'metrics': metrics
                })

                # 更新最佳结果
                is_better = (score < best_score) if metric in ['rmse', 'mae'] else (score > best_score)
                if is_better:
                    best_score = score
                    best_tunable_params = tunable_params.copy()
                    best_full_params = full_params.copy()
                    best_metrics = metrics
                    print(f"  ★ 新的最佳结果!")

            except Exception as e:
                print(f"  ❌ 错误: {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"随机搜索完成!")
        print(f"{'=' * 60}")
        print(f"最佳可调节超参数:")
        for k, v in best_tunable_params.items():
            print(f"  {k}: {v}")
        print(
            f"\n最佳结果: RMSE={best_metrics['rmse']:.4f}, MAE={best_metrics['mae']:.4f}, R²={best_metrics['r2']:.4f}")

        return best_tunable_params, best_metrics, all_results, best_full_params


# ===================== 微调主类（带超参数优化） =====================
class FlashPointFineTuner:
    """闪点预测微调器（带安全的超参数优化）- 支持基于type的分组划分"""

    def __init__(self, pretrained_model_path, finetune_data_path, save_dir):
        """初始化微调器"""
        self.pretrained_model_path = pretrained_model_path
        self.finetune_data_path = finetune_data_path
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # 加载预训练模型信息
        print("加载预训练模型...")
        self.pretrain_data = torch.load(pretrained_model_path, map_location='cpu')
        self.pretrain_hyperparams = self.pretrain_data['hyperparams']
        self.pretrain_target_scaler = self.pretrain_data['target_scaler']
        self.pretrain_descriptor_scaler = self.pretrain_data.get('descriptor_scaler', None)
        self.descriptor_cols = self.pretrain_data.get('descriptor_cols', None)
        self.use_descriptors = self.pretrain_data.get('use_descriptors', True)
        self.use_cross_attention = self.pretrain_data.get('use_cross_attention', False)
        self.num_attention_heads = self.pretrain_data.get('num_attention_heads', 8)
        self.descriptor_dim = self.pretrain_data.get('descriptor_dim', 0)
        self.cnn_config = self.pretrain_data.get('cnn_config', {
            'filters1': 64,
            'filters2': 128,
            'kernel_size': 7
        })

        print("\n预训练模型信息:")
        print(f"  使用描述符: {self.use_descriptors}")
        print(f"  使用交叉注意力: {self.use_cross_attention}")
        print(f"  描述符维度: {self.descriptor_dim}")
        print(f"  Hidden Size: {self.pretrain_hyperparams['hidden_size']}")
        print(f"  Depth: {self.pretrain_hyperparams['depth']}")

        # 加载微调数据（带type列）
        print("\n加载微调数据...")
        self.finetune_dataset = FineTuneDataset(
            finetune_data_path,
            type_col='type',  # 添加type列参数
            descriptor_cols=self.descriptor_cols,
            use_descriptors=self.use_descriptors,
            pretrain_descriptor_scaler=self.pretrain_descriptor_scaler
        )

        # 保存原始数据用于获取SMILES
        self.original_data = pd.read_csv(finetune_data_path)
        self.smiles_col = 'smiles'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n使用设备: {self.device}")

    def hyperparameter_tuning_strategy1(self, n_outer_folds=5, param_space=None,
                                        search_method='random', n_trials=20,
                                        epochs=50, patience=10, freeze_encoder=False):
        """超参数调优策略1: k折交叉验证 (基于type分组)"""
        print("\n" + "=" * 80)
        print(f"超参数调优策略1: {n_outer_folds}折交叉验证 (基于type分组)")
        print(f"每折: 1折测试 + 1折验证 + {n_outer_folds - 2}折训练")
        print("=" * 80)

        if param_space is None:
            param_space = HyperparameterSpace.get_compact_safe_space()
            print("\n使用默认安全超参数空间（仅可调节参数）")

        # 显示搜索空间
        print("\n可调节的超参数搜索空间:")
        for k, v in param_space.items():
            print(f"  {k}: {v}")

        # 使用基于type的K折划分
        splits = self.finetune_dataset.get_grouped_kfold_splits(
            n_splits=n_outer_folds,
            random_state=42
        )

        fold_results = []
        all_best_tunable_params = []

        for fold_idx, split in enumerate(splits):
            print(f"\n{'=' * 60}")
            print(f"外层第 {fold_idx + 1}/{n_outer_folds} 折")
            print(f"{'=' * 60}")

            train_idx = split['train_idx']
            val_idx = split['val_idx']
            test_idx = split['test_idx']

            print(f"数据划分:")
            print(f"  训练集: {len(train_idx)} 样本 ({len(split['train_types'])} types)")
            print(f"  验证集: {len(val_idx)} 样本 ({len(split['val_types'])} types)")
            print(f"  测试集: {len(test_idx)} 样本 ({len(split['test_types'])} types)")

            # 获取数据
            train_data, _ = self.finetune_dataset.get_splits(train_idx, test_idx)
            val_data, _ = self.finetune_dataset.get_splits(val_idx, test_idx)
            _, test_data = self.finetune_dataset.get_splits(train_idx, test_idx)

            # 超参数搜索
            optimizer = HyperparameterOptimizer(
                self.pretrain_data,
                self.finetune_dataset,
                self.device
            )

            if search_method == 'random':
                best_tunable_params, best_val_metrics, search_results, best_full_params = optimizer.random_search(
                    param_space, train_data, val_data,
                    n_trials=n_trials, epochs=epochs, patience=patience,
                    freeze_encoder=freeze_encoder
                )
            else:
                best_tunable_params, best_val_metrics, search_results, best_full_params = optimizer.grid_search(
                    param_space, train_data, val_data,
                    epochs=epochs, patience=patience,
                    freeze_encoder=freeze_encoder,
                    max_trials=n_trials
                )

            all_best_tunable_params.append(best_tunable_params)

            # 使用最佳超参数进行最终训练
            print(f"\n使用最佳超参数进行最终训练和全面评估...")

            # 重新划分训练和验证集（合并原训练和验证集，再重新分配）
            train_val_combined_idx = np.concatenate([train_idx, val_idx])

            # 基于type重新划分
            unique_types_in_train_val = np.unique(self.finetune_dataset.types[train_val_combined_idx])
            n_val_types = len(unique_types_in_train_val) // 5  # 20%的type作为验证集

            np.random.seed(42)
            np.random.shuffle(unique_types_in_train_val)
            final_val_types = unique_types_in_train_val[:n_val_types]
            final_train_types = unique_types_in_train_val[n_val_types:]

            # 映射回样本索引
            final_train_idx = []
            final_val_idx = []
            for idx in train_val_combined_idx:
                mol_type = self.finetune_dataset.types[idx]
                if mol_type in final_train_types:
                    final_train_idx.append(idx)
                else:
                    final_val_idx.append(idx)

            final_train_idx = np.array(final_train_idx)
            final_val_idx = np.array(final_val_idx)

            final_train_data, _ = self.finetune_dataset.get_splits(final_train_idx, test_idx)
            final_val_data, _ = self.finetune_dataset.get_splits(final_val_idx, test_idx)
            _, final_test_data = self.finetune_dataset.get_splits(final_train_idx, test_idx)

            print(f"最终数据划分 (基于type):")
            print(f"  最终训练集: {len(final_train_idx)} 样本")
            print(f"  最终验证集: {len(final_val_idx)} 样本")
            print(f"  最终测试集: {len(test_idx)} 样本")

            # 训练并评估所有数据集
            all_metrics, full_params, all_predictions_df, _ = self._train_and_evaluate_final(
                best_tunable_params, final_train_data, final_val_data, final_test_data,
                final_train_idx, final_val_idx, test_idx,
                epochs=epochs * 2, patience=patience * 2,
                freeze_encoder=freeze_encoder,
                fold=fold_idx + 1
            )

            fold_results.append({
                'fold': fold_idx + 1,
                'best_tunable_params': best_tunable_params,
                'best_full_params': best_full_params,
                'val_metrics': best_val_metrics,
                'all_metrics': all_metrics,
                'search_results': search_results,
                'all_predictions_df': all_predictions_df
            })

        # 汇总结果
        recommended_params = self._summarize_cv_results(fold_results, all_best_tunable_params)

        # 保存结果
        results_path = os.path.join(self.save_dir, 'strategy1_results.json')
        self._save_results(fold_results, results_path)

        # 保存所有折的预测结果到Excel
        self._save_all_fold_predictions(fold_results, 'strategy1')

        print(f"\n结果已保存到: {results_path}")

        return fold_results, all_best_tunable_params, recommended_params

    def hyperparameter_tuning_strategy2(self, n_outer_folds=5, n_inner_folds=5,
                                        param_space=None, search_method='random',
                                        n_trials=20, epochs=50, patience=10,
                                        freeze_encoder=False):
        """超参数调优策略2: 外层k折 + 内层k折交叉验证（基于type分组）"""
        print("\n" + "=" * 80)
        print(f"超参数调优策略2: 外层{n_outer_folds}折 + 内层{n_inner_folds}折 (基于type分组)")
        print("=" * 80)

        if param_space is None:
            param_space = HyperparameterSpace.get_compact_safe_space()
            print("\n使用默认安全超参数空间（仅可调节参数）")

        # 显示搜索空间
        print("\n可调节的超参数搜索空间:")
        for k, v in param_space.items():
            print(f"  {k}: {v}")

        # 使用基于type的K折划分
        outer_splits = self.finetune_dataset.get_grouped_kfold_splits(
            n_splits=n_outer_folds,
            random_state=42
        )

        fold_results = []
        all_best_tunable_params = []

        for outer_fold_idx, outer_split in enumerate(outer_splits):
            print(f"\n{'=' * 60}")
            print(f"外层第 {outer_fold_idx + 1}/{n_outer_folds} 折")
            print(f"{'=' * 60}")

            train_val_idx = np.concatenate([outer_split['train_idx'], outer_split['val_idx']])
            test_idx = outer_split['test_idx']

            print(f"测试集: {len(test_idx)} 样本 ({len(outer_split['test_types'])} types)")
            print(f"训练+验证集: {len(train_val_idx)} 样本 (将进行{n_inner_folds}折交叉验证)")

            # 内层交叉验证进行超参数搜索
            best_tunable_params, best_full_params = self._inner_cv_hyperparameter_search(
                train_val_idx, n_inner_folds, param_space,
                search_method, n_trials, epochs, patience, freeze_encoder
            )

            all_best_tunable_params.append(best_tunable_params)

            # 使用最佳超参数进行最终训练
            print(f"\n使用最佳超参数进行最终训练和全面评估...")

            # 基于type重新划分训练和验证集
            unique_types_in_train_val = np.unique(self.finetune_dataset.types[train_val_idx])
            n_val_types = len(unique_types_in_train_val) // 5

            np.random.seed(42)
            np.random.shuffle(unique_types_in_train_val)
            final_val_types = unique_types_in_train_val[:n_val_types]
            final_train_types = unique_types_in_train_val[n_val_types:]

            final_train_idx = []
            final_val_idx = []
            for idx in train_val_idx:
                mol_type = self.finetune_dataset.types[idx]
                if mol_type in final_train_types:
                    final_train_idx.append(idx)
                else:
                    final_val_idx.append(idx)

            final_train_idx = np.array(final_train_idx)
            final_val_idx = np.array(final_val_idx)

            final_train_data, _ = self.finetune_dataset.get_splits(final_train_idx, test_idx)
            final_val_data, _ = self.finetune_dataset.get_splits(final_val_idx, test_idx)
            _, final_test_data = self.finetune_dataset.get_splits(final_train_idx, test_idx)

            print(f"最终数据划分 (基于type):")
            print(f"  最终训练集: {len(final_train_idx)} 样本")
            print(f"  最终验证集: {len(final_val_idx)} 样本")
            print(f"  最终测试集: {len(test_idx)} 样本")

            # 训练并评估所有数据集
            all_metrics, full_params, all_predictions_df, _ = self._train_and_evaluate_final(
                best_tunable_params, final_train_data, final_val_data, final_test_data,
                final_train_idx, final_val_idx, test_idx,
                epochs=epochs * 2, patience=patience * 2,
                freeze_encoder=freeze_encoder,
                fold=outer_fold_idx + 1
            )

            fold_results.append({
                'fold': outer_fold_idx + 1,
                'best_tunable_params': best_tunable_params,
                'best_full_params': best_full_params,
                'all_metrics': all_metrics,
                'all_predictions_df': all_predictions_df
            })

        # 汇总结果
        recommended_params = self._summarize_cv_results(fold_results, all_best_tunable_params)

        # 保存结果
        results_path = os.path.join(self.save_dir, 'strategy2_results.json')
        self._save_results(fold_results, results_path)

        # 保存所有折的预测结果到Excel
        self._save_all_fold_predictions(fold_results, 'strategy2')

        print(f"\n结果已保存到: {results_path}")

        return fold_results, all_best_tunable_params, recommended_params

    def _inner_cv_hyperparameter_search(self, train_val_idx, n_inner_folds,
                                        param_space, search_method, n_trials,
                                        epochs, patience, freeze_encoder):
        """内层交叉验证进行超参数搜索（基于type分组）"""
        print(f"\n内层{n_inner_folds}折交叉验证进行超参数搜索 (基于type)...")

        import random

        # 生成超参数候选
        if search_method == 'random':
            param_candidates = []
            for _ in range(n_trials):
                params = {k: random.choice(v) for k, v in param_space.items()}
                param_candidates.append(params)
        else:
            keys = list(param_space.keys())
            values = [param_space[k] for k in keys]
            all_combinations = list(product(*values))

            if len(all_combinations) > n_trials:
                random.shuffle(all_combinations)
                all_combinations = all_combinations[:n_trials]

            param_candidates = [dict(zip(keys, combo)) for combo in all_combinations]

        print(f"共 {len(param_candidates)} 组候选超参数")

        # 获取训练验证集中的唯一types
        unique_types_in_train_val = np.unique(self.finetune_dataset.types[train_val_idx])
        inner_kfold = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

        best_score = float('inf')
        best_tunable_params = None
        best_full_params = None

        for param_idx, tunable_params in enumerate(param_candidates, 1):
            print(f"\n  [{param_idx}/{len(param_candidates)}] 测试超参数:")
            for k, v in tunable_params.items():
                print(f"    {k}: {v}")

            fold_scores = []

            # 对types进行K折划分
            for inner_fold, (inner_train_types_idx, inner_val_types_idx) in enumerate(
                    inner_kfold.split(unique_types_in_train_val), 1
            ):
                inner_train_types = unique_types_in_train_val[inner_train_types_idx]
                inner_val_types = unique_types_in_train_val[inner_val_types_idx]

                # 映射回样本索引
                actual_train_idx = []
                actual_val_idx = []
                for idx in train_val_idx:
                    mol_type = self.finetune_dataset.types[idx]
                    if mol_type in inner_train_types:
                        actual_train_idx.append(idx)
                    elif mol_type in inner_val_types:
                        actual_val_idx.append(idx)

                actual_train_idx = np.array(actual_train_idx)
                actual_val_idx = np.array(actual_val_idx)

                # 创建数据
                train_data, _ = self.finetune_dataset.get_splits(
                    actual_train_idx, actual_val_idx
                )
                _, val_data = self.finetune_dataset.get_splits(
                    actual_train_idx, actual_val_idx
                )

                # 评估
                optimizer = HyperparameterOptimizer(
                    self.pretrain_data,
                    self.finetune_dataset,
                    self.device
                )

                try:
                    val_metrics, full_params = optimizer.evaluate_hyperparams(
                        tunable_params, train_data, val_data,
                        epochs=epochs, patience=patience,
                        freeze_encoder=freeze_encoder,
                        verbose=False
                    )
                    fold_scores.append(val_metrics['rmse'])
                except Exception as e:
                    print(f"    内层第{inner_fold}折出错: {e}")
                    fold_scores.append(float('inf'))

            # 计算平均分数
            avg_score = np.mean(fold_scores)
            print(f"    内层CV平均RMSE: {avg_score:.4f} (±{np.std(fold_scores):.4f})")

            if avg_score < best_score:
                best_score = avg_score
                best_tunable_params = tunable_params.copy()
                best_full_params = full_params.copy()
                print(f"    ★ 新的最佳结果!")

        print(f"\n  内层CV最佳超参数 (平均RMSE: {best_score:.4f}):")
        for k, v in best_tunable_params.items():
            print(f"    {k}: {v}")

        return best_tunable_params, best_full_params

    def fine_tune_with_best_params(self, best_tunable_params, test_size=0.2,
                                   epochs=100, patience=20, freeze_encoder=False):
        """使用最佳超参数进行最终微调（基于type分组划分）"""
        print("\n" + "=" * 80)
        print("使用最佳超参数进行最终微调 (基于type分组)")
        print("=" * 80)

        # 使用基于type的数据划分
        train_val_idx, test_idx = self.finetune_dataset.get_grouped_splits_train_test(
            test_size=test_size, random_state=42
        )

        # 从训练验证集中再划分出验证集（基于type）
        unique_types_in_train_val = np.unique(self.finetune_dataset.types[train_val_idx])
        n_val_types = int(len(unique_types_in_train_val) * 0.2)  # 20%的type作为验证集

        np.random.seed(42)
        np.random.shuffle(unique_types_in_train_val)
        val_types = unique_types_in_train_val[:n_val_types]
        train_types = unique_types_in_train_val[n_val_types:]

        # 映射回样本索引
        train_idx = []
        val_idx = []
        for idx in train_val_idx:
            mol_type = self.finetune_dataset.types[idx]
            if mol_type in train_types:
                train_idx.append(idx)
            else:
                val_idx.append(idx)

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        print(f"\n数据划分 (基于type):")
        print(f"  训练集: {len(train_idx)} 样本 ({len(train_types)} types)")
        print(f"  验证集: {len(val_idx)} 样本 ({len(val_types)} types)")
        print(f"  测试集: {len(test_idx)} 样本")

        # 验证划分
        train_types_set = set(self.finetune_dataset.types[train_idx])
        val_types_set = set(self.finetune_dataset.types[val_idx])
        test_types_set = set(self.finetune_dataset.types[test_idx])

        print(f"\n验证type划分:")
        print(f"  训练集与验证集type重叠: {len(train_types_set & val_types_set)} (应为0)")
        print(f"  训练集与测试集type重叠: {len(train_types_set & test_types_set)} (应为0)")
        print(f"  验证集与测试集type重叠: {len(val_types_set & test_types_set)} (应为0)")

        # 获取数据
        train_data, _ = self.finetune_dataset.get_splits(train_idx, test_idx)
        val_data, _ = self.finetune_dataset.get_splits(val_idx, test_idx)
        _, test_data = self.finetune_dataset.get_splits(train_idx, test_idx)

        # 训练并评估所有数据集
        all_metrics, full_params, all_predictions_df, trained_model= self._train_and_evaluate_final(
            best_tunable_params,
            train_data, val_data, test_data,
            train_idx, val_idx, test_idx,
            epochs=epochs, patience=patience,
            freeze_encoder=freeze_encoder,
            fold=0  # 0表示最终模型
        )

        # 保存预测结果到Excel
        excel_path = os.path.join(self.save_dir, 'final_model_all_predictions.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            all_predictions_df.to_excel(writer, sheet_name='All Data', index=False)

            train_df = all_predictions_df[all_predictions_df['dataset'] == 'train']
            val_df = all_predictions_df[all_predictions_df['dataset'] == 'validation']
            test_df = all_predictions_df[all_predictions_df['dataset'] == 'test']

            if len(train_df) > 0:
                train_df.to_excel(writer, sheet_name='Train', index=False)
            if len(val_df) > 0:
                val_df.to_excel(writer, sheet_name='Validation', index=False)
            if len(test_df) > 0:
                test_df.to_excel(writer, sheet_name='Test', index=False)

        print(f"\n所有数据集的预测结果已保存到: {excel_path}")

        csv_path = os.path.join(self.save_dir, 'final_model_all_predictions.csv')
        all_predictions_df.to_csv(csv_path, index=False)
        print(f"所有数据集的预测结果已保存到: {csv_path}")

        # 获取用于保存模型的数据
        test_metrics = all_metrics['test']

        model_save_path = os.path.join(self.save_dir, 'final_finetuned_model.pth')
        torch.save({
            'model_state_dict': trained_model.state_dict(),  # 使用训练好的模型
            'hyperparams': full_params,
            'tunable_params': best_tunable_params,
            'target_scaler': self.finetune_dataset.target_scaler,
            'descriptor_scaler': self.finetune_dataset.descriptor_scaler,
            'descriptor_cols': self.descriptor_cols,
            'all_metrics': {
                'train': {k: float(v) for k, v in all_metrics['train'].items()
                          if k in ['rmse', 'mae', 'r2']},
                'val': {k: float(v) for k, v in all_metrics['val'].items()
                        if k in ['rmse', 'mae', 'r2']},
                'test': {k: float(v) for k, v in all_metrics['test'].items()
                         if k in ['rmse', 'mae', 'r2']}
            },
            'use_descriptors': self.use_descriptors,
            'use_cross_attention': self.use_cross_attention,
            'descriptor_dim': self.descriptor_dim,
            'freeze_encoder': freeze_encoder,
            'atom_dim': 15,  # 明确指定原子特征维度
            'bond_dim': 7,  # 明确指定键特征维度
            'output_size': 1  # 输出维度
        }, model_save_path)

        return model_save_path, all_metrics

    # 其余的辅助方法保持不变
    def _get_smiles_for_indices(self, indices, fold_data_indices):
        """根据索引获取对应的SMILES字符串"""
        smiles_list = []
        for idx in indices:
            if idx < len(fold_data_indices):
                original_idx = fold_data_indices[idx]
                if original_idx < len(self.original_data):
                    smiles = self.original_data.iloc[original_idx][self.smiles_col]
                    smiles_list.append(smiles)
                else:
                    smiles_list.append('Unknown')
            else:
                smiles_list.append('Index_Error')
        return smiles_list

    def _evaluate_all_sets(self, model, train_loader, val_loader, test_loader,
                           criterion, scaler, train_indices, val_indices, test_indices, fold=1):
        """评估训练集、验证集和测试集的性能"""
        trainer = FineTuneTrainer(model, self.device)

        print(f"\n评估训练集...")
        train_metrics = trainer.evaluate(train_loader, criterion, scaler, return_details=True)

        print(f"评估验证集...")
        val_metrics = trainer.evaluate(val_loader, criterion, scaler, return_details=True)

        print(f"评估测试集...")
        test_metrics = trainer.evaluate(test_loader, criterion, scaler, return_details=True)

        # 获取SMILES
        train_smiles = self._get_smiles_for_indices(train_metrics['indices'], train_indices)
        val_smiles = self._get_smiles_for_indices(val_metrics['indices'], val_indices)
        test_smiles = self._get_smiles_for_indices(test_metrics['indices'], test_indices)

        # 创建DataFrame
        train_df = pd.DataFrame({
            'fold': fold,
            'dataset': 'train',
            'smiles': train_smiles,
            'true_value_K': train_metrics['targets'],
            'predicted_value_K': train_metrics['predictions'],
            'true_value_C': train_metrics['targets'] - 273.15,
            'predicted_value_C': train_metrics['predictions'] - 273.15,
            'residual_K': train_metrics['targets'] - train_metrics['predictions'],
            'residual_C': (train_metrics['targets'] - train_metrics['predictions'])
        })

        val_df = pd.DataFrame({
            'fold': fold,
            'dataset': 'validation',
            'smiles': val_smiles,
            'true_value_K': val_metrics['targets'],
            'predicted_value_K': val_metrics['predictions'],
            'true_value_C': val_metrics['targets'] - 273.15,
            'predicted_value_C': val_metrics['predictions'] - 273.15,
            'residual_K': val_metrics['targets'] - val_metrics['predictions'],
            'residual_C': (val_metrics['targets'] - val_metrics['predictions'])
        })

        test_df = pd.DataFrame({
            'fold': fold,
            'dataset': 'test',
            'smiles': test_smiles,
            'true_value_K': test_metrics['targets'],
            'predicted_value_K': test_metrics['predictions'],
            'true_value_C': test_metrics['targets'] - 273.15,
            'predicted_value_C': test_metrics['predictions'] - 273.15,
            'residual_K': test_metrics['targets'] - test_metrics['predictions'],
            'residual_C': (test_metrics['targets'] - test_metrics['predictions'])
        })

        # 合并三个数据集
        all_predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # 打印性能指标
        print(f"\n第 {fold} 折性能指标:")
        print(
            f"  训练集: RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R²={train_metrics['r2']:.4f}")
        print(f"  验证集: RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, R²={val_metrics['r2']:.4f}")
        print(f"  测试集: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")

        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }, all_predictions_df

    def _train_and_evaluate_final(self, tunable_params, train_data, val_data, test_data,
                                  train_indices, val_indices, test_indices,
                                  epochs=100, patience=20,
                                  freeze_encoder=False, fold=1):
        """最终训练和评估"""
        # 代码与原来相同，不需要修改
        optimizer_obj = HyperparameterOptimizer(
            self.pretrain_data,
            self.finetune_dataset,
            self.device
        )

        model, full_params = optimizer_obj.create_model(
            tunable_params,
            load_pretrained_weights=True,
            freeze_encoder=freeze_encoder,
            allow_partial_load=True
        )

        train_loader, val_loader = optimizer_obj.create_data_loaders(
            train_data, val_data, tunable_params['batch_size']
        )

        test_data_list = []
        for graph, target in zip(test_data[0], test_data[1]):
            graph.y = target
            test_data_list.append(graph)

        def custom_collate_fn(batch):
            from torch_geometric.data import Batch
            mol_descriptors = []
            for data in batch:
                if hasattr(data, 'mol_descriptors'):
                    mol_descriptors.append(data.mol_descriptors)
                    temp_desc = data.mol_descriptors
                    delattr(data, 'mol_descriptors')
                else:
                    mol_descriptors.append(None)

            batched_data = Batch.from_data_list(batch)

            if any(desc is not None for desc in mol_descriptors):
                valid_descriptors = [desc for desc in mol_descriptors if desc is not None]
                if valid_descriptors:
                    batched_descriptors = torch.stack(valid_descriptors, dim=0)
                    batched_data.mol_descriptors = batched_descriptors

            for data, desc in zip(batch, mol_descriptors):
                if desc is not None:
                    data.mol_descriptors = desc

            return batched_data

        test_loader = DataLoader(test_data_list, batch_size=tunable_params['batch_size'],
                                 shuffle=False, collate_fn=custom_collate_fn)

        trainer = FineTuneTrainer(model, self.device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tunable_params['learning_rate']
        )
        criterion = nn.MSELoss()

        print(f"\n开始最终训练 (第{fold}折)...")
        train_history, _ = trainer.train_with_early_stopping(
            train_loader, val_loader, optimizer, criterion,
            epochs=epochs, patience=patience,
            scaler=self.finetune_dataset.target_scaler,
            verbose=True
        )

        all_metrics, all_predictions_df = self._evaluate_all_sets(
            model, train_loader, val_loader, test_loader,
            criterion, self.finetune_dataset.target_scaler,
            train_indices, val_indices, test_indices, fold
        )

        return all_metrics, full_params, all_predictions_df, model

    def _summarize_cv_results(self, fold_results, all_best_tunable_params):
        """汇总交叉验证结果"""
        # 代码与原来相同
        print("\n" + "=" * 80)
        print("交叉验证汇总结果")
        print("=" * 80)

        train_rmse = [r['all_metrics']['train']['rmse'] for r in fold_results]
        train_mae = [r['all_metrics']['train']['mae'] for r in fold_results]
        train_r2 = [r['all_metrics']['train']['r2'] for r in fold_results]

        val_rmse = [r['all_metrics']['val']['rmse'] for r in fold_results]
        val_mae = [r['all_metrics']['val']['mae'] for r in fold_results]
        val_r2 = [r['all_metrics']['val']['r2'] for r in fold_results]

        test_rmse = [r['all_metrics']['test']['rmse'] for r in fold_results]
        test_mae = [r['all_metrics']['test']['mae'] for r in fold_results]
        test_r2 = [r['all_metrics']['test']['r2'] for r in fold_results]

        print(f"\n训练集性能:")
        print(f"  RMSE: {np.mean(train_rmse):.4f} ± {np.std(train_rmse):.4f}")
        print(f"  MAE:  {np.mean(train_mae):.4f} ± {np.std(train_mae):.4f}")
        print(f"  R²:   {np.mean(train_r2):.4f} ± {np.std(train_r2):.4f}")

        print(f"\n验证集性能:")
        print(f"  RMSE: {np.mean(val_rmse):.4f} ± {np.std(val_rmse):.4f}")
        print(f"  MAE:  {np.mean(val_mae):.4f} ± {np.std(val_mae):.4f}")
        print(f"  R²:   {np.mean(val_r2):.4f} ± {np.std(val_r2):.4f}")

        print(f"\n测试集性能:")
        print(f"  RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f}")
        print(f"  MAE:  {np.mean(test_mae):.4f} ± {np.std(test_mae):.4f}")
        print(f"  R²:   {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")

        print(f"\n可调节超参数统计:")
        param_names = list(all_best_tunable_params[0].keys())

        for param_name in param_names:
            values = [params[param_name] for params in all_best_tunable_params]

            if isinstance(values[0], (int, float)):
                print(f"  {param_name}: {np.mean(values):.6f} ± {np.std(values):.6f}")
            else:
                counter = Counter(values)
                print(f"  {param_name}: {dict(counter)}")

        print(f"\n推荐的超参数配置 (基于众数):")
        recommended_params = {}
        for param_name in param_names:
            values = [params[param_name] for params in all_best_tunable_params]
            counter = Counter(values)
            recommended_params[param_name] = counter.most_common(1)[0][0]
            print(f"  {param_name}: {recommended_params[param_name]}")

        return recommended_params

    def _save_results(self, fold_results, results_path):
        """保存结果到JSON"""
        serializable_results = []
        for result in fold_results:
            serializable_result = {
                'fold': result['fold'],
                'best_tunable_params': result.get('best_tunable_params', {}),
                'best_full_params': result.get('best_full_params', {}),
            }

            if 'all_metrics' in result:
                serializable_result['all_metrics'] = {
                    'train': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in result['all_metrics']['train'].items()
                              if
                              k not in ['predictions', 'targets', 'predictions_scaled', 'targets_scaled', 'indices']},
                    'val': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in result['all_metrics']['val'].items()
                            if k not in ['predictions', 'targets', 'predictions_scaled', 'targets_scaled', 'indices']},
                    'test': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in result['all_metrics']['test'].items()
                             if k not in ['predictions', 'targets', 'predictions_scaled', 'targets_scaled', 'indices']}
                }

            serializable_results.append(serializable_result)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def _save_all_fold_predictions(self, fold_results, strategy_name):
        """保存所有折的预测结果到Excel"""
        all_predictions = []

        for result in fold_results:
            if 'all_predictions_df' in result:
                df = result['all_predictions_df']
                all_predictions.append(df)

        if all_predictions:
            combined_df = pd.concat(all_predictions, ignore_index=True)

            excel_path = os.path.join(self.save_dir, f'{strategy_name}_all_predictions.xlsx')

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='All Data', index=False)

                train_df = combined_df[combined_df['dataset'] == 'train']
                val_df = combined_df[combined_df['dataset'] == 'validation']
                test_df = combined_df[combined_df['dataset'] == 'test']

                if len(train_df) > 0:
                    train_df.to_excel(writer, sheet_name='Train', index=False)
                if len(val_df) > 0:
                    val_df.to_excel(writer, sheet_name='Validation', index=False)
                if len(test_df) > 0:
                    test_df.to_excel(writer, sheet_name='Test', index=False)

            print(f"\n所有数据集的预测结果已保存到: {excel_path}")

            csv_path = os.path.join(self.save_dir, f'{strategy_name}_all_predictions.csv')
            combined_df.to_csv(csv_path, index=False)
            print(f"所有数据集的预测结果已保存到: {csv_path}")

            print(f"\n预测结果统计 ({strategy_name}):")
            print(f"  总样本数: {len(combined_df)}")
            for dataset in ['train', 'validation', 'test']:
                dataset_df = combined_df[combined_df['dataset'] == dataset]
                if len(dataset_df) > 0:
                    print(f"\n  {dataset.capitalize()}集:")
                    print(f"    样本数: {len(dataset_df)}")
                    print(f"    平均真实值: {dataset_df['true_value_K'].mean():.2f} K")
                    print(f"    平均预测值: {dataset_df['predicted_value_K'].mean():.2f} K")
                    print(f"    平均残差: {dataset_df['residual_K'].mean():.2f} K")
                    print(
                        f"    RMSE: {np.sqrt(mean_squared_error(dataset_df['true_value_K'], dataset_df['predicted_value_K'])):.2f} K")

    def _plot_training_results(self, all_metrics, all_predictions_df):
        """绘制训练结果"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        datasets = ['train', 'validation', 'test']
        colors = ['blue', 'green', 'red']

        for idx, (dataset, color) in enumerate(zip(datasets, colors)):
            dataset_df = all_predictions_df[all_predictions_df['dataset'] == dataset]

            if len(dataset_df) == 0:
                continue

            predictions = dataset_df['predicted_value_K'].values
            targets = dataset_df['true_value_K'].values
            residuals = dataset_df['residual_K'].values

            axes[0, idx].scatter(targets, predictions, alpha=0.6, s=50,
                                 edgecolors='black', linewidth=0.5, color=color)
            axes[0, idx].plot([targets.min(), targets.max()],
                              [targets.min(), targets.max()],
                              'r--', lw=2, label='Perfect Prediction')
            axes[0, idx].set_xlabel('True Values (K)', fontsize=12)
            axes[0, idx].set_ylabel('Predictions (K)', fontsize=12)

            metrics = all_metrics[dataset.replace('validation', 'val')]
            axes[0, idx].set_title(
                f'{dataset.capitalize()} Set\nRMSE: {metrics["rmse"]:.4f}, R²: {metrics["r2"]:.4f}',
                fontsize=14, fontweight='bold')
            axes[0, idx].legend(fontsize=10)
            axes[0, idx].grid(True, alpha=0.3)

            axes[1, idx].scatter(predictions, residuals, alpha=0.6, s=50,
                                 edgecolors='black', linewidth=0.5, color=color)
            axes[1, idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1, idx].set_xlabel('Predictions (K)', fontsize=12)
            axes[1, idx].set_ylabel('Residuals (K)', fontsize=12)
            axes[1, idx].set_title(f'{dataset.capitalize()} Residual Plot',
                                   fontsize=14, fontweight='bold')
            axes[1, idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'final_training_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n可视化结果保存在: {os.path.join(self.save_dir, 'final_training_results.png')}")

# ===================== 预测器类 =====================
class FlashPointPredictor:
    """闪点预测器"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化预测器"""
        self.model_path = model_path
        self.device = device

        # 加载模型
        print("加载模型...")
        model_data = torch.load(model_path, map_location='cpu')

        self.hyperparams = model_data['hyperparams']
        self.target_scaler = model_data['target_scaler']
        self.descriptor_scaler = model_data.get('descriptor_scaler', None)
        self.descriptor_cols = model_data.get('descriptor_cols', None)
        self.use_descriptors = model_data.get('use_descriptors', True)
        self.use_cross_attention = model_data.get('use_cross_attention', False)
        self.descriptor_dim = model_data.get('descriptor_dim', 0)

        # 创建模型
        self.model = DMPNN(
            descriptor_dim=self.descriptor_dim,
            hidden_size=self.hyperparams['hidden_size'],
            depth=self.hyperparams['depth'],
            ffn_hidden_size=self.hyperparams['ffn_hidden_size'],
            dropout=self.hyperparams['dropout'],
            use_descriptors=self.use_descriptors,
            num_attention_heads=self.hyperparams['num_attention_heads'],
            use_cross_attention=self.use_cross_attention,
            cnn_filters1=self.hyperparams['cnn_filters1'],
            cnn_filters2=self.hyperparams['cnn_filters2'],
            cnn_kernel_size=self.hyperparams['cnn_kernel_size']
        )

        self.model.load_state_dict(model_data['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        print("模型加载完成")

    def predict_single(self, smiles, descriptors=None):
        """预测单个分子的闪点"""
        graph = MolecularFeatures.smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"无法解析SMILES: {smiles}")

        if self.use_descriptors and descriptors is not None:
            if self.descriptor_scaler is not None:
                descriptors = self.descriptor_scaler.transform(
                    np.array(descriptors).reshape(1, -1)
                ).flatten()
            graph.mol_descriptors = torch.FloatTensor(descriptors)
        else:
            graph.mol_descriptors = None

        graph.y = torch.FloatTensor([0])

        def custom_collate_fn(batch):
            from torch_geometric.data import Batch
            mol_descriptors = []
            for data in batch:
                if hasattr(data, 'mol_descriptors'):
                    mol_descriptors.append(data.mol_descriptors)
                    temp_desc = data.mol_descriptors
                    delattr(data, 'mol_descriptors')
                else:
                    mol_descriptors.append(None)

            batched_data = Batch.from_data_list(batch)

            if any(desc is not None for desc in mol_descriptors):
                valid_descriptors = [desc for desc in mol_descriptors if desc is not None]
                if valid_descriptors:
                    batched_descriptors = torch.stack(valid_descriptors, dim=0)
                    batched_data.mol_descriptors = batched_descriptors

            for data, desc in zip(batch, mol_descriptors):
                if desc is not None:
                    data.mol_descriptors = desc

            return batched_data

        dataloader = DataLoader([graph], batch_size=1, collate_fn=custom_collate_fn)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)
                prediction = output.squeeze().cpu().numpy()

        # 反标准化
        prediction = self.target_scaler.inverse_transform([[prediction]])[0][0]

        return prediction

    def predict_batch(self, smiles_list, descriptors_list=None):
        """批量预测闪点"""
        predictions = []

        for i, smiles in enumerate(tqdm(smiles_list, desc="预测中")):
            try:
                desc = descriptors_list[i] if descriptors_list is not None else None
                pred = self.predict_single(smiles, desc)
                predictions.append(pred)
            except Exception as e:
                print(f"预测失败 (SMILES: {smiles}): {e}")
                predictions.append(None)

        return predictions


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 配置参数
    PRETRAINED_MODEL_PATH = "flashpoint_pretrain_1dcnn/final_pretrained_model_1dcnn.pth"
    FINETUNE_DATA_PATH = "finetune_data.csv"
    SAVE_DIR = "flashpoint_finetune_tuned"

    # 检查文件
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"错误: 预训练模型文件不存在: {PRETRAINED_MODEL_PATH}")
        exit(1)

    if not os.path.exists(FINETUNE_DATA_PATH):
        print(f"错误: 微调数据文件不存在: {FINETUNE_DATA_PATH}")
        exit(1)

    print("\n" + "=" * 80)
    print("闪点预测模型微调 - 带安全的超参数优化")
    print("=" * 80)

    print("\n选择运行模式:")
    print("1. 超参数调优策略1 (k折: 训练+验证+测试)")
    print("2. 超参数调优策略2 (外层k折 + 内层k折CV)")
    print("3. 使用自定义超参数直接微调")
    print("4. 使用微调后的模型进行预测")

    choice = input("\n请输入选择 (1-4): ").strip()

    if choice == "1":
        print("\n>>> 策略1: k折交叉验证 (训练+验证+测试)")

        n_folds = int(input("输入折数 (默认5): ").strip() or "5")

        print("\n选择搜索方法:")
        print("1. 随机搜索 (推荐)")
        print("2. 网格搜索")
        search_choice = input("请选择 (1-2, 默认1): ").strip() or "1"
        search_method = 'random' if search_choice == '1' else 'grid'

        n_trials = int(input(f"搜索次数 (默认20): ").strip() or "20")

        freeze_choice = input("是否冻结编码器? (y/n, 默认n): ").strip().lower()
        freeze_encoder = (freeze_choice == 'y')

        print("\n选择超参数搜索空间:")
        print("1. 安全空间 - 精简版 (推荐，快速)")
        print("2. 安全空间 - 完整版")
        print("3. 部分调节空间 - 精简版 (包含FFN)")
        print("4. 部分调节空间 - 完整版 (包含FFN)")
        space_choice = input("请选择 (1-4, 默认1): ").strip() or "1"

        if space_choice == '1':
            param_space = HyperparameterSpace.get_compact_safe_space()
            print("✅ 使用安全精简空间: learning_rate, batch_size, dropout")
        elif space_choice == '2':
            param_space = HyperparameterSpace.get_safe_tunable_space()
            print("✅ 使用安全完整空间: learning_rate, batch_size, dropout")
        elif space_choice == '3':
            param_space = HyperparameterSpace.get_compact_partial_space()
            print("⚠️  使用部分调节精简空间: + ffn_hidden_size")
        else:
            param_space = HyperparameterSpace.get_partial_tunable_space()
            print("⚠️  使用部分调节完整空间: + ffn_hidden_size")

        # 创建微调器
        finetuner = FlashPointFineTuner(
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            finetune_data_path=FINETUNE_DATA_PATH,
            save_dir=SAVE_DIR
        )

        # 执行策略1
        fold_results, all_best_params, recommended_params = finetuner.hyperparameter_tuning_strategy1(
            n_outer_folds=n_folds,
            param_space=param_space,
            search_method=search_method,
            n_trials=n_trials,
            epochs=50,
            patience=10,
            freeze_encoder=freeze_encoder
        )

        # 询问是否使用推荐超参数进行最终训练
        final_train = input("\n是否使用推荐超参数进行最终训练? (y/n): ").strip().lower()
        if final_train == 'y':
            model_path, all_metrics = finetuner.fine_tune_with_best_params(
                best_tunable_params=recommended_params,
                test_size=0.2,
                epochs=100,
                patience=20,
                freeze_encoder=freeze_encoder
            )

    elif choice == "2":
        print("\n>>> 策略2: 外层k折 + 内层k折交叉验证（嵌套CV）")

        n_outer_folds = int(input("输入外层折数 (默认5): ").strip() or "5")
        n_inner_folds = int(input("输入内层折数 (默认5): ").strip() or "5")

        print("\n选择搜索方法:")
        print("1. 随机搜索 (推荐)")
        print("2. 网格搜索")
        search_choice = input("请选择 (1-2, 默认1): ").strip() or "1"
        search_method = 'random' if search_choice == '1' else 'grid'

        n_trials = int(input(f"每次内层CV的搜索次数 (默认15): ").strip() or "15")

        freeze_choice = input("是否冻结编码器? (y/n, 默认n): ").strip().lower()
        freeze_encoder = (freeze_choice == 'y')

        print("\n选择超参数搜索空间:")
        print("1. 安全空间 - 精简版 (推荐)")
        print("2. 安全空间 - 完整版")
        print("3. 部分调节空间 - 精简版")
        print("4. 部分调节空间 - 完整版")
        space_choice = input("请选择 (1-4, 默认1): ").strip() or "1"

        if space_choice == '1':
            param_space = HyperparameterSpace.get_compact_safe_space()
        elif space_choice == '2':
            param_space = HyperparameterSpace.get_safe_tunable_space()
        elif space_choice == '3':
            param_space = HyperparameterSpace.get_compact_partial_space()
        else:
            param_space = HyperparameterSpace.get_partial_tunable_space()

        # 创建微调器
        finetuner = FlashPointFineTuner(
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            finetune_data_path=FINETUNE_DATA_PATH,
            save_dir=SAVE_DIR
        )

        # 执行策略2
        fold_results, all_best_params, recommended_params = finetuner.hyperparameter_tuning_strategy2(
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            param_space=param_space,
            search_method=search_method,
            n_trials=n_trials,
            epochs=50,
            patience=10,
            freeze_encoder=freeze_encoder
        )

        # 询问是否使用推荐超参数进行最终训练
        final_train = input("\n是否使用推荐超参数进行最终训练? (y/n): ").strip().lower()
        if final_train == 'y':
            model_path, all_metrics = finetuner.fine_tune_with_best_params(
                best_tunable_params=recommended_params,
                test_size=0.2,
                epochs=100,
                patience=20,
                freeze_encoder=freeze_encoder
            )

    elif choice == "3":
        print("\n>>> 使用自定义超参数直接微调")

        print("\n请输入可调节的超参数 (直接回车使用默认值):")

        custom_params = {
            'learning_rate': float(input("学习率 (默认1e-4): ").strip() or "1e-4"),
            'batch_size': int(input("批大小 (默认32): ").strip() or "32"),
            'dropout': float(input("Dropout (默认0.1): ").strip() or "0.1")
        }

        tune_ffn = input("\n是否调节FFN层大小? (y/n, 默认n): ").strip().lower()
        if tune_ffn == 'y':
            custom_params['ffn_hidden_size'] = int(input("FFN隐藏层大小 (默认300): ").strip() or "300")

        freeze_choice = input("\n是否冻结编码器? (y/n, 默认n): ").strip().lower()
        freeze_encoder = (freeze_choice == 'y')

        # 创建微调器
        finetuner = FlashPointFineTuner(
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            finetune_data_path=FINETUNE_DATA_PATH,
            save_dir=SAVE_DIR
        )

        # 执行微调
        model_path, all_metrics = finetuner.fine_tune_with_best_params(
            best_tunable_params=custom_params,
            test_size=0.2,
            epochs=100,
            patience=20,
            freeze_encoder=freeze_encoder
        )