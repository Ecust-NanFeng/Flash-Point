import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import itertools
import json
import os
from tqdm import tqdm
import warnings
import math

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

        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(MolecularFeatures.atom_features(atom))
        atom_features = torch.FloatTensor(atom_features)

        # 边和边特征
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
    """
    1D-CNN描述符编码器
    用1D-CNN替代原来的简单全连接层，增强描述符特征提取能力
    """

    def __init__(self, descriptor_dim, output_dim,
                 filters1=64, filters2=128, kernel_size=7,
                 pool_size=2, dropout=0.1):
        super(CNN1DDescriptorEncoder, self).__init__()

        self.descriptor_dim = descriptor_dim
        self.output_dim = output_dim

        # 第一个卷积块
        self.conv1 = nn.Conv1d(
            in_channels=1,  # 输入通道数
            out_channels=filters1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2  # same padding
        )
        self.bn1 = nn.BatchNorm1d(filters1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)

        # 第二个卷积块
        self.conv2 = nn.Conv1d(
            in_channels=filters1,
            out_channels=filters2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(filters2)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size)

        # 计算卷积后的特征维度
        # 经过两次池化后: descriptor_dim -> descriptor_dim/pool_size -> descriptor_dim/(pool_size^2)
        pooled_dim = descriptor_dim // (pool_size ** 2)
        flattened_dim = filters2 * pooled_dim

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch_size, descriptor_dim)
        Returns:
            (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # 重塑为1D-CNN输入格式: (batch_size, 1, descriptor_dim)
        x = x.unsqueeze(1)

        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # 展平
        x = x.view(batch_size, -1)

        # 全连接层
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

        # 【修复】使用可学习的融合权重
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

        # 【修复】使用可学习的权重进行加权平均
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
    """D-MPNN模型（融合1D-CNN描述符编码器 + 多头注意力机制）"""

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

        # 输入投影
        self.atom_encoder = nn.Linear(atom_dim, hidden_size)

        # 【核心修改】使用1D-CNN编码分子描述符
        if self.use_descriptors and descriptor_dim > 0:
            # 1D-CNN描述符编码器
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

        # D-MPNN层
        self.conv_layers = nn.ModuleList([
            DMPNNConv(hidden_size, bond_dim) for _ in range(depth)
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 前馈神经网络
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

        # 原子特征编码
        x = self.atom_encoder(x)
        x = self.activation(x)

        # D-MPNN消息传递
        for conv_layer in self.conv_layers:
            x_new = conv_layer(x, edge_index, edge_attr)
            x = x + x_new
            x = self.activation(x)
            x = self.dropout(x)

        # 图级别pooling
        from torch_geometric.nn import global_add_pool
        graph_repr = global_add_pool(x, batch)

        # 【核心修改】使用1D-CNN处理分子描述符
        if (self.use_descriptors and
                self.descriptor_encoder is not None and
                hasattr(data, 'mol_descriptors') and
                data.mol_descriptors is not None):

            mol_desc = data.mol_descriptors
            batch_size = graph_repr.size(0)

            # 重新整形分子描述符
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

            # 【核心修改】通过1D-CNN处理描述符
            descriptor_repr = self.descriptor_encoder(mol_desc)

            # 选择融合方式
            if self.use_cross_attention and self.attention_fusion is not None:
                combined_repr = self.attention_fusion(graph_repr, descriptor_repr)
            else:
                combined_repr = torch.cat([graph_repr, descriptor_repr], dim=-1)
        else:
            combined_repr = graph_repr

        # 前馈网络预测
        output = self.ffn(combined_repr)

        return output


# ===================== 数据处理 =====================
class FlashPointDataset:
    """闪点数据集处理类"""

    def __init__(self, csv_path, smiles_col='smiles', target_col='FP (K)',
                 descriptor_cols=None, use_descriptors=True):
        """初始化数据集

        Args:
            csv_path: 数据文件路径
            smiles_col: SMILES列名
            target_col: 目标值列名
            descriptor_cols: 描述符列名列表
            use_descriptors: 是否使用描述符
        """
        self.data = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.descriptor_cols = descriptor_cols
        self.use_descriptors = use_descriptors

        # 删除缺失值
        self.data = self.data.dropna(subset=[smiles_col, target_col])

        # 【修复】修改条件判断，明确检查 descriptor_cols 是否非空
        if (self.use_descriptors and
                self.descriptor_cols is not None and
                len(self.descriptor_cols) > 0):  # 明确检查长度

            # 确保 descriptor_cols 是列表
            if hasattr(self.descriptor_cols, 'tolist'):
                self.descriptor_cols = self.descriptor_cols.tolist()

            missing_cols = [col for col in self.descriptor_cols if col not in self.data.columns]
            if missing_cols:
                print(f"警告: 以下描述符列不存在: {missing_cols}")
                self.descriptor_cols = [col for col in self.descriptor_cols if col in self.data.columns]

            if len(self.descriptor_cols) > 0:
                self.data = self.data.dropna(subset=self.descriptor_cols)
                print(f"使用 {len(self.descriptor_cols)} 个描述符: {self.descriptor_cols}")
            else:
                print("警告: 没有有效的描述符列，将不使用描述符")
                self.use_descriptors = False
        else:
            self.use_descriptors = False
            if self.descriptor_cols is not None and len(self.descriptor_cols) > 0:
                print("描述符列已提供但 use_descriptors=False，将忽略描述符")
            else:
                print("未提供描述符列，将仅使用图结构数据")

        print(f"加载了 {len(self.data)} 条数据")

        self.graphs = []
        self.targets = []
        self.descriptor_data = []

        print("转换SMILES为图数据...")
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            graph = MolecularFeatures.smiles_to_graph(row[smiles_col])
            if graph is not None:
                self.graphs.append(graph)
                self.targets.append(row[target_col])

                if self.use_descriptors and len(self.descriptor_cols) > 0:
                    descriptors = row[self.descriptor_cols].values.astype(np.float32)
                    self.descriptor_data.append(descriptors)
                    graph.mol_descriptors = torch.FloatTensor(descriptors)
                else:
                    graph.mol_descriptors = None

        print(f"成功转换 {len(self.graphs)} 条数据")

        if len(self.targets) > 0:
            self.targets = torch.FloatTensor(self.targets)
            # 标准化目标值
            self.target_scaler = StandardScaler()
            self.targets = torch.FloatTensor(
                self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
            )
        else:
            self.targets = torch.FloatTensor()
            self.target_scaler = StandardScaler()

        # 标准化分子描述符
        if self.use_descriptors and len(self.descriptor_data) > 0:
            self.descriptor_scaler = StandardScaler()
            descriptor_array = np.array(self.descriptor_data)
            scaled_descriptors = self.descriptor_scaler.fit_transform(descriptor_array)

            for i, scaled_desc in enumerate(scaled_descriptors):
                self.graphs[i].mol_descriptors = torch.FloatTensor(scaled_desc)

            self.descriptor_dim = descriptor_array.shape[1]
            print(f"分子描述符维度: {self.descriptor_dim}")
        else:
            self.descriptor_scaler = None
            self.descriptor_dim = 0
            self.use_descriptors = False  # 确保没有描述符时关闭使用

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]

    def get_splits(self, train_idx, val_idx, test_idx=None):
        train_graphs = [self.graphs[i] for i in train_idx]
        train_targets = self.targets[train_idx]

        val_graphs = [self.graphs[i] for i in val_idx]
        val_targets = self.targets[val_idx]

        if test_idx is not None:
            test_graphs = [self.graphs[i] for i in test_idx]
            test_targets = self.targets[test_idx]
            return (train_graphs, train_targets), (val_graphs, val_targets), (test_graphs, test_targets)

        return (train_graphs, train_targets), (val_graphs, val_targets)


# ===================== 训练器 =====================
class DMPNNTrainer:
    """D-MPNN训练器"""

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
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, dataloader, criterion, scaler=None):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = criterion(output.squeeze(), batch.y)

                total_loss += loss.item()
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(batch.y.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        return {
            'loss': total_loss / len(dataloader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def train_with_early_stopping(self, train_loader, val_loader, optimizer, criterion,
                                  epochs=300, patience=15, scaler=None):
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"早停在第 {epoch + 1} epoch")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return train_history, val_metrics


# ===================== 预训练器主类 =====================
class FlashPointPretrainer:
    """闪点预测预训练器（使用1D-CNN描述符编码器）"""

    def __init__(self, data_path, save_dir, descriptor_cols=None, use_descriptors=True,
                 use_cross_attention=True, num_attention_heads=8,
                 cnn_filters1=64, cnn_filters2=128, cnn_kernel_size=7):
        self.data_path = data_path
        self.save_dir = save_dir
        self.descriptor_cols = descriptor_cols
        self.use_descriptors = use_descriptors
        self.use_cross_attention = use_cross_attention
        self.num_attention_heads = num_attention_heads
        self.cnn_filters1 = cnn_filters1
        self.cnn_filters2 = cnn_filters2
        self.cnn_kernel_size = cnn_kernel_size

        os.makedirs(save_dir, exist_ok=True)

        self.dataset = FlashPointDataset(
            data_path,
            descriptor_cols=descriptor_cols,
            use_descriptors=use_descriptors
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        print(f"使用分子描述符: {self.use_descriptors}")
        print(f"使用1D-CNN编码器: filters=({cnn_filters1}, {cnn_filters2}), kernel={cnn_kernel_size}")
        print(f"使用交叉注意力: {self.use_cross_attention}")
        if self.use_cross_attention:
            print(f"注意力头数: {self.num_attention_heads}")

        # 超参数搜索空间
        self.hyperparameter_space = {
            'hidden_size': [300, 500],
            'depth': [3, 4, 5],
            'ffn_hidden_size': [300, 500],
            'dropout': [0.0, 0.1],
            'batch_size': [32, 64],
            'learning_rate': [1e-4]
        }

        if self.use_cross_attention:
            self.hyperparameter_space['num_attention_heads'] = [4, 8]

        # 固定参数
        self.fixed_params = {
            'epochs': 100,
            'patience': 10,
            'num_folds': 5,
            'seed': 42
        }

    def create_data_loaders(self, train_data, val_data, batch_size):
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

    def cross_validation_with_early_stopping(self, hyperparams):
        """5折交叉验证"""
        kfold = KFold(n_splits=self.fixed_params['num_folds'], shuffle=True,
                      random_state=self.fixed_params['seed'])

        fold_results = []
        indices = np.arange(len(self.dataset))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            if fold !=1:
                pass
            else:
                print(f"  运行第 {fold + 1} 折...")

                train_data, val_data = self.dataset.get_splits(train_idx, val_idx)

                train_loader, val_loader = self.create_data_loaders(
                    train_data, val_data, hyperparams['batch_size']
                )

                num_heads = hyperparams.get('num_attention_heads', self.num_attention_heads)

                # 【核心修改】创建带有1D-CNN编码器的模型
                model = DMPNN(
                    descriptor_dim=self.dataset.descriptor_dim,
                    hidden_size=hyperparams['hidden_size'],
                    depth=hyperparams['depth'],
                    ffn_hidden_size=hyperparams['ffn_hidden_size'],
                    dropout=hyperparams['dropout'],
                    use_descriptors=self.use_descriptors,
                    num_attention_heads=num_heads,
                    use_cross_attention=self.use_cross_attention,
                    cnn_filters1=self.cnn_filters1,
                    cnn_filters2=self.cnn_filters2,
                    cnn_kernel_size=self.cnn_kernel_size
                )

                trainer = DMPNNTrainer(model, self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
                criterion = nn.MSELoss()

                try:
                    _, val_metrics = trainer.train_with_early_stopping(
                        train_loader, val_loader, optimizer, criterion,
                        epochs=self.fixed_params['epochs'],
                        patience=self.fixed_params['patience'],
                        scaler=self.dataset.target_scaler
                    )

                    rmse_score = val_metrics['rmse']
                    fold_results.append(rmse_score)
                    print(f"    第 {fold + 1} 折 RMSE: {rmse_score:.4f}")

                except Exception as e:
                    print(f"    第 {fold + 1} 折训练失败: {str(e)}")
                    fold_results.append(float('inf'))

            mean_score = np.mean(fold_results)
            std_score = np.std(fold_results)

        return mean_score, std_score, fold_results

    def hyperparameter_search(self):
        """超参数搜索"""
        print("开始超参数搜索...")

        keys = list(self.hyperparameter_space.keys())
        values = list(self.hyperparameter_space.values())
        combinations = list(itertools.product(*values))

        hyperparameter_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            hyperparameter_combinations.append(param_dict)

        total_combinations = len(hyperparameter_combinations)
        print(f"总共 {total_combinations} 个超参数组合")
        print("-" * 80)

        best_score = float('inf')
        best_params = None
        all_results = []

        for i, hyperparams in enumerate(hyperparameter_combinations, 1):
            print(f"\n超参数搜索组合 {i}/{total_combinations}")
            print("当前超参数配置:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")

            mean_score, std_score, fold_results = self.cross_validation_with_early_stopping(hyperparams)

            result = {
                'combination': i,
                'hyperparams': hyperparams.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_results': fold_results
            }
            all_results.append(result)

            print(f"5折交叉验证结果:")
            for j, score in enumerate(fold_results, 1):
                print(f"  Fold {j}: {score:.4f}")
            print(f"平均 RMSE: {mean_score:.4f} ± {std_score:.4f}")

            if mean_score < best_score:
                best_score = mean_score
                best_params = hyperparams.copy()
                print(f"*** 新的最佳结果! ***")

            print("-" * 80)

        search_results = {
            'best_score': best_score,
            'best_params': best_params,
            'all_results': all_results,
            'use_descriptors': self.use_descriptors,
            'use_cross_attention': self.use_cross_attention,
            'num_attention_heads': self.num_attention_heads,
            'descriptor_cols': self.descriptor_cols,
            'descriptor_dim': self.dataset.descriptor_dim,
            'cnn_config': {
                'filters1': self.cnn_filters1,
                'filters2': self.cnn_filters2,
                'kernel_size': self.cnn_kernel_size
            }
        }

        results_path = os.path.join(self.save_dir, 'hyperparameter_search_results_1dcnn.json')
        with open(results_path, 'w') as f:
            json.dump(search_results, f, indent=2)

        print(f"\n超参数搜索完成!")
        print(f"最佳 RMSE: {best_score:.4f}")
        print("最佳超参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        return best_params, best_score

    def train_final_model(self, best_params):
        """训练最终预训练模型"""
        print("\n开始训练最终预训练模型...")

        total_size = len(self.dataset)
        indices = np.arange(total_size)
        np.random.seed(self.fixed_params['seed'])
        np.random.shuffle(indices)

        split_idx = int(0.8 * total_size)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_data, val_data = self.dataset.get_splits(train_idx, val_idx)

        train_loader, val_loader = self.create_data_loaders(
            train_data, val_data, best_params['batch_size']
        )

        num_heads = best_params.get('num_attention_heads', self.num_attention_heads)

        # 【核心修改】创建带有1D-CNN编码器的最终模型
        final_model = DMPNN(
            descriptor_dim=self.dataset.descriptor_dim,
            hidden_size=best_params['hidden_size'],
            depth=best_params['depth'],
            ffn_hidden_size=best_params['ffn_hidden_size'],
            dropout=best_params['dropout'],
            use_descriptors=self.use_descriptors,
            num_attention_heads=num_heads,
            use_cross_attention=self.use_cross_attention,
            cnn_filters1=self.cnn_filters1,
            cnn_filters2=self.cnn_filters2,
            cnn_kernel_size=self.cnn_kernel_size
        )

        trainer = DMPNNTrainer(final_model, self.device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        train_history, final_metrics = trainer.train_with_early_stopping(
            train_loader, val_loader, optimizer, criterion,
            epochs=self.fixed_params['epochs'],
            patience=self.fixed_params['patience'],
            scaler=self.dataset.target_scaler
        )

        model_path = os.path.join(self.save_dir, 'final_pretrained_model_1dcnn.pth')
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'hyperparams': best_params,
            'target_scaler': self.dataset.target_scaler,
            'descriptor_scaler': self.dataset.descriptor_scaler,
            'descriptor_cols': self.descriptor_cols,
            'train_history': train_history,
            'final_metrics': final_metrics,
            'use_descriptors': self.use_descriptors,
            'use_cross_attention': self.use_cross_attention,
            'num_attention_heads': num_heads,
            'descriptor_dim': self.dataset.descriptor_dim,
            'cnn_config': {
                'filters1': self.cnn_filters1,
                'filters2': self.cnn_filters2,
                'kernel_size': self.cnn_kernel_size
            }
        }, model_path)

        print("最终预训练模型训练完成!")
        print(f"最终验证 RMSE: {final_metrics['rmse']:.4f}")
        print(f"最终验证 MAE: {final_metrics['mae']:.4f}")
        print(f"最终验证 R²: {final_metrics['r2']:.4f}")

        return model_path, final_metrics

    def run_pretraining(self):
        """运行完整预训练流程"""
        print("=" * 80)
        print("D-MPNN 闪点预测模型 - 预训练阶段 (1D-CNN描述符编码器)")
        print("=" * 80)

        best_params, best_score = self.hyperparameter_search()
        model_path, final_metrics = self.train_final_model(best_params)

        print("\n" + "=" * 80)
        print("预训练阶段完成!")
        print(f"模型保存路径: {model_path}")
        print("=" * 80)

        return model_path, best_params, final_metrics


# ===================== 模型加载器 =====================
class PretrainedModelLoader:
    """预训练模型加载器（支持1D-CNN编码器）"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model_data = torch.load(model_path, map_location='cpu')

    def load_model(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """加载预训练模型"""
        hyperparams = self.model_data['hyperparams']
        use_descriptors = self.model_data.get('use_descriptors', True)
        use_cross_attention = self.model_data.get('use_cross_attention', False)
        num_attention_heads = self.model_data.get('num_attention_heads', 8)
        descriptor_dim = self.model_data.get('descriptor_dim', 0)

        # 获取1D-CNN配置
        cnn_config = self.model_data.get('cnn_config', {
            'filters1': 64,
            'filters2': 128,
            'kernel_size': 7
        })

        model = DMPNN(
            descriptor_dim=descriptor_dim,
            hidden_size=hyperparams['hidden_size'],
            depth=hyperparams['depth'],
            ffn_hidden_size=hyperparams['ffn_hidden_size'],
            dropout=hyperparams['dropout'],
            use_descriptors=use_descriptors,
            num_attention_heads=num_attention_heads,
            use_cross_attention=use_cross_attention,
            cnn_filters1=cnn_config['filters1'],
            cnn_filters2=cnn_config['filters2'],
            cnn_kernel_size=cnn_config['kernel_size']
        )

        model.load_state_dict(self.model_data['model_state_dict'])
        model = model.to(device)

        return model

    def get_scalers(self):
        """获取标准化器"""
        target_scaler = self.model_data['target_scaler']
        descriptor_scaler = self.model_data.get('descriptor_scaler', None)
        return target_scaler, descriptor_scaler

    def get_model_info(self):
        """获取模型信息"""
        return {
            'hyperparams': self.model_data['hyperparams'],
            'final_metrics': self.model_data['final_metrics'],
            'use_descriptors': self.model_data.get('use_descriptors', True),
            'use_cross_attention': self.model_data.get('use_cross_attention', False),
            'num_attention_heads': self.model_data.get('num_attention_heads', 8),
            'descriptor_dim': self.model_data.get('descriptor_dim', 0),
            'descriptor_cols': self.model_data.get('descriptor_cols', None),
            'cnn_config': self.model_data.get('cnn_config', {})
        }


# ===================== 使用示例 =====================
if __name__ == "__main__":
    DATA_PATH = "pretrain_data_rank_filtration.csv"
    SAVE_DIR = "flashpoint_pretrain_1dcnn"

    # 1D-CNN配置
    CNN_FILTERS1 = 64
    CNN_FILTERS2 = 128
    CNN_KERNEL_SIZE = 7

    # 注意力机制配置
    USE_CROSS_ATTENTION = True
    NUM_ATTENTION_HEADS = 8

    # 描述符数量搜索空间
    DESCRIPTOR_NUM_CANDIDATES = [100, 120, 150, 180, 210]

    df = pd.read_csv(DATA_PATH)

    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据文件 {DATA_PATH} 不存在!")
        exit(1)

    try:
        print("CSV文件中的所有列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        exit(1)

    # 运行模式选择
    print("\n选择运行模式:")
    print("1. 训练使用1D-CNN编码器 + 交叉注意力的模型")
    print("2. 训练使用1D-CNN编码器 + 传统拼接的模型")
    print("3. 训练仅使用图结构的模型 (无描述符)")
    print("4. 比较1D-CNN编码器 vs 原始MLP编码器")

    choice = input("请输入选择 (1-4): ").strip()

    if choice in ["1", "2", "4"]:
        # ===================== 描述符数量搜索 =====================
        print(f"\n开始描述符数量搜索，候选数量: {DESCRIPTOR_NUM_CANDIDATES}")
        print("=" * 80)

        descriptor_search_results = []
        best_descriptor_score = float('inf')
        best_descriptor_num = None
        best_descriptor_cols = None

        for desc_num in DESCRIPTOR_NUM_CANDIDATES:
            print(f"\n>>> 测试描述符数量: {desc_num}")

            # 动态生成对应数量的描述符列
            DESCRIPTOR_COLS = df.iloc[:, 1:desc_num + 1].columns.tolist()

            # 检查描述符列是否存在
            available_descriptors = [col for col in DESCRIPTOR_COLS if col in df.columns]
            if len(available_descriptors) < len(DESCRIPTOR_COLS):
                print(f"  警告: 指定 {len(DESCRIPTOR_COLS)} 个描述符，实际找到 {len(available_descriptors)} 个")
                DESCRIPTOR_COLS = available_descriptors

            print(f"  实际使用描述符数量: {len(DESCRIPTOR_COLS)}")

            # 为当前描述符数量创建独立的保存目录
            desc_save_dir = os.path.join(SAVE_DIR, f"desc_{desc_num}")

            use_cross_attention = (choice != "2")

            # 针对当前描述符数量创建预训练器并运行超参数搜索
            pretrainer = FlashPointPretrainer(
                DATA_PATH, desc_save_dir,
                descriptor_cols=DESCRIPTOR_COLS if DESCRIPTOR_COLS else None,
                use_descriptors=bool(DESCRIPTOR_COLS),
                use_cross_attention=use_cross_attention,
                num_attention_heads=NUM_ATTENTION_HEADS,
                cnn_filters1=CNN_FILTERS1,
                cnn_filters2=CNN_FILTERS2,
                cnn_kernel_size=CNN_KERNEL_SIZE
            )

            best_params, best_score = pretrainer.hyperparameter_search()

            descriptor_search_results.append({
                'descriptor_num': desc_num,
                'descriptor_cols': DESCRIPTOR_COLS,
                'best_score': best_score,
                'best_params': best_params
            })

            print(f"  描述符数量 {desc_num} -> 最佳 RMSE: {best_score:.4f}")

            if best_score < best_descriptor_score:
                best_descriptor_score = best_score
                best_descriptor_num = desc_num
                best_descriptor_cols = DESCRIPTOR_COLS
                best_descriptor_params = best_params
                best_pretrainer = pretrainer
                print(f"  *** 新的最佳描述符数量: {desc_num} ***")

            print("-" * 80)

        # 汇总描述符搜索结果
        print("\n描述符数量搜索结果汇总:")
        print(f"{'描述符数量':<15} {'最佳RMSE':<15}")
        print("-" * 30)
        for result in descriptor_search_results:
            marker = " <-- 最佳" if result['descriptor_num'] == best_descriptor_num else ""
            print(f"{result['descriptor_num']:<15} {result['best_score']:<15.4f}{marker}")

        print(f"\n最佳描述符数量: {best_descriptor_num}")
        print(f"最佳 RMSE: {best_descriptor_score:.4f}")

        # 保存描述符搜索结果
        descriptor_search_summary = {
            'descriptor_candidates': DESCRIPTOR_NUM_CANDIDATES,
            'best_descriptor_num': best_descriptor_num,
            'best_descriptor_score': best_descriptor_score,
            'all_results': [
                {
                    'descriptor_num': r['descriptor_num'],
                    'best_score': r['best_score'],
                    'best_params': r['best_params']
                }
                for r in descriptor_search_results
            ]
        }
        summary_path = os.path.join(SAVE_DIR, 'descriptor_search_summary.json')
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(descriptor_search_summary, f, indent=2)
        print(f"\n描述符搜索汇总已保存至: {summary_path}")

        # 使用最佳描述符数量训练最终模型
        print(f"\n使用最佳描述符数量 {best_descriptor_num} 训练最终模型...")
        model_path, final_metrics = best_pretrainer.train_final_model(best_descriptor_params)
        print(f"\n预训练完成! 模型保存在: {model_path}")

    elif choice == "3":
        print(">>> 选择模式3: 仅图结构 (无描述符)")
        pretrainer = FlashPointPretrainer(
            DATA_PATH, SAVE_DIR,
            descriptor_cols=None,
            use_descriptors=False
        )
        model_path, best_params, final_metrics = pretrainer.run_pretraining()
        print(f"\n预训练完成! 模型保存在: {model_path}")

    else:
        print("无效选择!")

    print("\n" + "=" * 80)
    print("程序执行完成!")
    print("=" * 80)