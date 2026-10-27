import os, random
import math
import numpy as np
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from focal_loss_GPU0 import MultiClassFocalLossWithAlpha

def try_gpu(i=0):  #@save
    """Return gpu(i) if it exists, cpu() otherwise."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_bin():
    f = open('../mm10.main.nochrM.chrom.sizes')
    index = {}
    resolution = 1000000
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index


def load_Feature_data(Feature, idX):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        feature = []
        for chr in chr_list:
            feature.append(Feature[cell[0]][chr])
        X.append(np.concatenate(feature).tolist())
    return X, np.array(X).shape[0]


'''
Position-based feed-forward networks
'''
#@save
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


'''
Residual connectivity and layer normalization
'''
class AddNorm(nn.Module):
    """Layer normalization after residual joining"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

'''
Encoder.
'''
class EncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        X = X.to(torch.float32)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

'''
Transformer Encoder
'''
class TransformerEncoder(d2l.Encoder):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, *args):
        self.attention_weights = [None] * len(self.blks)
        valid_lens = None
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


def com_linearsize(linear_size, Con_layer, kernel_size):
    for i in range(Con_layer):
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size


# ============ 通用消融实验模型 ============
# 支持动态数量的特征集
class AblationModel(nn.Module):
    def __init__(self, model_para, feature_names):
        """
        model_para: 模型参数列表
        feature_names: 启用的特征集名称列表，如 ['ENBCP', 'RBCP', 'MRBCP', ...]
        """
        super(AblationModel, self).__init__()
        
        num_features, key_size, query_size, value_size, num_hiddens, \
        norm_shape, ffn_num_input, ffn_num_hiddens, \
        num_heads, num_layers, dp, use_bias, kernel_size, out_channels, out_feature, Con_layer, linear_layer, num_types = model_para
        
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.encoder = TransformerEncoder(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dp, use_bias)
        self.linear_layer = linear_layer
        self.out_channels = out_channels
        
        # 为每个启用的特征集创建卷积层
        self.conv_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.relu_layers = nn.ModuleDict()
        self.pool_layers = nn.ModuleDict()
        self.dropout_layers = nn.ModuleDict()
        self.con_layers = nn.ModuleDict()
        self.linear_sizes = {}
        self.con_layer_nums = {}
        
        linear_size_init = num_hiddens
        
        for i, name in enumerate(feature_names):
            con_layer_num = Con_layer[i] if i < len(Con_layer) else 2
            self.con_layer_nums[name] = con_layer_num
            linear_size = com_linearsize(linear_size_init, con_layer_num, kernel_size)
            self.linear_sizes[name] = linear_size
            
            if con_layer_num != 0:
                self.conv_layers[name] = nn.Conv1d(
                    in_channels=1,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1
                )
                self.bn_layers[name] = nn.BatchNorm1d(num_features=out_channels)
                self.relu_layers[name] = nn.ReLU()
                self.pool_layers[name] = nn.MaxPool1d(kernel_size=2)
                self.dropout_layers[name] = nn.Dropout(dp)
                
                con_seq = nn.Sequential()
                for j in range(con_layer_num - 1):
                    layer_id = str(j + 2)
                    con_seq.add_module("conv%s" % layer_id,
                                      nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=1, padding=1))
                    con_seq.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                    con_seq.add_module("relu%s" % layer_id, nn.ReLU())
                    con_seq.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                    con_seq.add_module("drop%s" % layer_id, nn.Dropout(dp))
                self.con_layers[name] = con_seq
        
        # 计算全连接层输入大小
        total_linear_size = sum(self.linear_sizes.values())
        
        if linear_layer == 1:
            self.fc = nn.Linear(in_features=out_channels * total_linear_size, out_features=num_types)
        else:
            self.fc1 = nn.Linear(in_features=out_channels * total_linear_size, out_features=out_feature)
            self.relu2 = nn.ReLU()
            self.dropout = nn.Dropout(dp)
            self.linear = nn.Sequential()
            self.out_feature = out_feature
            for i in range(linear_layer-2):
                l_layer_id = str(i + 2)
                self.linear.add_module("linear%s" % l_layer_id, nn.Linear(in_features=self.out_feature, out_features=int(self.out_feature/2)))
                self.out_feature = int(self.out_feature/2)
                self.linear.add_module("linear_relu%s" % l_layer_id, nn.ReLU())
                self.linear.add_module("linear_dropout%s" % l_layer_id, nn.Dropout(dp))
            self.fc2 = nn.Linear(in_features=self.out_feature, out_features=num_types)

    def forward(self, X):
        X = self.encoder(X)
        
        outputs = []
        for i, name in enumerate(self.feature_names):
            xi = X[:, i:i+1, :]
            
            if self.con_layer_nums[name] != 0:
                xi = self.relu_layers[name](self.bn_layers[name](self.conv_layers[name](xi)))
                xi = self.pool_layers[name](xi)
                xi = self.dropout_layers[name](xi)
                xi = self.con_layers[name](xi)
                xi = xi.view(-1, self.out_channels * self.linear_sizes[name])
                outputs.append(xi)
        
        x = torch.cat(outputs, 1)
        
        if self.linear_layer == 1:
            x = self.fc(x)
        else:
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu2(x)
            x = self.linear(x)
            x = self.fc2(x)
        
        x = nn.functional.log_softmax(x, dim=1)
        return x


def load_loader(train_dataset, test_dataset, test_size, batch_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    return train_loader, val_loader


def CNN_train(epoch, model, optimizer, train_loader, loss_fn, device):
    i = 0
    for(tensors, labels) in train_loader:
        optimizer.zero_grad()
        labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        tensors = tensors.to(device)
        outputs = model(tensors)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(tensors), len(train_loader.dataset),
                   100. * i / len(train_loader), train_loss.cpu().data))
        i = i + 1
    return model, optimizer


def CNN_test(epoch, model, test_loader, loss_fn, test_size, device):
    torch.cuda.empty_cache()
    
    all_predictions = []
    all_labels = []
    all_outputs = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for (tensors, labels) in test_loader:
            labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
            labels = labels.to(device)
            tensors = tensors.to(device)
            outputs = model(tensors)
            test_loss = loss_fn(outputs, labels)
            total_loss += test_loss.item()
            num_batches += 1
            
            _, prediction = torch.max(outputs.data, 1)
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.data.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())
    
    label_pred = np.array(all_predictions)
    label = np.array(all_labels)
    test_result_matrix = np.vstack(all_outputs)
    
    prediction_num = np.sum(label_pred == label)
    test_accuracy = prediction_num / test_size
    avg_loss = total_loss / num_batches
    
    print('Test Epoch: {} \tAvg Loss: {:.6f} \tAccuracy: {:.4f}'.format(
        epoch, avg_loss, test_accuracy))
    
    return label_pred, label, avg_loss, model, test_accuracy, test_result_matrix


def CNN_1D_ablation(feature_data_dict, feature_names, x_train, y_train, X_test, y_test, 
                    lr, model_para, alpha, gamma, batch_size):
    """
    通用消融实验训练函数
    
    参数:
        feature_data_dict: 字典，键为特征集名称，值为特征数据
        feature_names: 启用的特征集名称列表
        其他参数同原版
    """
    # 加载启用的特征集数据
    train_features = []
    test_features = []
    test_size = None
    
    for name in feature_names:
        train_X, train_size = load_Feature_data(feature_data_dict[name], x_train)
        test_X, test_size_tmp = load_Feature_data(feature_data_dict[name], X_test)
        
        train_X = torch.unsqueeze(torch.from_numpy(np.array(train_X).astype(float)), dim=1)
        test_X = torch.unsqueeze(torch.from_numpy(np.array(test_X).astype(float)), dim=1)
        
        train_features.append(train_X)
        test_features.append(test_X)
        
        if test_size is None:
            test_size = test_size_tmp
    
    # 打印特征维度信息
    print(f"\n=== 启用的特征集: {feature_names} ===")
    for i, name in enumerate(feature_names):
        print(f"{name}: {train_features[i].shape[2]} dimensions")
    
    # 自适应池化到统一维度
    num_heads = model_para[8]
    target_dim = 2660
    print(f"Target dimension: {target_dim}")
    
    adaptive_pool = torch.nn.AdaptiveAvgPool1d(target_dim)
    
    train_features = [adaptive_pool(f.float()) for f in train_features]
    test_features = [adaptive_pool(f.float()) for f in test_features]
    
    # 拼接特征
    train_X = torch.cat(train_features, 1)
    test_X = torch.cat(test_features, 1)
    print(f"Final train_X shape: {train_X.shape}")
    print(f"Final test_X shape: {test_X.shape}")
    
    # 更新model_para
    num_features = len(feature_names)
    model_para[0] = num_features
    model_para[1] = target_dim
    model_para[2] = target_dim
    model_para[3] = target_dim
    model_para[4] = target_dim
    model_para[5] = [num_features, target_dim]
    model_para[6] = target_dim
    model_para[7] = target_dim
    
    # 调整Con_layer长度
    if len(model_para[15]) < num_features:
        model_para[15] = model_para[15] + [2] * (num_features - len(model_para[15]))
    else:
        model_para[15] = model_para[15][:num_features]
    
    # 创建数据集
    train_dataset = TensorDataset(train_X, torch.from_numpy(np.array(y_train).astype(int)))
    test_dataset = TensorDataset(test_X, torch.from_numpy(np.array(y_test).astype(int)))
    
    train_loader, test_loader = load_loader(train_dataset, test_dataset, test_size, batch_size)
    
    # 创建模型
    model = AblationModel(model_para, feature_names)
    device = try_gpu(0)
    model.to(device)
    print(model)
    
    # 训练
    num_epochs = 30
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
    
    # 保存最佳结果
    best_acc = 0
    best_test_label = None
    best_real_label = None
    best_result_matrix = None
    
    for epoch in range(num_epochs):
        model.train()
        model, optimizer = CNN_train(epoch, model, optimizer, train_loader, loss_fn, device)
        model.eval()
        test_label, label, test_loss, model, test_accuracy, test_result_matrix = CNN_test(
            epoch, model, test_loader, loss_fn, test_size, device)
        
        # 保存最佳结果
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_test_label = test_label.copy()
            best_real_label = label.copy()
            best_result_matrix = test_result_matrix.copy()
    
    torch.cuda.empty_cache()
    
    return best_acc, best_test_label, best_real_label, best_result_matrix