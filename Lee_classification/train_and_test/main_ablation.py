"""
通用消融实验主函数
只需修改 feature_config 配置即可控制使用哪些特征集
"""
import pandas as pd
import xlsxwriter
from scipy import stats
import numpy as np
from method_ablation import CNN_1D_ablation
from sklearn.model_selection import train_test_split
import random, os, torch
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score, recall_score, matthews_corrcoef
from collections import Counter

# ============================================================
# =============== 【配置区域 - 只需修改这里】 ===============
# ============================================================

# 特征集开关配置: True=使用, False=移除
feature_config = {
    'ENBCP': False,    # 新特征集1
    'RBCP': False,      # 新特征集2
    'MRBCP': False,    # 新特征集3
    'DBCP': False,      # 新特征集4
    'NBCP': True,     # 原始特征集1
    'SBCP': True,     # 原始特征集2
    'PSDCP': True,    # 原始特征集3
    'SSDCP': True,     # 原始特征集4
}

# ============================================================
# =============== 【配置区域结束】 ===========================
# ============================================================


def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_output_filename(feature_config):
    """根据配置自动生成输出文件名"""
    enabled = [k for k, v in feature_config.items() if v]
    disabled = [k for k, v in feature_config.items() if not v]
    
    if len(disabled) == 0:
        return 'train&test_ablation_ALL.xlsx'
    elif len(disabled) == 1:
        return f'train&test_ablation_no_{disabled[0]}.xlsx'
    else:
        return f'train&test_ablation_no_{"_".join(disabled)}.xlsx'


def main():
    # 获取启用的特征集列表
    enabled_features = [k for k, v in feature_config.items() if v]
    disabled_features = [k for k, v in feature_config.items() if not v]
    
    print("=" * 60)
    print("消融实验配置")
    print("=" * 60)
    print(f"启用的特征集 ({len(enabled_features)}个): {enabled_features}")
    print(f"移除的特征集 ({len(disabled_features)}个): {disabled_features}")
    print("=" * 60)
    
    if len(enabled_features) == 0:
        print("错误: 至少需要启用一个特征集!")
        return
    
    # 定义不同细胞类型的样本数量
    types = {'Astro': 449, 'Endo': 204, 'L4': 131, 'L5': 180, 'L6': 86, 'L23': 551, 'MG': 422, 'Ndnf': 144,
             'NN1': 100, 'ODC': 1243, 'OPC': 203, 'Pvalb': 134, 'Sst': 216, 'Vip': 171}
    
    # alpha 计算
    alpha = []
    for value in types.values():
        ds = 1 / value
        alpha.append(ds)

    Y = []
    X = []
    resolution = 1000000
    tps = sorted(types)
    
    f = open('../combo_hg19.genomesize') 
    index = {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
    f.seek(0)
    f.close()

    cell_number = 0
    for type in tps:
        for c_num in range(types[type]):
            cell_number += 1
            X.append(str(cell_number))
            if type == 'Astro':
                Y.append(0)
            elif type == 'Endo':
                Y.append(1)
            elif type == 'L4':
                Y.append(2)
            elif type == 'L5':
                Y.append(3)
            elif type == 'L6':
                Y.append(4)
            elif type == 'L23':
                Y.append(5)
            elif type == 'MG':
                Y.append(6)
            elif type == 'Ndnf':
                Y.append(7)
            elif type == 'NN1':
                Y.append(8)
            elif type == 'ODC':
                Y.append(9)
            elif type == 'OPC':
                Y.append(10)
            elif type == 'Pvalb':
                Y.append(11)
            elif type == 'Sst':
                Y.append(12)
            elif type == 'Vip':
                Y.append(13)
    
    X = np.array(X).reshape(cell_number, 1)
    Y = np.array(Y)

    # CNN网络配置
    num_features = len(enabled_features)
    num_types = 14
    Con_layer = [2] * num_features  # 每个特征集2层卷积
    
    random.seed(2023)
    testseed_list = [884807, 408141, 640905, 353617, 129898, 682656, 318109, 879136,
                     730637, 557627, 753260, 553225, 972593, 185272, 221168, 380684,
                     820294, 209523, 162059, 503280, 468642, 557513, 527263, 522696, 32041,
                     668028, 111881, 503384, 677611, 631852, 221929, 304937, 914569, 282637,
                     664421, 751662, 124891, 954318, 917249, 5748, 2624, 5008, 820, 6427, 5334,
                     2486, 5885, 3662, 2975, 576]

    # 特征集路径
    feature_path = "../Lee_features"
    
    # 加载所有启用的特征集数据
    print("\n加载特征集数据...")
    feature_data_dict = {}
    for name in enabled_features:
        fpath = f"{feature_path}/{name}.npy"
        print(f"  加载 {name} from {fpath}")
        feature_data_dict[name] = np.load(fpath, allow_pickle=True).item()
    print("特征集数据加载完成!\n")

    # 创建输出目录和文件
    output_dir = "/Lee_train_test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = get_output_filename(feature_config)
    file = os.path.join(output_dir, output_filename)
    print(f"结果将保存到: {file}\n")
    
    workbook = xlsxwriter.Workbook(file)
    worksheet1 = workbook.add_worksheet('results')
    
    # 写入表头
    worksheet1.write(0, 0, 'test_seed')
    worksheet1.write(0, 1, 'acc')
    worksheet1.write(0, 2, 'micro_F1')
    worksheet1.write(0, 3, 'macro_F1')
    worksheet1.write(0, 4, 'micro_Precision')
    worksheet1.write(0, 5, 'macro_Precision')
    worksheet1.write(0, 6, 'bacc')
    worksheet1.write(0, 7, 'mcc')
    worksheet1.write(0, 8, 'micro_Recall')
    worksheet1.write(0, 9, 'macro_Recall')
    worksheet1.write(0, 10, 'counter')
    worksheet1.write(0, 11, 'enabled_features')

    row = 0
    for testseed in testseed_list:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=testseed, stratify=Y)
        row += 1
        
        # 模型参数
        num_hiddens = 3060
        norm_shape = [num_features, num_hiddens]
        ffn_num_input = num_hiddens
        ffn_num_hiddens = 3060
        num_heads = 20
        num_layers = 2
        dp = 0.1
        use_bias = False
        kernel_size = 7
        out_channels = 32
        out_feature = 1330
        linear_layer = 1
        lr = 0.00005
        gamma = 1
        batch_size = 64
        key_size, query_size, value_size = num_hiddens, num_hiddens, num_hiddens
        
        model_para = [num_features, key_size, query_size,
                      value_size,
                      num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dp,
                      use_bias,
                      kernel_size, out_channels, out_feature,
                      Con_layer,
                      linear_layer, num_types]
        
        print(f"\n{'='*60}")
        print(f"实验 {row}/50 - testseed: {testseed}")
        print(f"启用特征集: {enabled_features}")
        print(f"{'='*60}")
        
        # 训练和测试
        test_acc, test_label, real_label, test_result_matrix = CNN_1D_ablation(
            feature_data_dict,
            enabled_features,
            x_train,
            y_train,
            x_test,
            y_test,
            lr,
            model_para,
            alpha,
            gamma,
            batch_size
        )
        
        # 计算评估指标
        label_count = []
        for i, j in zip(test_label, real_label):
            if i == j:
                label_count.append(i)
        
        micro_F1 = f1_score(real_label, test_label, average='micro')
        macro_F1 = f1_score(real_label, test_label, average='macro')
        micro_Precision = precision_score(real_label, test_label, average='micro')
        macro_Precision = precision_score(real_label, test_label, average='macro')
        micro_Recall = recall_score(real_label, test_label, average='micro')
        macro_Recall = recall_score(real_label, test_label, average='macro')
        bacc = balanced_accuracy_score(real_label, test_label)
        mcc = matthews_corrcoef(real_label, test_label)
        
        print("=" * 40)
        print(f"testseed: {testseed}")
        print(f"row: {row}")
        print(f"acc: {test_acc:.4f}")
        print(f"bacc: {bacc:.4f}")
        print(f"macro_F1: {macro_F1:.4f}")
        print("=" * 40)

        # 写入结果
        worksheet1.write(row, 0, testseed)
        worksheet1.write(row, 1, test_acc)
        worksheet1.write(row, 2, micro_F1)
        worksheet1.write(row, 3, macro_F1)
        worksheet1.write(row, 4, micro_Precision)
        worksheet1.write(row, 5, macro_Precision)
        worksheet1.write(row, 6, bacc)
        worksheet1.write(row, 7, mcc)
        worksheet1.write(row, 8, micro_Recall)
        worksheet1.write(row, 9, macro_Recall)
        worksheet1.write(row, 10, str(Counter(label_count)))
        worksheet1.write(row, 11, str(enabled_features))
    
    workbook.close()
    print(f"\n实验完成! 结果已保存到: {file}")


if __name__ == '__main__':
    main()