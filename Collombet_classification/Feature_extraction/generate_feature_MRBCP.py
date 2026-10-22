import numpy as np
from methods import sum_matrix, sum_block, downsample_matrix
import time
import math

# 该函数用于计算单一染色体的MRBCP特征
# MRBCP: Multi-Resolution Block Contact Proportion（多分辨率块接触比例）
# 在不同分辨率（2R, 3R等）下计算RBCP特征，捕捉多尺度的接触模式
def calculate_MRBCP(contact_matrix, max_length, scale_factor, block_size=10):
    """
    计算MRBCP特征
    
    参数:
        contact_matrix: n×n的接触矩阵（原始1M分辨率）
        max_length: 原始矩阵维度
        scale_factor: 分辨率缩放因子（2表示2R，3表示3R）
        block_size: 块的大小（修改为10×10正方形区域）
    
    返回:
        MRBCP: 上三角区域（含对角线）所有块的接触比例列表
    """
    MRBCP = []
    
    # 第一步：对矩阵进行下采样（合并相邻bin）
    downsampled_matrix, new_length = downsample_matrix(contact_matrix, max_length, scale_factor)
    
    # 计算下采样后矩阵的接触总数
    chr_total = sum_matrix(downsampled_matrix)
    
    # 如果总接触数为0，返回全0列表
    if chr_total == 0:
        num_blocks = math.ceil(new_length / block_size)
        num_features = num_blocks * (num_blocks + 1) // 2
        return [0] * num_features
    
    # 计算块的数量
    num_blocks = math.ceil(new_length / block_size)
    
    # 遍历上三角区域的所有块（包含对角线块）
    for block_i in range(num_blocks):
        for block_j in range(block_i, num_blocks):  # block_j >= block_i 确保是上三角
            # 计算当前块在矩阵中的起始和结束位置
            row_start = block_i * block_size
            row_end = min((block_i + 1) * block_size, new_length)
            col_start = block_j * block_size
            col_end = min((block_j + 1) * block_size, new_length)
            
            # 提取当前块
            block = downsampled_matrix[row_start:row_end, col_start:col_end]
            
            # 计算块内接触总和
            if block_i == block_j:
                # 对角线块：只计算上三角+对角线（避免重复计算）
                block_sum = sum_block(block, is_diagonal=True)
            else:
                # 非对角线块：计算整个块
                block_sum = sum_block(block, is_diagonal=False)
            
            # 计算该块的接触比例
            block_ratio = block_sum / chr_total
            MRBCP.append(block_ratio)
    
    return MRBCP


def con_ran(cell_id, type, chr_name, max_length, scale_factor, block_size=10):
    """
    读取接触矩阵文件并计算MRBCP特征
    """
    # 定义接触矩阵文件的路径
    file_path = f"/MFFE/Collombet_classification/Collombet_Data/{type}/cell_{cell_id}_{chr_name}.txt"
    
    # 打开接触矩阵文件
    print(f"Opening file: {file_path}")
    
    # 读取文件的所有行
    with open(file_path) as chr_file:
        lines = chr_file.readlines()

    # 初始化接触矩阵为零矩阵
    contact_matrix = np.zeros((max_length, max_length))

    for line in lines:
        bin1, bin2, num = line.split()
        bin1 = int(bin1)
        bin2 = int(bin2)
        
        # 跳过无效接触
        if bin1 >= max_length or bin2 >= max_length:
            continue
        
        contact_matrix[bin1, bin2] += int(float(num))
        if bin1 != bin2:
            contact_matrix[bin2, bin1] += int(float(num))

    # 计算MRBCP特征
    mrbcp = calculate_MRBCP(contact_matrix, max_length, scale_factor, block_size)
    return mrbcp


def format_time(seconds):
    """将秒数转换为小时、分钟、秒的格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)} h "
    if minutes > 0:
        time_str += f"{int(minutes)} min "
    if seconds > 0:
        time_str += f"{int(seconds)} s"
    return time_str.strip()


def main():
    start_time = time.time()
    
    # 定义不同类型的细胞数量
    types = {'1CSE': 85, '2CSE': 114, '4CSE': 121, '64CSE': 205, '8CSE': 123}
    resolution = 1000000  # 原始1M分辨率
    block_size = 10  # 【修改】块大小：从3改为10，减少特征维度
    
    # 定义要计算的分辨率尺度（2R、3R、4R、5R）
    scale_factors = [2, 3, 4, 5]  # 分别对应2M、3M、4M、5M分辨率
    
    # 创建染色体和其长度的索引
    index = {}
    with open('/MFFE/Collombet_classification/mm10.main.nochrM.chrom.sizes') as f:
        lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1

    print(f"Chromosome index: {index}")
    print(f"Block size: {block_size}×{block_size} bins")
    print(f"Scale factors: {scale_factors} (corresponding to 2R, 3R, 4R, 5R resolutions)")

    # 初始化存储细胞数据的字典
    cell_dict = {}
    cell_number = 0
    chr_list = sorted(index.keys())
    
    # 遍历每个数据集
    for type in types:
        for c_num in range(types[type]):
            cell_id = c_num + 1
            cell_number += 1
            cell_dict[str(cell_number)] = {}
            print(f"Processing cell {cell_id} of type {type}")
            
            # 处理当前细胞的所有染色体
            for chr in chr_list:
                max_len = index[chr]
                
                # 收集所有尺度的特征并拼接
                all_scale_features = []
                for scale_factor in scale_factors:
                    MRBCP = con_ran(cell_id, type, chr, max_len, scale_factor, block_size)
                    all_scale_features.extend(MRBCP)
                
                # 将拼接后的特征存入字典
                cell_dict[str(cell_number)][chr] = all_scale_features

    # 定义输出路径（单一文件包含所有尺度）
    out_path = '/MFFE/Collombet_classification/Collombet_features/MRBCP.npy'
    
    # 保存计算出的特征
    np.save(out_path, cell_dict)
    print(f"\nFeature extraction completed. Saved to {out_path}")
    print(f"Features include scales: {scale_factors} (2R, 3R, 4R, 5R)")

    # 计算并打印运行的总时间
    elapsed_time = time.time() - start_time
    print(f"Total running time: {format_time(elapsed_time)}")


if __name__ == '__main__':
    main()