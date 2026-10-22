import numpy as np
from methods import sum_matrix, sum_block
import time
import math

# 该函数用于计算单一染色体的RBCP特征
# RBCP: Regional Block Contact Proportion（区域块接触比例）
# 将接触矩阵划分为不重叠的小块，计算每个块对整体接触的贡献比例
# 只计算上三角区域（包含对角线块），因为Hi-C矩阵是对称的
def calculate_RBCP(contact_matrix, max_length, block_size=10):
    """
    计算RBCP特征
    
    参数:
        contact_matrix: n×n的接触矩阵
        max_length: 矩阵维度
        block_size: 块的大小（修改为10×10正方形区域，减少特征维度）
    
    返回:
        RBCP: 上三角区域（含对角线）所有块的接触比例列表
    """
    RBCP = []
    
    # 计算整个矩阵的接触总数
    chr_total = sum_matrix(contact_matrix)
    
    # 如果总接触数为0，返回全0列表
    if chr_total == 0:
        num_blocks = math.ceil(max_length / block_size)
        num_features = num_blocks * (num_blocks + 1) // 2
        return [0] * num_features
    
    # 计算块的数量
    num_blocks = math.ceil(max_length / block_size)
    
    # 遍历上三角区域的所有块（包含对角线块）
    for block_i in range(num_blocks):
        for block_j in range(block_i, num_blocks):  # block_j >= block_i 确保是上三角
            # 计算当前块在原矩阵中的起始和结束位置
            row_start = block_i * block_size
            row_end = min((block_i + 1) * block_size, max_length)
            col_start = block_j * block_size
            col_end = min((block_j + 1) * block_size, max_length)
            
            # 提取当前块
            block = contact_matrix[row_start:row_end, col_start:col_end]
            
            # 计算块内接触总和
            if block_i == block_j:
                # 对角线块：只计算上三角+对角线（避免重复计算）
                block_sum = sum_block(block, is_diagonal=True)
            else:
                # 非对角线块：计算整个块
                block_sum = sum_block(block, is_diagonal=False)
            
            # 计算该块的接触比例
            block_ratio = block_sum / chr_total
            RBCP.append(block_ratio)
    
    return RBCP


def con_ran(cell_id, type, chr_name, max_length, block_size=10):
    """
    读取接触矩阵文件并计算RBCP特征
    """
    # 定义接触矩阵文件的路径
    file_path = f"/home/graduates/dwt/MFFE/Cell_classification/4DN/4DN_Data/{type}/cell_{cell_id}_{chr_name}.txt"
    
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

    # 计算RBCP特征
    rbcp = calculate_RBCP(contact_matrix, max_length, block_size)
    return rbcp


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
    types = {'GM12878': 1591, 'H1Esc': 1240, 'HAP1': 916, 'HFF': 356, 'IMR90': 12}
    resolution = 1000000  # 1M分辨率
    block_size = 10  # 【修改】块大小：从3改为10，减少特征维度（预计从27537降到约3000）
    
    # 创建染色体和其长度的索引
    index = {}
    with open('/home/graduates/dwt/MFFE/Cell_classification/4DN/combo_hg19.genomesize') as f:
        lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1

    print(f"Chromosome index: {index}")
    print(f"Block size: {block_size}×{block_size} bins (square region)")

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
                RBCP = con_ran(cell_id, type, chr, max_len, block_size)
                cell_dict[str(cell_number)][chr] = RBCP

    # 定义输出路径
    out_path = '/home/graduates/dwt/MFFE/Cell_classification/4DN/4DN_features/RBCP.npy'
    
    # 保存计算出的特征
    np.save(out_path, cell_dict)
    print(f"Feature extraction completed. Saved to {out_path}")

    # 计算并打印运行的总时间
    elapsed_time = time.time() - start_time
    print(f"Total running time: {format_time(elapsed_time)}")


if __name__ == '__main__':
    main()