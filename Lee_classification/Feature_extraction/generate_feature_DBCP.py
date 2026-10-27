import numpy as np
from methods import sum_matrix, sum_diagonal_band
import time
import math

# 该函数用于计算单一染色体的DBCP特征
# DBCP: Diagonal Band Contact Proportion（对角线带接触比例）
# 将接触矩阵按距离对角线的远近分成若干带状区域，计算每个带的接触贡献比例
# 这能反映短程、中程、长程染色质相互作用的分布特征
def calculate_DBCP(contact_matrix, max_length, band_width=1):
    """
    计算DBCP特征
    
    参数:
        contact_matrix: n×n的接触矩阵
        max_length: 矩阵维度
        band_width: 每个带的宽度（修改为1，每个距离单独一个特征，增加特征密度）
    
    返回:
        DBCP: 每个带的接触比例列表
              带0: 距离0的对角线
              带1: 距离1的对角线
              ...
    """
    DBCP = []
    
    # 计算整个矩阵的接触总数（只计算上三角+对角线）
    chr_total = sum_matrix(contact_matrix)
    
    # 如果总接触数为0，返回全0列表
    if chr_total == 0:
        num_bands = math.ceil(max_length / band_width)
        return [0] * num_bands
    
    # 计算带的数量
    num_bands = math.ceil(max_length / band_width)
    
    # 遍历每个带
    for band_idx in range(num_bands):
        # 计算当前带包含的对角线距离范围
        dist_start = band_idx * band_width
        dist_end = min((band_idx + 1) * band_width, max_length)
        
        # 使用辅助函数计算带内接触总和
        band_sum = sum_diagonal_band(contact_matrix, max_length, dist_start, dist_end)
        
        # 计算该带的接触比例
        band_ratio = band_sum / chr_total
        DBCP.append(band_ratio)
    
    return DBCP


def con_ran(cell_id, type, chr_name, max_length, band_width=1):
    """
    读取接触矩阵文件并计算DBCP特征
    """
    # 定义接触矩阵文件的路径
    file_path = f"../Lee_Data/{type}/cell_{cell_id}_{chr_name}.txt"
    
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

    # 计算DBCP特征
    dbcp = calculate_DBCP(contact_matrix, max_length, band_width)
    return dbcp


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
    types = {'Astro': 449, 'Endo': 204, 'L4': 131, 'L5': 180, 'L6': 86, 'L23': 551, 'MG': 422, 'Ndnf': 144,
             'NN1': 100, 'ODC': 1243, 'OPC': 203, 'Pvalb': 134, 'Sst': 216, 'Vip': 171}
    resolution = 1000000  # 1M分辨率
    band_width = 1  # 【修改】带宽：从5改为1，每个距离单独一个特征（预计从621升到约3053）
    
    # 创建染色体和其长度的索引
    index = {}
    with open('../combo_hg19.genomesize') as f:
        lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1

    print(f"Chromosome index: {index}")
    print(f"Band width: {band_width} diagonal(s) per band")

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
                DBCP = con_ran(cell_id, type, chr, max_len, band_width)
                cell_dict[str(cell_number)][chr] = DBCP

    # 定义输出路径
    out_path = '../Lee_features/DBCP.npy'
    
    # 保存计算出的特征
    np.save(out_path, cell_dict)
    print(f"\nFeature extraction completed. Saved to {out_path}")

    # 计算并打印运行的总时间
    elapsed_time = time.time() - start_time
    print(f"Total running time: {format_time(elapsed_time)}")


if __name__ == '__main__':
    main()