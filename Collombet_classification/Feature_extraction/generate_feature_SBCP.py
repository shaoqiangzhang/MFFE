import numpy as np
from methods import find_contact_list_row, sum_matrix, smooth_mean
import time  

# 该函数用于计算单一染色体的SBCP特征
def calculate_SBCP(contact_matrix, max_length):
    # 首先对接触矩阵进行平滑处理
    matrix1 = contact_matrix
    for i in range(len(contact_matrix)):
        # space_neighbour_contact_infor 存储当前bin的空间邻接信息
        space_neighbour_contact_infor = []
        # index1 存储与当前bin线性邻接的bin的索引
        index1 = []
        space_neighbour_contact_infor.append(contact_matrix[i, :])  # 添加当前bin的接触信息

        # 获取与当前bin线性邻接的bin的接触信息
        for m in range(i - 1, i + 2):
            # 确保 m 不越界
            if m < 0 or m > len(contact_matrix) - 1 or m == i:
                continue
            space_neighbour_contact_infor.append(contact_matrix[m, :])
            index1.append(m)

        # 获取与当前bin空间邻接的bin的接触信息
        for n in range(len(contact_matrix[0])):
            if n in index1:
                continue
            if contact_matrix[i, n] != 0 and i != n:
                space_neighbour_contact_infor.append(contact_matrix[n, :])

        # 使用平滑后的接触信息替换原接触信息
        matrix1[i, :] = smooth_mean(space_neighbour_contact_infor)
    
    # 将矩阵对称化，保证上下三角对称
    m = np.triu(matrix1, 1)
    n = m.T
    smoothed_matrix = m + n + np.diag(np.diag(matrix1))

    # 计算SBCP特征
    SBCP = []
    chr_total = sum_matrix(smoothed_matrix)  # 计算总接触数
    for i in range(max_length):
        con_list = find_contact_list_row(smoothed_matrix[i])  # 获取当前bin的接触信息
        if len(con_list) == 0:
            SBCP.append(0)
        else:
            con_pro = sum(con_list) / chr_total
            SBCP.append(con_pro)
    return SBCP

def con_ran(cell_id, type, chr_name, max_length):
    # 定义接触矩阵文件路径
    file_path = f"/MFFE/Collombet_classification/Collombet_Data/{type}/cell_{cell_id}_{chr_name}.txt"
    print(f"Opening file: {file_path}")  # 打印路径信息

    with open(file_path) as chr_file:
        lines = chr_file.readlines()
    
    # 初始化接触矩阵
    contact_matrix = np.zeros((max_length, max_length))

    # 遍历文件中的每一行，更新接触矩阵
    for line in lines:
        bin1, bin2, num = line.split()
        bin1 = int(bin1)
        bin2 = int(bin2)
        
        if bin1 >= max_length or bin2 >= max_length:
            continue
        
        contact_matrix[bin1, bin2] += int(float(num))
        if bin1 != bin2:
            contact_matrix[bin2, bin1] += int(float(num))

    # 计算SBCP特征
    SBCP = calculate_SBCP(contact_matrix, max_length)
    return SBCP

def format_time(seconds):
    # 将秒数转换为小时、分钟、秒的格式
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
    start_time = time.time()  # 记录开始时间
    
    # 定义细胞类型及其数量
    types = {'1CSE': 85, '2CSE': 114, '4CSE': 121, '64CSE': 205, '8CSE': 123}
    resolution = 1000000  # 1M分辨率
    
    # 创建染色体和其长度的索引
    index = {}
    with open('/MFFE/Collombet_classification/mm10.main.nochrM.chrom.sizes') as f:
        lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1  # 加1是因为除法会向下取整

    print(f"Chromosome index: {index}")

    # 初始化存储细胞数据的字典
    cell_dict = {}
    cell_number = 0
    chr_list = sorted(index.keys())

    # 遍历每个数据集
    for type in types:
        for c_num in range(types[type]):  # 处理所有细胞
            cell_id = c_num + 1
            cell_number += 1
            cell_dict[str(cell_number)] = {}
            print(f"Processing cell {cell_id} of type {type}")
            
            # 处理当前细胞的所有染色体
            for chr in chr_list:
                max_len = index[chr]
                SBCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = SBCP

    # 定义输出路径
    out_path = '/MFFE/Collombet_classification/Collombet_features/SBCP.npy'
    
    # 保存计算出的特征
    np.save(out_path, cell_dict)
    print(f"Feature extraction completed. Saved to {out_path}")

    # 计算并打印运行的总时间
    elapsed_time = time.time() - start_time
    print(f"Total running time: {format_time(elapsed_time)}")

if __name__ == '__main__':
    main()
