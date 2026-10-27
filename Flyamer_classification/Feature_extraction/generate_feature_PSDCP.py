import numpy as np
from methods import sum_matrix
import time  

# 该函数用于计算单一染色体的PSDCP特征
def calculate_PSDCP(contact_matrix, index, scale):
    contact_matrix = np.array(contact_matrix)
    new_matrix = np.zeros((index + 2 * scale, index + 2 * scale))  # 初始化一个新的矩阵，用于计算PSDCP
    PSDCP = []
    chr_total = sum_matrix(contact_matrix)  # 计算染色体的接触总数
    
    # 将原始接触矩阵移到新矩阵中
    for i in range(index):
        for j in range(index):
            new_matrix[i + scale, j + scale] = contact_matrix[i, j]
    
    # 计算PSDCP特征
    for i in range(index):
        bin = i + scale
        # 对新的矩阵区域计算总接触数
        a = sum_matrix(new_matrix[bin - scale:bin + scale + 1, bin - scale:bin + scale + 1])
        if a == 0:
            PSDCP.append(float(0))
        else:
            PSDCP.append(float(a / chr_total))
    return PSDCP


def con_ran(cell_id, type, chr_name, max_length):
    # 定义接触矩阵文件的路径
    file_path = f"../Flyamer_Data/{type}/cell_{cell_id}_{chr_name}.txt"
    print(f"Opening file: {file_path}")  # 打印路径信息以便调试

    # 打开文件并读取所有行
    with open(file_path) as chr_file:
        lines = chr_file.readlines()
    
    # 初始化接触矩阵为零矩阵
    contact_matrix = np.zeros((max_length, max_length))

    # 遍历文件中的每一行数据，更新接触矩阵
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

    # 计算PSDCP特征
    PSDCP = calculate_PSDCP(contact_matrix, max_length, scale=2)
    return PSDCP

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
    types = {'oocytes': 114, 'ZygM': 32, 'ZygP': 32}
    resolution = 1000000  # 1M分辨率
    
    # 创建染色体和其长度的索引
    index = {}
    with open('../mm10.main.nochrM.chrom.sizes') as f:
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
                PSDCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = PSDCP

    # 定义输出路径
    out_path = '../Flyamer_features/PSDCP.npy'
    
    # 保存计算出的特征
    np.save(out_path, cell_dict)
    print(f"Feature extraction completed. Saved to {out_path}")

    # 计算并打印运行的总时间
    elapsed_time = time.time() - start_time
    print(f"Total running time: {format_time(elapsed_time)}")

if __name__ == '__main__':
    main()
