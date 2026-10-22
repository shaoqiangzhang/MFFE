import numpy as np
from methods import sum_matrix, find_contact_list, find_contact_list_row
import time  

# 该函数用于计算单一染色体的NBCP特征
def calculate_NBCP(contact_matrix, max_length):
    NBCP = []
    # sum_matrix用于计算单一染色体的接触总数，矩阵是对称的，计算染色体的接触总数只需要计算上三角元素和对角元素的和
    chr_total = sum_matrix(contact_matrix)
    
    # 第0行没有上邻居，所以只考虑它自身和下邻居
    con_list1 = find_contact_list_row(contact_matrix[0])
    con_list2 = find_contact_list_row(contact_matrix[1])
    sum_list = con_list1 + con_list2
    if len(sum_list) == 0:
        NBCP.append(0)
    else:
        con_pro = sum(sum_list) / chr_total
        NBCP.append(con_pro)

    # 由于max_length-1行没有下邻居，所以考虑1到max_length-2行
    for i in range(1, max_length-1):
        con_list1 = find_contact_list_row(contact_matrix[i])
        con_list2 = find_contact_list_row(contact_matrix[i-1])
        con_list3 = find_contact_list_row(contact_matrix[i+1])
        final_list = con_list1 + con_list2 + con_list3
        if len(final_list) == 0:
            NBCP.append(0)
        else:
            con_pro = sum(final_list) / chr_total
            NBCP.append(con_pro)
    
    # max_length-1行没有下邻居，只考虑它自身和上邻居
    con_list1 = find_contact_list_row(contact_matrix[max_length-2])
    con_list2 = find_contact_list_row(contact_matrix[max_length-1])
    sum_list = con_list1 + con_list2
    if len(sum_list) == 0:
        NBCP.append(0)
    else:
        con_pro = sum(sum_list) / chr_total
        NBCP.append(con_pro)
    
    return NBCP


def con_ran(cell_id, type, chr_name, max_length):
    # 定义接触矩阵文件的路径
    file_path = f"/MFFE/4DN_classification/4DN_Data/{type}/cell_{cell_id}_{chr_name}.txt"
    
    # 打开接触矩阵文件
    print(f"Opening file: {file_path}")  # 打印调试信息
    
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

    # 计算NBCP特征
    nbcp = calculate_NBCP(contact_matrix, max_length)
    return nbcp

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
    
    # 定义不同类型的细胞数量
    types = {'GM12878': 1591, 'H1Esc': 1240, 'HAP1': 916, 'HFF': 356, 'IMR90': 12}
    resolution = 1000000  # 1M分辨率
    
    # 创建染色体和其长度的索引
    index = {}
    with open('/MFFE/4DN_classification/4DN/combo_hg19.genomesize') as f:
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
    
    # 遍历每个数据集（GM12878, H1Esc, HAP1 等）
    for type in types:
        for c_num in range(types[type]):  # 处理所有细胞
            cell_id = c_num + 1
            cell_number += 1
            cell_dict[str(cell_number)] = {}
            print(f"Processing cell {cell_id} of type {type}")
            
            # 处理当前细胞的所有染色体
            for chr in chr_list:
                max_len = index[chr]
                NBCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = NBCP

    # 定义输出路径
    out_path = '/MFFE/4DN_classification/4DN_features/NBCP.npy'
    
    # 保存计算出的特征
    np.save(out_path, cell_dict)
    print(f"Feature extraction completed. Saved to {out_path}")

    # 计算并打印运行的总时间
    elapsed_time = time.time() - start_time
    print(f"Total running time: {format_time(elapsed_time)}")

if __name__ == '__main__':
    main()
