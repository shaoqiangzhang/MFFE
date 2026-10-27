import numpy as np

# This function is used to calculate the total number of contacts on a single chromatin, since the matrix is symmetric, counting the total number of contacts on a chromatin only requires counting the upper triangular elements + diagonal elements.
def sum_matrix(matrix):
    U = np.triu(matrix, 1)
    D = np.diag(np.diag(matrix))
    return sum(sum(U + D))


# Get the sum of the contacts of the elements on the main diagonal of the matrix matrix
def sum_diagonal_matrix(matrix):
    # Take only the diagonal elements of the matrix, the rest of the elements are 0
    D = np.diag(np.diag(matrix))
    return sum(sum(D))


# This function is used to return all contact information for the current bin
# 注意：此函数用于处理整个二维矩阵
def find_contact_list(contact_matrix):
    contact_list = []
    # 遍历矩阵的上三角部分（不包含对角线）
    for i in range(contact_matrix.shape[0]):
        for j in range(i + 1, contact_matrix.shape[1]):
            if contact_matrix[i, j] > 0:
                contact_list.append(contact_matrix[i, j])
    return contact_list


# 新增函数：用于处理单行数据（一维数组）
# This function is used to return all non-zero contact values from a single row
def find_contact_list_row(row):
    """
    返回单行中所有非零的接触值
    
    参数:
        row: 一维数组，表示接触矩阵的一行
    
    返回:
        contact_list: 包含所有非零接触值的列表
    """
    contact_list = []
    for value in row:
        if value > 0:
            contact_list.append(value)
    return contact_list


# 用于计算块内接触总和
# This function is used to calculate the sum of contacts within a block
def sum_block(block, is_diagonal=False):
    """
    计算块内的接触总和
    
    参数:
        block: 二维数组，表示接触矩阵的一个子块
        is_diagonal: 布尔值，表示该块是否位于对角线上
                     如果是对角线块，只计算上三角+对角线（避免重复计算）
                     如果不是对角线块，计算整个块
    
    返回:
        block_sum: 块内接触总和
    """
    if is_diagonal:
        # 对角线块：只计算上三角（含对角线）
        U = np.triu(block, 1)  # 上三角（不含对角线）
        D = np.diag(np.diag(block))  # 对角线
        return np.sum(U + D)
    else:
        # 非对角线块：计算整个块
        return np.sum(block)


# 用于对接触矩阵进行下采样（降低分辨率）
# This function is used to downsample the contact matrix to lower resolution
def downsample_matrix(matrix, original_length, scale_factor):
    """
    对接触矩阵进行下采样，合并相邻的bin
    
    参数:
        matrix: 原始n×n接触矩阵
        original_length: 原始矩阵维度
        scale_factor: 缩放因子（2表示每2×2合并为1，3表示每3×3合并为1）
    
    返回:
        downsampled_matrix: 下采样后的矩阵
        new_length: 新矩阵的维度
    
    示例:
        scale_factor=2: 6×6矩阵 → 3×3矩阵（每2×2区域求和合并为1个bin）
        scale_factor=3: 9×9矩阵 → 3×3矩阵（每3×3区域求和合并为1个bin）
    """
    import math
    
    # 计算新矩阵的维度
    new_length = math.ceil(original_length / scale_factor)
    
    # 初始化新矩阵
    downsampled_matrix = np.zeros((new_length, new_length))
    
    # 遍历新矩阵的每个位置
    for new_i in range(new_length):
        for new_j in range(new_length):
            # 计算对应的原始矩阵区域
            orig_i_start = new_i * scale_factor
            orig_i_end = min((new_i + 1) * scale_factor, original_length)
            orig_j_start = new_j * scale_factor
            orig_j_end = min((new_j + 1) * scale_factor, original_length)
            
            # 提取原始矩阵的对应区域并求和
            region = matrix[orig_i_start:orig_i_end, orig_j_start:orig_j_end]
            downsampled_matrix[new_i, new_j] = np.sum(region)
    
    return downsampled_matrix, new_length


# 用于计算对角线带内的接触总和
# This function is used to calculate the sum of contacts within a diagonal band
def sum_diagonal_band(matrix, max_length, dist_start, dist_end):
    """
    计算指定距离范围内的对角线带的接触总和
    
    参数:
        matrix: n×n接触矩阵
        max_length: 矩阵维度
        dist_start: 起始距离（包含）
        dist_end: 结束距离（不包含）
    
    返回:
        band_sum: 该带内的接触总和
    
    示例:
        dist_start=0, dist_end=5: 计算距离0,1,2,3,4的对角线元素之和
        
        距离的定义: distance = |j - i|
        距离0: 主对角线 (i,i)
        距离1: 第一条副对角线 (i,i+1)
        距离2: 第二条副对角线 (i,i+2)
        ...
    """
    band_sum = 0
    
    for dist in range(dist_start, dist_end):
        if dist >= max_length:
            break
        # 对于距离为dist的对角线，遍历上三角的所有元素
        # 元素位置: (i, i+dist) 其中 i 从 0 到 max_length-dist-1
        for i in range(max_length - dist):
            j = i + dist
            band_sum += matrix[i, j]
    
    return band_sum


# This function is used to perform smoothing calculations on the current bin
def smooth_mean(list):
    # In numpy's sum function axis is 0 is compressed rows, that is, the elements of each column are added, the matrix will be compressed into a line
    # to m = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) for example
    # print(numpy.sum(m,axis = 0)) results in [22 26 30].
    # In the following line of code the function sums each column of a matrix consisting of contact information from spatially adjacent (linear + interacting) bins of the current bin.
    # The sum is then divided by the length of the List list, which is b+1, where b is the number of neighbors, plus itself.
    return np.sum(np.array(list), axis=0) / len(list)


# This function is used to implement the restart randomized wandering algorithm, where the argument rp is the restart probability.
def random_walk_imp(matrix, rp):
    row, _ = matrix.shape
    row_sum = np.sum(matrix, axis=1)
    for i in range(row_sum.shape[0]):
        if row_sum[i] == 0:
            row_sum[i] = 0.001
    nor_matrix = np.divide(matrix.T, row_sum).T
    Q = np.eye(row)
    I = np.eye(row)
    for i in range(30):
        # The random wandering process can be represented by the following matrix operation.
        Q_new = rp * np.dot(Q, nor_matrix) + (1 - rp) * I
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        # When delta is less than the threshold, the restart random walk process converges and the restart random walk process terminates.
        if delta < 1e-6:
            break
    return Q