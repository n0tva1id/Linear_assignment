from scipy.optimize import linear_sum_assignment
import numpy as np
from numba import jit
from time import time

pcl_a = np.fromfile("/Users/exthardwaremac/Desktop/cadc_seq/0000000000.bin",
                    dtype=np.float32).reshape((-1, 4))[:, :3]


pcl_b = np.fromfile("/Users/exthardwaremac/Desktop/cadc_seq/0000000004.bin",
                    dtype=np.float32).reshape((-1, 4))[:,:3]

max_length = min(pcl_a.shape[0], pcl_b.shape[0])


pcl_a = pcl_a[:max_length]
pcl_b = pcl_b[:max_length]

# 计算单帧点云中部分点的cost矩阵
def compute_cost(pcl_a, pcl_b, start_pos, n):
    cost_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(pcl_a[start_pos:n].shape[0]):
        for j in range(pcl_b[start_pos:n].shape[0]):
            cost_matrix[i, j] = np.linalg.norm(pcl_a[i] - pcl_b[j])
    return cost_matrix

# 大规模的linear sum assignment运算速度会很慢, 也许可以使用GPU加速
cost_matrix = compute_cost(pcl_a, pcl_b, 100, 500)

print(cost_matrix.shape)


t1 = time()
row_ind, col_ind = linear_sum_assignment(cost_matrix)
t2 = time()
# 1000*1000: 183.6s
print("Time: ", t2 - t1)
# print(row_ind)  # 开销矩阵对应的行索引
# print(col_ind)  # 对应行索引的最优指派的列索引
# print(cost[row_ind, col_ind])  # 提取每个行索引的最优指派列索引所在的元素，形成数组
# print(cost[row_ind, col_ind].sum())  # 数组求和
