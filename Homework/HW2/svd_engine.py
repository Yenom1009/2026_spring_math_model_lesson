import numpy as np

def my_svd(A, k=None):
    """
    使用 eigh 实现高性能、高精度的 SVD
    原理：A^T * A = V * S^2 * V^T
    """
    # 确保矩阵是 float 类型
    A = A.astype(float)
    m, n = A.shape
    
    # 构建对称矩阵 ATA
    ATA = A.T @ A
    
    # 调用 eigh 计算对称矩阵的特征值和特征向量 (这是最快的方法)
    # 此时 eigenvalues 为升序
    eigenvalues, V = np.linalg.eigh(ATA)
    
    # 反转顺序，使其变为降序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # 奇异值 S = sqrt(特征值)
    # 处理数值误差导致的小负数
    S = np.sqrt(np.maximum(eigenvalues, 0))
    
    # 截断：如果指定了 k，只保留前 k 个
    if k is not None:
        k = min(k, len(S))
        S = S[:k]
        V = V[:, :k]
    
    # 计算 U = A * V * S^-1
    # 只有当 S > 0 时才除以 S，避免除零错误
    S_inv = np.where(S > 1e-10, 1.0 / S, 0)
    U = (A @ V) * S_inv
    
    return U, S, V.T