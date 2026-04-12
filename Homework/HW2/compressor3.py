import time
import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from svd_engine import my_svd 

def calc_fair_metrics(orig_arr, comp_img, required_values_count):
    """
    统一的公平计算公式：
    原始字节 = H * W * 3
    压缩字节 = 必需保留的浮点数个数 * 4 (float32占4字节)
    """
    orig_total_bytes = orig_arr.size 
    compressed_total_bytes = required_values_count * 4
    
    p = psnr(orig_arr.astype(np.uint8), comp_img)
    s = ssim(orig_arr.astype(np.uint8), comp_img, channel_axis=2)
    cr = orig_total_bytes / max(1, compressed_total_bytes)
    
    return p, s, cr

# --- 1. SVD ---
def process_image_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    A = orig_A / 255.0
    H, W = A.shape[:2]
    
    compressed_channels = []
    # SVD 每个通道存：U(H,k), S(k), V(k,W) -> 总数: k * (H + W + 1)
    values_per_channel = k * (H + W + 1)
    
    for i in range(3):
        U, S, VT = my_svd(A[:,:,i], k=k)
        compressed_channels.append(U @ np.diag(S) @ VT)
        
    comp = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    p, s, cr = calc_fair_metrics(orig_A, comp, values_per_channel * 3)
    return comp, p, s, cr, (time.time() - start) * 1000

# --- 2. PCA ---
def process_pca_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    H, W = orig_A.shape[:2]
    
    compressed_channels = []
    # PCA 每个通道存：U, S, VT 和 均值向量 mean(W) -> 总数: k*(H+W+1) + W
    values_per_channel = k * (H + W + 1) + W
    
    for i in range(3):
        ch = orig_A[:, :, i]
        mean = np.mean(ch, axis=0)
        U, S, VT = my_svd((ch - mean)/255.0, k=k)
        compressed_channels.append((U @ np.diag(S) @ VT)*255.0 + mean)
        
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    p, s, cr = calc_fair_metrics(orig_A, comp, values_per_channel * 3)
    return comp, p, s, cr, (time.time() - start) * 1000

# --- 3. DCT ---
def process_dct_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    H, W = orig_A.shape[:2]
    
    # 将 k (1-64) 映射为比例，计算实际保留的系数个数
    ratio = k / 64.0
    h_k, w_k = int(H * ratio), int(W * ratio)
    # DCT 每个通道存：左上角 h_k * w_k 的系数块
    values_per_channel = h_k * w_k
    
    compressed_channels = []
    for i in range(3):
        dct_c = dct(dct(orig_A[:,:,i].T, norm='ortho').T, norm='ortho')
        sparse_dct = np.zeros_like(dct_c)
        sparse_dct[:h_k, :w_k] = dct_c[:h_k, :w_k]
        comp = idct(idct(sparse_dct.T, norm='ortho').T, norm='ortho')
        compressed_channels.append(comp)
        
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    p, s, cr = calc_fair_metrics(orig_A, comp, values_per_channel * 3)
    return comp, p, s, cr, (time.time() - start) * 1000

# --- 4. DWT ---
def process_dwt_compression(image_file, threshold):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_size, orig_A = img.size, np.array(img, dtype=float)
    
    compressed_channels = []
    total_non_zero = 0
    for i in range(3):
        coeffs = pywt.wavedec2(orig_A[:,:,i]/255.0, 'db1', level=1)
        cA, (cH, cV, cD) = coeffs
        cH = pywt.threshold(cH, threshold, mode='soft')
        cV = pywt.threshold(cV, threshold, mode='soft')
        cD = pywt.threshold(cD, threshold, mode='soft')
        
        # 统计非零系数个数（这是DWT重构必需的）
        total_non_zero += np.count_nonzero(cA) + np.count_nonzero(cH) + np.count_nonzero(cV) + np.count_nonzero(cD)
        compressed_channels.append(pywt.waverec2((cA, (cH, cV, cD)), 'db1'))
        
    comp_raw = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    comp = np.array(Image.fromarray(comp_raw).resize(orig_size, Image.BILINEAR))
    p, s, cr = calc_fair_metrics(orig_A, comp, total_non_zero)
    return comp, p, s, cr, (time.time() - start) * 1000