import time
import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from svd_engine import my_svd 

# --- 通用指标计算 ---
def calc_metrics(orig_arr, comp_img, alg_name, param):
    H, W, C = orig_arr.shape
    total_pixels = H * W * C
    
    # 指标
    p = psnr(orig_arr.astype(np.uint8), comp_img)
    s = ssim(orig_arr.astype(np.uint8), comp_img, channel_axis=2)
    
    # 压缩比 (CR) 计算逻辑
    if alg_name in ['SVD', 'PCA']:
        # 存储 k 个奇异值/特征值及对应向量
        params = param * (H + W + 1)
        cr = total_pixels / (params * C)
    elif alg_name == 'DCT':
        # 保留 k*k 低频系数
        cr = total_pixels / ((param * param) * C)
    else: # DWT
        # 简单估计：保留系数比例大约为 (1 - threshold)
        cr = 1.0 / (0.1 + param * 0.5)
        
    return p, s, cr

# SVD算法压缩
def process_image_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    A = orig_A / 255.0
    comp_channels = [my_svd(A[:,:,i], k=k)[0] @ np.diag(my_svd(A[:,:,i], k=k)[1]) @ my_svd(A[:,:,i], k=k)[2] for i in range(3)]
    comp = np.clip(np.stack(comp_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    p, s, cr = calc_metrics(orig_A, comp, 'SVD', k)
    return comp, p, s, cr, (time.time() - start) * 1000

def process_pca_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    comp_channels = []
    for i in range(3):
        ch = orig_A[:, :, i]
        mean = np.mean(ch, axis=0)
        U, S, VT = my_svd(ch - mean, k=k)
        comp_channels.append((U @ np.diag(S) @ VT) + mean)
    comp = np.clip(np.stack(comp_channels, axis=2), 0, 255).astype(np.uint8)
    p, s, cr = calc_metrics(orig_A, comp, 'PCA', k)
    return comp, p, s, cr, (time.time() - start) * 1000

# def process_dct_compression(image_file, k):
#     start = time.time()
#     img = Image.open(image_file).convert('RGB')
#     orig_A = np.array(img, dtype=float)
#     comp_channels = []
#     for i in range(3):
#         c = orig_A[:, :, i]
#         dct_c = dct(dct(c.T, norm='ortho').T, norm='ortho')
#         mask = np.zeros_like(dct_c)
#         mask[:k*2, :k*2] = 1 # 稍微放大以保证效果
#         comp = idct(idct((dct_c * mask).T, norm='ortho').T, norm='ortho')
#         comp_channels.append(comp)
#     comp = np.clip(np.stack(comp_channels, axis=2), 0, 255).astype(np.uint8)
#     p, s, cr = calc_metrics(orig_A, comp, 'DCT', k)
#     return comp, p, s, cr, (time.time() - start) * 1000
def process_dct_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    H, W = orig_A.shape[:2]
    
    compressed_channels = []
    for i in range(3):
        c = orig_A[:, :, i]
        # 二维 DCT
        dct_c = dct(dct(c.T, norm='ortho').T, norm='ortho')
        
        # --- 优化点：不再是固定大小的 k*2，而是根据图片分辨率的比例 ---
        # k 越大，保留的比例越高
        ratio = min(k / 64.0, 1.0) 
        h_k, w_k = int(H * ratio), int(W * ratio)
        
        mask = np.zeros_like(dct_c)
        mask[:h_k, :w_k] = 1.0 
        
        # 增加一个平滑处理：保留极少量极高频信息，减少振铃现象
        mask[h_k:h_k+2, w_k:w_k+2] = 0.5 
        
        dct_comp = dct_c * mask
        
        # 逆 DCT
        comp = idct(idct(dct_comp.T, norm='ortho').T, norm='ortho')
        compressed_channels.append(comp)
        
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    p, s, cr = calc_metrics(orig_A, comp, 'DCT', k)
    return comp, p, s, cr, (time.time() - start) * 1000

def process_dwt_compression(image_file, threshold):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_size, orig_A = img.size, np.array(img, dtype=float)
    A = orig_A / 255.0
    comp_channels = []
    for i in range(3):
        coeffs = pywt.wavedec2(A[:, :, i], 'db1', level=1)
        coeffs_list = list(coeffs)
        for j in range(1, len(coeffs_list)):
            coeffs_list[j] = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs_list[j]))
        comp_channels.append(pywt.waverec2(coeffs_list, 'db1'))
    comp_raw = np.clip(np.stack(comp_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    comp = np.array(Image.fromarray(comp_raw).resize(orig_size, Image.BILINEAR))
    p, s, cr = calc_metrics(orig_A, comp, 'DWT', threshold)
    return comp, p, s, cr, (time.time() - start) * 1000