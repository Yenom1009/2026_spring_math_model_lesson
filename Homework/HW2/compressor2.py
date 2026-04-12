import time
import numpy as np
import pywt
import io
from PIL import Image
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from svd_engine import my_svd 

# --- 新增：真实存储体积计算工具 ---
def get_real_storage_size(data_dict):
    """
    模拟真实文件存储：将算法必须存储的参数放入二进制流，计算其占用的字节数。
    使用 savez_compressed 模拟工业级的二进制压缩存储。
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, **data_dict)
    return len(buf.getvalue())

# --- 修改：通用指标计算 ---
def calc_metrics(orig_arr, comp_img, real_comp_size):
    # 原始图像大小 (假设为 Raw 存储: H * W * 3 字节)
    orig_size = orig_arr.nbytes
    
    # 指标计算
    p = psnr(orig_arr.astype(np.uint8), comp_img)
    s = ssim(orig_arr.astype(np.uint8), comp_img, channel_axis=2)
    
    # 真实压缩比 = 原始字节数 / 压缩参数存储字节数
    cr = orig_size / real_comp_size
    return p, s, cr

# --- SVD 算法压缩 ---
def process_image_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    A = orig_A / 255.0
    
    compressed_channels = []
    storage_data = {}
    
    for i in range(3):
        # 注意：这里改为只调用一次 my_svd，提高效率
        U, S, VT = my_svd(A[:,:,i], k=k)
        compressed_channels.append(U @ np.diag(S) @ VT)
        
        # 记录该通道必须存储的数据
        storage_data[f'u_{i}'] = U
        storage_data[f's_{i}'] = S
        storage_data[f'vt_{i}'] = VT
        
    comp = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    
    # 计算真实压缩大小
    real_size = get_real_storage_size(storage_data)
    p, s, cr = calc_metrics(orig_A, comp, real_size)
    
    return comp, p, s, cr, (time.time() - start) * 1000

# --- PCA 算法压缩 ---
def process_pca_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    
    compressed_channels = []
    storage_data = {}
    
    for i in range(3):
        ch = orig_A[:, :, i]
        mean = np.mean(ch, axis=0)
        # 中心化处理
        U, S, VT = my_svd((ch - mean)/255.0, k=k)
        compressed_channels.append((U @ np.diag(S) @ VT)*255.0 + mean)
        
        # 记录存储数据：U, S, VT 以及 均值mean(恢复图像必需)
        storage_data[f'u_{i}'] = U
        storage_data[f's_{i}'] = S
        storage_data[f'vt_{i}'] = VT
        storage_data[f'm_{i}'] = mean
        
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    
    real_size = get_real_storage_size(storage_data)
    p, s, cr = calc_metrics(orig_A, comp, real_size)
    
    return comp, p, s, cr, (time.time() - start) * 1000

# --- DCT 算法压缩 ---
def process_dct_compression(image_file, k):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    H, W = orig_A.shape[:2]
    
    compressed_channels = []
    storage_data = {}
    
    ratio = min(k / 64.0, 1.0) 
    h_k, w_k = int(H * ratio), int(W * ratio)
    
    for i in range(3):
        c = orig_A[:, :, i]
        dct_c = dct(dct(c.T, norm='ortho').T, norm='ortho')
        
        mask = np.zeros_like(dct_c)
        mask[:h_k, :w_k] = 1.0 
        mask[h_k:h_k+2, w_k:w_k+2] = 0.5 
        
        # 存储截断后的系数块 (这是 DCT 压缩实际存的东西)
        storage_data[f'dct_{i}'] = dct_c[:h_k+2, :w_k+2]
        
        dct_comp = dct_c * mask
        comp = idct(idct(dct_comp.T, norm='ortho').T, norm='ortho')
        compressed_channels.append(comp)
        
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    
    real_size = get_real_storage_size(storage_data)
    p, s, cr = calc_metrics(orig_A, comp, real_size)
    
    return comp, p, s, cr, (time.time() - start) * 1000

# --- DWT 算法压缩 ---
def process_dwt_compression(image_file, threshold):
    start = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_size, orig_A = img.size, np.array(img, dtype=float)
    A = orig_A / 255.0
    
    compressed_channels = []
    storage_data = {}
    
    for i in range(3):
        coeffs = pywt.wavedec2(A[:, :, i], 'db1', level=1)
        coeffs_list = list(coeffs)
        # 阈值处理
        for j in range(1, len(coeffs_list)):
            coeffs_list[j] = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs_list[j]))
        
        # 存储阈值化后的稀疏小波系数
        storage_data[f'dwt_{i}_approx'] = coeffs_list[0]
        storage_data[f'dwt_{i}_detail'] = coeffs_list[1]
        
        compressed_channels.append(pywt.waverec2(coeffs_list, 'db1'))
        
    comp_raw = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    comp = np.array(Image.fromarray(comp_raw).resize(orig_size, Image.BILINEAR))
    
    real_size = get_real_storage_size(storage_data)
    p, s, cr = calc_metrics(orig_A, comp, real_size)
    
    return comp, p, s, cr, (time.time() - start) * 1000