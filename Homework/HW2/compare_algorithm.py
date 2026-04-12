import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from compressor2 import *

def generate_report_plots(image_path):
    # 预处理
    img = Image.open(image_path).convert('RGB').resize((300, 300))
    img.save("temp_test.jpg")
    
    results = {alg: {"cr": [], "psnr": []} for alg in ["SVD", "PCA", "DCT", "DWT"]}
    
    # 扫描参数
    ks = [2, 5, 10, 20, 40, 60, 100, 150]
    ths = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]

    for k in tqdm(ks, desc="扫描 SVD/PCA/DCT"):
        _, p, _, cr, _ = process_image_compression("temp_test.jpg", k)
        results["SVD"]["cr"].append(cr); results["SVD"]["psnr"].append(p)
        _, p, _, cr, _ = process_pca_compression("temp_test.jpg", k)
        results["PCA"]["cr"].append(cr); results["PCA"]["psnr"].append(p)
        _, p, _, cr, _ = process_dct_compression("temp_test.jpg", int(k/200*64)+1)
        results["DCT"]["cr"].append(cr); results["DCT"]["psnr"].append(p)

    for th in tqdm(ths, desc="扫描 DWT"):
        _, p, _, cr, _ = process_dwt_compression("temp_test.jpg", th)
        results["DWT"]["cr"].append(cr); results["DWT"]["psnr"].append(p)

    plt.figure(figsize=(12, 7))
    
    # 再次确保配置生效
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    colors = ['red', 'blue', 'green', 'orange']
    alg_names = ["SVD", "PCA", "DCT", "DWT"]

    for alg, color in zip(alg_names, colors):
        if not results[alg]["cr"]: continue
        
        idx = np.argsort(results[alg]["cr"])
        x = np.array(results[alg]["cr"])[idx]
        y = np.array(results[alg]["psnr"])[idx]
        
        # 给 PCA 一个极小的偏移以便观察重合线
        if alg == "PCA":
            y = y + 0.2
            label = "PCA (微调偏移)"
        else:
            label = alg
            
        plt.plot(x, y, label=label, color=color, marker='o', markersize=4, alpha=0.8)

    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # 使用中文字符串
    plt.xlabel("压缩比 (CR) - 对数轴 [原始大小 / 压缩参数量]", fontsize=12)
    plt.ylabel("图像质量 [PSNR (dB)]", fontsize=12)
    plt.title("多算法率失真曲线对比 (Rate-Distortion)", fontsize=14)
    plt.legend()
    
    # 强制保存为图片，方便你贴入报告
    plt.savefig('comparison_fair_rd_curve.png', dpi=300)
    print("\n✅ 图片已保存为 'comparison_fair_rd_curve.png'，请在文件夹中直接打开查看。")
    plt.show()

if __name__ == "__main__":
    generate_report_plots("peppers.jpg")