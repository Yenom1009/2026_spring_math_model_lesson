# SVD 图像压缩实验 - 快速启动指南

## 快速开始

### Step 1：安装依赖（仅需一次）

```bash
pip install -r requirements.txt
```

### Step 2：准备测试图片

放置任意 JPG/PNG 图片到项目目录，或使用标准测试图 `peppers.jpg`

### Step 3：启动交互应用

```bash
streamlit run app1.py
```

浏览器自动打开 http://localhost:8501 

### Step 4：上传图片 → 调整参数 → 查看结果

完成！

---

## 三种使用场景

### 场景 A：交互式 GUI 实验

**目标：** 实时调整参数，观察压缩效果变化

```bash
streamlit run app2.py     # 进阶版，支持目标压缩比
```

**特点：**
- 直观的滑块参数调整
- 实时图像预览
- 数值指标（PSNR、SSIM、压缩比）
- 无需编码，点击即用

**使用流程：**
1. 在侧边栏上传图片
2. 选择压缩算法
3. 拖动滑块调整参数
4. 观察压缩结果和指标变化

---

### 场景 B：自动性能分析

**目标：** 生成 PSNR、SSIM、CR、耗时的完整分析图表

```bash
python generate_plots.py
```

**输出文件：** `svd_final_analysis.png`

**包含内容：**
- ✅ PSNR 随 k 值变化曲线
- ✅ SSIM 随 k 值变化曲线
- ✅ 真实压缩比 (CR) 随 k 变化
- ✅ 重构耗时随 k 变化

**用途：** 直接粘贴到实验报告 

**前置条件：** 工作目录需要 `peppers.jpg` 文件

---

### 场景 C：多算法对比

**目标：** 在相同压缩比下比较 4 种算法的性能

```bash
python compare_algorithm.py
```

**输出文件：** `comparison_fair_rd_curve.png`

**包含内容：**
- 横轴：压缩比（对数坐标）
- 纵轴：PSNR（图像质量）
- 4 条曲线分别代表：SVD、PCA、DCT、DWT

**解读说明：**
- 曲线越高越好（质量高）
- 曲线越向右越好（压缩比高）
- "左上角"是最优区域

**前置条件：** 工作目录需要 `peppers.jpg` 文件


## 文件对应关系速查

| 你想要... | 运行此文件 |
|---------|----------|
| 用 GUI 调参数 | `streamlit run app2.py` |
| 生成性能曲线 | `python generate_plots.py` |
| 算法对比曲线 | `python compare_algorithm.py` |
| 理解 SVD 原理 | 打开 `svd_engine.py` |
| 查看所有算法 | 打开 `compressor2.py` |


---

## 代码片段复用

### 片段 1：快速压缩一张图片

```python
from compressor2 import process_image_compression
from PIL import Image

# 压缩图片
comp_img, psnr, ssim, cr, time_ms = process_image_compression("input.jpg", k=50)

# 显示或保存
Image.fromarray(comp_img).save("output_compressed.jpg")
print(f"质量 (PSNR): {psnr:.1f} dB")
print(f"压缩比: {cr:.1f}x")
```

---

### 片段 2：批量测试多个 k 值

```python
from compressor2 import process_image_compression
import matplotlib.pyplot as plt

ks = [10, 20, 50, 100, 150]
psnrs = []

for k in ks:
    _, psnr, _, _, _ = process_image_compression("peppers.jpg", k=k)
    psnrs.append(psnr)

plt.plot(ks, psnrs, marker='o')
plt.xlabel("Rank (k)")
plt.ylabel("PSNR (dB)")
plt.savefig("psnr_curve.png")
```

---

### 片段 3：手动使用 SVD 分解

```python
from svd_engine import my_svd
import numpy as np
from PIL import Image

# 加载图片并取单通道
img = Image.open("peppers.jpg").convert('L')  # 转灰度
A = np.array(img, dtype=float) / 255.0

# SVD 分解（保留 k=50 个分量）
U, S, VT = my_svd(A, k=50)

# 重建
reconstructed = U @ np.diag(S) @ VT
reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

Image.fromarray(reconstructed_uint8).save("svd_reconstructed.jpg")
```

