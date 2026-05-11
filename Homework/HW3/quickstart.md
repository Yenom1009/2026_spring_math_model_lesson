# 快速开始指南

## 项目概述

本项目实现了平面点列曲线拟合算法，包含参数化方法、插值/拟合算法、评价指标和可视化功能。

**对应报告章节：**
- 实验1（4.1节）：参数化方式对比
- 实验2（4.2节）：噪声鲁棒性测试
- 实验3（4.3节）：采样密度影响
- 实验4（4.4节）：多种基函数对比
- 选做1（第5章）：傅里叶级数可视化

---

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包：**
- `numpy>=1.20.0` - 数值计算
- `scipy>=1.7.0` - 科学计算（插值、样条）
- `matplotlib>=3.4.0` - 可视化

---

## 代码文件说明

### 核心库模块（5个）
| 文件 | 功能 | 报告对应 |
|------|------|---------|
| `data_generator.py` | 生成测试曲线（心形、星形等） | 所有实验 |
| `parameterization.py` | 三种参数化方法（Uniform/Chord/Centripetal） | 4.1节 |
| `curve_fitting.py` | 拟合算法（三次样条插值/平滑样条） | 4.1-4.3节 |
| `metrics.py` | 评价指标（RMSE、Hausdorff等） | 所有实验 |
| `visualize.py` | 可视化函数 | 所有实验 |

### 实验脚本（4个）
| 文件 | 功能 | 报告对应 |
|------|------|---------|
| `main_experiments.py` | 实验1+2 | 4.1节+4.2节 |
| `regenerate_exp3_revised.py` | 实验3 | 4.3节 |
| `experiment_9_final.py` | 实验4 | 4.4节 |
| `fourier_star_gui.py` | 傅里叶可视化（星形） | 5.1节 |
| `fourier_draw_gui.py` | 傅里叶可视化（自由绘制） | 5.2节 |

### 辅助模块
| 文件 | 功能 |
|------|------|
| `experiment_logger.py` | 实验日志记录 |

---

## 运行实验

### 方式1：运行所有实验（推荐）

```bash
# 运行实验1+2（生成exp1和exp2图片）
python main_experiments.py

# 运行实验3（生成exp3_revised图片）
python regenerate_exp3_revised.py

# 运行实验4（生成exp9_final图片）
python experiment_9_final.py
```

### 方式2：运行单个实验

#### 实验1：参数化方式对比（报告4.1节）
```bash
python main_experiments.py
```
**生成图片：** `figures/exp1_*.png`（21张）
- 对比Uniform、Chord Length、Centripetal三种参数化方法
- 测试曲线：Circle、Heart、Star、S-Curve
- 包含均匀采样和非均匀采样

#### 实验2：噪声鲁棒性测试（报告4.2节）
```bash
python main_experiments.py
```
**生成图片：** `figures/exp2_*.png`（6张）
- 对比插值方法 vs 平滑拟合方法
- 噪声水平：2%、5%、10%
- 测试曲线：Heart、Ellipse

#### 实验3：采样密度影响（报告4.3节）
```bash
python regenerate_exp3_revised.py
```
**生成图片：** `figures/exp3_revised_*.png`（3张）
- 对比插值 vs 拟合在不同采样密度下的表现
- 采样密度：n=10, 20, 40, 80
- 测试曲线：Heart、Fixed_Random_Blob、S-Curve

#### 实验4：多种基函数对比（报告4.4节）
```bash
python experiment_9_final.py
```
**生成图片：** `figures/exp9_final_*.png`（9张）
- 对比多项式、FFT、RBF、样条等方法
- 测试场景：干净数据、2%噪声、5%噪声
- 测试曲线：Heart、Star

---

## 傅里叶可视化（选做1，报告第5章）

### 星形曲线傅里叶级数可视化
```bash
python fourier_star_gui.py
```
**功能：**
- 实时显示傅里叶级数的圆周叠加过程
- 可调节傅里叶项数（1-50项）
- 动画演示曲线绘制过程

### 自由绘制曲线傅里叶分析
```bash
python fourier_draw_gui.py
```
**功能：**
- 鼠标绘制任意闭合曲线
- 自动进行傅里叶级数分解
- 显示圆周叠加动画

---

## 输出结果

### 图片输出
所有图片保存在 `figures/` 目录：
- `exp1_*.png` - 实验1结果（21张）
- `exp2_*.png` - 实验2结果（6张）
- `exp3_revised_*.png` - 实验3结果（3张）
- `exp9_final_*.png` - 实验4结果（9张）

### 数据日志
实验数据保存在 `experiment_logs/` 目录：
- `experiment_summary.csv` - 所有实验的指标汇总
- `*_comparison.md` - Markdown格式的对比表格

## 代码结构

```
HW3/
├── 核心库模块/
│   ├── data_generator.py      # 曲线生成
│   ├── parameterization.py    # 参数化方法
│   ├── curve_fitting.py       # 拟合算法
│   ├── metrics.py             # 评价指标
│   └── visualize.py           # 可视化
├── 实验脚本/
│   ├── main_experiments.py           # 实验1+2
│   ├── regenerate_exp3_revised.py    # 实验3
│   ├── experiment_9_final.py         # 实验4
│   ├── fourier_star_gui.py           # 选做1a
│   └── fourier_draw_gui.py           # 选做1b
├── 辅助模块/
│   └── experiment_logger.py   # 日志记录
├── 输出目录/
│   ├── figures/               # 图片输出
│   └── experiment_logs/       # 数据日志
└── 文档/
    ├── requirements.txt       # 依赖列表
    ├── quickstart.md         # 本文档
    └── README.md             # 项目说明
```
