# Cahn-Net：受 Cahn–Hilliard 启发的单幅图像去雾

Cahn-Net 是一个用于单幅图像去雾的 U 形网络。编码器与解码器的每个阶段在若干残差块之后接入 Cahn 注意力引导模块（Cahn-AG），并通过多尺度特征融合与三尺度输出完成图像重建。本仓库包含训练、测试、数据加载与指标计算的完整代码。

> 论文：*Cahn-Net: Cahn–Hilliard Inspired Feature Evolution for Single Image Dehazing*（投稿中）

---

## 1. 目录结构

```
CahnNet/
├── train.py                 # 训练入口（PyTorch Lightning）
├── test.py                  # 测试入口：PSNR / SSIM、结果保存、单图推理
├── option.py                # 命令行参数
├── data_utils.py            # 三个数据集类：real_Dataset / RESIDE_Dataset / outRESIDE_Dataset
├── C2R.py                   # 对比正则项（可选，默认不启用）
├── model/
│   └── cahnet/
│       ├── cahnet.py        # CahnNet 主体：EBlock / DBlock / SCM / FAM / 三尺度输出
│       └── layers.py        # BasicConv / ResBlock / CahnAG / LayerNorm 等基础层
├── utils/
│   ├── val_utils.py         # PSNR / SSIM / NIQE 计算与计时工具
│   └── schedulers.py        # CosineAnnealingRestartCyclicLR 学习率调度
└── pytorch-gradual-warmup-lr/   # 学习率 warmup 工具（可选）
```

---

## 2. 环境依赖

已验证环境：Python 3.12.4、PyTorch 2.5.1、PyTorch Lightning 2.6.6；训练使用单卡或多卡（DDP）NVIDIA GPU。

```bash
pip install torch torchvision
pip install lightning
pip install einops pytorch_wavelets thop tqdm pillow numpy
pip install scikit-image pytorch_msssim scikit-video
```

- `pytorch_wavelets`：`model/cahnet/layers.py` 中小波下采样（`DWTForward`）使用；
- `thop`：`cahnet.py` 末尾统计 FLOPs / 参数量使用；
- `scikit-video`：`utils/val_utils.py` 中 NIQE 使用（部分平台需要额外安装 ffmpeg）。

---

## 3. 数据准备

数据目录由各类数据集类的构造函数决定，切换数据集只需改路径。

### 3.1 真实场景配对数据（NH-HAZE、Dense-HAZE、O-Haze 等）

使用 `real_Dataset`，目录约定：

```
<dataset_root>/
├── hazy/
│   ├── dataset/        # 训练/测试用的有雾图（origin=False）
│   └── origin/         # 原始分辨率版本（origin=True）
└── clear/
    ├── dataset/        # 与有雾图同名的清晰图
    └── origin/
```

- `origin=False`：清晰图与有雾图**同名**（`0001.png` ↔ `0001.png`）；
- `origin=True`：清晰图命名为 `<id>_GT<后缀>`（`44_013.png` ↔ `44_GT.png`），`id` 取文件名第一个 `_` 之前的部分。

### 3.2 合成配对数据（Haze4K、RESIDE 等）

使用 `RESIDE_Dataset`：

```
<dataset_root>/
├── hazy/      # 有雾图
└── clear/     # 训练集清晰图
```

测试集把清晰图放在 `gt/` 下（`train=False` 时读取 `gt/`）。文件名以 `_` 前的编号匹配，例如 `1_0.9_0.2.jpg` ↔ `1.png`。

### 3.3 O-Haze 风格数据（可选）

`outRESIDE_Dataset` 读取 `hazy/hazy_images` 与 `clear/clear_images`（训练），测试时读取 `hazy/` 与 `gt2/`。

> 数据增强：训练时随机水平翻转 + 随机 90° 旋转；输入按 `mean=[0.64, 0.60, 0.58]`、`std=[0.14, 0.15, 0.152]` 归一化，清晰图只转张量不做归一化。

---

## 4. 训练

```bash
python train.py
```

---

## 5. 测试

```bash
python test.py
```

---

## 6. 模型结构

```
输入 I（3×H×W）
├── patch_embed: Conv3×3 → 32 通道
├── Encoder
│   ├── EBlock(32)   → 下采样 ×2 + FAM 融合 SCM(2×)
│   ├── EBlock(64)   → 下采样 ×2 + FAM 融合 SCM(4×)
│   └── EBlock(128)
├── Decoder
│   ├── DBlock(128)  → 输出 1/4 尺度结果
│   ├── DBlock(64)   → 与编码器特征拼接、1×1 归约 → 输出 1/2 尺度结果
│   └── DBlock(32)   → 与编码器特征拼接、1×1 归约 → 输出全尺度结果
└── 三尺度输出：[1/4 残差, 1/2 残差, 全尺度残差]（各与对应尺度输入相加）
```

- `EBlock` / `DBlock`：6 个 `ResBlock`（Conv3×3 + GELU）之后接一个 `CahnAG` 模块（`model/cahnet/layers.py`），用于在块级特征上做调制；`CahnAG` 的输入为块输出与块输入两个张量。
- `SCM`：对 1/2、1/4 尺度输入做浅层特征提取，经 `FAM` 与主干下采样特征融合，补充多尺度信息。
- 训练与测试默认取三尺度输出中的**全尺度结果**（`net(x)[2]`），另外两个尺度可作为深监督目标。

**参数量与 FLOPs**：

```bash
python model/cahnet/cahnet.py
```

脚本会打印参数量、`thop` 统计的 FLOPs（输入 1×3×256×256）、30 次前向的平均耗时，以及三个输出的尺寸。

---

## 7. 命令行参数（option.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset_dir` | `/media/StudentGroup/LZ/Dataset/` | 数据根目录（入口脚本中实际使用的是硬编码路径） |
| `--model_dir` | `./trained_models/ITS_0707` | 模型/日志目录 |
| `--bs` | 8 | batch size |
| `--crop` / `--crop_size` | True / 128 | 是否随机裁剪及裁剪尺寸 |
| `--cl_lambda` | 0.25 | 对比正则项系数 |
| `--loss_weight` | 0.3 | 对比正则项权重 |
| `--clcrloss` | True | 是否启用对比正则项 |
| `--norm` | True | 输入是否归一化 |
| `--ablation_type` | 0 | 消融类型开关 |
| `--clip` | False | 是否做梯度裁剪（最大范数 0.2） |
| `--beta1` / `--beta2` | 0.9 / 0.999 | AdamW 的 beta 参数 |

---

## 8. 引用

```bibtex
@article{cahnnet2026,
  title   = {Cahn-Net: Cahn--Hilliard Inspired Feature Evolution for Single Image Dehazing},
  author  = {Liu, Zhang and Ren, Zhengwei and Fu, Feiran and Fang, Ming},
  journal = {IEEE Transactions on Computational Imaging},
  year    = {2026},
  note    = {under review}
}
```
