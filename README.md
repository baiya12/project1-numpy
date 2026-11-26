# 项目一：用 NumPy 复现 OpenCLIP ViT-B-32 视觉编码器

用**纯 NumPy**（无 PyTorch、无 OpenCLIP 包、无 GPU）实现 OpenCLIP ViT-B-32 视觉塔的前向计算，
输出与独立 FP32 参考实现数值一致。

`python check.py --stage all` → **7/7 PASS**｜输出 `[10,512]` 单位长度向量｜10 图前向 **3.19 s**（CPU）

---

## 功能与接口

| 项 | 内容 |
| --- | --- |
| 输入 | `float32 [B,3,224,224]`（课程提供的 resize + center crop + CLIP 归一化之后） |
| 输出 | `float32 [B,512]`，每张图一个单位长度向量 |
| 内部 | 49 个 32×32 patch + 1 个 CLS = 50 token；D=768，H=12，d_h=64，12 层，MLP 3072 |
| 判定 | 形状一致、`float32`、无 NaN/Inf，且 `|Δ| ≤ atol + rtol·|ref|`（output 阶段 atol=2e-5） |

六个实现点全部在 `image_encoder.py`：

| 编号 | 方法 | 内容 | 关键形状变化 |
| --- | --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | 切 32×32 块，块内按 C,H,W 展平 | `[B,3,224,224]` → `[B,49,3072]` |
| TODO-02 | `MultiHeadAttention.split_heads` | 特征轴切成 12 个连续 64 维头 | `[B,N,768]` → `[B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | 缩放点积打分、加权、合并头、输出投影 | `[B,12,N,N]` → `[B,N,768]` |
| TODO-04 | `TransformerBlock.__call__` | 两个残差 + MLP | `[B,50,3072]` → `[B,50,768]` |
| TODO-05 | `ImageEncoder.embed` | patch 投影（无 bias）、拼 CLS、加位置编码 | `[B,49,768]` → `[B,50,768]` |
| TODO-06 | `ImageEncoder.__call__` | 取 CLS、ln_post、乘 `visual.proj`、L2 归一化 | `[B,50,768]` → `[B,512]` |

## 快速开始

```bash
python -m pip install -r requirements.txt     # numpy>=1.24,<3  Pillow>=9.2,<13
python check.py --stage assets                # 校验 152 个张量的 sha256
python check.py --stage all                   # 7 项分阶段对拍，应全部 PASS
python run.py --output output_tokens.npz      # 生成 [10,512] 输出
```

命令与工作目录无关（脚本按自身路径定位模型与图片）。退出码：`0` 全过、`1` 数值错、`2` 还有未实现的 TODO。

> 复现需要在课程附件的 `weights/image_encoder.npz` 与 `assets/reference_outputs.npz` 存在时才成立，
> 这两个文件体积大/属课程材料，未纳入本仓库（见下文“模型与数据”）。

## 实测结果（`check.py --stage all` 原始输出）

| 阶段 | 输出形状 | max_abs | mean_abs | atol | 结论 |
| --- | --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0.000e+00 | 0.000e+00 | 2e-6 | PASS |
| patches | (2,49,3072) | 0.000e+00 | 0.000e+00 | 2e-4 | PASS |
| tokens | (2,50,768) | 4.387e-05 | 1.335e-06 | 2e-4 | PASS |
| block0 | (2,50,768) | 7.147e-06 | 3.348e-07 | 2e-4 | PASS |
| last_tokens | (2,50,768) | 4.196e-05 | 2.633e-06 | 2e-4 | PASS |
| output | (10,512) | 5.662e-07 | 8.688e-08 | 2e-5 | PASS |
| batch `[1,0,1]` | (3,512) | 4.061e-07 | 7.848e-08 | 2e-5 | PASS |

- `‖x‖₂` 与 1 的最大偏差：**2.38e-07**；批大小 1 时返回 `[1,512]`，重排 + 重复图结果一致。
- 10 图完整前向：**3.19 s**（AMD Ryzen 7 H 260，单线程 NumPy；精确 GELU 逐元素实现是主要开销）。
- `patches` 误差恰为 0：patchify 只重排内存，不引入算术，因此应逐位相同——可作为轴序正确性的判据。

## 目录结构

```
project1-numpy/
├── README.md                  # 本文件
├── requirements.txt
├── .gitignore                 # 排除 326 MB 权重与参考数组
├── image_encoder.py           # 唯一修改的源码：六组 TODO 实现
├── helpers.py                 # 给定：LayerNorm / 精确 GELU / softmax / unit_vector
├── preprocess_image.py        # 给定：resize + center crop + 归一化
├── runtime.py  run.py  check.py           # 给定：加载、命令行、分阶段对拍
├── config.json                # 固定架构参数
├── assets/                    # 10 张图片 + 参考数组（参考数组不入库）
├── weights/                   # 152 个视觉张量（不入库）
├── docs/
│   ├── 实验报告.docx           # 课程 Word 模板填写稿（正文宋体五号，6 页：封面+4 页正文+批阅页）
│   ├── 实验报告.pdf            # 上者导出的 PDF
│   ├── figures/               # 图 1 形状流水线、图 2 注意力系数
│   ├── metrics.json           # 报告中全部指标与消融数据的原始输出
│   └── 课程原始README.md       # 课程附件的原始说明，留存备查
├── practice/                  # 课程练习题 PDF
└── lecture-demos/             # 课堂示例脚本
```

## 实现要点

1. **patchify 必须三步走**：`reshape(B,3,7,32,7,32)` → `transpose(0,2,4,1,3,5)` → `reshape(B,49,3072)`。
   `reshape` 只给轴命名、不移动数据；必须先转置让块内三轴（C,32,32）在内存上连续，
   否则每 3072 个数会横跨多行像素（实测误差 3.81）。
2. **头是连续切段**：`reshape(B,N,12,64).transpose(0,2,1,3)`；合并是它的逆操作
   `transpose(0,2,1,3).reshape(B,N,768)`，封装成 `merge_heads` 以免忘记。
3. **权重方向不统一**：注意力/MLP 权重是 `[out,in]`（用 `.T`），
   但 `visual.proj` 已经是 `[in,out]=[768,512]`（**不要** `.T`）。
4. **两处 dtype/数值陷阱**：`k` 是 4 维，只能用 `transpose(0,1,3,2)`（`.T` 会反转全部轴）；
   缩放要写 `np.sqrt(np.float32(dh))`，`np.sqrt(python_int)` 是 float64，会把整条链路升级成 float64。
5. 残差基准是块输入 `x`（不是 `norm1(x)`），第二个残差加在 `residual` 上；本模型无 LayerScale、无 dropout。

## 模型与数据

- `weights/image_encoder.npz`：152 个视觉张量、87,849,216 个参数、**326 MB**，`sha256` 前 16 位
  `8567f805696fb374`。体积超过 GitHub 单文件上限，未入库；从课程附件获取后放在 `weights/` 即可。
- `assets/reference_outputs.npz`：各阶段参考数组（10 图），属课程评测材料，未入库；
  仅在 `check.py` 内部使用，**不作为编码器输入**。
- `assets/images/`：10 张课程图片，来源与 sha256 见 `assets/images.json`。

## 复现环境

Windows 11 家庭版 26200（AMD Ryzen 7 H 260，15.3 GB 内存），Python 3.12.7，
numpy 2.3.5，Pillow 12.3.0，无 GPU。

## 实验报告

`docs/实验报告.docx` 直接填写课程下发的 Word 模板（封面 + 四个固定小节 + 指导教师批阅页），
**未改动模板原有字体**，正文统一为宋体五号（10.5 pt）。PDF 见 `docs/实验报告.pdf`。

| 小节 | 内容 |
| --- | --- |
| 实验目的与要求 | 目标、输入输出契约、验收容差、固定结构参数、数据与权重规模 |
| 实验内容与方法 | 环境表（表 1）、形状流水线（图 1）、4 个公式、运行步骤与退出码 |
| 实验步骤与过程 | 六组 TODO 的实现顺序、四个陷阱、分阶段误差表（表 2）、输出性质证据、注意力热图（图 2）、消融表（表 3） |
| 实验结论或体会 | 结论、一个 shape 变化（patchify 六维重组）、一个 bug（头合并漏转置）、局限与改进、贡献与 AI 使用说明 |

全文 6 页：封面 1 页 + **正文 4 页**（不超过 5 页）+ 批阅页 1 页。文中所有数值可在
`docs/metrics.json` 与 `check.py --stage all` 的输出中复核。

## 贡献与 AI 使用说明

- 本人完成六组 TODO 的实现与调试、README 与仓库整理、报告数据采集（`docs/metrics.json`）。
- 本实现在 AI 助手（DeepSeek Harness）辅助下完成，主要用于核对 reshape/transpose 轴序、
  解释多头拆分的内存布局与整理报告文字；每条建议均以 `check.py` 的分阶段对拍验证后才保留。

## 参考

1. Ilharco G., et al. OpenCLIP, 2021. <https://github.com/mlfoundations/open_clip>
2. Dosovitskiy A., et al. An Image is Worth 16x16 Words. ICLR 2021.
3. Radford A., et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
4. NumPy v2 Reference: reshape, transpose, matmul. <https://numpy.org/doc/stable/>
