<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <a href="README.md">English</a> |
  <strong>简体中文</strong> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ro.md">Română</a>
</p>

本仓库提供 **NeuroTS-Net** 的训练、评估、推理和后处理代码。NeuroTS-Net 是一种面向多模态 MRI 儿童脑肿瘤分割的三维多类别语义分割架构。

NeuroTS-Net 架构由 Darius Peteleaza 开发。如对论文或代码有任何疑问，请联系 [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

本代码对应论文：**NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

该论文目前正作为 [BraTS 挑战赛](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097)参赛工作在 MICCAI 2026 审稿中。

- MICCAI/BraTS 论文链接：待添加。
- arXiv 预印本链接：待添加。

> [**引用。**](#how-to-cite) 如果您使用 NeuroTS-Net 代码、架构或配套论文中的材料，请引用我们的论文。本仓库采用 [知识共享署名 4.0 国际许可协议](LICENSE)。

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="NeuroTS-Net 架构" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="NeuroTS 模块" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="NeuroTS 下采样机制" width="32%"></a>
</p>
<p align="center"><em>从左到右：NeuroTS-Net 架构概览、NeuroTS 模块和 NeuroTS 下采样机制。</em></p>

## 环境要求

- 已测试的操作系统：Windows 11 和 Ubuntu 22.04 LTS（本代码设计为可在 Python 3.11、PyTorch 及所列依赖项支持的任何操作系统上运行）。
- Python 3.11。
- 强烈建议使用支持 CUDA 的 GPU 进行训练和推理。也支持 CPU 执行，但速度会明显更慢。

标准安装请安装运行时依赖：

```bash
python -m pip install -r requirements.txt
```

如需可编辑的开发安装（包括测试依赖），请使用：

```bash
python -m pip install -e ".[dev]"
```

## 配置

本项目使用 YAML 配置文件定义数据路径与划分方式、模型架构、损失函数、训练、数据增强、推理和输出设置。

有关所有字段的逐项说明，请参阅[配置参考](configs/CONFIGURATION.md)。

## 数据集

默认配置面向 BraTS-PED 数据，使用以下模态和标签：

| 通道/标签 | 含义 |
|---|---|
| t1n | 原生 T1 加权 MRI |
| t1c | 对比增强 T1 加权 MRI |
| t2w | T2 加权 MRI |
| t2f | T2-FLAIR MRI |
| 0 | 背景 |
| 1 | 增强肿瘤（ET） |
| 2 | 非增强肿瘤（NET） |
| 3 | 囊性成分（CC） |
| 4 | 水肿（ED） |

预期的数据集目录结构如下：

~~~text
data/
|-- BraTS26_PED_training/
|   |-- BraTS-PED-00001-000/
|   |   |-- BraTS-PED-00001-000-t1n.nii.gz
|   |   |-- BraTS-PED-00001-000-t1c.nii.gz
|   |   |-- BraTS-PED-00001-000-t2w.nii.gz
|   |   |-- BraTS-PED-00001-000-t2f.nii.gz
|   |   |-- BraTS-PED-00001-000-seg.nii.gz
|   |-- ...
|-- BraTS26_PED_training_Batch2_Release/
|   |-- ...
|-- BraTS26_PED_validation/
    |-- BraTS-PED-XXXXX-XXX/
    |   |-- BraTS-PED-XXXXX-XXX-t1n.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t1c.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2w.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2f.nii.gz
    |-- ...
~~~

在进行数据划分之前，系统会按病例 ID 对多个训练根目录中的受试者进行去重。

## 数据预处理

预处理流程如下：

1. 读取四种 NIfTI 模态，不进行重采样；
2. 在所有模态上计算非零前景包围盒；
3. 按配置的边距扩展包围盒（默认为 32 个体素）；
4. 按前景强度百分位数裁剪每种模态的强度；
5. 仅在前景上执行 z-score 标准化；
6. 将图像保存为 float32 数组，同时保存标签和空间元数据。

构建可复用缓存：

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

重建已有缓存条目：

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

预测结果会被粘贴回原始图像形状，并使用源图像的仿射矩阵、体素间距、空间方向、qform 和 sform 元数据保存。

## 模型训练

同一个训练入口同时支持分组交叉验证和一个显式训练/验证划分。

### 五折交叉验证

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

如果折叠定义文件不存在，系统会根据缓存的训练病例自动生成。仅训练一个指定折叠：

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### 单一训练/验证划分

创建按标签分层的高信号划分：

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

在该划分上训练：

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

当 `data.split_mode` 为 `single` 时，必须提供划分文件。若文件缺失，训练会明确报错，而不会静默回退到折叠模式。

## 采样与目标函数

公开配置复现了 NeuroTS-Net 使用的稀有区域感知训练方案：

- 对 ET、NET、CC 和 ED 进行基于连通分量的均衡块采样；
- 在无 ED 病例的非 ED 肿瘤组织上采样困难负例块；
- 加权交叉熵和逐样本标签 Dice；
- 针对 ET、NET、CC、ED、TC 和 WT 评估区域的 Tversky/Dice 项；
- 针对 ET、CC 和 ED 的类别缺失概率惩罚；
- 深监督、余弦学习率衰减和线性预热。

采样请求、成功、回退、类别存在情况和目标体素统计会写入训练日志和 `patch_sampling_latest.json`。

## 检查点与指标

每个训练目录可包含：

- `best_voxel_dice.pt`：块级验证平均区域 Dice 最优检查点；
- `best_rare_present_score.pt`：全体积稀有类别存在评分最优检查点。

稀有类别存在评估器会分别报告 GT 存在、预测存在、TP、FP 和 FN 病例。对于稀有区域，GT 与预测均为空的病例仅用于诊断，不会提高检查点选择分数。

训练输出包括紧凑的 CSV/JSON 指标历史、学习曲线、全体积验证报告、日志和资源快照。分支选择遥测信息不会被打印或导出。

验证 YAML 文件，包括重复键检测：

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## 推理

推理前先预处理验证数据：

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### 原始检查点或折叠集成

发现并平均一次运行中的所有稀有类别存在检查点：

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

可重复指定显式检查点以构建任意集成：

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### ET/CC 连通分量后处理

调优后的公开后处理会调整 ET 和 CC 的概率及连通分量，同时保留主模型的 ED 预测：

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

重复使用 `--checkpoint` 可平均多个检查点。NIfTI 文件会写入预测目录，`--zip` 会创建适合上传到挑战赛平台的扁平归档文件。

## 输出

生成的数据会写入配置的根目录下：

~~~text
cache/       预处理数组以及数据划分/折叠定义
outputs/     检查点、日志、指标、预测结果和归档文件
~~~

这些目录以及原始数据集、检查点和医学图像均已通过 `.gitignore` 排除。

## 测试

运行快速测试套件：

~~~bash
python -m pytest -q
~~~

运行模型前向/反向传播冒烟测试：

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## 引用

使用本仓库时，请引用我们的论文：

### 纯文本

```

```

### BibTeX

```

```
