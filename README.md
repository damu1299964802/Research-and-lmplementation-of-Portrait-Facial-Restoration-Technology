# Research and Implementation of Portrait Facial Restoration Technology

本项目实现了一个面向人脸图像修复的双阶段/多阶段 GAN 修复系统，并提供训练、损失函数搜索、推理、超分辨率增强、质量评估和简易 Web/GUI 演示代码。项目主要用于在遮挡、破损或缺失区域上恢复人脸内容，并可在推理后追加超分辨率与锐化等后处理。

> 注意：仓库不包含训练数据集、训练输出、实验结果、InsightFace 模型缓存和大体积权重文件。请按下文准备本地数据和模型文件。

## 模型思路

核心流程由生成器、判别器、可配置损失函数和后处理模块组成：

1. **遮挡模拟与输入构造**：训练时随机生成 mask，将原图被遮挡区域替换为训练集平均像素值，再与 mask 拼接成 4 通道输入。
2. **生成器修复**：`Generator` 使用编码器-空洞卷积瓶颈-解码器结构输出修复图像；`GeneratorV2` 在此基础上增加 skip connection、残差瓶颈和 SE 通道注意力，以提升细节保留能力。
3. **判别器约束**：`Discriminator` 同时包含局部判别器和全局判别器，分别关注 mask 区域细节和整张人脸的整体一致性。
4. **多项损失组合**：支持重建损失、VGG 感知损失、人脸关键点损失、L1 损失、频域损失和 SSIM 损失，可通过 `--loss_combination` 切换组合。
5. **Poisson 融合与超分辨率**：推理阶段将生成区域与已知区域融合，可选 bicubic、OpenCV、SRCNN、ESPCN 等超分辨率增强策略。

训练脚本采用三段式流程：

- **Phase 1**：仅训练生成器，使模型先学会基础修复。
- **Phase 2**：冻结/使用生成器输出，预训练判别器。
- **Phase 3**：生成器与判别器联合训练，提升真实感和局部一致性。

## 项目结构

```text
Face-study/
├── train.py                  # Windows 训练脚本
├── train_linux.py            # Linux/CUDA 服务器训练脚本
├── predict.py                # 推理、自动查找模型、可选超分辨率和评估
├── test.py                   # 图像质量评估，包含 MSE/PSNR/SSIM 等指标
├── GUI.py                    # 桌面 GUI
├── models.py                 # Generator / GeneratorV2 / Discriminator
├── utils.py                  # 数据集、mask、裁剪、Poisson 融合等工具
├── losses.py                 # 混合损失
├── loss_functions.py         # 可配置损失函数组合
├── loss_search.py            # Windows 损失组合搜索
├── loss_search_linux.py      # Linux 损失组合搜索
├── super_resolution.py       # 超分辨率增强模块
├── metrics.py                # 评估指标
├── run_experiment.py         # 实验流水线
├── run_linux.sh              # Linux 运行脚本
├── config/                   # 示例配置
├── demo/                     # 示例模型/配置
└── web_app/                  # Flask Web 演示
```

以下目录需要本地准备或运行后自动生成，默认不上传到 GitHub：

```text
datasets/             # 训练集和测试集
result/               # 训练输出、checkpoint、可视化图片
results/              # 推理输出
evaluation_results/   # 评估报告
insightface_models/   # InsightFace buffalo_l 模型缓存
F_/                   # 本地备份/缓存目录
```

## 环境配置

推荐使用 Python 3.8、PyTorch 2.0.1 和 CUDA 11.8 版本的 PyTorch wheel。你的服务器 CUDA 12.2 可以运行 CUDA 11.8 wheel。

### 方式一：使用 env.yml

```bash
conda env create -f env.yml -n face-study
conda activate face-study
```

如果 PyTorch 没有安装到 CUDA 11.8 wheel，请用下面的手动方式重新安装 PyTorch。

### 方式二：手动安装

```bash
conda create -n face-study python=3.8 -y
conda activate face-study
python -m pip install --upgrade pip

pip install setuptools==65.5.0 packaging==21.3

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install onnxruntime-gpu==1.15.1
pip install numpy==1.23.5
pip install opencv-python-headless==4.8.0.74
pip install insightface==0.7.3
pip install albumentations==1.4.18
pip install tqdm Pillow scikit-image matplotlib scipy flask
```

## 数据准备

将人脸图片放入以下目录：

```text
datasets/train/
datasets/test/
```

代码默认读取 `.jpg` 和 `.png` 图片。Linux 服务器上也可以使用绝对路径，例如：

```text
/local_data/Face-study/datasets/train
/local_data/Face-study/datasets/test
```

InsightFace 会使用 `buffalo_l` 人脸检测和关键点模型。可通过环境变量指定缓存目录：

```bash
export INSIGHTFACE_ROOT=./insightface_models
```

首次运行时如网络可用，InsightFace 会自动下载模型；也可以手动把模型放到 `insightface_models/models/buffalo_l/`。

## 训练

Linux 服务器训练：

```bash
cd Face-study/Face-study
python train_linux.py \
  --phase1 8 \
  --phase2 2 \
  --phase3 40 \
  --loss_combination all \
  --data_dir /local_data/Face-study/datasets/train \
  --test_dir /local_data/Face-study/datasets/test \
  --result_dir ./result/
```

Windows 本地训练：

```bash
python train.py \
  --phase1 8 \
  --phase2 2 \
  --phase3 40 \
  --loss_combination all \
  --data_dir ./datasets/train \
  --test_dir ./datasets/test \
  --result_dir ./result/
```

可选损失组合：

| 参数 | 说明 |
| --- | --- |
| `reconstruction` | MSE 重建损失 |
| `perceptual` | VGG 感知损失 |
| `landmark` | 人脸关键点损失 |
| `recon_percep` | 重建 + 感知 |
| `recon_landmark` | 重建 + 关键点 |
| `percep_landmark` | 感知 + 关键点 |
| `all` | 重建 + 感知 + 关键点 |
| `v2_balanced` | V2 平衡组合 |
| `v2_detail` | V2 细节组合 |
| `v2_ssim` | V2 SSIM 优化组合 |
| `v2_full` | V2 完整组合 |

训练输出通常保存在：

```text
result/config.json
result/phase_1/
result/phase_2/
result/phase_3/
```

## Screen 训练管理

```bash
screen -S train
conda activate face-study
cd Face-study/Face-study
python train_linux.py

screen -ls
screen -r train
```

查看和停止训练进程：

```bash
ps aux | grep train_linux.py
pkill -f "python train_linux.py"
```

## 损失函数搜索

快速跳过完整训练，仅做损失组合搜索：

```bash
python loss_search_linux.py --skip_full_train
```

也可以使用完整实验流程：

```bash
python run_experiment.py \
  --train_dir ./datasets/train \
  --test_dir ./datasets/test \
  --result_dir ./result/experiments \
  --small_dataset_size 500 \
  --search_epochs 10 \
  --full_epochs 50
```

## 推理

如果 `result/` 下存在训练产物，`predict.py` 会自动尝试查找最新的生成器 checkpoint 和配置：

```bash
python predict.py --data ./datasets/test/ --sr_enable --sr_method bicubic --sr_scale 2
```

手动指定模型与配置：

```bash
python predict.py \
  ./result/phase_3/phase_3_model_generator_epoch40 \
  ./result/config.json \
  ./results/my_test/ \
  --data ./datasets/test/ \
  --sr_enable \
  --sr_method bicubic \
  --sr_scale 2 \
  --eval \
  --eval_hr
```

常用推理参数：

| 参数 | 说明 |
| --- | --- |
| `--data` | 输入图片目录 |
| `--mode cat` | 拼接批处理输出 |
| `--sr_enable` | 开启超分辨率 |
| `--sr_method` | `bicubic` / `srcnn` / `espcn` / `opencv` |
| `--sr_scale` | 超分倍率 |
| `--sharpen` / `--no-sharpen` | 是否锐化 |
| `--eval` | 评估普通分辨率输出 |
| `--eval_hr` | 评估超分输出 |
| `--model_version` | `auto` / `v1` / `v2` |

## 评估

```bash
python test.py \
  --original ./datasets/test \
  --restored ./results/my_test/completed \
  --output ./evaluation_results \
  --save_comparison
```

## Web/GUI 演示

桌面 GUI：

```bash
python GUI.py
```

Flask Web 应用：

```bash
cd web_app/web_app
pip install -r requirements.txt
python app.py
```

## GitHub 上传说明

`.gitignore` 已排除数据、训练结果、缓存和大体积模型文件，避免触发 GitHub 单文件 100MB 限制。若后续需要发布大模型权重，建议使用 GitHub Releases、对象存储或 Git LFS。
