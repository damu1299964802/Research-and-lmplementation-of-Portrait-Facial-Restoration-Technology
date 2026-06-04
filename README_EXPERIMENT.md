# 人脸修复与超分辨率增强系统 - 使用说明

## 项目简介

基于 GAN 的人脸修复系统，修复人脸图像中被遮挡/损坏的区域，可选超分辨率增强。采用三阶段训练流程（Generator 预训练 → Discriminator 预训练 → 联合训练），支持多种损失函数组合实验。

## 项目结构

```
Face-study/
├── train.py                # Windows 训练脚本（三阶段）
├── train_linux.py          # Linux 服务器训练脚本
├── predict.py              # 推理预测 + 评估
├── test.py                 # 图像质量评估（MSE/PSNR/SSIM）
├── GUI.py                  # 桌面 GUI 界面
├── models.py               # Generator / Discriminator 网络定义
├── utils.py                # 数据集加载、Mask生成、Poisson融合等工具
├── losses.py               # HybridLoss（Landmark + 感知损失）
├── loss_functions.py       # 可配置损失函数模块（6种组合）
├── loss_search.py          # Windows 损失函数自动搜索
├── loss_search_linux.py    # Linux 损失函数自动搜索
├── run_experiment.py       # 完整实验流程（搜索+训练）
├── run_linux.sh            # Linux 一键运行脚本
├── metrics.py              # PSNR / SSIM / FID 评估指标
├── super_resolution.py     # 超分辨率模块（SRCNN/ESPCN/Bicubic/OpenCV）
├── plot_model_architecture.py  # 模型架构图生成（需PlotNeuralNet）
├── demo/
│   ├── model_cn            # 预训练生成器模型（旧数据集）
│   └── config.json         # 预训练配置（旧数据集）
├── config/
│   └── config.json         # 本地配置文件
├── datasets/
│   ├── train/              # 训练集图片目录
│   └── test/               # 测试集图片目录
├── result/                 # 训练输出目录
│   ├── config.json         # 训练生成的配置（含新 mpv）
│   ├── phase_1/            # Phase 1 模型 + 可视化
│   ├── phase_2/            # Phase 2 模型 + 可视化
│   ├── phase_3/            # Phase 3 模型 + 可视化（最终模型在此）
│   └── loss_search/        # 损失函数搜索实验结果
├── results/                # 推理输出目录
├── insightface_models/     # InsightFace 人脸检测模型
└── web_app/                # Flask Web 应用
```

## 环境要求

- Python 3.8+
- PyTorch (CUDA 11.8)
- torchvision
- insightface
- opencv-python (cv2)
- scikit-image
- matplotlib
- numpy, pillow, tqdm, scipy

## 使用流程

### 一、准备数据集

将人脸图片（.jpg / .png）放入：
- 训练集：`./datasets/train/`
- 测试集：`./datasets/test/`

### 二、模型训练

#### 方式 A：标准三阶段训练

**Windows 本地：**
```bash
python train.py --phase1 8 --phase2 2 --phase3 40 \
    --loss_combination all \
    --data_dir ./datasets/train \
    --test_dir ./datasets/test \
    --result_dir ./result/
```

**Linux 服务器：**
```bash
python train_linux.py --phase1 8 --phase2 2 --phase3 40 \
    --loss_combination all \
    --data_dir /local_data/Face-study/datasets/train \
    --test_dir /local_data/Face-study/datasets/test \
    --result_dir ./result/
```

可选损失函数组合（`--loss_combination`）：
| 组合名 | 包含损失 |
|--------|----------|
| `reconstruction` | 仅 MSE 重建损失 |
| `perceptual` | 仅 VGG 感知损失 |
| `recon_percep` | 重建 + 感知 |
| `recon_landmark` | 重建 + 人脸关键点 |
| `percep_landmark` | 感知 + 人脸关键点 |
| `all` | 重建 + 感知 + 关键点（默认） |

训练输出：
- 模型文件：`./result/phase_3/phase_3_model_generator_epoch{N}`
- 配置文件：`./result/config.json`（包含该次训练的 mpv 值）
- 可视化：`./result/phase_{1,2,3}/phase_{1,2,3}_epoch{N}.png`

#### 方式 B：损失函数自动搜索 + 最优训练

```bash
# 完整流程（搜索 + 训练）
python run_experiment.py \
    --train_dir ./datasets/train \
    --test_dir ./datasets/test \
    --result_dir ./result/experiments \
    --small_dataset_size 500 \
    --search_epochs 10 \
    --full_epochs 50

# Linux 一键运行
bash run_linux.sh
```

### 三、推理预测

**⚠️ 重要：必须指定新训练的模型和对应的 config！**

```bash
python predict.py \
    ./result/phase_3/phase_3_model_generator_epoch40 \
    ./result/config.json \
    ./results/output/ \
    --data ./images/test/
```

常用参数：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode cat` | 批处理模式（所有图片拼接输出） | 单图模式 |
| `--sr_enable` | 启用超分辨率增强 | 关闭 |
| `--sr_method` | 超分辨率方法：bicubic/srcnn/espcn/opencv | bicubic |
| `--sr_scale` | 放大倍数 | 2 |
| `--sharpen` | 启用锐化 | 默认开启 |
| `--sharpen_strength` | 锐化强度 (0.1-1.0) | 0.5 |
| `--eval` | 自动评估修复质量 | 关闭 |
| `--eval_hr` | 评估超分辨率后的结果 | 关闭 |

完整示例（含超分辨率 + 评估）：
```bash
python predict.py \
    ./result/phase_3/phase_3_model_generator_epoch40 \
    ./result/config.json \
    ./results/my_test/ \
    --data ./images/test/ \
    --sr_enable --sr_method bicubic --sr_scale 2 \
    --eval --eval_hr
```

### 四、独立评估

```bash
# 文件夹批量对比
python test.py \
    --original ./datasets/test \
    --restored ./results/my_test/completed \
    --output ./evaluation_results \
    --save_comparison

# 单文件对比
python test.py \
    --original img_original.png \
    --restored img_restored.png \
    --output ./evaluation_results \
    --single_mode --save_comparison
```

### 五、GUI 界面

```bash
python GUI.py
```

界面中可配置：
- config 路径（指向 `./result/config.json`）
- model 路径（指向 `./result/phase_3/phase_3_model_generator_epoch40`）
- 数据路径、结果路径
- 超分辨率开关与方法

## 常见问题

### Q: 重新训练后预测结果与旧数据集一致？
**A:** 检查 predict.py 的 model 和 config 参数是否指向新训练产物：
- model → `./result/phase_3/phase_3_model_generator_epochN`
- config → `./result/config.json`

不要使用 `./demo/model_cn` 和 `./demo/config.json`，那是旧数据集的预训练模型。

### Q: InsightFace 模型找不到？
**A:** 确保 `insightface_models/models/` 下有检测模型文件。Windows 路径在代码中设为 `F:/insightface_models`，Linux 通过环境变量 `INSIGHTFACE_ROOT` 配置。

### Q: CUDA 错误？
**A:** 检查 CUDA 11.8 是否正确安装，DLL 路径是否与 `train.py` 顶部的 `os.add_dll_directory()` 一致。
