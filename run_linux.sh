#!/bin/bash
# Linux 服务器运行脚本

# ========================================
# 配置区域 - 请根据实际情况修改
# ========================================

# 数据集路径
TRAIN_DIR="./datasets/train"
TEST_DIR="./datasets/test"

# 结果保存路径
RESULT_DIR="./result/experiments"

# 预训练模型路径
INIT_MODEL="./demo/model_cn"

# insightface 模型路径 (需要修改为你的实际路径)
export INSIGHTFACE_ROOT="./insightface_models"

# ========================================
# 运行选项
# ========================================

# 小数据集大小 (用于损失函数搜索)
SMALL_DATASET_SIZE=500

# 搜索阶段训练轮数
SEARCH_EPOCHS=10

# 完整数据集训练轮数
FULL_EPOCHS=50

# 批大小
BATCH_SIZE=16

# ========================================
# 运行命令
# ========================================

echo "============================================"
echo "两阶段面部修复模型 - Linux 服务器运行"
echo "============================================"
echo "训练数据: $TRAIN_DIR"
echo "测试数据: $TEST_DIR"
echo "结果目录: $RESULT_DIR"
echo "============================================"

# 检查 GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 运行完整实验流程
python run_experiment.py \
    --train_dir $TRAIN_DIR \
    --test_dir $TEST_DIR \
    --result_dir $RESULT_DIR \
    --init_model_G $INIT_MODEL \
    --small_dataset_size $SMALL_DATASET_SIZE \
    --search_epochs $SEARCH_EPOCHS \
    --full_epochs $FULL_EPOCHS \
    --batch_size $BATCH_SIZE

echo "============================================"
echo "实验完成！结果保存在: $RESULT_DIR"
echo "============================================"
