"""
两阶段面部修复模型 - 完整实验运行脚本

本脚本实现研究计划中的完整流程:
1. 从完整数据集随机采样小数据集
2. 在小数据集上测试不同损失函数组合
3. 评估并选择最优组合
4. 使用最优组合在完整数据集上训练
5. 最终评估并生成报告
"""

import torch
import os
import sys
if sys.platform == 'win32':
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
    os.add_dll_directory("F:\\anaconda\\envs\\face-study\\Lib\\site-packages\\onnxruntime\\capi")

import argparse
import json
from datetime import datetime
from loss_search import LossSearchExperiment
from loss_functions import get_all_loss_combinations


def main():
    parser = argparse.ArgumentParser(description='两阶段面部修复模型 - 完整实验流程')
    
    # 数据路径
    parser.add_argument('--train_dir', default='./datasets/train', help='训练数据目录')
    parser.add_argument('--test_dir', default='./datasets/test', help='测试数据目录')
    parser.add_argument('--result_dir', default='./result/experiments', help='实验结果目录')
    parser.add_argument('--init_model_G', default='./demo/model_cn', help='预训练生成器路径')
    
    # 实验配置
    parser.add_argument('--small_dataset_size', type=int, default=500, 
                       help='小数据集大小 (用于损失函数搜索)')
    parser.add_argument('--search_epochs', type=int, default=10, 
                       help='搜索阶段每种组合的训练轮数')
    parser.add_argument('--full_epochs', type=int, default=50, 
                       help='完整数据集训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--input_size', type=int, default=160, help='输入图像大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    # 实验选项
    parser.add_argument('--skip_search', action='store_true', 
                       help='跳过搜索阶段，直接使用指定组合训练')
    parser.add_argument('--skip_full_train', action='store_true', 
                       help='跳过完整数据集训练')
    parser.add_argument('--loss_combination', default=None, 
                       help='指定损失组合 (跳过搜索时使用)')
    parser.add_argument('--combinations', nargs='+', default=None,
                       help='指定要搜索的损失组合列表')
    parser.add_argument('--calculate_fid', action='store_true', default=True,
                       help='是否计算FID指标')
    
    args = parser.parse_args()
    
    print("="*70)
    print("两阶段面部修复模型 - 自动化实验流程")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练数据: {args.train_dir}")
    print(f"测试数据: {args.test_dir}")
    print(f"结果目录: {args.result_dir}")
    print("="*70)
    
    # 实验配置
    config = {
        'train_dir': args.train_dir,
        'test_dir': args.test_dir,
        'result_dir': args.result_dir,
        'init_model_G': args.init_model_G,
        'small_dataset_size': args.small_dataset_size,
        'search_epochs': args.search_epochs,
        'full_epochs': args.full_epochs,
        'batch_size': args.batch_size,
        'input_size': args.input_size,
        'lr': args.lr,
        'calculate_fid': args.calculate_fid,
        'mpv': [0.5109094257003173, 0.41451381534316145, 0.3723845290805799],
    }
    
    # 创建实验实例
    experiment = LossSearchExperiment(config)
    
    if args.skip_search:
        # 直接使用指定的损失组合
        if args.loss_combination is None:
            args.loss_combination = 'all'
        print(f"\n跳过搜索阶段，直接使用损失组合: {args.loss_combination}")
        best_combo = args.loss_combination
    else:
        # 执行损失函数组合搜索
        print("\n" + "="*70)
        print("阶段1: 损失函数组合搜索")
        print("="*70)
        
        combinations = args.combinations
        if combinations is None:
            combinations = get_all_loss_combinations()
        
        print(f"待测试组合: {combinations}")
        print(f"小数据集大小: {args.small_dataset_size}")
        print(f"每组合训练轮数: {args.search_epochs}")
        
        best_combo, results = experiment.run_search(combinations)
        
        print("\n" + "="*70)
        print("搜索阶段完成!")
        print(f"最优损失函数组合: {best_combo}")
        print("="*70)
    
    # 使用最优组合在完整数据集上训练
    if not args.skip_full_train:
        print("\n" + "="*70)
        print("阶段2: 完整数据集训练")
        print("="*70)
        print(f"使用损失组合: {best_combo}")
        print(f"训练轮数: {args.full_epochs}")
        
        final_model, final_metrics = experiment.train_full_dataset(
            best_combo, args.full_epochs
        )
        
        print("\n" + "="*70)
        print("训练完成!")
        print("="*70)
        experiment.metrics_calculator.print_metrics(final_metrics, prefix="最终模型 ")
    
    print("\n" + "="*70)
    print("实验完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所有结果保存在: {experiment.experiment_dir}")
    print("="*70)
    
    return experiment.experiment_dir


if __name__ == '__main__':
    main()
