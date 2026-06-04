"""
Linux 版本 - 损失函数组合自动搜索脚本

流程:
1. 从完整数据集中随机采样小数据集
2. 在小数据集上用不同损失函数组合训练模型
3. 评估各组合的效果 (PSNR, SSIM, FID)
4. 选择最优组合
5. 用最优组合在完整数据集上进行最终训练
"""

import torch
import os
import json
import random
import argparse
import shutil
from tqdm import tqdm
from datetime import datetime
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

from models import Generator, Discriminator
from loss_functions import ConfigurableLoss, get_all_loss_combinations, create_loss_function
from metrics import MetricsCalculator, calculate_psnr, calculate_ssim
from utils import (
    MyDataset,
    gen_input_MaskLayer,
    gen_Mask,
    poisson_blend,
    crop,
    sample_random_batch,
)
from PIL import Image
from insightface.app import FaceAnalysis

# insightface 模型路径 - 通过环境变量配置
INSIGHTFACE_ROOT = os.environ.get('INSIGHTFACE_ROOT', './insightface_models')


class LossSearchExperiment:
    """损失函数组合搜索实验类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建实验结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(config['result_dir'], f'loss_search_{timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化insightface
        print(f"初始化FaceAnalysis (模型路径: {INSIGHTFACE_ROOT})...")
        self.app = FaceAnalysis(
            allowed_modules=['detection', 'landmark_2d_106'],
            root=INSIGHTFACE_ROOT
        )
        self.app.prepare(ctx_id=0, det_size=(256, 256))
        
        # 初始化评估器
        self.metrics_calculator = MetricsCalculator(
            device=self.device,
            calculate_fid_flag=config.get('calculate_fid', True)
        )
        
        # 存储实验结果
        self.results = {}
    
    def create_small_dataset(self, full_dataset, sample_size):
        """从完整数据集中随机采样创建小数据集"""
        total_size = len(full_dataset)
        if sample_size > total_size:
            sample_size = total_size
            print(f"警告: 采样大小超过数据集大小，使用全部 {total_size} 张图片")
        
        indices = random.sample(range(total_size), sample_size)
        return Subset(full_dataset, indices), indices
    
    def train_with_loss_combination(self, loss_combination, train_loader, test_dataset, 
                                     num_epochs, init_model_path=None):
        """使用指定损失函数组合训练模型"""
        print(f"\n{'='*60}")
        print(f"训练损失组合: {loss_combination}")
        print(f"{'='*60}")
        
        # 创建该组合的结果目录
        combo_dir = os.path.join(self.experiment_dir, loss_combination)
        os.makedirs(combo_dir, exist_ok=True)
        
        # 初始化模型
        G = Generator()
        if init_model_path and os.path.exists(init_model_path):
            G.load_state_dict(torch.load(init_model_path, map_location='cpu'))
            print(f"加载预训练模型: {init_model_path}")
        G = G.to(self.device)
        
        # 初始化损失函数
        loss_fn = create_loss_function(loss_combination)
        print(f"损失函数配置: {loss_fn.get_description()}")
        
        # 优化器
        optimizer = torch.optim.Adam(G.parameters(), lr=self.config.get('lr', 1e-4))
        
        # 计算训练数据的均值像素值
        mpv = torch.tensor(self.config.get('mpv', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(self.device)
        
        # 训练历史
        history = {'loss': [], 'epoch_loss': []}
        
        # 训练循环
        best_loss = float('inf')
        for epoch in range(num_epochs):
            G.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, X in enumerate(pbar):
                X = X.to(self.device)
                
                # 生成mask
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location
                ).to(self.device)
                
                # 创建被遮挡的输入
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                
                # 前向传播
                output_G = G(input_G)
                
                # 计算损失
                total_loss, loss_dict = loss_fn(X, output_G, masklayer, self.app)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                history['loss'].append(total_loss.item())
                
                pbar.set_postfix({'loss': f"{total_loss.item():.5f}"})
            
            avg_epoch_loss = np.mean(epoch_losses)
            history['epoch_loss'].append(avg_epoch_loss)
            print(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.5f}")
            
            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(G.state_dict(), os.path.join(combo_dir, 'best_model.pth'))
            
            # 定期保存可视化结果
            if (epoch + 1) % max(1, num_epochs // 5) == 0:
                self._save_visualization(G, test_dataset, mpv, combo_dir, epoch + 1)
        
        # 保存最终模型
        torch.save(G.state_dict(), os.path.join(combo_dir, 'final_model.pth'))
        
        # 保存训练历史
        with open(os.path.join(combo_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        return G, history
    
    def evaluate_model(self, G, test_dataset, mpv):
        """评估模型性能"""
        G.eval()
        
        all_real = []
        all_fake = []
        
        with torch.no_grad():
            for i in range(len(test_dataset)):
                x = test_dataset[i].unsqueeze(0).to(self.device)
                
                mask_size, mask_location = gen_Mask(x)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location
                ).to(self.device)
                
                x_mask = x - x * masklayer + mpv * masklayer
                input_G = torch.cat((x_mask, masklayer), dim=1)
                output_G = G(input_G)
                
                # Poisson融合
                completed = poisson_blend(x_mask, output_G, masklayer)
                
                all_real.append(x.cpu())
                all_fake.append(completed.cpu())
        
        real_images = torch.cat(all_real, dim=0)
        fake_images = torch.cat(all_fake, dim=0)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_all(real_images, fake_images)
        
        return metrics
    
    def _save_visualization(self, G, test_dataset, mpv, save_dir, epoch):
        """保存可视化结果"""
        G.eval()
        with torch.no_grad():
            num_vis = min(4, len(test_dataset))
            indices = random.sample(range(len(test_dataset)), num_vis)
            
            vis_images = []
            for idx in indices:
                x = test_dataset[idx].unsqueeze(0).to(self.device)
                
                mask_size, mask_location = gen_Mask(x)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location
                ).to(self.device)
                
                x_mask = x - x * masklayer + mpv * masklayer
                input_G = torch.cat((x_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(x_mask, output_G, masklayer).to(self.device)
                
                vis_images.extend([x.cpu(), x_mask.cpu(), completed.cpu()])
            
            imgs = torch.cat(vis_images, dim=0)
            save_image(imgs, os.path.join(save_dir, f'epoch_{epoch}.png'), nrow=3)
        G.train()
    
    def run_search(self, loss_combinations=None):
        """运行损失函数组合搜索"""
        if loss_combinations is None:
            loss_combinations = get_all_loss_combinations()
        
        print(f"\n开始损失函数组合搜索实验")
        print(f"待测试组合: {loss_combinations}")
        print(f"小数据集大小: {self.config['small_dataset_size']}")
        print(f"搜索阶段训练轮数: {self.config['search_epochs']}")
        
        # 加载数据
        transform = transforms.Compose([
            transforms.Resize(self.config['input_size']),
            transforms.CenterCrop((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
        ])
        
        full_train_dataset = MyDataset(self.config['train_dir'], transform)
        full_test_dataset = MyDataset(self.config['test_dir'], transform)
        
        print(f"完整训练集大小: {len(full_train_dataset)}")
        print(f"完整测试集大小: {len(full_test_dataset)}")
        
        # 创建小数据集
        small_train_dataset, train_indices = self.create_small_dataset(
            full_train_dataset, self.config['small_dataset_size']
        )
        small_test_dataset, test_indices = self.create_small_dataset(
            full_test_dataset, min(self.config['small_dataset_size'] // 5, len(full_test_dataset))
        )
        
        print(f"小训练集大小: {len(small_train_dataset)}")
        print(f"小测试集大小: {len(small_test_dataset)}")
        
        # 保存采样索引
        with open(os.path.join(self.experiment_dir, 'sampled_indices.json'), 'w') as f:
            json.dump({'train': train_indices, 'test': test_indices}, f)
        
        # 创建数据加载器
        train_loader = DataLoader(
            small_train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        mpv = torch.tensor(self.config.get('mpv', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(self.device)
        
        # 对每种损失组合进行训练和评估
        for combo in loss_combinations:
            print(f"\n{'#'*70}")
            print(f"测试损失组合: {combo}")
            print(f"{'#'*70}")
            
            try:
                G, history = self.train_with_loss_combination(
                    loss_combination=combo,
                    train_loader=train_loader,
                    test_dataset=small_test_dataset,
                    num_epochs=self.config['search_epochs'],
                    init_model_path=self.config.get('init_model_G')
                )
                
                print(f"\n评估 {combo} 组合...")
                metrics = self.evaluate_model(G, small_test_dataset, mpv)
                
                self.results[combo] = {
                    'metrics': metrics,
                    'final_loss': history['epoch_loss'][-1] if history['epoch_loss'] else float('inf'),
                    'loss_history': history['epoch_loss']
                }
                
                self.metrics_calculator.print_metrics(metrics, prefix=f"{combo} ")
                
            except Exception as e:
                print(f"组合 {combo} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                self.results[combo] = {
                    'metrics': {'psnr': 0, 'ssim': 0, 'fid': float('inf'), 'mse': float('inf')},
                    'error': str(e)
                }
        
        self._save_results()
        best_combo = self.select_best_combination()
        
        return best_combo, self.results
    
    def select_best_combination(self):
        """根据评估指标选择最优损失函数组合"""
        print(f"\n{'='*60}")
        print("损失函数组合搜索结果汇总")
        print(f"{'='*60}")
        
        scores = {}
        for combo, result in self.results.items():
            if 'error' in result:
                scores[combo] = -float('inf')
                continue
            
            metrics = result['metrics']
            psnr_score = metrics.get('psnr', 0) / 30.0
            ssim_score = metrics.get('ssim', 0)
            fid_score = 1.0 - min(metrics.get('fid', 100), 100) / 100.0
            
            score = 0.4 * psnr_score + 0.4 * ssim_score + 0.2 * fid_score
            scores[combo] = score
            
            print(f"\n{combo}:")
            print(f"  PSNR: {metrics.get('psnr', 0):.4f}")
            print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
            print(f"  FID:  {metrics.get('fid', float('nan')):.4f}")
            print(f"  综合得分: {score:.4f}")
        
        best_combo = max(scores, key=scores.get)
        print(f"\n{'='*60}")
        print(f"最优损失函数组合: {best_combo}")
        print(f"综合得分: {scores[best_combo]:.4f}")
        print(f"{'='*60}")
        
        with open(os.path.join(self.experiment_dir, 'best_combination.json'), 'w') as f:
            json.dump({
                'best_combination': best_combo,
                'score': scores[best_combo],
                'all_scores': scores
            }, f, indent=2)
        
        return best_combo
    
    def _save_results(self):
        """保存实验结果"""
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(os.path.join(self.experiment_dir, 'all_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n结果已保存至: {self.experiment_dir}")
    
    def train_full_dataset(self, best_combination, num_epochs):
        """使用最优损失组合在完整数据集上训练"""
        print(f"\n{'#'*70}")
        print(f"使用最优组合 '{best_combination}' 在完整数据集上训练")
        print(f"训练轮数: {num_epochs}")
        print(f"{'#'*70}")
        
        transform = transforms.Compose([
            transforms.Resize(self.config['input_size']),
            transforms.CenterCrop((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
        ])
        
        full_train_dataset = MyDataset(self.config['train_dir'], transform)
        full_test_dataset = MyDataset(self.config['test_dir'], transform)
        
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        final_dir = os.path.join(self.experiment_dir, 'final_training')
        os.makedirs(final_dir, exist_ok=True)
        
        G, history = self.train_with_loss_combination(
            loss_combination=best_combination,
            train_loader=train_loader,
            test_dataset=full_test_dataset,
            num_epochs=num_epochs,
            init_model_path=self.config.get('init_model_G')
        )
        
        mpv = torch.tensor(self.config.get('mpv', [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(self.device)
        final_metrics = self.evaluate_model(G, full_test_dataset, mpv)
        
        print("\n最终模型评估结果:")
        self.metrics_calculator.print_metrics(final_metrics, prefix="最终 ")
        
        with open(os.path.join(final_dir, 'final_metrics.json'), 'w') as f:
            json.dump({
                'best_combination': best_combination,
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in final_metrics.items()}
            }, f, indent=2)
        
        shutil.copy(
            os.path.join(final_dir, 'final_model.pth'),
            os.path.join(self.experiment_dir, 'best_final_model.pth')
        )
        
        return G, final_metrics


def main():
    parser = argparse.ArgumentParser(description='损失函数组合自动搜索 (Linux版)')
    parser.add_argument('--train_dir', default='/local_data/Face-study/datasets/train', help='训练数据目录')
    parser.add_argument('--test_dir', default='/local_data/Face-study/datasets/test', help='测试数据目录')
    parser.add_argument('--result_dir', default='./result/loss_search', help='结果保存目录')
    parser.add_argument('--init_model_G', default='./demo/model_cn', help='预训练生成器路径')
    parser.add_argument('--small_dataset_size', type=int, default=500, help='小数据集大小')
    parser.add_argument('--search_epochs', type=int, default=10, help='搜索阶段训练轮数')
    parser.add_argument('--full_epochs', type=int, default=50, help='完整数据集训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--input_size', type=int, default=160, help='输入图像大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--skip_full_train', action='store_true', help='跳过完整数据集训练')
    parser.add_argument('--combinations', nargs='+', default=None, 
                       help='指定要测试的损失组合')
    parser.add_argument('--no_fid', action='store_true', help='不计算FID (节省内存)')
    
    args = parser.parse_args()
    
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
        'calculate_fid': not args.no_fid,
        'mpv': [0.5109094257003173, 0.41451381534316145, 0.3723845290805799],
    }
    
    experiment = LossSearchExperiment(config)
    best_combo, results = experiment.run_search(args.combinations)
    
    if not args.skip_full_train:
        experiment.train_full_dataset(best_combo, args.full_epochs)
    
    print("\n实验完成！")
    print(f"最优损失组合: {best_combo}")
    print(f"所有结果保存在: {experiment.experiment_dir}")


if __name__ == '__main__':
    main()
