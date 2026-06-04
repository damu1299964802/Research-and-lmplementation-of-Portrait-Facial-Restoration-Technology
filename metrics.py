"""
评估指标模块
包含 PSNR, SSIM, FID 等定量评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from scipy import linalg
from PIL import Image


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        img1: 原始图像 tensor (N, C, H, W) 或 numpy array
        img2: 修复图像 tensor (N, C, H, W) 或 numpy array
        max_val: 像素最大值
    
    Returns:
        psnr: PSNR值 (dB)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    计算结构相似性指数 (SSIM)
    
    Args:
        img1: 原始图像 tensor (N, C, H, W)
        img2: 修复图像 tensor (N, C, H, W)
        window_size: 窗口大小
    
    Returns:
        ssim: SSIM值 [0, 1]
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # 创建高斯窗口
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    
    window = gaussian_window(window_size)
    window = window.expand(img1.shape[1], 1, window_size, window_size).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


class InceptionV3Features(nn.Module):
    """InceptionV3特征提取器用于FID计算"""
    
    def __init__(self, local_model_path=None):
        super().__init__()
        from torchvision.models import Inception_V3_Weights
        
        if local_model_path and os.path.exists(local_model_path):
            # 使用本地模型文件
            inception = models.inception_v3(weights=None)
            inception.load_state_dict(torch.load(local_model_path, map_location='cpu'))
            print(f"使用本地InceptionV3模型: {local_model_path}")
        else:
            # 尝试使用预训练模型（需要网络）
            try:
                inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
                print("使用在线下载的InceptionV3模型")
            except Exception as e:
                print(f"无法下载InceptionV3模型: {e}")
                print("请使用 --no_fid 参数跳过FID计算，或提供本地模型文件")
                raise e
        
        # 提取到avgpool层之前的特征
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        # InceptionV3期望输入尺寸为299x299
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 归一化到ImageNet标准
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        features = self.blocks(x)
        return features.view(features.size(0), -1)


class FIDCalculator:
    """FID计算器"""
    
    def __init__(self, device='cuda', local_model_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.inception = InceptionV3Features(local_model_path).to(self.device)
    
    def extract_features(self, images, batch_size=32):
        """从图像批次提取InceptionV3特征"""
        self.inception.eval()
        features_list = []
        
        if isinstance(images, torch.Tensor):
            n_samples = images.shape[0]
            for i in range(0, n_samples, batch_size):
                batch = images[i:i+batch_size].to(self.device)
                with torch.no_grad():
                    feat = self.inception(batch)
                features_list.append(feat.cpu())
            features = torch.cat(features_list, dim=0).numpy()
        else:
            # 假设是图像路径列表或数据集
            for batch in images:
                if isinstance(batch, (str, Image.Image)):
                    # 单个图像
                    if isinstance(batch, str):
                        batch = Image.open(batch).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((299, 299)),
                        transforms.ToTensor()
                    ])
                    batch = transform(batch).unsqueeze(0)
                batch = batch.to(self.device)
                with torch.no_grad():
                    feat = self.inception(batch)
                features_list.append(feat.cpu())
            features = torch.cat(features_list, dim=0).numpy()
        
        return features
    
    def calculate_statistics(self, features):
        """计算特征的均值和协方差"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_images, fake_images, batch_size=32):
        """
        计算FID分数
        
        Args:
            real_images: 真实图像 tensor (N, C, H, W)
            fake_images: 生成图像 tensor (N, C, H, W)
            batch_size: 批处理大小
        
        Returns:
            fid: FID分数 (越低越好)
        """
        # 提取特征
        real_features = self.extract_features(real_images, batch_size)
        fake_features = self.extract_features(fake_images, batch_size)
        
        # 计算统计量
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(fake_features)
        
        # 计算FID
        diff = mu1 - mu2
        
        # 计算 sqrt(sigma1 @ sigma2)
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # 处理数值不稳定性
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        
        # 处理虚数部分
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError("Imaginary component in sqrtm")
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


class MetricsCalculator:
    """综合评估指标计算器"""
    
    def __init__(self, device='cuda', calculate_fid_flag=True):
        self.device = device
        self.calculate_fid_flag = calculate_fid_flag
        if calculate_fid_flag:
            self.fid_calculator = FIDCalculator(device)
    
    def calculate_all(self, real_images, fake_images, batch_size=32):
        """
        计算所有评估指标
        
        Args:
            real_images: 真实图像 tensor (N, C, H, W)
            fake_images: 修复图像 tensor (N, C, H, W)
        
        Returns:
            dict: 包含所有指标的字典
        """
        metrics = {}
        
        # PSNR
        psnr_values = []
        for i in range(real_images.shape[0]):
            psnr = calculate_psnr(real_images[i], fake_images[i])
            psnr_values.append(psnr)
        metrics['psnr'] = np.mean(psnr_values)
        metrics['psnr_std'] = np.std(psnr_values)
        
        # SSIM
        ssim_values = []
        for i in range(real_images.shape[0]):
            ssim = calculate_ssim(
                real_images[i:i+1].to(self.device), 
                fake_images[i:i+1].to(self.device)
            )
            ssim_values.append(ssim)
        metrics['ssim'] = np.mean(ssim_values)
        metrics['ssim_std'] = np.std(ssim_values)
        
        # FID (需要足够多的样本)
        if self.calculate_fid_flag and real_images.shape[0] >= 10:
            try:
                metrics['fid'] = self.fid_calculator.calculate_fid(
                    real_images, fake_images, batch_size
                )
            except Exception as e:
                print(f"FID计算失败: {e}")
                metrics['fid'] = float('inf')
        else:
            metrics['fid'] = float('nan')
        
        # MSE
        mse = F.mse_loss(fake_images, real_images).item()
        metrics['mse'] = mse
        
        return metrics
    
    def print_metrics(self, metrics, prefix=""):
        """打印评估指标"""
        print(f"{prefix}评估结果:")
        print(f"  PSNR: {metrics['psnr']:.4f} ± {metrics.get('psnr_std', 0):.4f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f} ± {metrics.get('ssim_std', 0):.4f}")
        print(f"  MSE:  {metrics['mse']:.6f}")
        if not np.isnan(metrics.get('fid', float('nan'))):
            print(f"  FID:  {metrics['fid']:.4f}")


if __name__ == '__main__':
    # 测试代码
    print("测试评估指标模块...")
    
    # 创建测试数据
    real = torch.rand(4, 3, 160, 160)
    fake = real + torch.randn_like(real) * 0.1
    fake = torch.clamp(fake, 0, 1)
    
    calculator = MetricsCalculator(calculate_fid_flag=False)
    metrics = calculator.calculate_all(real, fake)
    calculator.print_metrics(metrics, prefix="测试")
