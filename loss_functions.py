"""
可配置的损失函数模块
支持不同损失函数组合的系统性实验

损失函数组合:
1. reconstruction: 仅基础重建损失 (MSE)
2. perceptual: 仅感知损失 (VGG)
3. landmark: 仅标志点损失
4. recon_percep: 重建 + 感知
5. recon_landmark: 重建 + 标志点
6. percep_landmark: 感知 + 标志点
7. all: 重建 + 感知 + 标志点 (完整组合)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np


def get_landmark(img, app):
    """提取人脸关键点"""
    device = img.device
    img = img.clone().cpu()
    num_samples = img.shape[0]
    ret = []
    for i in range(num_samples):
        try:
            dstimg = transforms.functional.to_pil_image(img[i])
            dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
            faces = app.get(dstimg)
            if len(faces) >= 1:
                lmk = faces[0].landmark_2d_106
                lmk = lmk.astype(np.float32)
            else:
                lmk = np.zeros((106, 2), dtype=np.float32)
        except Exception as e:
            lmk = np.zeros((106, 2), dtype=np.float32)
        ret.append(lmk)
    return torch.tensor(np.array(ret), device=device, dtype=torch.float32)


class SSIMLoss(nn.Module):
    """
    SSIM损失：直接优化结构相似性，显著提升SSIM指标。
    使用高斯窗口计算局部统计量，输出 1 - SSIM，越小越好。
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        sigma = 1.5
        gauss = torch.tensor(
            [math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
             for x in range(window_size)], dtype=torch.float32)
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', kernel_2d)

    def forward(self, x, y):
        C = x.size(1)
        pad = self.window_size // 2
        window = self.window.expand(C, 1, -1, -1)
        mu_x = F.conv2d(x, window, padding=pad, groups=C)
        mu_y = F.conv2d(y, window, padding=pad, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2))
        return 1.0 - ssim_map.mean()


class FrequencyLoss(nn.Module):
    """
    频域损失：通过FFT变换惩罚高频细节差异，直接提升PSNR
    高频分量对应图像的边缘、纹理等细节信息
    """
    def __init__(self):
        super().__init__()

    def forward(self, real_img, fake_img):
        # 对每个通道做2D FFT
        real_fft = torch.fft.rfft2(real_img, norm='ortho')
        fake_fft = torch.fft.rfft2(fake_img, norm='ortho')
        # 比较频谱幅值的差异
        real_mag = torch.abs(real_fft)
        fake_mag = torch.abs(fake_fft)
        return F.l1_loss(fake_mag, real_mag)


class ConfigurableLoss(nn.Module):
    """
    可配置的损失函数类
    
    支持的损失组合:
    - reconstruction: 基础重建损失 (MSE)
    - perceptual: 感知损失 (VGG特征匹配)
    - landmark: 人脸关键点损失
    - l1: L1重建损失（比MSE更锐利）
    - freq: 频域损失（保留高频细节）
    """
    
    # 预定义的损失组合
    LOSS_COMBINATIONS = {
        'reconstruction': {'recon': 1.0, 'percep': 0.0, 'landmark': 0.0, 'l1': 0.0, 'freq': 0.0, 'ssim': 0.0},
        'perceptual':     {'recon': 0.0, 'percep': 1.0, 'landmark': 0.0, 'l1': 0.0, 'freq': 0.0, 'ssim': 0.0},
        'recon_percep':   {'recon': 1.0, 'percep': 0.05,'landmark': 0.0, 'l1': 0.0, 'freq': 0.0, 'ssim': 0.0},
        'recon_landmark': {'recon': 1.0, 'percep': 0.0, 'landmark': 0.1, 'l1': 0.0, 'freq': 0.0, 'ssim': 0.0},
        'percep_landmark':{'recon': 0.0, 'percep': 0.05,'landmark': 0.1, 'l1': 0.0, 'freq': 0.0, 'ssim': 0.0},
        'all':            {'recon': 1.0, 'percep': 0.05,'landmark': 0.1, 'l1': 0.0, 'freq': 0.0, 'ssim': 0.0},
        # ===== V2专用组合（针对PSNR优化）=====
        'v2_balanced': {
            'recon': 0.7, 'l1': 0.3, 'percep': 0.05, 'landmark': 0.1, 'freq': 0.1, 'ssim': 0.0
        },
        'v2_detail': {
            'recon': 0.5, 'l1': 0.5, 'percep': 0.05, 'landmark': 0.1, 'freq': 0.2, 'ssim': 0.0
        },
        # ===== 针对SSIM优化的新组合 =====
        'v2_ssim': {
            'recon': 0.5, 'l1': 0.3, 'percep': 0.05, 'landmark': 0.1, 'freq': 0.05, 'ssim': 0.5
        },
        'v2_full': {
            'recon': 0.4, 'l1': 0.3, 'percep': 0.05, 'landmark': 0.1, 'freq': 0.1, 'ssim': 0.4
        },
    }
    
    def __init__(self, combination='all', custom_weights=None):
        """
        初始化可配置损失函数
        
        Args:
            combination: 损失组合名称，见 LOSS_COMBINATIONS
            custom_weights: 自定义权重字典 {'recon': w1, 'percep': w2, 'landmark': w3, 'l1': w4, 'freq': w5}
        """
        super().__init__()
        
        if custom_weights is not None:
            self.weights = custom_weights
        elif combination in self.LOSS_COMBINATIONS:
            self.weights = self.LOSS_COMBINATIONS[combination]
        else:
            raise ValueError(f"未知的损失组合: {combination}，可选: {list(self.LOSS_COMBINATIONS.keys())}")
        
        # 兼容旧版组合（没有l1/freq/ssim字段）
        self.weights.setdefault('l1', 0.0)
        self.weights.setdefault('freq', 0.0)
        self.weights.setdefault('ssim', 0.0)
        
        self.combination_name = combination
        
        # 初始化VGG用于感知损失
        if self.weights['percep'] > 0:
            from torchvision.models import VGG16_Weights
            self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg.eval()
        else:
            self.vgg = None
        
        # 初始化频域损失
        if self.weights['freq'] > 0:
            self.freq_loss = FrequencyLoss()
        else:
            self.freq_loss = None

        # 初始化SSIM损失
        if self.weights['ssim'] > 0:
            self.ssim_loss_fn = SSIMLoss(window_size=11)
        else:
            self.ssim_loss_fn = None
    
    def _normalize_for_vgg(self, x):
        """归一化到VGG输入范围"""
        return (x - 0.5) * 2
    
    def reconstruction_loss(self, real_img, fake_img, mask):
        """基础重建损失 (MSE)"""
        return F.mse_loss(fake_img * mask, real_img * mask)
    
    def l1_reconstruction_loss(self, real_img, fake_img, mask):
        """L1重建损失（产生更锐利的结果）"""
        return F.l1_loss(fake_img * mask, real_img * mask)
    
    def perceptual_loss(self, real_img, fake_img):
        """感知损失 (VGG特征匹配)"""
        if self.vgg is None:
            return torch.tensor(0.0, device=real_img.device)
        
        self.vgg = self.vgg.to(real_img.device)
        real_norm = self._normalize_for_vgg(real_img)
        fake_norm = self._normalize_for_vgg(fake_img)
        return F.l1_loss(self.vgg(fake_norm), self.vgg(real_norm))
    
    def landmark_loss(self, real_img, fake_img, app):
        """人脸关键点损失"""
        if app is None:
            return torch.tensor(0.0, device=real_img.device)
        
        real_lmk = get_landmark(real_img, app)
        fake_lmk = get_landmark(fake_img, app)
        return F.l1_loss(real_lmk, fake_lmk)
    
    def frequency_loss(self, real_img, fake_img):
        """频域损失"""
        if self.freq_loss is None:
            return torch.tensor(0.0, device=real_img.device)
        return self.freq_loss(real_img, fake_img)

    def ssim_loss(self, real_img, fake_img):
        """SSIM结构相似性损失（1 - SSIM，直接优化SSIM指标）"""
        if self.ssim_loss_fn is None:
            return torch.tensor(0.0, device=real_img.device)
        self.ssim_loss_fn = self.ssim_loss_fn.to(real_img.device)
        return self.ssim_loss_fn(fake_img, real_img)
    
    def forward(self, real_img, fake_img, mask, app=None):
        """
        计算总损失
        
        Args:
            real_img: 原始真实图像 (N, 3, H, W)
            fake_img: 生成的修复图像 (N, 3, H, W)
            mask: 遮挡mask (N, 1, H, W)
            app: insightface FaceAnalysis 实例 (用于landmark)
        
        Returns:
            total_loss: 加权总损失
            loss_dict: 各项损失的详细信息
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=real_img.device)
        
        # MSE重建损失
        if self.weights['recon'] > 0:
            recon_loss = self.reconstruction_loss(real_img, fake_img, mask)
            loss_dict['recon'] = recon_loss.item()
            total_loss = total_loss + self.weights['recon'] * recon_loss
        else:
            loss_dict['recon'] = 0.0
        
        # L1重建损失
        if self.weights['l1'] > 0:
            l1_loss = self.l1_reconstruction_loss(real_img, fake_img, mask)
            loss_dict['l1'] = l1_loss.item()
            total_loss = total_loss + self.weights['l1'] * l1_loss
        else:
            loss_dict['l1'] = 0.0
        
        # 感知损失
        if self.weights['percep'] > 0:
            percep_loss = self.perceptual_loss(real_img, fake_img)
            loss_dict['percep'] = percep_loss.item()
            total_loss = total_loss + self.weights['percep'] * percep_loss
        else:
            loss_dict['percep'] = 0.0
        
        # 关键点损失
        if self.weights['landmark'] > 0 and app is not None:
            lmk_loss = self.landmark_loss(real_img, fake_img, app)
            loss_dict['landmark'] = lmk_loss.item()
            total_loss = total_loss + self.weights['landmark'] * lmk_loss
        else:
            loss_dict['landmark'] = 0.0
        
        # 频域损失
        if self.weights['freq'] > 0:
            freq_loss = self.frequency_loss(real_img, fake_img)
            loss_dict['freq'] = freq_loss.item()
            total_loss = total_loss + self.weights['freq'] * freq_loss
        else:
            loss_dict['freq'] = 0.0

        # SSIM损失
        if self.weights['ssim'] > 0:
            s_loss = self.ssim_loss(real_img, fake_img)
            loss_dict['ssim'] = s_loss.item()
            total_loss = total_loss + self.weights['ssim'] * s_loss
        else:
            loss_dict['ssim'] = 0.0

        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_description(self):
        """返回当前损失组合的描述"""
        active = [f"{k}={v}" for k, v in self.weights.items() if v > 0]
        return f"{self.combination_name}: [{', '.join(active)}]"


def get_all_loss_combinations():
    """返回所有预定义的损失组合名称"""
    return list(ConfigurableLoss.LOSS_COMBINATIONS.keys())


def create_loss_function(combination='all', custom_weights=None):
    """工厂函数：创建指定组合的损失函数"""
    return ConfigurableLoss(combination=combination, custom_weights=custom_weights)


if __name__ == '__main__':
    # 测试代码
    print("可用的损失函数组合:")
    for name in get_all_loss_combinations():
        loss_fn = create_loss_function(name)
        print(f"  - {loss_fn.get_description()}")
