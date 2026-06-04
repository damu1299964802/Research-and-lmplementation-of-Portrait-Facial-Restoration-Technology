import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
import numpy as np

def get_landmark(img, app):
    device = img.device  # 从输入图像获取设备信息
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
            print(f"Error processing image {i}: {e}")
            lmk = np.zeros((106, 2), dtype=np.float32)
        ret.append(lmk)
    return torch.tensor(np.array(ret), device=device, dtype=torch.float32)

def completion_network_loss(input, output, mask):
    return F.mse_loss(output * mask, input * mask)

class HybridLoss(torch.nn.Module):
    def __init__(self, landmark_weight=0.1, percep_weight=0.05):  # 降低权重
        super().__init__()
        # 使用新的权重初始化方式
        from torchvision.models import VGG16_Weights
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()
        
        self.landmark_weight = landmark_weight
        self.percep_weight = percep_weight

    def _normalize(self, x):
        return (x - 0.5) * 2  # 适配VGG输入范围

    def perceptual_loss(self, fake, real):
        fake_norm = self._normalize(fake)
        real_norm = self._normalize(real)
        return F.l1_loss(self.vgg(fake_norm), self.vgg(real_norm))

    def forward(self, real_img, fake_img, app):
        # 设备自动适配
        self.vgg = self.vgg.to(real_img.device)
        
        # Landmark损失
        real_lmk = get_landmark(real_img, app)
        fake_lmk = get_landmark(fake_img, app)
        lmk_loss = F.l1_loss(real_lmk, fake_lmk)
        
        # 感知损失
        percep_loss = self.perceptual_loss(fake_img, real_img)
        
        return (self.landmark_weight * lmk_loss +
                self.percep_weight * percep_loss)