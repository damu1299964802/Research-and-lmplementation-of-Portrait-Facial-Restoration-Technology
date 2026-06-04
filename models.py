import torch.nn as nn
import torch
import math


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # input_shape = (N, 4, H, W)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()

        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()

        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()

        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()

        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()

        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()

        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()

        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()
        # output = (N, 3, H, W)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class GeneratorV2(nn.Module):
    """
    改进版生成器：在原有架构基础上增加
    1. Encoder→Decoder Skip Connection（保留空间细节，提升PSNR）
    2. SE通道注意力（让网络聚焦重要特征通道）
    3. 瓶颈层残差连接（防止信息退化）
    原有的17层卷积结构和空洞卷积瓶颈层完全保留。
    """
    def __init__(self):
        super().__init__()
        # ===== Encoder（与V1完全一致）=====
        # Block1: (4,160,160) → (64,160,160)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # Block2: (64,160,160) → (128,80,80)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        # Block3: (128,80,80) → (128,80,80)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()

        # Block4: (128,80,80) → (256,40,40)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()

        # Block5-6: (256,40,40)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()

        # ===== 空洞卷积瓶颈层（与V1完全一致）=====
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()

        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()

        # 瓶颈层后续卷积（与V1一致）
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()

        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()

        # ===== Decoder（修改：增加skip connection融合层）=====
        # 上采样 40→80
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()

        # [新增] Skip融合：128(decoder) + 128(encoder e3) = 256 → 128
        self.skip_conv_80 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.skip_bn_80 = nn.BatchNorm2d(128)
        self.skip_act_80 = nn.ReLU()

        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()

        # 上采样 80→160
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()

        # [新增] Skip融合：64(decoder) + 64(encoder e1) = 128 → 64
        self.skip_conv_160 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.skip_bn_160 = nn.BatchNorm2d(64)
        self.skip_act_160 = nn.ReLU()

        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()

        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()

        # [新增] SE通道注意力
        self.se_bottleneck = SEBlock(256)
        self.se_dec_80 = SEBlock(128)
        self.se_dec_160 = SEBlock(64)

    def forward(self, x):
        # ===== Encoder =====
        e1 = self.bn1(self.act1(self.conv1(x)))       # (64,  160, 160)
        e2 = self.bn2(self.act2(self.conv2(e1)))       # (128, 80,  80)
        e3 = self.bn3(self.act3(self.conv3(e2)))       # (128, 80,  80)
        e4 = self.bn4(self.act4(self.conv4(e3)))       # (256, 40,  40)
        e5 = self.bn5(self.act5(self.conv5(e4)))       # (256, 40,  40)
        e6 = self.bn6(self.act6(self.conv6(e5)))       # (256, 40,  40)

        # ===== 空洞卷积瓶颈层 =====
        b = self.bn7(self.act7(self.conv7(e6)))
        b = self.bn8(self.act8(self.conv8(b)))
        b = self.bn9(self.act9(self.conv9(b)))
        b = self.bn10(self.act10(self.conv10(b)))
        b = self.bn11(self.act11(self.conv11(b)))
        b = self.bn12(self.act12(self.conv12(b)))

        # [新增] 瓶颈残差 + SE注意力
        b = self.se_bottleneck(b + e6)                 # (256, 40,  40)

        # ===== Decoder + Skip Connections =====
        d = self.bn13(self.act13(self.deconv13(b)))    # (128, 80,  80)
        d = torch.cat([d, e3], dim=1)                  # (256, 80,  80)
        d = self.skip_bn_80(self.skip_act_80(self.skip_conv_80(d)))  # (128,80,80)
        d = self.se_dec_80(d)
        d = self.bn14(self.act14(self.conv14(d)))      # (128, 80,  80)

        d = self.bn15(self.act15(self.deconv15(d)))    # (64,  160, 160)
        d = torch.cat([d, e1], dim=1)                  # (128, 160, 160)
        d = self.skip_bn_160(self.skip_act_160(self.skip_conv_160(d)))  # (64,160,160)
        d = self.se_dec_160(d)
        d = self.bn16(self.act16(self.conv16(d)))      # (32,  160, 160)

        out = self.act17(self.conv17(d))               # (3,   160, 160)
        return out

    def load_pretrained_v1(self, v1_state_dict):
        """从V1模型加载兼容的权重（共享的conv1-17层），新增层使用随机初始化"""
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in v1_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own_state)
        print(f"[V2模型] 从V1加载 {loaded} 个参数, 跳过 {skipped} 个, "
              f"新增层({len(own_state)-loaded}个参数)使用随机初始化")


class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.C = input_shape[0]
        self.H = input_shape[1]
        self.W = input_shape[2]
        # input_shape: (No, 3, H, W)
        self.conv1 = nn.Conv2d(self.C, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()

        in_features = 512 * math.ceil(self.H / 32) * math.ceil(self.W / 32)
        self.flatten6 = Flatten()

        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()
        # output_shape: (N, 1024)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.C = input_shape[0]
        self.H = input_shape[1]
        self.W = input_shape[2]

        # input_shape = (N, 3, H, W)
        self.conv1 = nn.Conv2d(self.C, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()

        # self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        # self.bn6 = nn.BatchNorm2d(512)
        # self.act6 = nn.ReLU()

        in_features = 512 * math.ceil(self.H // 32) * math.ceil(self.W // 32)
        self.flatten6 = Flatten()
        self.linear6 = nn.Linear(in_features, 1024)
        self.act6 = nn.ReLU()
        # output_shape: (N, 1024)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        # x = self.bn6(self.act6(self.conv6(x)))
        x = self.act6(self.linear6(self.flatten6(x)))
        return x


class Discriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape):
        super().__init__()
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_local_discriminator = LocalDiscriminator(local_input_shape)
        self.model_global_discriminator = GlobalDiscriminator(global_input_shape)

        self.concat1 = Concatenate()
        self.linear1 = nn.Linear(2048, 1)
        self.act1 = nn.Sigmoid()

    def forward(self, X):
        ld_X, gd_X = X
        return self.act1(self.linear1(self.concat1([self.model_global_discriminator(gd_X),
                                                    self.model_local_discriminator(ld_X)
                                                    ])))
