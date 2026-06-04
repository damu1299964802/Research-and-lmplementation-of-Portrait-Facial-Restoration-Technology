import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return torch.sigmoid(x)


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return torch.sigmoid(x)


def apply_unsharp_mask(image, strength=0.5, radius=1, threshold=0):
    """Apply unsharp masking on a BCHW tensor in [0, 1]."""
    device = image.device
    batch_size = image.shape[0]
    result = torch.zeros_like(image)

    for idx in range(batch_size):
        img_np = image[idx].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        img_np = np.clip(img_np, 0.0, 1.0)

        blurred = cv2.GaussianBlur(img_np, (radius * 2 + 1, radius * 2 + 1), radius)
        sharpened = cv2.addWeighted(img_np, 1.0 + strength, blurred, -strength, 0.0)

        if threshold > 0:
            mask = np.abs(img_np - blurred) >= float(threshold)
            sharpened = np.where(mask, sharpened, img_np)

        sharpened = np.clip(sharpened, 0.0, 1.0)
        sharpened_tensor = torch.from_numpy(sharpened).permute(2, 0, 1)
        result[idx] = sharpened_tensor

    return result.to(device)


class SuperResolutionEnhancer:
    def __init__(
        self,
        method='bicubic',
        scale_factor=2,
        device='cuda',
        sharpen=False,
        sharpen_strength=0.5,
        detail_strength=0.12,
        ibp_iterations=4,
        ibp_step=0.65,
        cycle_tolerance=0.02,
        residual_clip=0.05,
        texture_suppress=0.20,
    ):
        self.method = method
        self.scale_factor = scale_factor

        if isinstance(device, torch.device):
            requested_device = device
        else:
            requested_device = torch.device(device)

        if requested_device.type == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = requested_device

        self.sharpen = sharpen
        self.sharpen_strength = sharpen_strength
        self.detail_strength = detail_strength
        self.ibp_iterations = ibp_iterations
        self.ibp_step = ibp_step
        self.cycle_tolerance = max(1e-6, cycle_tolerance)
        self.residual_clip = max(1e-4, residual_clip)
        self.texture_suppress = min(max(texture_suppress, 0.0), 1.0)

        print(f"Initialize SR enhancer: method={method}, scale={scale_factor}, device={self.device}")
        print(
            f"Post-process: sharpen={'on' if sharpen else 'off'} "
            f"(strength={sharpen_strength}), detail_strength={detail_strength}, "
            f"ibp_iterations={ibp_iterations}, ibp_step={ibp_step}, "
            f"residual_clip={self.residual_clip}, texture_suppress={self.texture_suppress}"
        )

        if method == 'srcnn':
            self.model = SRCNN().to(self.device)
            print('Loaded SRCNN model structure')
        elif method == 'espcn':
            self.model = ESPCN(scale_factor=scale_factor).to(self.device)
            print('Loaded ESPCN model structure')
        elif method == 'bicubic':
            print('Using bicubic interpolation')
        elif method == 'opencv':
            print('Using OpenCV Lanczos4 interpolation')
        else:
            print(f"[Warning] Unknown SR method '{method}', fallback to bicubic.")

    def _gaussian_blur_tensor(self, tensor):
        """Lightweight Gaussian blur implemented by depthwise convolution."""
        kernel = torch.tensor(
            [[1.0, 4.0, 6.0, 4.0, 1.0],
             [4.0, 16.0, 24.0, 16.0, 4.0],
             [6.0, 24.0, 36.0, 24.0, 6.0],
             [4.0, 16.0, 24.0, 16.0, 4.0],
             [1.0, 4.0, 6.0, 4.0, 1.0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ) / 256.0
        channels = tensor.shape[1]
        kernel = kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
        return F.conv2d(tensor, kernel, padding=2, groups=channels)

    def _edge_magnitude(self, tensor):
        """Compute normalized edge magnitude map from RGB tensor (B,C,H,W)."""
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return edge

    def _edge_guided_detail_boost(self, high_res, low_res):
        """Inject upsampled LR details with edge-aware gain control."""
        if self.detail_strength <= 0:
            return high_res

        low_res_smooth = self._gaussian_blur_tensor(low_res)
        low_res_detail = low_res - low_res_smooth
        detail_up = F.interpolate(low_res_detail, size=high_res.shape[2:], mode='bicubic', align_corners=False)

        edge_map = self._edge_magnitude(low_res)
        edge_up = F.interpolate(edge_map, size=high_res.shape[2:], mode='bilinear', align_corners=False)

        gain = self.detail_strength * (0.4 + 0.6 * edge_up)
        boosted = high_res + gain * detail_up
        return torch.clamp(boosted, 0.0, 1.0)

    def _iterative_back_projection(self, high_res, low_res):
        """Enforce LR-HR consistency to suppress hallucinated artifacts."""
        if self.ibp_iterations <= 0:
            return high_res

        refined = high_res
        for _ in range(self.ibp_iterations):
            low_res_recon = F.interpolate(refined, size=low_res.shape[2:], mode='bicubic', align_corners=False)
            residual = low_res - low_res_recon
            residual_up = F.interpolate(residual, size=refined.shape[2:], mode='bicubic', align_corners=False)
            refined = torch.clamp(refined + self.ibp_step * residual_up, 0.0, 1.0)

        return refined

    def _stabilize_detail(self, candidate, baseline, low_res):
        """
        Clip and smooth SR residuals to reduce ringing/over-sharpen artifacts.
        This tends to improve SSIM/GMSD stability.
        """
        residual = candidate - baseline

        edge_map = self._edge_magnitude(low_res)
        edge_up = F.interpolate(edge_map, size=candidate.shape[2:], mode='bilinear', align_corners=False)
        clip_map = self.residual_clip * (0.5 + 0.5 * edge_up)
        residual = torch.maximum(torch.minimum(residual, clip_map), -clip_map)

        smooth_residual = self._gaussian_blur_tensor(residual)
        residual = (1.0 - self.texture_suppress) * residual + self.texture_suppress * smooth_residual

        stabilized = baseline + residual
        return torch.clamp(stabilized, 0.0, 1.0)

    def _stability_blend(self, candidate, baseline, cycle_ratio):
        """Blend candidate with baseline when candidate is only slightly worse in cycle consistency."""
        if cycle_ratio <= 1.0:
            return candidate

        degrade = min(max((cycle_ratio - 1.0) / self.cycle_tolerance, 0.0), 1.0)
        baseline_weight = 0.25 + 0.5 * degrade
        blended = (1.0 - baseline_weight) * candidate + baseline_weight * baseline
        return torch.clamp(blended, 0.0, 1.0)

    def _cycle_consistency_mse(self, high_res, low_res):
        """
        Downscale SR output back to low resolution and compare with the input.
        Lower MSE means fewer artifacts were introduced by the SR stage.
        """
        low_res_recon = F.interpolate(
            high_res,
            size=low_res.shape[2:],
            mode='bicubic',
            align_corners=False,
        )
        low_res_recon = torch.clamp(low_res_recon, 0.0, 1.0)
        return F.mse_loss(low_res_recon, low_res).item()

    def _gradient_consistency_error(self, high_res, low_res):
        """
        Compare Sobel-edge maps after downsampling.
        Lower is better and correlates with SSIM/GMSD stability.
        """
        low_res_recon = F.interpolate(
            high_res,
            size=low_res.shape[2:],
            mode='bicubic',
            align_corners=False,
        )
        low_res_recon = torch.clamp(low_res_recon, 0.0, 1.0)

        edge_ref = self._edge_magnitude(low_res)
        edge_rec = self._edge_magnitude(low_res_recon)
        return F.l1_loss(edge_rec, edge_ref).item()

    def enhance_opencv(self, img_tensor, scale_factor):
        """OpenCV Lanczos4 upsampling with optional mild sharpening."""
        output_samples = []

        for sample in img_tensor:
            img_np = sample.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            img_np = np.clip(img_np, 0.0, 1.0)
            height, width = img_np.shape[:2]

            img_up = cv2.resize(
                img_np,
                (width * scale_factor, height * scale_factor),
                interpolation=cv2.INTER_LANCZOS4,
            )

            if self.sharpen:
                mild_strength = min(self.sharpen_strength * 0.4, 0.2)
                blurred = cv2.GaussianBlur(img_up, (3, 3), 1.0)
                img_up = cv2.addWeighted(img_up, 1.0 + mild_strength, blurred, -mild_strength, 0.0)

            img_up = np.clip(img_up, 0.0, 1.0)
            output_samples.append(torch.from_numpy(img_up).permute(2, 0, 1).float())

        return torch.stack(output_samples, dim=0).to(img_tensor.device)

    def bicubic_with_sharpening(self, img_tensor, scale_factor):
        """Bicubic upsampling with optional unsharp mask."""
        upscaled = F.interpolate(
            img_tensor,
            scale_factor=scale_factor,
            mode='bicubic',
            align_corners=False,
        )
        upscaled = torch.clamp(upscaled, 0.0, 1.0)

        if self.sharpen:
            upscaled = apply_unsharp_mask(
                upscaled,
                strength=min(self.sharpen_strength * 0.5, 0.25),
            )

        return upscaled

    def enhance(self, img_tensor):
        """
        Enhance image resolution.
        Input tensor shape must be (B, C, H, W) with values in [0, 1].
        """
        try:
            if not isinstance(img_tensor, torch.Tensor):
                raise TypeError(f'Input must be torch.Tensor, got {type(img_tensor)}')

            if img_tensor.ndim != 4:
                raise ValueError(f'Input tensor must be 4D (B,C,H,W), got shape={img_tensor.shape}')

            if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                print('[Warning] Input contains NaN/Inf; replaced with finite values.')
                img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0)

            if img_tensor.min() < 0 or img_tensor.max() > 1:
                print(
                    f"[Warning] Input value range out of [0,1], clamped. "
                    f"min={img_tensor.min().item():.6f}, max={img_tensor.max().item():.6f}"
                )
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

            _, _, height, width = img_tensor.shape
            print(f'SR processing size: {height}x{width} -> {height * self.scale_factor}x{width * self.scale_factor}')

            original_device = img_tensor.device
            work_tensor = img_tensor.to(self.device)

            baseline = F.interpolate(
                work_tensor,
                scale_factor=self.scale_factor,
                mode='bicubic',
                align_corners=False,
            )
            baseline = torch.clamp(baseline, 0.0, 1.0)

            if self.method == 'bicubic':
                candidate = self.bicubic_with_sharpening(work_tensor, self.scale_factor)
                print(f'Bicubic interpolation done: {candidate.shape[2]}x{candidate.shape[3]}')
            elif self.method == 'opencv':
                candidate = self.enhance_opencv(work_tensor, self.scale_factor)
                print(f'OpenCV Lanczos4 done: {candidate.shape[2]}x{candidate.shape[3]}')
            elif self.method in ['srcnn', 'espcn']:
                print(f"[Info] {self.method.upper()} has no pretrained weights; fallback to bicubic.")
                candidate = self.bicubic_with_sharpening(work_tensor, self.scale_factor)
                print(f'Bicubic fallback done: {candidate.shape[2]}x{candidate.shape[3]}')
            else:
                candidate = self.bicubic_with_sharpening(work_tensor, self.scale_factor)
                print(f'Default bicubic done: {candidate.shape[2]}x{candidate.shape[3]}')

            # Core SR reinforcement: detail boost + iterative back projection.
            candidate = self._edge_guided_detail_boost(candidate, work_tensor)
            candidate = self._iterative_back_projection(candidate, work_tensor)
            candidate = self._stabilize_detail(candidate, baseline, work_tensor)
            candidate = torch.clamp(candidate, 0.0, 1.0)

            with torch.no_grad():
                base_cycle_mse = self._cycle_consistency_mse(baseline, work_tensor)
                cand_cycle_mse = self._cycle_consistency_mse(candidate, work_tensor)
                cycle_ratio = cand_cycle_mse / (base_cycle_mse + 1e-12)

                base_grad_err = self._gradient_consistency_error(baseline, work_tensor)
                cand_grad_err = self._gradient_consistency_error(candidate, work_tensor)
                grad_ratio = cand_grad_err / (base_grad_err + 1e-12)
                fused_ratio = max(cycle_ratio, grad_ratio)

                if fused_ratio <= 0.995:
                    enhanced_tensor = candidate
                    print(
                        f"[Quality Gate] Improved consistency "
                        f"(cycle {cand_cycle_mse:.6f}/{base_cycle_mse:.6f}, "
                        f"grad {cand_grad_err:.6f}/{base_grad_err:.6f}), keep enhanced SR"
                    )
                elif fused_ratio <= 1.0 + self.cycle_tolerance:
                    enhanced_tensor = self._stability_blend(candidate, baseline, fused_ratio)
                    print(
                        f"[Quality Gate] Slight consistency increase "
                        f"(cycle {cand_cycle_mse:.6f}/{base_cycle_mse:.6f}, "
                        f"grad {cand_grad_err:.6f}/{base_grad_err:.6f}), apply stability blend"
                    )
                else:
                    enhanced_tensor = baseline
                    print(
                        f"[Quality Gate] Degraded consistency "
                        f"(cycle {cand_cycle_mse:.6f}/{base_cycle_mse:.6f}, "
                        f"grad {cand_grad_err:.6f}/{base_grad_err:.6f}), fallback to bicubic baseline"
                    )

            enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)

            if torch.isnan(enhanced_tensor).any() or torch.isinf(enhanced_tensor).any():
                print('[Warning] Output contains NaN/Inf; replaced with finite values.')
                enhanced_tensor = torch.nan_to_num(enhanced_tensor, nan=0.0, posinf=1.0, neginf=0.0)

            return enhanced_tensor.to(original_device)

        except Exception as err:
            print(f'SR processing error: {err}')
            print('Falling back to bicubic upsampling.')

            try:
                _, _, height, width = img_tensor.shape
                backup = F.interpolate(
                    img_tensor.to(self.device),
                    size=(height * self.scale_factor, width * self.scale_factor),
                    mode='bicubic',
                    align_corners=False,
                )
                backup = torch.clamp(backup, 0.0, 1.0)
                return backup.to(img_tensor.device)
            except Exception as backup_err:
                print(f'Backup upsampling failed: {backup_err}. Return original tensor.')
                return img_tensor

    def enhance_batch(self, img_batch):
        return self.enhance(img_batch)
