import argparse
import glob
import json
import os
import re

import matplotlib
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator, GeneratorV2
from super_resolution import SuperResolutionEnhancer
from test import evaluate_images, find_matching_pairs, load_image
from utils import MyDataset, gen_input_MaskLayer, gen_Mask, poisson_blend


def find_latest_model(result_dir='./result/'):
    """Find the latest generator checkpoint, preferring phase_3 over phase_1."""
    for phase in ['phase_3', 'phase_1']:
        phase_dir = os.path.join(result_dir, phase)
        if not os.path.isdir(phase_dir):
            continue

        pattern = os.path.join(phase_dir, f'{phase}_model_generator_epoch*')
        models = glob.glob(pattern)
        if not models:
            continue

        def extract_epoch(path):
            match = re.search(r'epoch(\d+)', os.path.basename(path))
            return int(match.group(1)) if match else -1

        models.sort(key=extract_epoch)
        best = models[-1]
        print(f'[Auto Detect] Use latest model: {best}')
        return best

    return None


def find_latest_config(result_dir='./result/'):
    """Find config.json in the training result directory."""
    config_path = os.path.join(result_dir, 'config.json')
    if os.path.isfile(config_path):
        print(f'[Auto Detect] Use latest config: {config_path}')
        return config_path
    return None


def resolve_model_and_config(args_model, args_config):
    """Resolve model/config paths and fallback to latest artifacts if possible."""
    model_path = args_model
    config_path = args_config

    if model_path is None or model_path == './demo/model_cn':
        detected = find_latest_model()
        if detected:
            model_path = detected
        elif model_path is None:
            model_path = './demo/model_cn'
            print('[Warning] No trained model found, fallback to ./demo/model_cn')

    if config_path is None or config_path == './demo/config.json':
        detected = find_latest_config()
        if detected:
            config_path = detected
        elif config_path is None:
            config_path = './demo/config.json'
            print('[Warning] No trained config found, fallback to ./demo/config.json')

    return model_path, config_path


parser = argparse.ArgumentParser()
parser.add_argument('model', nargs='?', default=None, help='Path to generator checkpoint. If omitted, auto-detect latest model.')
parser.add_argument('config', nargs='?', default=None, help='Path to config JSON. If omitted, auto-detect latest config.')
parser.add_argument('result_dir', nargs='?', default='./results/', help='Output directory for prediction results.')
parser.add_argument('--mode')
parser.add_argument(
    '--sr_method',
    default='bicubic',
    choices=['bicubic', 'srcnn', 'espcn', 'opencv'],
    help='Super-resolution method: bicubic, srcnn, espcn, opencv',
)
parser.add_argument('--sr_scale', type=int, default=2, help='Super-resolution upscale factor.')
parser.add_argument('--sr_enable', action='store_true', help='Enable super-resolution stage.')
parser.add_argument('--sr_detail_strength', type=float, default=0.08, help='Detail injection strength for SR (lower improves stability).')
parser.add_argument('--sr_ibp_iterations', type=int, default=3, help='Iterative back-projection steps.')
parser.add_argument('--sr_ibp_step', type=float, default=0.55, help='Iterative back-projection step size.')
parser.add_argument('--sr_cycle_tolerance', type=float, default=0.015, help='Quality-gate tolerance for SR consistency.')
parser.add_argument('--sr_residual_clip', type=float, default=0.05, help='Clip limit for SR residual (artifact control).')
parser.add_argument('--sr_texture_suppress', type=float, default=0.20, help='Residual smoothing ratio for SR artifact suppression.')
parser.add_argument('--sr_mask_blur', type=float, default=1.5, help='Boundary blur radius for SR mask fusion.')

sharpen_group = parser.add_mutually_exclusive_group()
sharpen_group.add_argument('--sharpen', dest='sharpen', action='store_true', help='Enable optional sharpening post-process.')
sharpen_group.add_argument('--no-sharpen', dest='sharpen', action='store_false', help='Disable sharpening (recommended for fair evaluation).')
parser.set_defaults(sharpen=False)

parser.add_argument('--sharpen_strength', type=float, default=0.5, help='Sharpening strength in [0.1, 1.0].')
parser.add_argument('--data', default='./datasets/test/', help='Input test image directory.')
parser.add_argument('--eval', action='store_true', help='Evaluate normal-resolution restored outputs.')
parser.add_argument('--eval_hr', action='store_true', help='Evaluate super-resolved outputs in HR mode.')
parser.add_argument('--model_version', default='auto', choices=['auto', 'v1', 'v2'], help='Generator version: auto / v1 / v2')


def run_evaluation(original_dir, restored_dir, eval_output_dir, suffix=None, image_size=160, hr_mode=False, sr_scale=2):
    """Evaluate restored outputs and generate report figures."""
    import matplotlib.pyplot as plt

    matplotlib.rcParams['axes.unicode_minus'] = False

    if not os.path.isdir(restored_dir):
        print(f'[Skip] Restored directory not found: {restored_dir}')
        return

    os.makedirs(eval_output_dir, exist_ok=True)
    pairs = find_matching_pairs(original_dir, restored_dir, suffix=suffix)

    if not pairs:
        print('[Skip] No matched image pairs found for evaluation.')
        return

    eval_image_size = image_size * sr_scale if hr_mode else image_size
    mode_text = 'HR (upsampled reference)' if hr_mode else 'LR (native reference)'

    print(f'\nStart evaluation: {len(pairs)} pairs')
    print(f'Mode: {mode_text}, eval image size={eval_image_size}')

    results = []

    for idx, (orig_file, rest_file) in enumerate(pairs):
        orig_path = os.path.join(original_dir, orig_file)
        rest_path = os.path.join(restored_dir, rest_file)

        orig_img = load_image(orig_path, image_size)
        rest_img = load_image(rest_path, eval_image_size)

        if orig_img is None or rest_img is None:
            continue

        if hr_mode:
            orig_img = F.interpolate(
                orig_img.unsqueeze(0),
                size=rest_img.shape[1:],
                mode='bicubic',
                align_corners=False,
            ).squeeze(0).clamp(0, 1)

        metrics = evaluate_images(orig_img, rest_img)
        results.append(
            {
                'original': orig_file,
                'restored': rest_file,
                'metrics': metrics,
                'orig_img': orig_img,
                'rest_img': rest_img,
            }
        )

        lpips_text = f" LPIPS={metrics['lpips']:.4f}" if 'lpips' in metrics else ''
        print(
            f"  [{idx + 1}/{len(pairs)}] {orig_file}: "
            f"MSE={metrics['mse']:.4f} "
            f"PSNR={metrics['psnr']:.2f}dB "
            f"SSIM={metrics['ssim']:.4f} "
            f"MS-SSIM={metrics.get('ms_ssim', 0):.4f} "
            f"GMSD={metrics.get('gmsd', 0):.4f}"
            f"{lpips_text}"
        )

    if not results:
        print('[Skip] No valid image pairs were evaluated.')
        return

    mse_list = [item['metrics']['mse'] for item in results]
    psnr_list = [item['metrics']['psnr'] for item in results]
    ssim_list = [item['metrics']['ssim'] for item in results]
    ms_ssim_list = [item['metrics'].get('ms_ssim', 0.0) for item in results]
    gmsd_list = [item['metrics'].get('gmsd', 0.0) for item in results]
    lpips_list = [item['metrics']['lpips'] for item in results if 'lpips' in item['metrics']]
    labels = [os.path.splitext(item['original'])[0][:10] for item in results]

    avg_mse = float(np.mean(mse_list))
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_ms_ssim = float(np.mean(ms_ssim_list))
    avg_gmsd = float(np.mean(gmsd_list))
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else None

    tag = suffix if suffix else ''

    # Figure 1: visual comparisons
    n_items = len(results)
    cols = min(4, n_items)
    rows = min(5, (n_items + cols - 1) // cols)
    show_n = min(n_items, rows * cols)

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3))
    if rows == 1 and cols * 2 == 2:
        axes = np.array([[axes[0], axes[1]]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Restoration Comparison{tag} (Original vs Restored)', fontsize=16, fontweight='bold')

    for i in range(show_n):
        row = i // cols
        col = i % cols
        item = results[i]

        orig_np = item['orig_img'].permute(1, 2, 0).numpy() if isinstance(item['orig_img'], torch.Tensor) else item['orig_img']
        rest_np = item['rest_img'].permute(1, 2, 0).numpy() if isinstance(item['rest_img'], torch.Tensor) else item['rest_img']

        ax_orig = axes[row, col * 2]
        ax_rest = axes[row, col * 2 + 1]

        ax_orig.imshow(np.clip(orig_np, 0, 1))
        ax_orig.set_title('Original', fontsize=8)
        ax_orig.axis('off')

        ax_rest.imshow(np.clip(rest_np, 0, 1))
        ax_rest.set_title(f"Restored PSNR={item['metrics']['psnr']:.1f}", fontsize=8)
        ax_rest.axis('off')

    for i in range(show_n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_output_dir, f'comparison_grid{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: per-image bars
    max_bars = 50
    if n_items > max_bars:
        sample_idx = np.linspace(0, n_items - 1, max_bars, dtype=int)
        sampled_mse = [mse_list[i] for i in sample_idx]
        sampled_psnr = [psnr_list[i] for i in sample_idx]
        sampled_ssim = [ssim_list[i] for i in sample_idx]
        sampled_ms_ssim = [ms_ssim_list[i] for i in sample_idx]
        sampled_gmsd = [gmsd_list[i] for i in sample_idx]
        sampled_labels = [labels[i] for i in sample_idx]
        bar_n = max_bars
        sample_note = f' (sampled {max_bars} from {n_items} images)'
    else:
        sampled_mse = mse_list
        sampled_psnr = psnr_list
        sampled_ssim = ssim_list
        sampled_ms_ssim = ms_ssim_list
        sampled_gmsd = gmsd_list
        sampled_labels = labels
        bar_n = n_items
        sample_note = ''

    fig_width = min(40, max(8, bar_n * 0.6))
    num_rows = 6 if avg_lpips is not None else 5
    fig, axes_bar = plt.subplots(num_rows, 1, figsize=(fig_width, num_rows * 4))
    fig.suptitle(f'Per-Image Restoration Metrics{tag}{sample_note}', fontsize=14, fontweight='bold')

    x = np.arange(bar_n)
    bar_width = 0.6

    bar_specs = [
        (sampled_mse, avg_mse, '#e74c3c', '#c0392b', 'MSE (lower is better)', 'Mean Squared Error (MSE)', None),
        (sampled_psnr, avg_psnr, '#2ecc71', '#27ae60', 'PSNR (dB, higher is better)', 'Peak Signal-to-Noise Ratio (PSNR)', f'{avg_psnr:.2f} dB'),
        (sampled_ssim, avg_ssim, '#3498db', '#2980b9', 'SSIM (higher is better)', 'Structural Similarity (SSIM)', None),
        (sampled_ms_ssim, avg_ms_ssim, '#9b59b6', '#8e44ad', 'MS-SSIM (higher is better)', 'Multi-Scale SSIM (MS-SSIM)', None),
        (
            sampled_gmsd,
            avg_gmsd,
            '#e67e22',
            '#d35400',
            'GMSD (lower is better)',
            'Gradient Magnitude Similarity Deviation (GMSD)',
            None,
        ),
    ]

    if avg_lpips is not None:
        if n_items <= max_bars:
            sampled_lpips = [item['metrics']['lpips'] for item in results if 'lpips' in item['metrics']]
        else:
            sampled_lpips = [
                results[i]['metrics']['lpips']
                for i in np.linspace(0, n_items - 1, max_bars, dtype=int)
                if 'lpips' in results[i]['metrics']
            ]
        bar_specs.append(
            (sampled_lpips, avg_lpips, '#1abc9c', '#16a085', 'LPIPS (lower is better)', 'Perceptual Similarity (LPIPS)', None)
        )

    for ax, (vals, avg_val, bar_color, line_color, ylabel, title, mean_fmt) in zip(axes_bar, bar_specs):
        ax.bar(x[:len(vals)], vals, bar_width, color=bar_color, alpha=0.8)
        mean_label = mean_fmt if mean_fmt else f'Mean: {avg_val:.4f}'
        ax.axhline(y=avg_val, color=line_color, linestyle='--', linewidth=2, label=mean_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x[:len(vals)])
        ax.set_xticklabels(sampled_labels[:len(vals)], rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_output_dir, f'metrics_per_image{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: distributions
    hist_specs = [
        (mse_list, avg_mse, '#e74c3c', '#c0392b', 'MSE', 'MSE Distribution', f'Mean: {avg_mse:.4f}'),
        (psnr_list, avg_psnr, '#2ecc71', '#27ae60', 'PSNR (dB)', 'PSNR Distribution', f'Mean: {avg_psnr:.2f} dB'),
        (ssim_list, avg_ssim, '#3498db', '#2980b9', 'SSIM', 'SSIM Distribution', f'Mean: {avg_ssim:.4f}'),
        (ms_ssim_list, avg_ms_ssim, '#9b59b6', '#8e44ad', 'MS-SSIM', 'MS-SSIM Distribution', f'Mean: {avg_ms_ssim:.4f}'),
        (gmsd_list, avg_gmsd, '#e67e22', '#d35400', 'GMSD', 'GMSD Distribution', f'Mean: {avg_gmsd:.4f}'),
    ]

    if avg_lpips is not None:
        hist_specs.append((lpips_list, avg_lpips, '#1abc9c', '#16a085', 'LPIPS', 'LPIPS Distribution', f'Mean: {avg_lpips:.4f}'))

    ncols = 3
    nrows_hist = (len(hist_specs) + ncols - 1) // ncols
    fig, axes_hist = plt.subplots(nrows_hist, ncols, figsize=(15, 5 * nrows_hist))
    axes_hist = np.array(axes_hist).reshape(-1)
    fig.suptitle(f'Metrics Distribution{tag}', fontsize=14, fontweight='bold')

    hist_bins = min(50, max(5, n_items // 3))
    for ax, (data, avg_val, bar_color, line_color, xlabel, title, mean_label) in zip(axes_hist, hist_specs):
        ax.hist(data, bins=hist_bins, color=bar_color, alpha=0.7, edgecolor='black')
        ax.axvline(avg_val, color=line_color, linestyle='--', linewidth=2, label=mean_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()

    for ax in axes_hist[len(hist_specs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_output_dir, f'metrics_distribution{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: dashboard
    metrics_info = [
        ('MSE', avg_mse, f'{avg_mse:.6f}', '#e74c3c', 'Lower is better'),
        ('PSNR', avg_psnr, f'{avg_psnr:.2f} dB', '#2ecc71', 'Higher is better'),
        ('SSIM', avg_ssim, f'{avg_ssim:.4f}', '#3498db', 'Higher is better'),
        ('MS-SSIM', avg_ms_ssim, f'{avg_ms_ssim:.4f}', '#9b59b6', 'Higher is better'),
        ('GMSD', avg_gmsd, f'{avg_gmsd:.4f}', '#e67e22', 'Lower is better'),
    ]

    if avg_lpips is not None:
        metrics_info.append(('LPIPS', avg_lpips, f'{avg_lpips:.4f}', '#1abc9c', 'Lower is better'))

    dash_cols = 3
    dash_rows = (len(metrics_info) + dash_cols - 1) // dash_cols
    fig, axes_dash = plt.subplots(dash_rows, dash_cols, figsize=(15, 5 * dash_rows))
    axes_dash = np.array(axes_dash).reshape(-1)
    fig.suptitle(f'Evaluation Summary{tag}', fontsize=16, fontweight='bold')

    for ax, (name, _, value_str, color, note) in zip(axes_dash, metrics_info):
        ax.text(0.5, 0.55, value_str, transform=ax.transAxes, fontsize=32, fontweight='bold', ha='center', va='center', color=color)
        ax.text(0.5, 0.82, name, transform=ax.transAxes, fontsize=18, fontweight='bold', ha='center', va='center', color='#2c3e50')
        ax.text(0.5, 0.28, note, transform=ax.transAxes, fontsize=11, ha='center', va='center', color='#7f8c8d')
        ax.text(0.5, 0.12, f'Evaluated: {len(results)} images', transform=ax.transAxes, fontsize=10, ha='center', va='center', color='#95a5a6')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#bdc3c7')
            spine.set_linewidth(2)

    for ax in axes_dash[len(metrics_info):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_output_dir, f'summary_dashboard{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: PSNR vs SSIM
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(psnr_list, ssim_list, c=mse_list, cmap='RdYlGn_r', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PSNR (dB)', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title(f'PSNR vs SSIM Correlation{tag}', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('MSE')

    ax.axhline(y=avg_ssim, color='#3498db', linestyle=':', alpha=0.5, label=f'Avg SSIM: {avg_ssim:.4f}')
    ax.axvline(x=avg_psnr, color='#2ecc71', linestyle=':', alpha=0.5, label=f'Avg PSNR: {avg_psnr:.2f}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_output_dir, f'psnr_vs_ssim{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Text report
    report_path = os.path.join(eval_output_dir, f'evaluation_results{tag}.txt')
    with open(report_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('=' * 60 + '\n')
        file_obj.write('Image Restoration Evaluation Report\n')
        file_obj.write('=' * 60 + '\n\n')

        file_obj.write(f'Number of evaluated pairs: {len(results)}\n')
        file_obj.write(f'Original images dir: {original_dir}\n')
        file_obj.write(f'Restored images dir: {restored_dir}\n\n')

        file_obj.write('-' * 40 + '\n')
        file_obj.write('Average Metrics\n')
        file_obj.write('-' * 40 + '\n')
        file_obj.write(f'Avg MSE:     {avg_mse:.6f} (lower is better)\n')
        file_obj.write(f'Avg PSNR:    {avg_psnr:.4f} dB (higher is better)\n')
        file_obj.write(f'Avg SSIM:    {avg_ssim:.6f} (higher is better, <= 1.0)\n')
        file_obj.write(f'Avg MS-SSIM: {avg_ms_ssim:.6f} (higher is better)\n')
        file_obj.write(f'Avg GMSD:    {avg_gmsd:.6f} (lower is better)\n')
        if avg_lpips is not None:
            file_obj.write(f'Avg LPIPS:   {avg_lpips:.6f} (lower is better)\n')

        file_obj.write('\n')
        file_obj.write(f'Best PSNR:   {max(psnr_list):.4f} dB\n')
        file_obj.write(f'Worst PSNR:  {min(psnr_list):.4f} dB\n')
        file_obj.write(f'Best SSIM:   {max(ssim_list):.4f}\n')
        file_obj.write(f'Worst SSIM:  {min(ssim_list):.4f}\n\n')

        file_obj.write('-' * 40 + '\n')
        file_obj.write('Per-Image Metrics\n')
        file_obj.write('-' * 40 + '\n')

        for i, item in enumerate(results):
            file_obj.write(f"\n[{i + 1}] {item['original']} -> {item['restored']}\n")
            file_obj.write(f"  MSE     = {item['metrics']['mse']:.6f}\n")
            file_obj.write(f"  PSNR    = {item['metrics']['psnr']:.4f} dB\n")
            file_obj.write(f"  SSIM    = {item['metrics']['ssim']:.6f}\n")
            file_obj.write(f"  MS-SSIM = {item['metrics'].get('ms_ssim', 0.0):.6f}\n")
            file_obj.write(f"  GMSD    = {item['metrics'].get('gmsd', 0.0):.6f}\n")
            if 'lpips' in item['metrics']:
                file_obj.write(f"  LPIPS   = {item['metrics']['lpips']:.6f}\n")

        file_obj.write('\n' + '=' * 60 + '\n')
        file_obj.write('Generated Figures\n')
        file_obj.write('=' * 60 + '\n')
        file_obj.write(f'1. comparison_grid{tag}.png\n')
        file_obj.write(f'2. metrics_per_image{tag}.png\n')
        file_obj.write(f'3. metrics_distribution{tag}.png\n')
        file_obj.write(f'4. summary_dashboard{tag}.png\n')
        file_obj.write(f'5. psnr_vs_ssim{tag}.png\n')

    for item in results:
        del item['orig_img']
        del item['rest_img']

    print(f"\n{'=' * 50}")
    print(f'Evaluation summary {tag}:')
    print(f'  Number of evaluated pairs: {len(results)}')
    print(f'  Avg MSE:     {avg_mse:.6f}')
    print(f'  Avg PSNR:    {avg_psnr:.4f} dB')
    print(f'  Avg SSIM:    {avg_ssim:.4f}')
    print(f'  Avg MS-SSIM: {avg_ms_ssim:.4f}')
    print(f'  Avg GMSD:    {avg_gmsd:.4f} (lower is better)')
    if avg_lpips is not None:
        print(f'  Avg LPIPS:   {avg_lpips:.4f} (lower is better)')

    print('  Generated files:')
    print(f'    - comparison_grid{tag}.png')
    print(f'    - metrics_per_image{tag}.png')
    print(f'    - metrics_distribution{tag}.png')
    print(f'    - summary_dashboard{tag}.png')
    print(f'    - psnr_vs_ssim{tag}.png')
    print(f'  Output directory: {eval_output_dir}')
    print(f"{'=' * 50}")


def fuse_sr_with_known_regions(completed_hr, reference_hr, mask_lr, blur_radius=1.5):
    """
    Keep known regions from the original input after SR.
    This prevents SR artifacts on non-masked regions.
    """
    mask_hr = F.interpolate(mask_lr, size=completed_hr.shape[2:], mode='bilinear', align_corners=False)
    if blur_radius > 0:
        kernel_size = int(max(3, 2 * round(blur_radius * 2) + 1))
        mask_hr = F.avg_pool2d(mask_hr, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    mask_hr = torch.clamp(mask_hr, 0.0, 1.0)
    fused = completed_hr * mask_hr + reference_hr * (1.0 - mask_hr)
    return torch.clamp(fused, 0.0, 1.0)


def main(
    input_generator_size=160,
    config=None,
    model=None,
    data='./datasets/test/',
    result_dir='./results/',
    mode=None,
    sr_enable=False,
    sr_method='bicubic',
    sr_scale=2,
    sr_detail_strength=0.08,
    sr_ibp_iterations=3,
    sr_ibp_step=0.55,
    sr_cycle_tolerance=0.015,
    sr_residual_clip=0.05,
    sr_texture_suppress=0.20,
    sr_mask_blur=1.5,
    sharpen=False,
    sharpen_strength=0.5,
    eval_normal=False,
    eval_hr=False,
    model_version='auto',
):
    model, config = resolve_model_and_config(model, config)

    os.makedirs(result_dir, exist_ok=True)

    print(
        f'Run settings: image_size={input_generator_size}, config={config}, '
        f'model={model}, data={data}'
    )
    print(f'Result directory: {result_dir}')
    print(f'SR enabled: {sr_enable}, method={sr_method}, scale={sr_scale}')
    print(
        'SR tuning: '
        f'detail={sr_detail_strength}, ibp_iters={sr_ibp_iterations}, ibp_step={sr_ibp_step}, '
        f'cycle_tol={sr_cycle_tolerance}, residual_clip={sr_residual_clip}, '
        f'texture_suppress={sr_texture_suppress}, mask_blur={sr_mask_blur}'
    )
    print(f'Sharpen enabled: {sharpen}, strength={sharpen_strength}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(config, 'r', encoding='utf-8') as file_obj:
        config_dict = json.load(file_obj)

    mpv = torch.tensor(config_dict['mpv']).view(1, 3, 1, 1).to(device)

    if model_version == 'auto':
        state_dict = torch.load(model, map_location='cpu')
        try:
            generator = GeneratorV2()
            generator.load_state_dict(state_dict)
            model_version = 'v2'
        except RuntimeError:
            generator = Generator()
            generator.load_state_dict(state_dict)
            model_version = 'v1'
        print(f'Auto-detected model version: {model_version}')
    elif model_version == 'v2':
        generator = GeneratorV2()
        generator.load_state_dict(torch.load(model, map_location='cpu'))
    else:
        generator = Generator()
        generator.load_state_dict(torch.load(model, map_location='cpu'))

    generator = generator.to(device)
    generator.eval()
    print(f'Loaded generator version: {model_version}')

    sr_enhancer = None
    if sr_enable:
        print(f'Initialize SR enhancer: method={sr_method}, scale={sr_scale}')
        sr_enhancer = SuperResolutionEnhancer(
            method=sr_method,
            scale_factor=sr_scale,
            device=device,
            sharpen=sharpen,
            sharpen_strength=sharpen_strength,
            detail_strength=sr_detail_strength,
            ibp_iterations=sr_ibp_iterations,
            ibp_step=sr_ibp_step,
            cycle_tolerance=sr_cycle_tolerance,
            residual_clip=sr_residual_clip,
            texture_suppress=sr_texture_suppress,
        )

    transform = transforms.Compose(
        [
            transforms.Resize(input_generator_size),
            transforms.CenterCrop((input_generator_size, input_generator_size)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = MyDataset(data, transform)
    print(f'Found {len(test_dataset)} images')

    if len(test_dataset) == 0:
        print(f'Error: no .jpg/.png files found in {data}')
        return

    with torch.no_grad():
        if mode == 'cat':
            print('Run in concatenated batch mode (cat).')
            batch = [torch.unsqueeze(test_dataset[i], dim=0) for i in range(len(test_dataset))]
            x = torch.cat(batch, dim=0).to(device)

            mask_size, mask_location = gen_Mask(x)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location,
            ).to(device)

            x_mask = x - x * masklayer + mpv * masklayer
            input_g = torch.cat((x_mask, masklayer), dim=1)
            output_g = generator(input_g)
            completed = poisson_blend(x_mask, output_g, masklayer).to(device)
            completed = torch.clamp(completed * masklayer + x * (1.0 - masklayer), 0.0, 1.0)

            if sr_enhancer is not None:
                print('Applying SR to concatenated outputs...')
                x_hr = sr_enhancer.enhance(x)
                x_mask_hr = sr_enhancer.enhance(x_mask)
                completed_hr_raw = sr_enhancer.enhance(completed)
                completed_hr = fuse_sr_with_known_regions(completed_hr_raw, x_hr, masklayer, blur_radius=sr_mask_blur)

                imgs = torch.cat((x_hr, x_mask_hr, completed_hr), dim=2)
                imgpath = os.path.join(result_dir, 'predict_hr.png')
                save_image(imgs, imgpath, nrow=1)
                print(f'Saved SR result image: {imgpath}')

                imgs_original = torch.cat((x, x_mask, completed), dim=2)
                imgpath_original = os.path.join(result_dir, 'predict_original.png')
                save_image(imgs_original, imgpath_original, nrow=1)
                print(f'Saved original-resolution result image: {imgpath_original}')
            else:
                imgs = torch.cat((x, x_mask, completed), dim=2)
                imgpath = os.path.join(result_dir, 'predict.png')
                save_image(imgs, imgpath, nrow=1)
                print(f'Saved result image: {imgpath}')
        else:
            print('Run in per-image mode.')
            for index, sample in enumerate(test_dataset):
                orig_filename = os.path.splitext(os.path.basename(test_dataset.images[index]))[0]
                print(f'Processing {index + 1}/{len(test_dataset)}: {orig_filename}')

                x = torch.unsqueeze(sample, dim=0).to(device)
                mask_size, mask_location = gen_Mask(x)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location,
                ).to(device)

                x_mask = x - x * masklayer + mpv * masklayer
                input_g = torch.cat((x_mask, masklayer), dim=1)
                output_g = generator(input_g)
                completed = poisson_blend(x_mask, output_g, masklayer).to(device)
                completed = torch.clamp(completed * masklayer + x * (1.0 - masklayer), 0.0, 1.0)

                if sr_enhancer is not None:
                    print(f'Applying SR for {orig_filename}...')
                    x_hr = sr_enhancer.enhance(x)
                    x_mask_hr = sr_enhancer.enhance(x_mask)
                    completed_hr_raw = sr_enhancer.enhance(completed)
                    completed_hr = fuse_sr_with_known_regions(completed_hr_raw, x_hr, masklayer, blur_radius=sr_mask_blur)

                    imgs_hr = torch.cat((x_hr, x_mask_hr, completed_hr), dim=2)
                    imgpath_hr = os.path.join(result_dir, f'{orig_filename}_hr.png')
                    save_image(imgs_hr, imgpath_hr, nrow=3)
                    print(f'Saved SR output: {imgpath_hr}')

                    imgs_original = torch.cat((x, x_mask, completed), dim=2)
                    imgpath_original = os.path.join(result_dir, f'{orig_filename}_original.png')
                    save_image(imgs_original, imgpath_original, nrow=3)
                    print(f'Saved original-resolution output: {imgpath_original}')

                    if eval_normal:
                        completed_only_dir = os.path.join(result_dir, 'completed')
                        os.makedirs(completed_only_dir, exist_ok=True)
                        save_image(completed, os.path.join(completed_only_dir, f'{orig_filename}.png'))

                    if eval_hr:
                        completed_hr_dir = os.path.join(result_dir, 'completed_hr')
                        os.makedirs(completed_hr_dir, exist_ok=True)
                        save_image(completed_hr, os.path.join(completed_hr_dir, f'{orig_filename}.png'))
                else:
                    imgs = torch.cat((x, x_mask, completed), dim=2)
                    imgpath = os.path.join(result_dir, f'{orig_filename}.png')
                    save_image(imgs, imgpath, nrow=3)
                    print(f'Saved output: {imgpath}')

                    if eval_normal:
                        completed_only_dir = os.path.join(result_dir, 'completed')
                        os.makedirs(completed_only_dir, exist_ok=True)
                        save_image(completed, os.path.join(completed_only_dir, f'{orig_filename}.png'))

            if sr_enhancer is not None:
                print(f'SR outputs saved in: {result_dir}')
                print(f'Original-resolution outputs saved in: {result_dir}')
            else:
                print(f'Outputs saved in: {result_dir}')

    if eval_normal:
        print(f"\n{'=' * 60}")
        print('Start normal-resolution evaluation...')
        print(f"{'=' * 60}")

        completed_only_dir = os.path.join(result_dir, 'completed')
        eval_output_dir = os.path.join(result_dir, 'eval_normal')
        run_evaluation(
            data,
            completed_only_dir,
            eval_output_dir,
            image_size=input_generator_size,
            hr_mode=False,
            sr_scale=sr_scale,
        )

    if eval_hr:
        print(f"\n{'=' * 60}")
        print('Start HR evaluation for SR outputs...')
        print(f"{'=' * 60}")

        completed_hr_dir = os.path.join(result_dir, 'completed_hr')
        eval_output_dir = os.path.join(result_dir, 'eval_hr')
        run_evaluation(
            data,
            completed_hr_dir,
            eval_output_dir,
            image_size=input_generator_size,
            hr_mode=True,
            sr_scale=sr_scale,
        )


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        config=args.config,
        model=args.model,
        result_dir=args.result_dir,
        mode=args.mode,
        sr_enable=args.sr_enable,
        sr_method=args.sr_method,
        sr_scale=args.sr_scale,
        sr_detail_strength=args.sr_detail_strength,
        sr_ibp_iterations=args.sr_ibp_iterations,
        sr_ibp_step=args.sr_ibp_step,
        sr_cycle_tolerance=args.sr_cycle_tolerance,
        sr_residual_clip=args.sr_residual_clip,
        sr_texture_suppress=args.sr_texture_suppress,
        sr_mask_blur=args.sr_mask_blur,
        sharpen=args.sharpen,
        sharpen_strength=args.sharpen_strength,
        data=args.data,
        eval_normal=args.eval,
        eval_hr=args.eval_hr,
        model_version=args.model_version,
    )
