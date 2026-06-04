import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def _setup_chinese_font():
    """配置matplotlib中文字体，支持Windows和Linux"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Noto Sans CJK',
        'Source Han Sans SC', 'Source Han Sans',
        'AR PL UMing CN', 'Droid Sans Fallback',
    ]
    # 额外扫描Linux常见字体目录
    extra_dirs = [
        '/usr/share/fonts', '/usr/local/share/fonts',
        os.path.expanduser('~/.fonts'),
        os.path.expanduser('~/.local/share/fonts'),
    ]
    for d in extra_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for fname in files:
                    if fname.endswith(('.ttf', '.ttc', '.otf')):
                        fpath = os.path.join(root, fname)
                        try:
                            font_manager.fontManager.addfont(fpath)
                        except Exception:
                            pass

    # 扫描所有已注册字体
    available = {}
    for f in font_manager.fontManager.ttflist:
        available[f.name] = f.fname

    # 按优先级查找
    for font_name in chinese_fonts:
        for avail_name in available:
            if font_name.lower() in avail_name.lower():
                plt.rcParams['font.sans-serif'] = [avail_name] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['axes.unicode_minus'] = False
                return avail_name

    # 未找到中文字体
    plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False
    print("[警告] 未找到中文字体，图表中的中文可能显示为方块。")
    print("  Linux安装方法: sudo apt install fonts-wqy-microhei fonts-wqy-zenhei")
    print("  或: sudo yum install wqy-microhei-fonts wqy-zenhei-fonts")
    print("  安装后请删除matplotlib字体缓存: rm -rf ~/.cache/matplotlib")
    return None

_font_used = _setup_chinese_font()
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import mean_squared_error as calculate_mse
from skimage.transform import resize as skimage_resize
from scipy import ndimage

try:
    import lpips as lpips_lib
    _lpips_fn = lpips_lib.LPIPS(net='alex', verbose=False)
    _lpips_available = True
except ImportError:
    _lpips_available = False


def calculate_ms_ssim(img1_np, img2_np, data_range=1.0, levels=3):
    """MS-SSIM（多尺度结构相似性），值域 [0,1]，越大越好"""
    weights = [0.2, 0.4, 0.4][:levels]
    img1 = img1_np.copy()
    img2 = img2_np.copy()
    ms_val = 1.0
    for i in range(levels):
        h, w = img1.shape[:2]
        win = min(7, h - 1 if (h - 1) % 2 == 0 else h - 2)
        win = max(win, 3)
        if win % 2 == 0:
            win -= 1
        try:
            s = calculate_ssim(img1, img2, channel_axis=2, data_range=data_range, win_size=win)
        except Exception:
            s = 0.0
        ms_val *= s ** weights[i]
        if i < levels - 1:
            img1 = skimage_resize(img1, (max(h // 2, 8), max(w // 2, 8)), anti_aliasing=True)
            img2 = skimage_resize(img2, (max(h // 2, 8), max(w // 2, 8)), anti_aliasing=True)
    return float(ms_val)


def calculate_gmsd(img1_np, img2_np):
    """GMSD（梯度幅度相似性偏差），越小越好，衡量全局锐度一致性"""
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3.0
    ky = kx.T
    def to_gray(img):
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    g1 = to_gray(img1_np)
    g2 = to_gray(img2_np)
    gm1 = np.sqrt(ndimage.convolve(g1, kx) ** 2 + ndimage.convolve(g1, ky) ** 2 + 1e-8)
    gm2 = np.sqrt(ndimage.convolve(g2, kx) ** 2 + ndimage.convolve(g2, ky) ** 2 + 1e-8)
    c = 0.0026
    gms_map = (2 * gm1 * gm2 + c) / (gm1 ** 2 + gm2 ** 2 + c)
    return float(np.std(gms_map))


def calculate_lpips_score(img1_tensor, img2_tensor):
    """LPIPS（感知图像块相似性），越小越好"""
    if not _lpips_available:
        return None
    try:
        # lpips 需要 [-1,1] 范围
        t1 = img1_tensor.unsqueeze(0) * 2 - 1
        t2 = img2_tensor.unsqueeze(0) * 2 - 1
        with torch.no_grad():
            score = _lpips_fn(t1, t2)
        return float(score.item())
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='评估图像修复质量')
    parser.add_argument('--original', type=str, help='原始图像文件夹或文件路径')
    parser.add_argument('--restored', type=str, help='修复后图像文件夹或文件路径')
    parser.add_argument('--output', type=str, default='./evaluation_results', help='评估结果输出路径')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--save_comparison', action='store_true', help='是否保存对比图')
    parser.add_argument('--single_mode', action='store_true', help='单文件比较模式')
    parser.add_argument('--suffix', type=str, default=None, help='只匹配restored文件夹中带有指定后缀的文件，如 _hr 或 _original')
    
    return parser.parse_args()

def load_image(path, image_size=160):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    try:
        img = Image.open(path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"无法加载图像 {path}: {e}")
        return None

def find_matching_pairs(original_dir, restored_dir, suffix=None):
    """查找原始和修复图像的匹配对"""
    original_files = [f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    restored_files = [f for f in os.listdir(restored_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 按后缀过滤restored文件 (如 _hr 或 _original)
    if suffix is not None:
        restored_files = [f for f in restored_files if os.path.splitext(f)[0].endswith(suffix)]
    
    print(f"在original文件夹中找到的文件({len(original_files)}个): {original_files[:5]}...")
    print(f"在restored文件夹中找到的文件({len(restored_files)}个): {restored_files[:5]}...")
    
    # 构建restored文件名到文件的映射 (stem -> filename)
    restored_stem_map = {}
    for rest_file in restored_files:
        stem = os.path.splitext(rest_file)[0]  # e.g. "061836.jpg_hr" from "061836.jpg_hr.png"
        # 如果指定了suffix，去掉suffix后再存入映射，便于与原始文件名匹配
        if suffix and stem.endswith(suffix):
            stem = stem[:-len(suffix)]  # e.g. "061836.jpg_hr" -> "061836.jpg"
        restored_stem_map[stem] = rest_file
        # 也存储去掉所有扩展名后的纯名字 e.g. "061836" from "061836.jpg.png"
        pure_stem = stem.split('.')[0] if '.' in stem else stem
        if pure_stem not in restored_stem_map:
            restored_stem_map[pure_stem] = rest_file
    
    pairs = []
    for orig_file in original_files:
        # 1. 直接完全匹配
        if orig_file in restored_files:
            pairs.append((orig_file, orig_file))
            continue
        
        # 2. 按stem匹配（去掉最后一个扩展名）
        orig_stem = os.path.splitext(orig_file)[0]  # e.g. "061836.jpg" from "061836.jpg.jpg"
        if orig_stem in restored_stem_map:
            pairs.append((orig_file, restored_stem_map[orig_stem]))
            continue
        
        # 3. 按纯数字/纯名字匹配（去掉所有扩展名）
        orig_pure = orig_stem.split('.')[0] if '.' in orig_stem else orig_stem
        if orig_pure in restored_stem_map:
            pairs.append((orig_file, restored_stem_map[orig_pure]))
            continue
    
    print(f"找到的匹配对: {len(pairs)} 对")
    if len(pairs) > 0:
        print(f"示例匹配: {pairs[0]}")
    return pairs

def evaluate_images(original_img, restored_img):
    """计算评估指标（MSE/PSNR/SSIM/MS-SSIM/GMSD/LPIPS）"""
    if isinstance(original_img, torch.Tensor):
        original_np = original_img.permute(1, 2, 0).numpy()
        restored_np = restored_img.permute(1, 2, 0).numpy()
        orig_tensor = original_img
        rest_tensor = restored_img
    else:
        original_np = original_img
        restored_np = restored_img
        orig_tensor = torch.from_numpy(original_img).permute(2, 0, 1)
        rest_tensor = torch.from_numpy(restored_img).permute(2, 0, 1)

    original_np = np.clip(original_np, 0, 1)
    restored_np = np.clip(restored_np, 0, 1)

    mse = calculate_mse(original_np, restored_np)
    psnr = calculate_psnr(original_np, restored_np, data_range=1.0)

    try:
        ssim = calculate_ssim(original_np, restored_np, channel_axis=2, data_range=1.0, win_size=7)
    except ValueError:
        try:
            ssim = calculate_ssim(original_np, restored_np, channel_axis=2, data_range=1.0, win_size=3)
        except ValueError:
            ssim = 0.0

    ms_ssim = calculate_ms_ssim(original_np, restored_np)
    gmsd = calculate_gmsd(original_np, restored_np)
    lpips_score = calculate_lpips_score(orig_tensor, rest_tensor)

    result = {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'ms_ssim': ms_ssim, 'gmsd': gmsd}
    if lpips_score is not None:
        result['lpips'] = lpips_score
    return result

def save_comparison_image(original_img, restored_img, output_path):
    """保存原始图像和修复图像的对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 将tensor转换为numpy进行显示
    if isinstance(original_img, torch.Tensor):
        original_np = original_img.permute(1, 2, 0).numpy()
        restored_np = restored_img.permute(1, 2, 0).numpy()
    else:
        original_np = original_img
        restored_np = restored_img
    
    ax1.imshow(original_np)
    ax1.set_title('原始图像')
    ax1.axis('off')
    
    ax2.imshow(restored_np)
    ax2.set_title('修复图像')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_single_files(original_path, restored_path, output_dir, image_size=160, save_comparison=True):
    """比较单个文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.isfile(original_path):
        print(f"错误: 原始图像文件不存在: {original_path}")
        return
    
    if not os.path.isfile(restored_path):
        print(f"错误: 修复图像文件不存在: {restored_path}")
        return
    
    print(f"正在比较:\n原始图像: {original_path}\n修复图像: {restored_path}")
    
    # 加载图像
    orig_img = load_image(original_path, image_size)
    rest_img = load_image(restored_path, image_size)
    
    if orig_img is None or rest_img is None:
        print("无法加载图像，请检查图像格式")
        return
    
    # 评估质量
    metrics = evaluate_images(orig_img, rest_img)
    
    # 保存对比图
    if save_comparison:
        orig_filename = os.path.splitext(os.path.basename(original_path))[0]
        rest_filename = os.path.splitext(os.path.basename(restored_path))[0]
        comp_path = os.path.join(output_dir, f"comparison_{orig_filename}_vs_{rest_filename}.png")
        save_comparison_image(orig_img, rest_img, comp_path)
        print(f"对比图已保存到: {comp_path}")
    
    results_path = os.path.join(output_dir, f"evaluation_results_{orig_filename}_vs_{rest_filename}.txt")
    with open(results_path, 'w') as f:
        f.write("图像修复质量评估结果\n")
        f.write("====================\n\n")
        f.write(f"原始图像: {original_path}\n")
        f.write(f"修复图像: {restored_path}\n\n")
        f.write(f"MSE:     {metrics['mse']:.6f}\n")
        f.write(f"PSNR:    {metrics['psnr']:.4f} dB\n")
        f.write(f"SSIM:    {metrics['ssim']:.4f}\n")
        f.write(f"MS-SSIM: {metrics['ms_ssim']:.4f}\n")
        f.write(f"GMSD:    {metrics['gmsd']:.4f} (越小越好)\n")
        if 'lpips' in metrics:
            f.write(f"LPIPS:   {metrics['lpips']:.4f} (越小越好)\n")

    print("\n修复质量评估结果:")
    print(f"  MSE:     {metrics['mse']:.6f}")
    print(f"  PSNR:    {metrics['psnr']:.4f} dB")
    print(f"  SSIM:    {metrics['ssim']:.4f}")
    print(f"  MS-SSIM: {metrics['ms_ssim']:.4f}")
    print(f"  GMSD:    {metrics['gmsd']:.4f}  (越小越好)")
    if 'lpips' in metrics:
        print(f"  LPIPS:   {metrics['lpips']:.4f}  (越小越好)")
    print(f"详细结果已保存到: {results_path}")

    metric_names = ['PSNR (dB)', 'SSIM', 'MS-SSIM']
    metric_vals  = [metrics['psnr'], metrics['ssim'], metrics['ms_ssim']]
    colors = ['green', 'blue', 'purple']
    plt.figure(figsize=(10, 5))
    plt.bar(metric_names, metric_vals, color=colors)
    plt.title('图像修复质量指标')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"metrics_{orig_filename}_vs_{rest_filename}.png"))
    print(f"指标图表已保存到: {os.path.join(output_dir, f'metrics_{orig_filename}_vs_{rest_filename}.png')}")

def main():
    args = parse_args()
    
    # 检查是否为单文件模式
    if args.single_mode or (os.path.isfile(args.original) and os.path.isfile(args.restored)):
        compare_single_files(args.original, args.restored, args.output, args.image_size, args.save_comparison)
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 查找匹配的图像对
    pairs = find_matching_pairs(args.original, args.restored, suffix=args.suffix)
    
    if not pairs:
        print("未找到匹配的图像对，请确认文件名格式")
        return
    
    print(f"找到 {len(pairs)} 对匹配的图像")
    
    # 评估结果
    results = []
    
    for orig_file, rest_file in pairs:
        print(f"处理: {orig_file} - {rest_file}")
        
        orig_path = os.path.join(args.original, orig_file)
        rest_path = os.path.join(args.restored, rest_file)
        
        # 加载图像
        orig_img = load_image(orig_path, args.image_size)
        rest_img = load_image(rest_path, args.image_size)
        
        if orig_img is None or rest_img is None:
            print(f"跳过{orig_file}-{rest_file}对")
            continue
        
        # 评估质量
        metrics = evaluate_images(orig_img, rest_img)
        results.append({
            'original': orig_file,
            'restored': rest_file,
            'metrics': metrics
        })
        
        # 保存对比图
        if args.save_comparison:
            comp_path = os.path.join(args.output, f"comparison_{os.path.splitext(orig_file)[0]}.png")
            save_comparison_image(orig_img, rest_img, comp_path)
    
    avg_mse     = np.mean([r['metrics']['mse'] for r in results])
    avg_psnr    = np.mean([r['metrics']['psnr'] for r in results])
    avg_ssim    = np.mean([r['metrics']['ssim'] for r in results])
    avg_ms_ssim = np.mean([r['metrics']['ms_ssim'] for r in results])
    avg_gmsd    = np.mean([r['metrics']['gmsd'] for r in results])
    has_lpips   = 'lpips' in results[0]['metrics'] if results else False
    avg_lpips   = np.mean([r['metrics']['lpips'] for r in results]) if has_lpips else None

    with open(os.path.join(args.output, 'evaluation_results.txt'), 'w') as f:
        f.write("图像修复质量评估结果\n")
        f.write("====================\n\n")
        f.write(f"评估图像对数量: {len(results)}\n\n")
        f.write(f"平均 MSE:     {avg_mse:.6f}\n")
        f.write(f"平均 PSNR:    {avg_psnr:.4f} dB\n")
        f.write(f"平均 SSIM:    {avg_ssim:.4f}\n")
        f.write(f"平均 MS-SSIM: {avg_ms_ssim:.4f}\n")
        f.write(f"平均 GMSD:    {avg_gmsd:.4f} (越小越好)\n")
        if avg_lpips is not None:
            f.write(f"平均 LPIPS:   {avg_lpips:.4f} (越小越好)\n")
        f.write("\n详细结果:\n")
        f.write("====================\n")
        for r in results:
            f.write(f"原始图像: {r['original']}\n")
            f.write(f"修复图像: {r['restored']}\n")
            f.write(f"  MSE={r['metrics']['mse']:.6f}  PSNR={r['metrics']['psnr']:.4f}dB  "
                    f"SSIM={r['metrics']['ssim']:.4f}  MS-SSIM={r['metrics']['ms_ssim']:.4f}  "
                    f"GMSD={r['metrics']['gmsd']:.4f}")
            if 'lpips' in r['metrics']:
                f.write(f"  LPIPS={r['metrics']['lpips']:.4f}")
            f.write("\n--------------------\n")

    print("\n修复质量评估完成!")
    print(f"  平均 MSE:     {avg_mse:.6f}")
    print(f"  平均 PSNR:    {avg_psnr:.4f} dB")
    print(f"  平均 SSIM:    {avg_ssim:.4f}")
    print(f"  平均 MS-SSIM: {avg_ms_ssim:.4f}")
    print(f"  平均 GMSD:    {avg_gmsd:.4f}  (越小越好)")
    if avg_lpips is not None:
        print(f"  平均 LPIPS:   {avg_lpips:.4f}  (越小越好)")
    print(f"详细结果已保存到 {os.path.join(args.output, 'evaluation_results.txt')}")

    # 平均指标条形图
    bar_names = ['PSNR (dB)', 'SSIM', 'MS-SSIM']
    bar_vals  = [avg_psnr, avg_ssim, avg_ms_ssim]
    bar_colors = ['green', 'blue', 'purple']
    plt.figure(figsize=(10, 5))
    plt.bar(bar_names, bar_vals, color=bar_colors)
    plt.title('平均修复质量指标')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.output, 'average_metrics.png'))
    plt.close()

    if len(results) > 1:
        plt.figure(figsize=(18, 4))
        for idx, (key, label, color) in enumerate([
            ('psnr', 'PSNR (dB)', 'green'),
            ('ssim', 'SSIM', 'blue'),
            ('ms_ssim', 'MS-SSIM', 'purple'),
            ('gmsd', 'GMSD (越小越好)', 'orange'),
        ]):
            plt.subplot(1, 4, idx + 1)
            plt.hist([r['metrics'][key] for r in results], bins=10, color=color, alpha=0.7)
            plt.title(f'{label} 分布')
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'metrics_distribution.png'))
        plt.close()

if __name__ == "__main__":
    main()
