"""
人脸修复模型 Web 应用
Flask后端：加载模型、处理图片、返回修复结果和评估指标
"""
import sys
import os

# 将项目根目录加入路径，以便导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import io
import uuid
import json
import time
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from torchvision import transforms
from torchvision.utils import save_image as tv_save_image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import mean_squared_error as calculate_mse

from models import Generator, GeneratorV2
from utils import gen_input_MaskLayer, gen_Mask, poisson_blend
from super_resolution import SuperResolutionEnhancer

# ========== 配置 ==========
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB上传限制

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# ========== 全局模型变量 ==========
device = None
generator = None
mpv = None
sr_enhancers = {}
INPUT_SIZE = 160


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path, config_path, model_version='auto'):
    """启动时加载模型"""
    global device, generator, mpv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[启动] 使用设备: {device}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1).to(device)

    state_dict = torch.load(model_path, map_location='cpu')
    if model_version == 'auto':
        try:
            generator = GeneratorV2()
            generator.load_state_dict(state_dict)
            model_version = 'v2'
        except RuntimeError:
            generator = Generator()
            generator.load_state_dict(state_dict)
            model_version = 'v1'
    elif model_version == 'v2':
        generator = GeneratorV2()
        generator.load_state_dict(state_dict)
    else:
        generator = Generator()
        generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()
    print(f"[启动] 模型加载完成: {model_path} (版本: {model_version})")


def get_sr_enhancer(method='bicubic', scale=2):
    """懒加载超分辨率增强器"""
    key = f"{method}_{scale}"
    if key not in sr_enhancers:
        sr_enhancers[key] = SuperResolutionEnhancer(
            method=method, scale_factor=scale, device=device,
            sharpen=True, sharpen_strength=0.5
        )
        print(f"[SR] 初始化超分辨率: {method}, scale={scale}")
    return sr_enhancers[key]


def tensor_to_base64(tensor_img):
    """将 tensor (1,3,H,W) 或 (3,H,W) 转为 base64 PNG"""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.clamp(0, 1).cpu()
    img = transforms.functional.to_pil_image(tensor_img)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def compute_metrics(original_tensor, restored_tensor):
    """计算评估指标（输入为 3,H,W tensor）"""
    orig_np = original_tensor.cpu().numpy().transpose(1, 2, 0)
    rest_np = restored_tensor.cpu().numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    rest_np = np.clip(rest_np, 0, 1)

    mse = float(calculate_mse(orig_np, rest_np))
    psnr = float(calculate_psnr(orig_np, rest_np, data_range=1.0))

    min_dim = min(orig_np.shape[0], orig_np.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    ssim = float(calculate_ssim(orig_np, rest_np, data_range=1.0,
                                channel_axis=2, win_size=win_size))
    return {'mse': round(mse, 6), 'psnr': round(psnr, 4), 'ssim': round(ssim, 6)}


def restore_image(pil_image, sr_enable=False, sr_method='bicubic', sr_scale=2):
    """
    核心修复流程：接收 PIL Image，返回结果字典
    """
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor()
    ])

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    x = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_size, mask_location = gen_Mask(x)
        masklayer = gen_input_MaskLayer(
            MaskLayer_shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
            mask_size=mask_size, mask_area=mask_location
        ).to(device)

        x_mask = x - x * masklayer + mpv * masklayer
        input_G = torch.cat((x_mask, masklayer), dim=1)
        output_G = generator(input_G)
        completed = poisson_blend(x_mask, output_G, masklayer).to(device)

    # 评估指标（原图 vs 修复图）
    metrics = compute_metrics(x.squeeze(0), completed.squeeze(0))

    result = {
        'original': tensor_to_base64(x),
        'masked': tensor_to_base64(x_mask),
        'restored': tensor_to_base64(completed),
        'metrics': metrics,
    }

    # 超分辨率
    if sr_enable:
        sr = get_sr_enhancer(sr_method, sr_scale)
        with torch.no_grad():
            completed_hr = sr.enhance(completed)
            x_hr = sr.enhance(x)
        metrics_hr = compute_metrics(x_hr.squeeze(0), completed_hr.squeeze(0))
        result['restored_hr'] = tensor_to_base64(completed_hr)
        result['original_hr'] = tensor_to_base64(x_hr)
        result['metrics_hr'] = metrics_hr

    return result


# ========== 路由 ==========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/restore', methods=['POST'])
def api_restore():
    """处理单张图片修复请求"""
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件格式，支持: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    sr_enable = request.form.get('sr_enable', 'false').lower() == 'true'
    sr_method = request.form.get('sr_method', 'bicubic')
    sr_scale = int(request.form.get('sr_scale', '2'))

    try:
        start_time = time.time()
        pil_image = Image.open(file.stream)
        result = restore_image(pil_image, sr_enable=sr_enable,
                               sr_method=sr_method, sr_scale=sr_scale)
        result['time'] = round(time.time() - start_time, 3)
        result['filename'] = file.filename
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/api/restore_batch', methods=['POST'])
def api_restore_batch():
    """处理批量图片修复请求"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '未上传文件'}), 400

    sr_enable = request.form.get('sr_enable', 'false').lower() == 'true'
    sr_method = request.form.get('sr_method', 'bicubic')
    sr_scale = int(request.form.get('sr_scale', '2'))

    results = []
    for file in files:
        if not allowed_file(file.filename):
            continue
        try:
            start_time = time.time()
            pil_image = Image.open(file.stream)
            result = restore_image(pil_image, sr_enable=sr_enable,
                                   sr_method=sr_method, sr_scale=sr_scale)
            result['time'] = round(time.time() - start_time, 3)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})

    return jsonify({'results': results, 'total': len(results)})


@app.route('/api/status', methods=['GET'])
def api_status():
    """返回服务状态"""
    return jsonify({
        'status': 'running',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'model_loaded': generator is not None,
    })


# ========== 启动 ==========
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='人脸修复 Web 服务')
    parser.add_argument('--model', type=str, required=True, help='生成器模型路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径 (含mpv)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    load_model(args.model, args.config)
    print(f"[启动] 服务运行在 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
