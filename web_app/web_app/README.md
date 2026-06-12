# 人脸修复 Web 应用

基于 Flask 的人脸修复原型系统，提供网页端图片上传、修复、评估功能。

## 功能

- **图片上传**：支持拖拽或点击上传，支持 JPG/PNG/BMP/WebP 格式
- **实时修复**：上传后自动进行人脸修复处理
- **质量评估**：自动计算 MSE、PSNR、SSIM 三项指标
- **超分辨率**：可选启用超分辨率增强（Bicubic/SRCNN/ESPCN/OpenCV）
- **批量处理**：支持同时上传多张图片批量修复
- **结果下载**：一键下载修复后的图片
- **图片预览**：点击图片可放大查看

## 目录结构

```
web_app/
├── app.py                 # Flask 后端主程序
├── requirements.txt       # Python 依赖
├── README.md              # 本文件
├── templates/
│   └── index.html         # 前端页面
├── static/
│   ├── css/
│   │   └── style.css      # 样式
│   └── js/
│       └── main.js        # 前端逻辑
├── uploads/               # 上传文件临时目录（自动创建）
└── results/               # 处理结果目录（自动创建）
```

## 部署步骤

### 1. 安装依赖

```bash
# 进入 web_app 目录
cd /path/to/Face-study/web_app

# 安装依赖（如果已有项目环境可跳过）
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 基本启动（使用你训练好的模型）
python app.py \
    --model ../result/phase_3/phase_3_model_generator_epoch40 \
    --config ../result/config.json \
    --port 5000

# 指定监听地址和端口
python app.py \
    --model ../result/phase_3/phase_3_model_generator_epoch40 \
    --config ../result/config.json \
    --host 0.0.0.0 \
    --port 8080

# 使用 demo 预训练模型
python app.py \
    --model ../demo/model_cn \
    --config ../demo/config.json \
    --port 5000
```

### 3. 访问网页

启动后在浏览器中访问：

```
http://服务器IP:5000
```

如果是本地测试：

```
http://localhost:5000
```

### 4. 后台运行（生产环境）

```bash
# 使用 nohup 后台运行
nohup python app.py \
    --model ../result/phase_3/phase_3_model_generator_epoch40 \
    --config ../result/config.json \
    --host 0.0.0.0 \
    --port 5000 \
    > web_app.log 2>&1 &

# 查看日志
tail -f web_app.log

# 停止服务
kill $(ps aux | grep 'app.py' | grep -v grep | awk '{print $2}')
```

### 5. 使用 Gunicorn 部署（推荐生产环境）

```bash
pip install gunicorn

# 先设置环境变量，让 app.py 加载模型
# 在 app.py 同目录创建 wsgi.py：
cat > wsgi.py << 'EOF'
from app import app, load_model
load_model('../result/phase_3/phase_3_model_generator_epoch40', '../result/config.json')
EOF

# 启动 gunicorn（4个worker，绑定所有接口的5000端口）
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 wsgi:app
```

> **注意**：由于模型占用 GPU 内存，建议 worker 数量设为 1。

## API 接口

### POST /api/restore

单张图片修复。

**请求**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 要修复的图片文件 |
| sr_enable | string | 否 | 是否启用超分辨率，`true`/`false` |
| sr_method | string | 否 | 超分辨率方法：`bicubic`/`srcnn`/`espcn`/`opencv` |
| sr_scale | string | 否 | 放大倍数：`2`/`4` |

**响应**（JSON）：

```json
{
    "filename": "test.jpg",
    "original": "base64...",
    "masked": "base64...",
    "restored": "base64...",
    "metrics": {
        "mse": 0.003421,
        "psnr": 24.6581,
        "ssim": 0.891234
    },
    "time": 0.523,
    "restored_hr": "base64...",
    "metrics_hr": {
        "mse": 0.002891,
        "psnr": 25.3912,
        "ssim": 0.903456
    }
}
```

### POST /api/restore_batch

批量图片修复。

**请求**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File[] | 是 | 多个图片文件 |
| sr_enable | string | 否 | 同上 |
| sr_method | string | 否 | 同上 |
| sr_scale | string | 否 | 同上 |

### GET /api/status

返回服务运行状态。

## 使用流程

1. 打开网页
2. 拖拽或点击上传需要修复的人脸图片（支持多张）
3. （可选）开启超分辨率增强并选择方法
4. 点击"开始修复"
5. 等待处理完成，查看修复效果和质量指标
6. 点击图片可放大查看细节
7. 点击"下载修复图"保存结果

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 无（必填） | 训练好的生成器模型路径 |
| `--config` | 无（必填） | 配置文件路径（包含 mpv 值） |
| `--host` | `0.0.0.0` | 监听地址，`0.0.0.0` 允许外部访问 |
| `--port` | `5000` | 监听端口 |
| `--debug` | 关闭 | 开启 Flask 调试模式 |

## 注意事项

1. **GPU 内存**：模型加载约需 100MB GPU 内存，每次推理约需额外 50MB
2. **并发处理**：默认单进程，不支持高并发，适合原型演示
3. **文件大小**：上传限制 16MB
4. **模型路径**：`--model` 和 `--config` 使用相对于 `web_app/` 目录的路径，或使用绝对路径
5. **防火墙**：确保服务器的对应端口已开放
6. **HTTPS**：生产环境建议配合 Nginx 反向代理使用 HTTPS
