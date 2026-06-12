/**
 * 人脸修复系统 - 前端逻辑
 */

// ===== DOM 元素 =====
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const submitBtn = document.getElementById('submitBtn');
const srEnable = document.getElementById('srEnable');
const srOptions = document.getElementById('srOptions');
const srMethod = document.getElementById('srMethod');
const srScale = document.getElementById('srScale');
const resultSection = document.getElementById('resultSection');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const loadingText = document.getElementById('loadingText');
const resultsList = document.getElementById('resultsList');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');

let selectedFiles = [];

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setupUploadZone();
    setupSettings();
    setupModal();
});

// ===== 服务状态检查 =====
async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.status === 'running' && data.model_loaded) {
            statusText.textContent = `运行中 (${data.device})`;
            statusBadge.classList.remove('offline');
        } else {
            statusText.textContent = '模型未加载';
            statusBadge.classList.add('offline');
        }
    } catch {
        statusText.textContent = '服务离线';
        statusBadge.classList.add('offline');
    }
}

// ===== 上传区域 =====
function setupUploadZone() {
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        addFiles(files);
    });

    fileInput.addEventListener('change', () => {
        addFiles(Array.from(fileInput.files));
        fileInput.value = '';
    });

    submitBtn.addEventListener('click', handleSubmit);
}

function addFiles(files) {
    for (const file of files) {
        if (!selectedFiles.find(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
        }
    }
    renderFileList();
    submitBtn.disabled = selectedFiles.length === 0;
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
    submitBtn.disabled = selectedFiles.length === 0;
}

function renderFileList() {
    fileList.innerHTML = '';
    selectedFiles.forEach((file, i) => {
        const url = URL.createObjectURL(file);
        const size = file.size < 1024 * 1024
            ? (file.size / 1024).toFixed(1) + ' KB'
            : (file.size / (1024 * 1024)).toFixed(1) + ' MB';
        const el = document.createElement('div');
        el.className = 'file-item';
        el.innerHTML = `
            <img class="file-thumb" src="${url}" alt="">
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${size}</div>
            </div>
            <button class="file-remove" onclick="removeFile(${i})" title="移除">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;
        fileList.appendChild(el);
    });
}

// ===== 设置 =====
function setupSettings() {
    srEnable.addEventListener('change', () => {
        srOptions.style.display = srEnable.checked ? 'block' : 'none';
    });
}

// ===== 提交处理 =====
async function handleSubmit() {
    if (selectedFiles.length === 0) return;

    // 显示加载状态
    emptyState.style.display = 'none';
    loadingState.style.display = 'block';
    resultsList.style.display = 'none';
    submitBtn.disabled = true;
    submitBtn.classList.add('loading');

    const useSR = srEnable.checked;

    try {
        if (selectedFiles.length === 1) {
            // 单张图片
            loadingText.textContent = '正在修复图片...';
            const formData = new FormData();
            formData.append('file', selectedFiles[0]);
            formData.append('sr_enable', useSR);
            if (useSR) {
                formData.append('sr_method', srMethod.value);
                formData.append('sr_scale', srScale.value);
            }

            const res = await window.fetch('/api/restore', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            showResults([data]);
        } else {
            // 批量处理
            loadingText.textContent = `正在修复 ${selectedFiles.length} 张图片...`;
            const formData = new FormData();
            selectedFiles.forEach(f => formData.append('files', f));
            formData.append('sr_enable', useSR);
            if (useSR) {
                formData.append('sr_method', srMethod.value);
                formData.append('sr_scale', srScale.value);
            }

            const res = await window.fetch('/api/restore_batch', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            showResults(data.results);
        }
    } catch (err) {
        alert('处理失败: ' + err.message);
        emptyState.style.display = 'block';
    } finally {
        loadingState.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
    }
}

// ===== 显示结果 =====
function showResults(results) {
    loadingState.style.display = 'none';
    resultsList.style.display = 'flex';
    resultsList.innerHTML = '';

    // 批量汇总
    if (results.length > 1) {
        const validResults = results.filter(r => !r.error && r.metrics);
        if (validResults.length > 0) {
            const avgPsnr = (validResults.reduce((s, r) => s + r.metrics.psnr, 0) / validResults.length).toFixed(2);
            const avgSsim = (validResults.reduce((s, r) => s + r.metrics.ssim, 0) / validResults.length).toFixed(4);
            const avgMse = (validResults.reduce((s, r) => s + r.metrics.mse, 0) / validResults.length).toFixed(6);
            const totalTime = validResults.reduce((s, r) => s + (r.time || 0), 0).toFixed(2);

            const summary = document.createElement('div');
            summary.className = 'batch-summary';
            summary.innerHTML = `
                <h3>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    批量处理汇总 (${validResults.length} 张)
                </h3>
                <div class="summary-stats">
                    <div class="summary-stat">
                        <div class="stat-label">平均 PSNR</div>
                        <div class="stat-value text-success">${avgPsnr} dB</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-label">平均 SSIM</div>
                        <div class="stat-value text-primary">${avgSsim}</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-label">平均 MSE</div>
                        <div class="stat-value text-danger">${avgMse}</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-label">总耗时</div>
                        <div class="stat-value">${totalTime}s</div>
                    </div>
                </div>
            `;
            resultsList.appendChild(summary);
        }
    }

    // 逐个结果卡片
    results.forEach(result => {
        if (result.error) {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <div class="result-header">
                    <span class="result-filename">${result.filename || '未知文件'}</span>
                    <span style="color:var(--danger)">处理失败: ${result.error}</span>
                </div>
            `;
            resultsList.appendChild(card);
            return;
        }
        resultsList.appendChild(createResultCard(result));
    });
}

function createResultCard(result) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const hasHR = !!result.restored_hr;
    const imgCols = hasHR ? 'has-hr' : '';

    let imagesHTML = `
        <div class="image-cell">
            <div class="image-cell-label">原图</div>
            <img src="data:image/png;base64,${result.original}" onclick="showModal(this)" alt="原图">
        </div>
        <div class="image-cell">
            <div class="image-cell-label">遮挡</div>
            <img src="data:image/png;base64,${result.masked}" onclick="showModal(this)" alt="遮挡图">
        </div>
        <div class="image-cell">
            <div class="image-cell-label">修复结果</div>
            <img src="data:image/png;base64,${result.restored}" onclick="showModal(this)" alt="修复结果">
        </div>
    `;
    if (hasHR) {
        imagesHTML += `
        <div class="image-cell">
            <div class="image-cell-label">超分辨率</div>
            <img src="data:image/png;base64,${result.restored_hr}" onclick="showModal(this)" alt="超分辨率">
        </div>
        `;
    }

    // 指标
    const m = result.metrics;
    let metricsHTML = `
        <div class="metrics-title">修复质量指标</div>
        <div class="metrics-grid">
            <div class="metric-card mse">
                <div class="metric-name">MSE</div>
                <div class="metric-value">${m.mse.toFixed(4)}</div>
                <div class="metric-hint">越低越好</div>
            </div>
            <div class="metric-card psnr">
                <div class="metric-name">PSNR</div>
                <div class="metric-value">${m.psnr.toFixed(2)} <small>dB</small></div>
                <div class="metric-hint">越高越好</div>
            </div>
            <div class="metric-card ssim">
                <div class="metric-name">SSIM</div>
                <div class="metric-value">${m.ssim.toFixed(4)}</div>
                <div class="metric-hint">越高越好 (满分1.0)</div>
            </div>
        </div>
    `;

    if (hasHR && result.metrics_hr) {
        const mh = result.metrics_hr;
        metricsHTML += `
        <div class="metrics-hr">
            <div class="metrics-title">超分辨率指标</div>
            <div class="metrics-grid">
                <div class="metric-card mse">
                    <div class="metric-name">MSE</div>
                    <div class="metric-value">${mh.mse.toFixed(4)}</div>
                </div>
                <div class="metric-card psnr">
                    <div class="metric-name">PSNR</div>
                    <div class="metric-value">${mh.psnr.toFixed(2)} <small>dB</small></div>
                </div>
                <div class="metric-card ssim">
                    <div class="metric-name">SSIM</div>
                    <div class="metric-value">${mh.ssim.toFixed(4)}</div>
                </div>
            </div>
        </div>
        `;
    }

    // 下载按钮
    let downloadBtns = `
        <button class="btn-download" onclick="downloadImage('${result.restored}', '${result.filename}_restored.png')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            下载修复图
        </button>
    `;
    if (hasHR) {
        downloadBtns += `
        <button class="btn-download" onclick="downloadImage('${result.restored_hr}', '${result.filename}_restored_hr.png')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            下载超分辨率图
        </button>
        `;
    }

    card.innerHTML = `
        <div class="result-header">
            <span class="result-filename">${result.filename || '上传图片'}</span>
            <span class="result-time">${result.time ? result.time + 's' : ''}</span>
        </div>
        <div class="result-images ${imgCols}">${imagesHTML}</div>
        <div class="metrics-panel">${metricsHTML}</div>
        <div class="result-actions">${downloadBtns}</div>
    `;

    return card;
}

// ===== 图片放大 =====
let modalOverlay;

function setupModal() {
    modalOverlay = document.createElement('div');
    modalOverlay.className = 'modal-overlay';
    modalOverlay.innerHTML = '<img src="" alt="放大预览">';
    modalOverlay.addEventListener('click', () => modalOverlay.classList.remove('active'));
    document.body.appendChild(modalOverlay);
}

function showModal(imgEl) {
    modalOverlay.querySelector('img').src = imgEl.src;
    modalOverlay.classList.add('active');
}

// ===== 下载图片 =====
function downloadImage(base64Data, filename) {
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + base64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
