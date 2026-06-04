import torch
import os
import json
from models import Generator
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import functional as F
from utils import (
    MyDataset,
    gen_input_MaskLayer,
    gen_Mask,
    poisson_blend,
)
from super_resolution import SuperResolutionEnhancer
from predict import find_latest_model, find_latest_config, resolve_model_and_config
import tkinter as tk
from tkinter import filedialog, ttk



###
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------测试--------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
###


def main(input_generator_size=160, config=None, model=None, data='./datasets/test/',
         result_dir='./results/', mode=None, sr_enable=False, sr_method='bicubic', sr_scale=2):
    model, config = resolve_model_and_config(model, config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    mpv = mpv.to(device)
    G = Generator()
    G.load_state_dict(torch.load(model, map_location='cpu'))
    G = G.to(device)
    G.eval()

    # 初始化超分辨率增强器
    sr_enhancer = None
    if sr_enable:
        sr_enhancer = SuperResolutionEnhancer(method=sr_method, scale_factor=sr_scale, device=device)

    transform = transforms.Compose([
        transforms.Resize(input_generator_size),
        transforms.CenterCrop((input_generator_size, input_generator_size)),
        transforms.ToTensor()
    ])
    test_dataset = MyDataset(data, transform)
    with torch.no_grad():
        if mode == 'cat':
            num_samples = len(test_dataset)
            batch = []
            for i in range(num_samples):
                x = torch.unsqueeze(test_dataset[i], dim=0)
                batch.append(x)
            x = torch.cat(batch, dim=0).to(device)
            mask_size, mask_location = gen_Mask(x)
            masklayer = gen_input_MaskLayer(MaskLayer_shape=(x.shape[0], 1, x.shape[2],
                                                                x.shape[3]),
                                            mask_size=mask_size,
                                            mask_area=mask_location)
            masklayer = masklayer.to(device)
            x_mask = x - x * masklayer + mpv * masklayer
            input_G = torch.cat((x_mask, masklayer), dim=1)
            output_G = G(input_G)
            completed = poisson_blend(x_mask, output_G, masklayer).to(device)
            
            # 应用超分辨率增强（如果启用）
            if sr_enhancer is not None:
                x_hr = sr_enhancer.enhance(x)
                x_mask_hr = sr_enhancer.enhance(x_mask)
                completed_hr = sr_enhancer.enhance(completed)
                imgs = torch.cat((x_hr, x_mask_hr, completed_hr), dim=2)
                imgpath = os.path.join(result_dir, 'predict_hr.png')
                save_image(imgs, imgpath, nrow=0)
                print(f"高分辨率图像已保存至 {imgpath}")
            else:
                imgs = torch.cat((x, x_mask, completed), dim=2)
                imgpath = os.path.join(result_dir, 'predict.png')
                save_image(imgs, imgpath, nrow=0)
                print(imgpath)
        else:
            number = 0
            for x in test_dataset:
                x = torch.unsqueeze(x, dim=0).to(device)
                mask_size, mask_location = gen_Mask(x)
                masklayer = gen_input_MaskLayer(MaskLayer_shape=(x.shape[0], 1, x.shape[2],
                                                                x.shape[3]),
                                                mask_size=mask_size,
                                                mask_area=mask_location)
                masklayer = masklayer.to(device)
                x_mask = x - x * masklayer + mpv * masklayer
                input_G = torch.cat((x_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(x_mask, output_G, masklayer).to(device)
                
                # 应用超分辨率增强（如果启用）
                if sr_enhancer is not None:
                    x_hr = sr_enhancer.enhance(x)
                    x_mask_hr = sr_enhancer.enhance(x_mask)
                    completed_hr = sr_enhancer.enhance(completed)
                    imgs = torch.cat((x_hr, x_mask_hr, completed_hr), dim=2)
                    imgpath = os.path.join(result_dir, f'{number}_hr.png')
                else:
                    imgs = torch.cat((x, x_mask, completed), dim=2)
                    imgpath = os.path.join(result_dir, f'{number}.png')
                
                save_image(imgs, imgpath, nrow=3)
                number += 1
            
            if sr_enhancer is not None:
                print(f'高分辨率图像已保存在 {result_dir}')
            else:
                print(f'图像已保存在 {result_dir}')

if __name__ == '__main__':
    root = tk.Tk()
    root.title("人脸修复与超分辨率增强系统")
    root.geometry("500x450")
    
    # 创建框架来组织组件
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill="both", expand=True)
    
    # 配置和模型路径
    path_frame = ttk.LabelFrame(main_frame, text="路径配置", padding="5")
    path_frame.pack(fill="x", padx=5, pady=5)
    
    # 自动检测最新训练的配置和模型
    detected_config = find_latest_config() or './demo/config.json'
    detected_model = find_latest_model() or './demo/model_cn'
    
    config_path = tk.StringVar()
    config_path.set(detected_config)
    ttk.Label(path_frame, text="config路径:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(path_frame, textvariable=config_path, width=40).grid(row=0, column=1, sticky="ew", pady=2)
    ttk.Button(path_frame, text="浏览", 
               command=lambda: config_path.set(filedialog.askopenfilename(filetypes=[("JSON files", "*.json")]))
              ).grid(row=0, column=2, padx=5, pady=2)
    
    model_path = tk.StringVar()
    model_path.set(detected_model)
    ttk.Label(path_frame, text="model路径:").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(path_frame, textvariable=model_path, width=40).grid(row=1, column=1, sticky="ew", pady=2)
    ttk.Button(path_frame, text="浏览", 
               command=lambda: model_path.set(filedialog.askopenfilename())
              ).grid(row=1, column=2, padx=5, pady=2)
    
    data_path = tk.StringVar()
    data_path.set('./datasets/test/')
    ttk.Label(path_frame, text="数据路径:").grid(row=2, column=0, sticky="w", pady=2)
    ttk.Entry(path_frame, textvariable=data_path, width=40).grid(row=2, column=1, sticky="ew", pady=2)
    ttk.Button(path_frame, text="浏览", 
               command=lambda: data_path.set(filedialog.askdirectory())
              ).grid(row=2, column=2, padx=5, pady=2)
    
    result_dir = tk.StringVar()
    result_dir.set('./results/')
    ttk.Label(path_frame, text="结果路径:").grid(row=3, column=0, sticky="w", pady=2)
    ttk.Entry(path_frame, textvariable=result_dir, width=40).grid(row=3, column=1, sticky="ew", pady=2)
    ttk.Button(path_frame, text="浏览", 
               command=lambda: result_dir.set(filedialog.askdirectory())
              ).grid(row=3, column=2, padx=5, pady=2)
    
    # 超分辨率设置
    sr_frame = ttk.LabelFrame(main_frame, text="超分辨率设置", padding="5")
    sr_frame.pack(fill="x", padx=5, pady=5)
    
    sr_enable = tk.BooleanVar()
    sr_enable.set(False)
    ttk.Checkbutton(sr_frame, text="启用超分辨率增强", variable=sr_enable).grid(row=0, column=0, sticky="w", pady=2, columnspan=2)
    
    ttk.Label(sr_frame, text="超分辨率方法:").grid(row=1, column=0, sticky="w", pady=2)
    sr_method = tk.StringVar()
    sr_method.set("bicubic")
    sr_methods = ttk.Combobox(sr_frame, textvariable=sr_method, values=["bicubic", "srcnn", "espcn"], state="readonly", width=10)
    sr_methods.grid(row=1, column=1, sticky="w", pady=2)
    
    ttk.Label(sr_frame, text="放大倍数:").grid(row=2, column=0, sticky="w", pady=2)
    sr_scale = tk.IntVar()
    sr_scale.set(2)
    sr_scales = ttk.Combobox(sr_frame, textvariable=sr_scale, values=[2, 3, 4], state="readonly", width=10)
    sr_scales.grid(row=2, column=1, sticky="w", pady=2)
    
    # 操作按钮
    btn_frame = ttk.Frame(main_frame, padding="5")
    btn_frame.pack(fill="x", padx=5, pady=10)
    
    process_mode = tk.StringVar()
    process_mode.set(None)
    
    ttk.Button(btn_frame, text="单张处理", 
               command=lambda: main(
                   config=config_path.get(), 
                   model=model_path.get(), 
                   data=data_path.get(),
                   result_dir=result_dir.get(),
                   sr_enable=sr_enable.get(),
                   sr_method=sr_method.get(),
                   sr_scale=sr_scale.get()
               )
              ).pack(side="left", padx=5)
    
    ttk.Button(btn_frame, text="批量处理", 
               command=lambda: main(
                   config=config_path.get(), 
                   model=model_path.get(), 
                   data=data_path.get(),
                   result_dir=result_dir.get(),
                   mode="cat",
                   sr_enable=sr_enable.get(),
                   sr_method=sr_method.get(),
                   sr_scale=sr_scale.get()
               )
              ).pack(side="left", padx=5)
    
    # 状态信息框
    status_frame = ttk.LabelFrame(main_frame, text="状态信息", padding="5")
    status_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    status_text = tk.Text(status_frame, height=5, wrap="word", state="disabled")
    status_text.pack(fill="both", expand=True)
    
    # 重定向打印输出到状态框
    import sys
    class TextRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget
            
        def write(self, string):
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", string)
            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")
            
        def flush(self):
            pass
    
    sys.stdout = TextRedirector(status_text)
    
    root.mainloop()