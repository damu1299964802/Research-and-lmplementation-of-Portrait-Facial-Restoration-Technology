"""
Linux 版本训练脚本
移除了 Windows 特有的 DLL 路径设置
"""

import torch
import os
from tqdm import tqdm
import json
import argparse
from models import Discriminator, Generator, GeneratorV2
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from loss_functions import ConfigurableLoss, get_all_loss_combinations
from PIL import Image
from insightface.app import FaceAnalysis
from utils import (
    MyDataset,
    gen_input_MaskLayer,
    gen_Mask,
    poisson_blend,
    crop,
    sample_random_batch,
    Discriminator_Network_Loss,
)

# insightface 模型路径 - Linux 版本
INSIGHTFACE_ROOT = os.environ.get('INSIGHTFACE_ROOT', './insightface_models')
app = FaceAnalysis(
    allowed_modules=['detection', 'landmark_2d_106'],
    root=INSIGHTFACE_ROOT
)
app.prepare(ctx_id=0, det_size=(256, 256))

parser = argparse.ArgumentParser()
parser.add_argument('--model_G', default='./demo/model_cn')
parser.add_argument('--model_D', default=None)
parser.add_argument('--phase1', type=int, default=8)
parser.add_argument('--phase2', type=int, default=2)
parser.add_argument('--phase3', type=int, default=40)
parser.add_argument('--loss_combination', default='all', 
                   choices=['reconstruction', 'perceptual', 'landmark', 
                           'recon_percep', 'recon_landmark', 'percep_landmark', 'all',
                           'v2_balanced', 'v2_detail', 'v2_ssim', 'v2_full'],
                   help='损失函数组合选择')
parser.add_argument('--data_dir', default='/local_data/Face-study/datasets/train', help='训练数据目录')
parser.add_argument('--test_dir', default='/local_data/Face-study/datasets/test', help='测试数据目录')
parser.add_argument('--result_dir', default='./result/', help='结果保存目录')
parser.add_argument('--model_version', default='v1', choices=['v1', 'v2'],
                   help='生成器版本: v1=原版, v2=带Skip Connection和SE注意力的改进版')


def train(input_generator_size=160, batch_size=16, alpha=4e-4,
          num_epochs_pregen=10, num_epochs_predis=10, num_epochs_step3=20,
          mpv=None, init_model_G=None, init_model_D=None, 
          loss_combination='all', data_dir='./datasets/train', 
          test_dir='./datasets/test', result_dir='./result/',
          model_version='v1'):
    """
    训练函数
    
    Args:
        loss_combination: 损失函数组合，可选:
            - reconstruction: 仅基础重建损失
            - perceptual: 仅感知损失
            - landmark: 仅标志点损失
            - recon_percep: 重建 + 感知
            - recon_landmark: 重建 + 标志点
            - percep_landmark: 感知 + 标志点
            - all: 全部组合 (默认)
    """
    print(f"使用损失函数组合: {loss_combination}")
    print(f"生成器版本: {model_version}")
    
    # 初始化可配置损失函数
    configurable_loss = ConfigurableLoss(combination=loss_combination)
    print(f"损失函数配置: {configurable_loss.get_description()}")
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(result_dir, phase)):
            os.makedirs(os.path.join(result_dir, phase))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    transform = transforms.Compose([
        transforms.Resize(input_generator_size),
        transforms.RandomCrop((input_generator_size, input_generator_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset')

    train_dataset = MyDataset(data_dir, transform)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = MyDataset(test_dir, transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    if mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(
            total=len(train_dataset.images),
            desc='computing mean pixel value of training dataset...')
        for imgpath in train_dataset.images:
            img = Image.open(imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(train_dataset.images)
        pbar.close()
    else:
        mpv = np.array(mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))
    args_dict = {"data_dir": data_dir, "test_dir": test_dir, "result_dir": result_dir,
                 "init_model_G": init_model_G, "init_model_D": init_model_D,
                 "num_epochs_pregen": num_epochs_pregen, "num_epochs_predis": num_epochs_predis,
                 "num_epochs_step3": num_epochs_step3, "input_generator_size": input_generator_size,
                 "batch_size": batch_size, "alpha": alpha, 'mpv': mpv_json,
                 "loss_combination": loss_combination}
    with open(os.path.join(result_dir, 'config.json'), mode='w') as f:
        json.dump(args_dict, f, indent=2)
    
    mpv = torch.tensor(mpv.reshape(1, 3, 1, 1), dtype=torch.float32).to(device)
    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

    # Training Phase 1
    print("\n" + "="*50)
    print("Phase 1: 训练生成器")
    print("="*50)
    
    if model_version == 'v2':
        G = GeneratorV2()
        if init_model_G is not None and os.path.exists(init_model_G):
            v1_state = torch.load(init_model_G, map_location='cpu')
            try:
                G.load_state_dict(v1_state)
                print(f"加载V2模型权重: {init_model_G}")
            except RuntimeError:
                G.load_pretrained_v1(v1_state)
                print(f"从V1权重部分加载到V2模型: {init_model_G}")
    else:
        G = Generator()
        if init_model_G is not None and os.path.exists(init_model_G):
            G.load_state_dict(torch.load(init_model_G, map_location='cpu'))
            print(f"加载预训练模型: {init_model_G}")
    G = G.to(device)
    configurable_loss = configurable_loss.to(device)
    optimizer_G = torch.optim.Adadelta(G.parameters())

    epoch = tqdm(total=num_epochs_pregen)
    update_epoch = max(1, num_epochs_pregen // 10)
    
    while epoch.n < num_epochs_pregen:
        for X in train_iter:
            X = X.to(device)
            mask_size, mask_location = gen_Mask(X)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location).to(device)
            X_mask = X - X * masklayer + mpv * masklayer
            input_G = torch.cat((X_mask, masklayer), dim=1)
            output_G = G(input_G)
            loss, loss_detail = configurable_loss(X, output_G, masklayer, app)
    
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            epoch.set_description('phase 1 | train loss: %.5f' % loss.cpu())
        epoch.update()
    
        if epoch.n % update_epoch == 0:
            G.eval()
            with torch.no_grad():
                X = sample_random_batch(test_dataset, batch_size=8).to(device)
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location).to(device)
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(X_mask, output_G, masklayer)
                imgs = torch.cat((X.cpu(), X_mask.cpu(), completed.cpu()), dim=0)
                imgpath = os.path.join(result_dir, 'phase_1', f'phase_1_epoch{epoch.n}.png')
                model_path = os.path.join(result_dir, 'phase_1', f'phase_1_model_generator_epoch{epoch.n}')
                save_image(imgs, imgpath, nrow=8)
                torch.save(G.state_dict(), model_path)
            G.train()
        if epoch.n >= num_epochs_pregen:
            break
    epoch.close()

    # Training Phase 2
    print("\n" + "="*50)
    print("Phase 2: 训练判别器")
    print("="*50)
    
    D = Discriminator(
        local_input_shape=(3, input_generator_size // 2, input_generator_size // 2),
        global_input_shape=(3, input_generator_size, input_generator_size))
    if init_model_D is not None and os.path.exists(init_model_D):
        D.load_state_dict(torch.load(init_model_D, map_location='cpu'))
    D = D.to(device)
    optimizer_D = torch.optim.Adadelta(D.parameters())
    loss_D = torch.nn.BCELoss()

    epoch = tqdm(total=num_epochs_predis)
    update_epoch = max(1, num_epochs_predis // 10)
    
    while epoch.n < num_epochs_predis:
        for X in train_iter:
            fake = torch.zeros((X.shape[0], 1)).to(device)
            real = torch.ones((X.shape[0], 1)).to(device)
            X = X.to(device)
            mask_size, mask_location = gen_Mask(X)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location).to(device)
            X_mask = X - X * masklayer + mpv * masklayer
            input_G = torch.cat((X_mask, masklayer), dim=1)
            output_G = G(input_G)
            input_global_D_fake = output_G.detach()
            input_local_D_fake = crop(input_global_D_fake, (mask_size, mask_location))
            output_D_fake = D((input_local_D_fake, input_global_D_fake))
            loss_fake = loss_D(output_D_fake, fake)

            input_global_D_real = X
            input_local_D_real = crop(input_global_D_real, (mask_size, mask_location))
            output_D_real = D((input_local_D_real, input_global_D_real))
            loss_real = loss_D(output_D_real, real)

            loss_landmark = Discriminator_Network_Loss(X, output_G, app)
            loss = loss_real + loss_fake + loss_landmark

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()
            epoch.set_description('phase 2 | train loss: %.5f' % loss.cpu())
        epoch.update()

        if epoch.n % update_epoch == 0:
            G.eval()
            with torch.no_grad():
                X = sample_random_batch(test_dataset, batch_size=8).to(device)
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location).to(device)
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(X_mask, output_G, masklayer)
                imgs = torch.cat((X.cpu(), X_mask.cpu(), completed.cpu()), dim=0)
                imgpath = os.path.join(result_dir, 'phase_2', f'phase_2_epoch{epoch.n}.png')
                model_path = os.path.join(result_dir, 'phase_2', f'phase_2_model_discriminator_epoch{epoch.n}')
                save_image(imgs, imgpath, nrow=8)
                torch.save(D.state_dict(), model_path)
            G.train()
        if epoch.n >= num_epochs_predis:
            break
    epoch.close()

    # Training Phase 3
    print("\n" + "="*50)
    print("Phase 3: 联合训练生成器和判别器")
    print("="*50)
    
    epoch = tqdm(total=num_epochs_step3)
    update_epoch = max(1, num_epochs_step3 // 10)
    
    while epoch.n < num_epochs_step3:
        for X in train_iter:
            X = X.to(device)
            mask_size, mask_location = gen_Mask(X)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location).to(device)

            fake = torch.zeros((len(X), 1)).to(device)
            real = torch.ones((X.shape[0], 1)).to(device)

            X_mask = X - X * masklayer + mpv * masklayer
            input_G = torch.cat((X_mask, masklayer), dim=1)
            output_G = G(input_G)
            input_global_D_fake = output_G.detach()
            input_local_D_fake = crop(input_global_D_fake, (mask_size, mask_location))
            output_D_fake = D((input_local_D_fake, input_global_D_fake))
            loss_fake = loss_D(output_D_fake, fake)
            input_global_D_real = X
            input_local_D_real = crop(input_global_D_real, (mask_size, mask_location))
            output_D_real = D((input_local_D_real, input_global_D_real))
            loss_real = loss_D(output_D_real, real)
            loss_landmark = Discriminator_Network_Loss(X, output_G, app)
            loss_d = (loss_fake + loss_real + loss_landmark) * alpha

            optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_D.step()

            loss_G_1, _ = configurable_loss(X, output_G, masklayer, app)
            input_gD_fake = output_G
            input_lD_fake = crop(input_gD_fake, (mask_size, mask_location))
            output_fake = D((input_lD_fake, input_gD_fake))
            loss_G_2 = loss_D(output_fake, real)
            loss = alpha * loss_G_2 + loss_G_1

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            epoch.set_description('phase 3 | loss (D): %.5f (G): %.5f' % (loss_d.cpu(), loss.cpu()))
        epoch.update()

        if epoch.n % update_epoch == 0:
            G.eval()
            with torch.no_grad():
                X = sample_random_batch(test_dataset, batch_size=8).to(device)
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location).to(device)
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(X_mask, output_G, masklayer)
                imgs = torch.cat((X.cpu(), X_mask.cpu(), completed.cpu()), dim=0)
                imgpath = os.path.join(result_dir, 'phase_3', f'phase_3_epoch{epoch.n}.png')
                model_D_path = os.path.join(result_dir, 'phase_3', f'phase_3_model_discriminator_epoch{epoch.n}')
                model_G_path = os.path.join(result_dir, 'phase_3', f'phase_3_model_generator_epoch{epoch.n}')
                save_image(imgs, imgpath, nrow=8)
                torch.save(G.state_dict(), model_G_path)
                torch.save(D.state_dict(), model_D_path)
            G.train()
        if epoch.n >= num_epochs_step3:
            break
    
    print("\n" + "="*50)
    print("训练完成!")
    print(f"模型保存在: {result_dir}")
    print("="*50)


if __name__ == '__main__':
    args = parser.parse_args()
    train(
        batch_size=32, 
        num_epochs_pregen=args.phase1, 
        num_epochs_predis=args.phase2, 
        num_epochs_step3=args.phase3,
        init_model_G=args.model_G, 
        init_model_D=args.model_D,
        loss_combination=args.loss_combination, 
        data_dir=args.data_dir,
        test_dir=args.test_dir, 
        result_dir=args.result_dir,
        model_version=args.model_version
    )
