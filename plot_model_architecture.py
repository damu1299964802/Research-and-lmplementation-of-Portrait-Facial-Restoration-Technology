import sys
import os
import numpy as np

# 添加正确的PlotNeuralNet目录到路径
sys.path.append('F:/PlotNeuralNet-master/PlotNeuralNet-master')
from pycore.tikzeng import *

def create_generator_arch():
    """创建生成器模型的立体架构图"""
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),
        
        # 标题
        to_TextBox("人脸修复生成器模型架构"),
        
        # 输入层
        to_input('input', 160, 160, 4),
        to_Conv("input_text", 1, 1, offset="(0,0,0)", to="(input-south)", height=1, depth=1, width=1),
        to_TextBox("输入图像+掩码", to="(input_text-south)"),
        
        # 编码器部分
        to_Conv("conv1", 160, 64, offset="(2,0,0)", to="(input-east)", height=160, depth=160, width=2),
        to_TextBox("Conv1 5×5, 64", to="(conv1-south)"),
        to_connection("input", "conv1"),
        
        to_Conv("conv2", 80, 128, offset="(2,0,0)", to="(conv1-east)", height=80, depth=80, width=2),
        to_TextBox("Conv2 3×3, 128, s=2", to="(conv2-south)"),
        to_connection("conv1", "conv2"),
        
        to_Conv("conv3", 80, 128, offset="(2,0,0)", to="(conv2-east)", height=80, depth=80, width=2),
        to_TextBox("Conv3 3×3, 128", to="(conv3-south)"),
        to_connection("conv2", "conv3"),
        
        to_Conv("conv4", 40, 256, offset="(2,0,0)", to="(conv3-east)", height=40, depth=40, width=2),
        to_TextBox("Conv4 3×3, 256, s=2", to="(conv4-south)"),
        to_connection("conv3", "conv4"),
        
        to_Conv("conv5", 40, 256, offset="(2,0,0)", to="(conv4-east)", height=40, depth=40, width=2),
        to_TextBox("Conv5 3×3, 256", to="(conv5-south)"),
        to_connection("conv4", "conv5"),
        
        to_Conv("conv6", 40, 256, offset="(2,0,0)", to="(conv5-east)", height=40, depth=40, width=2),
        to_TextBox("Conv6 3×3, 256", to="(conv6-south)"),
        to_connection("conv5", "conv6"),
        
        # 空洞卷积部分
        to_Conv("dconv7", 40, 256, offset="(2,0,0)", to="(conv6-east)", height=40, depth=40, width=2),
        to_TextBox("DilatedConv7 3×3, d=2", to="(dconv7-south)"),
        to_connection("conv6", "dconv7"),
        
        to_Conv("dconv8", 40, 256, offset="(2,0,0)", to="(dconv7-east)", height=40, depth=40, width=2),
        to_TextBox("DilatedConv8 3×3, d=4", to="(dconv8-south)"),
        to_connection("dconv7", "dconv8"),
        
        to_Conv("dconv9", 40, 256, offset="(2,0,0)", to="(dconv8-east)", height=40, depth=40, width=2),
        to_TextBox("DilatedConv9 3×3, d=8", to="(dconv9-south)"),
        to_connection("dconv8", "dconv9"),
        
        to_Conv("dconv10", 40, 256, offset="(2,0,0)", to="(dconv9-east)", height=40, depth=40, width=2),
        to_TextBox("DilatedConv10 3×3, d=16", to="(dconv10-south)"),
        to_connection("dconv9", "dconv10"),
        
        # 解码器部分
        to_Conv("conv11", 40, 256, offset="(2,0,0)", to="(dconv10-east)", height=40, depth=40, width=2),
        to_TextBox("Conv11 3×3, 256", to="(conv11-south)"),
        to_connection("dconv10", "conv11"),
        
        to_Conv("conv12", 40, 256, offset="(2,0,0)", to="(conv11-east)", height=40, depth=40, width=2),
        to_TextBox("Conv12 3×3, 256", to="(conv12-south)"),
        to_connection("conv11", "conv12"),
        
        to_Conv("deconv13", 80, 128, offset="(2,0,0)", to="(conv12-east)", height=80, depth=80, width=2),
        to_TextBox("TransposedConv13 4×4, 128, s=2", to="(deconv13-south)"),
        to_connection("conv12", "deconv13"),
        
        to_Conv("conv14", 80, 128, offset="(2,0,0)", to="(deconv13-east)", height=80, depth=80, width=2),
        to_TextBox("Conv14 3×3, 128", to="(conv14-south)"),
        to_connection("deconv13", "conv14"),
        
        to_Conv("deconv15", 160, 64, offset="(2,0,0)", to="(conv14-east)", height=160, depth=160, width=2),
        to_TextBox("TransposedConv15 4×4, 64, s=2", to="(deconv15-south)"),
        to_connection("conv14", "deconv15"),
        
        to_Conv("conv16", 160, 32, offset="(2,0,0)", to="(deconv15-east)", height=160, depth=160, width=2),
        to_TextBox("Conv16 3×3, 32", to="(conv16-south)"),
        to_connection("deconv15", "conv16"),
        
        # 输出层
        to_Conv("conv17", 160, 3, offset="(2,0,0)", to="(conv16-east)", height=160, depth=160, width=1),
        to_TextBox("Conv17 3×3, 3, Sigmoid", to="(conv17-south)"),
        to_connection("conv16", "conv17"),
        
        to_output("output", 160, 160, offset="(2,0,0)", to="(conv17-east)"),
        to_TextBox("输出图像 3×160×160", to="(output-south)"),
        to_connection("conv17", "output"),
        
        to_end()
    ]
    return arch

def create_srcnn_arch():
    """创建SRCNN超分辨率模型的立体架构图"""
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),
        
        # 标题
        to_TextBox("SRCNN超分辨率模型架构"),
        
        # 输入层
        to_input('input', 64, 64, 3),
        to_TextBox("输入图像 3×H×W", to="(input-south)"),
        
        # 双三次插值
        to_Conv("upsample", 128, 128, offset="(3,0,0)", to="(input-east)", height=128, depth=128, width=1),
        to_TextBox("双三次插值 放大s倍", to="(upsample-south)"),
        to_connection("input", "upsample"),
        
        # SRCNN网络
        to_Conv("conv1", 128, 64, offset="(3,0,0)", to="(upsample-east)", height=128, depth=128, width=2),
        to_TextBox("Conv1 9×9, 64, ReLU", to="(conv1-south)"),
        to_connection("upsample", "conv1"),
        
        to_Conv("conv2", 128, 32, offset="(3,0,0)", to="(conv1-east)", height=128, depth=128, width=2),
        to_TextBox("Conv2 5×5, 32, ReLU", to="(conv2-south)"),
        to_connection("conv1", "conv2"),
        
        to_Conv("conv3", 128, 3, offset="(3,0,0)", to="(conv2-east)", height=128, depth=128, width=1),
        to_TextBox("Conv3 5×5, 3, Sigmoid", to="(conv3-south)"),
        to_connection("conv2", "conv3"),
        
        # 输出层
        to_output("output", 128, 128, offset="(3,0,0)", to="(conv3-east)"),
        to_TextBox("输出高分辨率图像 3×(H*s)×(W*s)", to="(output-south)"),
        to_connection("conv3", "output"),
        
        to_end()
    ]
    return arch

def create_espcn_arch():
    """创建ESPCN超分辨率模型的立体架构图"""
    scale_factor = 2  # 放大倍数
    
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),
        
        # 标题
        to_TextBox("ESPCN超分辨率模型架构"),
        
        # 输入层
        to_input('input', 64, 64, 3),
        to_TextBox("输入图像 3×H×W", to="(input-south)"),
        
        # ESPCN网络
        to_Conv("conv1", 64, 64, offset="(3,0,0)", to="(input-east)", height=64, depth=64, width=2),
        to_TextBox("Conv1 5×5, 64, ReLU", to="(conv1-south)"),
        to_connection("input", "conv1"),
        
        to_Conv("conv2", 64, 64, offset="(3,0,0)", to="(conv1-east)", height=64, depth=64, width=2),
        to_TextBox("Conv2 3×3, 64, ReLU", to="(conv2-south)"),
        to_connection("conv1", "conv2"),
        
        to_Conv("conv3", 64, 32, offset="(3,0,0)", to="(conv2-east)", height=64, depth=64, width=2),
        to_TextBox("Conv3 3×3, 32, ReLU", to="(conv3-south)"),
        to_connection("conv2", "conv3"),
        
        to_Conv("conv4", 64, 3*(scale_factor**2), offset="(3,0,0)", to="(conv3-east)", height=64, depth=64, width=2),
        to_TextBox(f"Conv4 3×3, {3*(scale_factor**2)}", to="(conv4-south)"),
        to_connection("conv3", "conv4"),
        
        # 子像素卷积层
        to_Conv("pixelshuffle", 128, 128, offset="(3,0,0)", to="(conv4-east)", height=128, depth=128, width=1),
        to_TextBox(f"PixelShuffle (放大{scale_factor}倍)", to="(pixelshuffle-south)"),
        to_connection("conv4", "pixelshuffle"),
        
        # Sigmoid激活
        to_Conv("sigmoid", 128, 128, offset="(2,0,0)", to="(pixelshuffle-east)", height=128, depth=128, width=1),
        to_TextBox("Sigmoid", to="(sigmoid-south)"),
        to_connection("pixelshuffle", "sigmoid"),
        
        # 输出层
        to_output("output", 128, 128, offset="(2,0,0)", to="(sigmoid-east)"),
        to_TextBox(f"输出高分辨率图像 3×(H*{scale_factor})×(W*{scale_factor})", to="(output-south)"),
        to_connection("sigmoid", "output"),
        
        to_end()
    ]
    return arch

def create_pipeline_arch():
    """创建完整的人脸修复与超分辨率流水线的立体架构图"""
    scale_factor = 2  # 放大倍数
    
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),
        
        # 标题
        to_TextBox("人脸修复与超分辨率增强流水线"),
        
        # 原始图像
        to_input('input', 64, 64, 3),
        to_TextBox("原始受损图像", to="(input-south)"),
        
        # 人脸修复部分
        to_ConvBlock("masklayer", 64, 64, offset="(3,0,0)", to="(input-east)", depth=64, height=64, width=1),
        to_TextBox("添加掩码层", to="(masklayer-south)"),
        to_connection("input", "masklayer"),
        
        to_ConvBlock("generator", 64, 64, offset="(3,0,0)", to="(masklayer-east)", depth=64, height=64, width=3),
        to_TextBox("生成器网络\n(17层CNN)", to="(generator-south)"),
        to_connection("masklayer", "generator"),
        
        to_ConvBlock("poisson", 64, 64, offset="(3,0,0)", to="(generator-east)", depth=64, height=64, width=1),
        to_TextBox("Poisson Blend\n融合", to="(poisson-south)"),
        to_connection("generator", "poisson"),
        
        to_Conv("restored", 64, 3, offset="(2,0,0)", to="(poisson-east)", height=64, depth=64, width=1),
        to_TextBox("修复后图像", to="(restored-south)"),
        to_connection("poisson", "restored"),
        
        # 超分辨率部分
        to_ConvBlock("sr", 64, 64, offset="(3,0,0)", to="(restored-east)", depth=64, height=64, width=3),
        to_TextBox("超分辨率增强\n(选择算法)", to="(sr-south)"),
        to_connection("restored", "sr"),
        
        # 锐化处理
        to_ConvBlock("sharpen", 128, 128, offset="(3,0,0)", to="(sr-east)", depth=128, height=128, width=1),
        to_TextBox("锐化处理\n(可选)", to="(sharpen-south)"),
        to_connection("sr", "sharpen"),
        
        # 最终输出
        to_output("output", 128, 128, offset="(2,0,0)", to="(sharpen-east)"),
        to_TextBox(f"最终高清修复图像 3×(H*{scale_factor})×(W*{scale_factor})", to="(output-south)"),
        to_connection("sharpen", "output"),
        
        to_end()
    ]
    return arch

# 添加自定义TextBox函数
def to_TextBox(text, to="(0,0,0)", offset="(0,0,0)", text_width=4, text_height=0.5, text_depth=0.5):
    return r"""
\node[text width=%fcm, align=center] at %s {%s};
""" % (text_width, to, text, )

def generate_text_descriptions(output_dir):
    """生成模型架构的文本描述"""
    print("生成模型架构文本描述...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成生成器模型架构文本描述
    with open(os.path.join(output_dir, 'generator_architecture.txt'), 'w', encoding='utf-8') as f:
        f.write("人脸修复生成器模型架构图:\n\n")
        f.write("输入: 带掩码的图像 (4×160×160)\n")
        f.write("编码器部分:\n")
        f.write("  - Conv1: 5×5, 64通道, ReLU (输出: 64×160×160)\n")
        f.write("  - Conv2: 3×3, 128通道, 步长=2, ReLU (输出: 128×80×80)\n")
        f.write("  - Conv3: 3×3, 128通道, ReLU (输出: 128×80×80)\n")
        f.write("  - Conv4: 3×3, 256通道, 步长=2, ReLU (输出: 256×40×40)\n")
        f.write("  - Conv5: 3×3, 256通道, ReLU (输出: 256×40×40)\n")
        f.write("  - Conv6: 3×3, 256通道, ReLU (输出: 256×40×40)\n")
        f.write("空洞卷积部分:\n")
        f.write("  - DilatedConv7: 3×3, 256通道, 空洞率=2, ReLU (输出: 256×40×40)\n")
        f.write("  - DilatedConv8: 3×3, 256通道, 空洞率=4, ReLU (输出: 256×40×40)\n")
        f.write("  - DilatedConv9: 3×3, 256通道, 空洞率=8, ReLU (输出: 256×40×40)\n")
        f.write("  - DilatedConv10: 3×3, 256通道, 空洞率=16, ReLU (输出: 256×40×40)\n")
        f.write("解码器部分:\n")
        f.write("  - Conv11: 3×3, 256通道, ReLU (输出: 256×40×40)\n")
        f.write("  - Conv12: 3×3, 256通道, ReLU (输出: 256×40×40)\n")
        f.write("  - TransposedConv13: 4×4, 128通道, 步长=2, ReLU (输出: 128×80×80)\n")
        f.write("  - Conv14: 3×3, 128通道, ReLU (输出: 128×80×80)\n")
        f.write("  - TransposedConv15: 4×4, 64通道, 步长=2, ReLU (输出: 64×160×160)\n")
        f.write("  - Conv16: 3×3, 32通道, ReLU (输出: 32×160×160)\n")
        f.write("输出层:\n")
        f.write("  - Conv17: 3×3, 3通道, Sigmoid (输出: 3×160×160)\n")
    
    # 生成超分辨率模型架构文本描述
    with open(os.path.join(output_dir, 'super_resolution_architecture.txt'), 'w', encoding='utf-8') as f:
        f.write("超分辨率模型架构图:\n\n")
        f.write("1. SRCNN模型:\n")
        f.write("   输入: 原始图像 (3×H×W)\n")
        f.write("   - 双三次插值放大到目标尺寸 (3×(H*s)×(W*s))\n")
        f.write("   - Conv1: 9×9, 64通道, ReLU\n")
        f.write("   - Conv2: 5×5, 32通道, ReLU\n")
        f.write("   - Conv3: 5×5, 3通道, Sigmoid\n")
        f.write("   输出: 高分辨率图像 (3×(H*s)×(W*s))\n\n")
        f.write("2. ESPCN模型:\n")
        f.write("   输入: 原始图像 (3×H×W)\n")
        f.write("   - Conv1: 5×5, 64通道, ReLU\n")
        f.write("   - Conv2: 3×3, 64通道, ReLU\n")
        f.write("   - Conv3: 3×3, 32通道, ReLU\n")
        f.write("   - Conv4: 3×3, 3*(s^2)通道\n")
        f.write("   - PixelShuffle: 子像素重排，放大s倍\n")
        f.write("   - Sigmoid激活\n")
        f.write("   输出: 高分辨率图像 (3×(H*s)×(W*s))\n")

def generate_model_diagrams(output_dir):
    """生成所有模型架构图"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成生成器模型架构图...")
    generator_arch = create_generator_arch()
    generator_tex = os.path.join(output_dir, 'generator_architecture.tex')
    to_generate(generator_arch, generator_tex)
    
    print("生成SRCNN模型架构图...")
    srcnn_arch = create_srcnn_arch()
    srcnn_tex = os.path.join(output_dir, 'srcnn_architecture.tex')
    to_generate(srcnn_arch, srcnn_tex)
    
    print("生成ESPCN模型架构图...")
    espcn_arch = create_espcn_arch()
    espcn_tex = os.path.join(output_dir, 'espcn_architecture.tex')
    to_generate(espcn_arch, espcn_tex)
    
    print("生成完整流水线架构图...")
    pipeline_arch = create_pipeline_arch()
    pipeline_tex = os.path.join(output_dir, 'pipeline_architecture.tex')
    to_generate(pipeline_arch, pipeline_tex)
    
    return [generator_tex, srcnn_tex, espcn_tex, pipeline_tex]

def main():
    # 在当前目录创建model_diagrams目录
    output_dir = 'model_diagrams'
    
    # 生成模型架构的文本描述
    generate_text_descriptions(output_dir)
    
    # 生成所有模型架构图
    tex_files = generate_model_diagrams(output_dir)
    
    print("\n所有模型架构图已生成！")
    print("文件保存在 'model_diagrams' 目录下")
    print("包括:")
    print("1. LaTeX格式的架构图源文件")
    print("2. 文本格式的架构描述")
    
    print("\n要将LaTeX文件转换为PDF，请使用以下命令:")
    for tex_file in tex_files:
        print(f"cd F:/PlotNeuralNet-master/PlotNeuralNet-master && python generate.py {os.path.abspath(tex_file)}")

if __name__ == "__main__":
    main() 