import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViTFeaturePyramid(nn.Module):
    """
    Simple Feature Pyramid for ViT: 将 ViT 的特征转换为多尺度金字塔
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Level 0 (1/16): 调整通道
        self.conv_0 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.norm_0 = nn.GroupNorm(32, out_dim)
        
        # Level 1 (1/32): 下采样 2x
        self.conv_1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm_1 = nn.GroupNorm(32, out_dim)
        
        # Level 2 (1/64): 再下采样 2x
        self.conv_2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm_2 = nn.GroupNorm(32, out_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        """
        x: [B, L, D] (去掉了 CLS token 的特征)
        H, W: 原始特征图的高度和宽度
        """
        B, L, C = x.shape
        # 1. 恢复空间结构: [B, L, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # 2. 构建金字塔
        p0 = F.gelu(self.norm_0(self.conv_0(x)))      # 1/16
        p1 = F.gelu(self.norm_1(self.conv_1(p0)))     # 1/32
        p2 = F.gelu(self.norm_2(self.conv_2(p1)))     # 1/64
        
        features = [p0, p1, p2]
        
        # 3. 准备 MultiScaleAttention 需要的展平特征和元数据
        src_flatten = []
        spatial_shapes = []
        
        for feat in features:
            _, _, h, w = feat.shape
            spatial_shapes.append((h, w))
            # 展平: [B, C, H, W] -> [B, H*W, C]
            src_flatten.append(feat.flatten(2).transpose(1, 2))
            
        src_flatten = torch.cat(src_flatten, 1) # [B, Total_Pixels, C]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device)
        # 计算每一层的起始索引
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        return src_flatten, spatial_shapes, level_start_index


def test_vit_feature_pyramid():
    """验证ViTFeaturePyramid的核心功能"""
    # 1. 定义测试参数（模拟ViT-Base的典型配置）
    BATCH_SIZE = 2          # 批次大小
    IN_DIM = 768            # ViT-Base的特征维度
    OUT_DIM = 256           # 金字塔输出维度
    H = 32                  # ViT特征图高度（对应原始图像512x512，patch size=16）
    W = 32                  # ViT特征图宽度
    L = H * W               # 特征序列长度（去掉CLS token后）
    
    # 2. 创建模型实例
    model = ViTFeaturePyramid(in_dim=IN_DIM, out_dim=OUT_DIM)
    model.eval()  # 评估模式，避免BatchNorm/GroupNorm的训练行为干扰
    
    # 3. 构造模拟输入（符合[B, L, D]的形状）
    # 随机生成输入，模拟ViT输出的去掉CLS token后的特征
    x = torch.randn(BATCH_SIZE, L, IN_DIM)
    print(f"=== 输入信息 ===")
    print(f"输入形状: {x.shape} (B={BATCH_SIZE}, L={L}, D={IN_DIM})")
    print(f"原始特征图尺寸: H={H}, W={W}\n")
    
    # 4. 执行前向传播
    with torch.no_grad():  # 禁用梯度计算，提升速度
        src_flatten, spatial_shapes, level_start_index = model(x, H, W)
    
    # 5. 验证关键输出（核心验证点）
    print(f"=== 输出验证 ===")
    
    # 5.1 验证各层金字塔的空间尺寸（下采样比例）
    print(f"\n1. 各层金字塔空间尺寸验证:")
    expected_shapes = [(32,32), (16,16), (8,8)]  # 预期的下采样比例：1x, 2x, 4x
    for i, (shape, expected) in enumerate(zip(spatial_shapes, expected_shapes)):
        h, w = shape.cpu().tolist()
        assert (h, w) == expected, f"第{i}层空间尺寸错误！预期{expected}，实际{(h,w)}"
        print(f"   第{i}层 (1/{16*(2**i)}): 尺寸({h}, {w}) ✅ (符合预期)")
    
    # 5.2 验证展平特征的总长度
    print(f"\n2. 展平特征总长度验证:")
    total_pixels = sum([h*w for h,w in spatial_shapes.cpu().tolist()])
    expected_total_pixels = 1344 # 32*32 + 16*16 + 8*8 = 1024 + 256 + 64 = 1344
    assert total_pixels == expected_total_pixels, f"总像素数错误！预期{expected_total_pixels}，实际{total_pixels}"
    print(f"   各层像素数之和: {total_pixels} ✅ (符合预期)")
    print(f"   src_flatten形状: {src_flatten.shape} (预期[2, 1344, 256]) ✅")
    
    # 5.3 验证层级起始索引
    print(f"\n3. 层级起始索引验证:")
    expected_indices = [0, 1024, 1280]  # 0层起始0，1层起始32*32=1024，2层起始1024+16*16=1280
    indices = level_start_index.cpu().tolist()
    assert indices == expected_indices, f"起始索引错误！预期{expected_indices}，实际{indices}"
    print(f"   层级起始索引: {indices} ✅ (符合预期)")
    
    # 5.4 验证数据类型和设备（工程细节）
    print(f"\n4. 数据类型与设备验证:")
    assert src_flatten.dtype == torch.float32, f"数据类型错误！预期float32，实际{src_flatten.dtype}"
    assert src_flatten.device == x.device, f"设备不匹配！输入设备{x.device}，输出设备{src_flatten.device}"
    print(f"   数据类型: {src_flatten.dtype} ✅")
    print(f"   设备: {src_flatten.device} ✅")
    
    print(f"\n=== 所有验证通过！ViTFeaturePyramid 代码正常运行 ===")


# 执行验证
if __name__ == "__main__":
    test_vit_feature_pyramid()