import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights


# -----------------------------
# 空间金字塔池化（SPP）
# -----------------------------
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

    def forward(self, x):
        B, C, H, W = x.size()
        outs = []
        for s in self.pool_sizes:
            pooled = F.adaptive_max_pool2d(x, (s, s))       # [B, C, s, s]
            outs.append(pooled.view(B, C, -1))              # [B, C, s*s]
        return torch.cat(outs, dim=2)                       # [B, C, sum(s*s)]


# -----------------------------
# PSP 风格融合
# -----------------------------
class PSPFusion(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.spp = SpatialPyramidPooling(pool_sizes)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(in_channels * 4, in_channels * 2, kernel_size=1),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        # features: list of 4 tensors [B, C, H_i, W_i]
        spp_list = [self.spp(feat) for feat in features]    # all -> [B, C, N]
        x = torch.cat(spp_list, dim=1)                      # [B, 4C, N]
        x = self.fusion_conv(x)                             # [B, out, N]
        return x.mean(dim=2)                                # [B, out]


# -----------------------------
# 子区域分割融合
# -----------------------------
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels, split_factor):
        super().__init__()
        assert split_factor >= 1
        self.split = split_factor
        self.num_regions = split_factor ** 2
        self.conv = nn.Conv2d(in_channels * self.num_regions, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        s = self.split
        assert H % s == 0 and W % s == 0, f"输入尺寸({H}x{W}) 必须能被分割因子 {s} 整除"

        subs = []
        for i in range(s):
            for j in range(s):
                subs.append(x[:, :, i::s, j::s])            # 跃步采样子图
        x = torch.cat(subs, dim=1)                          # [B, C*s*s, H/s, W/s]
        x = self.act(self.bn(self.conv(x)))
        return x


# -----------------------------
# 方向注意力（水平/垂直）+ 通道注意力
# 不再使用非法的 (None, 1)/(1, None) 自适应池化
# -----------------------------
class H_Attention(nn.Module):
    """沿宽度 W 聚合，得到形状 [B, C, H, 1] 的权重"""
    def __init__(self, in_channels):
        super().__init__()
        hidden = max(1, in_channels // 16)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均/最大沿 dim=3（W 轴）聚合，形状 [B, C, H, 1]
        avg_h = torch.mean(x, dim=3, keepdim=True)
        max_h = torch.amax(x, dim=3, keepdim=True)
        out = self.fc(avg_h) + self.fc(max_h)
        return self.sigmoid(out)


class W_Attention(nn.Module):
    """沿高度 H 聚合，得到形状 [B, C, 1, W] 的权重"""
    def __init__(self, in_channels):
        super().__init__()
        hidden = max(1, in_channels // 16)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均/最大沿 dim=2（H 轴）聚合，形状 [B, C, 1, W]
        avg_v = torch.mean(x, dim=2, keepdim=True)
        max_v = torch.amax(x, dim=2, keepdim=True)
        out = self.fc(avg_v) + self.fc(max_v)
        return self.sigmoid(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        return self.sigmoid(out)


class AttentionProbability(nn.Module):
    """用全局 mean/std 统计估计该分支“有用性”的标量权重（逐样本）"""
    def __init__(self, in_channels):
        super().__init__()
        hidden = max(1, in_channels // 16)
        self.prob_fc = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        stats = torch.cat([
            torch.mean(x, dim=(2, 3), keepdim=True),        # [B, C, 1, 1]
            torch.std(x,  dim=(2, 3), keepdim=True),        # [B, C, 1, 1]
        ], dim=1)                                           # [B, 2C, 1, 1]
        return self.prob_fc(stats)                          # [B, 1, 1, 1]


class ThreeDAttention(nn.Module):
    """(channel, H, W) 三分支注意力 + 概率加权（修改后：x 向 guide 对齐）"""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        # 初始化组件（与原版本完全一致，无修改）
        self.ca = ChannelAttention(in_planes, ratio)
        self.ha = H_Attention(in_planes)
        self.wa = W_Attention(in_planes)
        self.p_ca = AttentionProbability(in_planes)
        self.p_ha = AttentionProbability(in_planes)
        self.p_wa = AttentionProbability(in_planes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guide):
        # -------------------------- 核心修改：x 向 guide 对齐 --------------------------
        # 若 x 的空间尺寸与 guide 不一致，将 x 插值到 guide 的尺寸
        if x.shape[2:] != guide.shape[2:]:
            x = F.interpolate(
                x,
                size=guide.shape[2:],  # 目标尺寸 = guide 的 H×W（不再是 x 的尺寸）
                mode='bilinear',       # 插值方式：双线性
                align_corners=False    # 避免边角像素偏移，保持与原代码的兼容性
            )
        # ------------------------------------------------------------------------------

        m_c = self.ca(guide)  # 通道掩码
        m_h = self.ha(guide)  # 水平掩码
        m_w = self.wa(guide)  # 垂直掩码

        # 2. 用这些掩码增强guide，得到初步增强特征（这个结果只用于给SAP模块评估有效性）
        x_c_for_sap = guide * m_c
        x_h_for_sap = guide * m_h
        x_w_for_sap = guide * m_w

        # 3. SAP模块评估各分支的有效性，得到概率标量
        w_c = self.p_ca(x_c_for_sap)
        w_h = self.p_ha(x_h_for_sap)
        w_w = self.p_wa(x_w_for_sap)

        # 4. 【关键修改】用概率标量去校准原始的注意力掩码，得到更可靠的掩码
        m_c_calibrated = m_c * w_c  # 校准后的通道掩码
        m_h_calibrated = m_h * w_h  # 校准后的水平掩码
        m_w_calibrated = m_w * w_w  # 校准后的垂直掩码

        # 5. 使用校准后的可靠掩码去增强目标特征x
        Ic = x * m_c_calibrated
        Ih = x * m_h_calibrated
        Iw = x * m_w_calibrated

        return Ic + Ih + Iw


# -----------------------------
# 主干网络
# -----------------------------
class SMPAN(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        self.layer0 = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1   # C=256
        self.layer2 = self.backbone.layer2   # C=512
        self.layer3 = self.backbone.layer3   # C=1024
        self.layer4 = self.backbone.layer4   # C=2048

        # 匹配到统一通道 512
        self.adj1 = nn.Conv2d(256, 512, 1)
        self.adj2 = nn.Conv2d(512, 512, 1)
        self.adj3 = nn.Conv2d(1024, 512, 1)
        self.adj4 = nn.Conv2d(2048, 512, 1)

        # 子区域分割（跃步采样）
        self.cut1 = Cut(256,  512, split_factor=8)
        self.cut2 = Cut(512,  512, split_factor=4)
        self.cut3 = Cut(1024, 512, split_factor=2)
        self.cut4 = Cut(2048, 512, split_factor=1)

        # 引导式三维注意力
        self.att1 = ThreeDAttention(512)
        self.att2 = ThreeDAttention(512)
        self.att3 = ThreeDAttention(512)
        self.att4 = ThreeDAttention(512)

        # PSP 融合
        self.psp = PSPFusion(512, 512, pool_sizes=(1, 2, 3, 6))

        # 分类头
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)    # [B,256,H/4,W/4]
        x2 = self.layer2(x1)    # [B,512,H/8,W/8]
        x3 = self.layer3(x2)    # [B,1024,H/16,W/16]
        x4 = self.layer4(x3)    # [B,2048,H/32,W/32]

        g1 = self.cut1(x1)
        g2 = self.cut2(x2)
        g3 = self.cut3(x3)
        g4 = self.cut4(x4)

        a1 = self.att1(self.adj1(x1), g1)
        a2 = self.att2(self.adj2(x2), g2)
        a3 = self.att3(self.adj3(x3), g3)
        a4 = self.att4(self.adj4(x4), g4)

        fused = self.psp([a1, a2, a3, a4])   # [B, 512]
        logits = self.fc(fused)              # [B, num_classes]
        return logits