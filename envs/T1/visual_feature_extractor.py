import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNVisionBackbone(nn.Module):
    def __init__(self, feature_dim=52):
        super(DQNVisionBackbone, self).__init__()
        # 卷积层参数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2) 
        
        # 展平维度计算（适配 160x120 输入）
        self.flatten_dim = 3456 
        
        # 隐藏层：512 个单元 [cite: 209]
        self.fc1 = nn.Linear(self.flatten_dim, 512) 
        # 输出层：直接输出 52 维特征向量 [cite: 210]
        self.fc_out = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc_out(x)

class VisualFeatureExtractor:
    def __init__(self, feature_dim=52, device="cuda"):
        self.device = device
        self.feature_dim = feature_dim
        # 初始化模型
        self.model = DQNVisionBackbone(feature_dim=feature_dim).to(device)
        self.model.eval()

    def extract_all_features(self, image_tensor):
        """
        处理流程：RGB -> Grayscale -> Resize(160, 120) -> CNN
        """
        with torch.no_grad():
            # 1. 预处理图像
            processed_img = self._preprocess(image_tensor)
            # 2. 提取特征
            features = self.model(processed_img)
        return features.squeeze(0)

    def _preprocess(self, img):
        # 如果是 (H, W, C) 格式，转为 (C, H, W) 并增加 batch 维度
        if img.dim() == 3:
            img = img.permute(2, 0, 1).unsqueeze(0).float()
        
        # 1. 转换为灰度图：使用加权平均公式
        # (Batch, 3, H, W) -> (Batch, 1, H, W)
        gray_img = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        
        # 2. 缩放到论文要求的 160x120 
        # 注意：论文中宽度是160，高度是120
        resized_img = F.interpolate(gray_img, size=(120, 160), mode='bilinear', align_corners=False)
        
        # 3. 归一化到 [0, 1]
        return resized_img / 255.0