# envs/T1/visual_feature_extractor.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms

class VisualFeatureExtractor(nn.Module):
    """
    从机器人摄像头图像中提取视觉特征
    输入: RGB图像 [H, W, 3]
    输出: 视觉特征向量
    """
    def __init__(self, input_height=480, input_width=640, feature_dim=32):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.feature_dim = feature_dim
        
        # 简单的CNN编码器
        self.encoder = nn.Sequential(
            # 输入: 3 x 480 x 640
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # 16 x 240 x 320
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 32 x 120 x 160
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 64 x 60 x 80
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 128 x 30 x 40
            
            nn.AdaptiveAvgPool2d((4, 4)),
            # 128 x 4 x 4 = 2048维
            
            nn.Flatten(),
            
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, feature_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )
    
    def forward(self, images):
        """
        前向传播
        images: [B, H, W, 3] 或 [H, W, 3] 的RGB图像
        返回: [B, feature_dim] 或 [feature_dim] 的特征向量
        """
        # 确保输入是float类型
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # 确保通道维度在正确的位置
        if images.dim() == 3:
            # 单张图片: [H, W, 3] -> [3, H, W]
            images = images.permute(2, 0, 1).unsqueeze(0)
        elif images.dim() == 4:
            # 批量图片: [B, H, W, 3] -> [B, 3, H, W]
            images = images.permute(0, 3, 1, 2)
        
        # 预处理
        images = images / 255.0  # 归一化到[0, 1]
        
        # 通过编码器
        features = self.encoder(images)
        
        return features

# 轻量级版本 - 用于快速测试
class SimpleVisualFeatureExtractor:
    """
    简化的视觉特征提取器，不依赖于深度学习
    """
    def __init__(self):
        self.feature_names = [
            'ball_visible', 'ball_x', 'ball_y', 'ball_size',
            'brightness_center', 'brightness_variance'
        ]
    
    def extract_features(self, image):
        """
        提取简单的视觉特征
        """
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # 球检测特征
        ball_features = self._detect_ball(image_np)
        
        # 亮度特征
        if image_np.shape[-1] == 3:
            gray = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image_np
        
        brightness_center = np.mean(gray[gray.shape[0]//4:3*gray.shape[0]//4, 
                                         gray.shape[1]//4:3*gray.shape[1]//4]) / 255.0
        brightness_variance = np.std(gray) / 255.0
        
        features = np.concatenate([
            ball_features,
            np.array([brightness_center, brightness_variance])
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _detect_ball(self, image):
        """
        检测球的位置和大小
        """
        # 简化的球检测
        h, w = image.shape[:2]
        
        # 假设球在图像中心附近
        center_x = w // 2
        center_y = h // 2
        
        # 检查中心区域的颜色
        center_region = image[center_y-20:center_y+20, center_x-20:center_x+20]
        
        # 简单判断：如果中心区域亮度较高，假设有球
        if len(center_region) > 0:
            avg_brightness = np.mean(center_region)
            ball_visible = 1.0 if avg_brightness > 150 else 0.0
        else:
            ball_visible = 0.0
        
        return np.array([
            ball_visible,
            (center_x / w) * 2 - 1,  # 归一化到[-1, 1]
            (center_y / h) * 2 - 1,
            0.5  # 假设的球大小
        ])