## 环境

需要根据[HTWK Gym](https://github.com/NaoHTWK/htwk-gym)，配置好环境

#### 踢足球
```sh
# Train ball kicking behavior
python train.py --task=T1/Kicking

# Test ball kicking
python play.py --task=T1/Kicking --checkpoint=-1
```
## 修改
### 2025/12/08

创建了绿色草地和两个球门

![踢足球的视频演示](./images/kick_with_goal.gif)

### 2025/12/09

添加了全局摄像头和局部摄像头（更改代码：kicking.py 和 kicking.yaml）

可以通过更改帧数间隔（save_interval）保存图片

#### 全局摄像头
![全局视角](./images/1209_global_pic.png)

#### 局部摄像头
![局部视角](./images/1209_local_pic.png)

### 2025/12/12

添加了visual_feature_extractor.py文件，用简单的CNN提取视觉特征，并融入到机器人的状态中（更改代码：visual_feature_extractor.py、kicking.py 和 kicking.yaml）

具体效果还没去测试，目前可以运行，后期可以替换更好的神经网络架构

### 2025/12/23

添加了detect_football模块，可以检测出是否有足球并且输出足球的位置

``` 
pip install ultralytics opencv-python pillow
```

![识别结果](./images/1223_detect_ball.jpg)

### 2025/12/30

修改了visual_feature_extractor.py文件，搭建了可用的网络结构，机器人目前可以碰到球，但是不能把球踢出去

![足球踢出演示](./images/1230_kick_ball.gif)

### 2026/01/06

在强化学习（尤其是机器人踢球任务）中，“视觉伺服（Visual Servoing）”能够显著降低算法学习的难度，因为它保证了目标始终在机器人的视野范围内。

由于已经可以使用 YOLOv8 获取足球在图像中的中心坐标，最简单且有效的方法是使用一个 PID 控制器（或简化的比例控制 P） 来动态调整摄像头的旋转角（Pitch 和 Yaw）。

在kicking.py中添加了detect_football模块，通过yolo8自适应调节角度，在第一次识别到足球以后，保证每次读取的画面都包含足球

![识别效果](./images/0106_detect_ball.jpg)


