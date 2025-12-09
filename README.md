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

![踢足球的视频演示](./kick_with_goal.gif)

### 2025/12/09

添加了全局摄像头和局部摄像头（更改代码：kicking.py 和 kicking.yaml）
可以通过更改帧数间隔（save_interval）保存图片

#### 全局摄像头
![全局视角](./1209_global_pic.png)

#### 局部摄像头
![局部视角](./1209_local_pic.png)

