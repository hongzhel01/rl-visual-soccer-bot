from ultralytics import YOLO
import cv2

# 1. 加载模型 (yolov8n 表示 nano 版本，速度最快)
model = YOLO('yolov8n.pt')

# 2. 执行检测
# source: 图片路径, conf: 置信度阈值（0.25表示只保留25%把握以上的物体）
results = model.predict(source='1209_local_pic.jpg', conf=0.25)

# 3. 解析结果
for result in results:
    # 获取检测到的所有框
    boxes = result.boxes
    for box in boxes:
        # 获取类别 ID
        cls_id = int(box.cls[0])
        # 在 COCO 数据集中，ID 32 代表 "sports ball" (通常就是足球)
        if cls_id == 32:
            # 获取坐标 (左上角x, 左上角y, 右下角x, 右下角y)
            # .xyxy 是像素值格式，.xywh 是中心点+宽高格式
            coords = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            
            print(f"--- 发现足球 ---")
            print(f"坐标位置 (x1, y1, x2, y2): {coords}")
            print(f"置信度: {confidence:.2f}")

            # 4. 可视化（可选：在原图上画框并保存）
            annotated_frame = result.plot()
            cv2.imwrite('result_soccer.jpg', annotated_frame)
            print("结果已保存为 result_soccer.jpg")