from ultralytics import YOLO
import cv2

# 加载模型（用 yolov8s，精度更高，适合识别草）
model = YOLO("yolov8s.pt")

# 打开摄像头（0 是默认摄像头）
cap = cv2.VideoCapture(0)

# 检查摄像头是否打开成功
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

# 实时循环读取画面
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面！")
        break

    # YOLO 推理（实时识别）
    results = model(frame, stream=True)

    # 绘制识别框
    for r in results:
        annotated_frame = r.plot()

    # 显示实时画面
    cv2.imshow("激光除草实时识别", annotated_frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()