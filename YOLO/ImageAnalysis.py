#画像分析簡易
from ultralytics import YOLO

# 学習済みモデルをロード
model = YOLO(r"C:\YOLO\YOLO\runs\detect\train-8\weights\best.pt")

url = r"C:\YOLOcamera\Camera01_20260804_132600.mp4"
results = model(url,save=True)


print("分析完了")