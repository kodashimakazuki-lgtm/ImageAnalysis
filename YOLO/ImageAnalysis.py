#画像分析簡易
from ultralytics import YOLO

# 学習済みモデルをロード
model = YOLO(r"C:\YOLO\YOLO\runs\detect\train-11\weights\best.pt")

url = r"C:\YOLOcamera\Camera01_20260805_104550.mp4"
results = model(url,save=True)


print("分析完了")