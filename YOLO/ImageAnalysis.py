from ultralytics import YOLO

# 学習済みモデルをロード
model = YOLO(r"C:\YOLO\YOLO\best.pt")

url = r"C:\Users\01A1561\Downloads\98996_640x360.mp4"
results = model(url,save=True)

for frame_idx, result in enumerate(results):
    boxes = result.boxes  # 検出結果のボックス情報
    for box in boxes:
        cls_id = int(box.cls[0])        # クラスID
        cls_name = model.names[cls_id]  # クラス名
        conf = float(box.conf[0])       # 信頼度 (0.0〜1.0)
        print(f"フレーム {frame_idx}: 検出={cls_name}, 信頼度={conf:.2f}")

print("分析完了")