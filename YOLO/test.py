import cv2
from ultralytics import YOLO

# 1. モデルのロード
model = YOLO(r"C:\YOLO\YOLO\best.pt")

# ==========================================
# ⚙️ 条件設定
# ==========================================
SOURCE = "rtsp://ebara:Ebara1234@192.168.0.10/Src/MediaInput/stream_1"    

TARGET_LABEL = "bottle"  # 検出したい対象
CONF_THRESHOLD = 0.9     # 信頼度のしきい値 (0.9以上)

# 💡 エリアの割合設定（画面全体に対する比率）
# 横幅を小さく(0.3)、縦幅を大きく(0.7)設定することで「縦長」になります
AREA_RATIO_W = 0.3  # 横幅：画面幅の30%
AREA_RATIO_H = 0.8  # 縦幅：画面高さの70%
# ==========================================

# 動画の読み込み開始
cap = cv2.VideoCapture(SOURCE)

# 画面の幅と高さを自動取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 💡 縦長の中央エリア座標を自動計算
area_w = int(width * AREA_RATIO_W)
area_h = int(height * AREA_RATIO_H)

area_x1 = int((width - area_w) / 2)
area_y1 = int((height - area_h) / 2)
area_x2 = area_x1 + area_w
area_y2 = area_y1 + area_h

print(f"🎥 動画サイズ: {width}x{height}")
print(f"🎯 縦長エリア座標: [{area_x1}, {area_y1}, {area_x2}, {area_y2}] (幅:{area_w}px, 高さ:{area_h}px)")
print("処理を開始します。停止するには画面上で 'q' キーを押してください。")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("動画の再生が終了したか、読み込めませんでした。")
        break

    # YOLOでフレームごとに分析
    results = model(frame, verbose=False)

    is_ok = False

    # 検出された物体のチェック
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        label_name = model.names[cls_id]
        conf = box.conf[0].item()

        # ラベルが一致し、信頼度が0.9以上の場合
        if label_name == TARGET_LABEL and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 物体の中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 縦長エリア内に入っているか判定
            if area_x1 <= center_x <= area_x2 and area_y1 <= center_y <= area_y2:
                is_ok = True
                break

    # --- 画面描画処理 ---
    annotated_frame = results[0].plot()

    if is_ok:
        color = (0, 255, 0)  # 緑
        status_text = "STATUS: OK"
    else:
        color = (0, 0, 255)  # 赤
        status_text = "STATUS: NG"

    # 縦長エリアの四角形を描画
    cv2.rectangle(
        annotated_frame, (area_x1, area_y1), (area_x2, area_y2), color, 3
    )

    # 画面にステータス文字を描画
    cv2.putText(
        annotated_frame,
        f"{status_text} ({TARGET_LABEL} >= {CONF_THRESHOLD})",
        (area_x1, area_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    # 画面にリアルタイム表示
    cv2.imshow("YOLO Vertical Area Detection", annotated_frame)

    # 'q' キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
print("処理を終了しました。")