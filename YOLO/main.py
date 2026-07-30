#画像検知,データベース更新用
import cv2
import time
import oracledb
from ultralytics import YOLO

# Oracle 接続設定
DB_USER = "TSM_MGR_ST1"
DB_PASSWORD = "abc1_def"
DB_HOST = "devexavm01-vi1jg1-vip.ebara.com"
DB_PORT = "1521"
DB_SERVICE_NAME = "TEXP0200.sqlc.dev.oraclevcn.com"

dsn = f"{DB_HOST}:{DB_PORT}/{DB_SERVICE_NAME}"

try:
    conn = oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=dsn)
    cursor = conn.cursor()
    print("Oracle Database への接続に成功しました。")
except Exception as e:
    print(f"DB接続エラー: {e}")
    exit()

# DBステータス更新
def update_status_in_db(status):
    try:
        # ※もし特定の行だけ更新したい場合は WHERE 句を追加してください
        sql = """
            UPDATE IMAGE_ANALYSIS 
            SET STATUS = :1, UPDATE_TIME = CURRENT_TIMESTAMP 
        """
        cursor.execute(sql, [status])
        conn.commit()
        print(f"[Oracle UPDATE] STATUS を {status} に更新しました。")
    except Exception as e:
        print(f"UPDATEエラー: {e}")

# YOLO 動画設定
model = YOLO(r"C:\YOLO\YOLO\best.pt")

SOURCE = "rtsp://ebara:Ebara1234@192.168.0.10/Src/MediaInput/stream_1"
TARGET_LABEL = "bottle"
CONF_THRESHOLD = 0.9
REQUIRED_TIME = 3.0

AREA_RATIO_W = 0.3
AREA_RATIO_H = 0.8

cap = cv2.VideoCapture(SOURCE)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

area_w = int(width * AREA_RATIO_W)
area_h = int(height * AREA_RATIO_H)
area_x1 = int((width - area_w) / 2)
area_y1 = int((height - area_h) / 2)
area_x2 = area_x1 + area_w
area_y2 = area_y1 + area_h

# タイマーと状態管理変数
in_area_start_time = None
current_status = 0
prev_status = 0  # 前回のステータスを保持（連打防止用）

print(f"動画サイズ: {width}x{height}")
print(f"縦長エリア座標: [{area_x1}, {area_y1}, {area_x2}, {area_y2}] (幅:{area_w}px, 高さ:{area_h}px)")
print("処理を開始します。停止するには画面上で 'q' キーを押してください。")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("動画の再生が終了したか、読み込めませんでした。")
        break

    results = model(frame, verbose=False)
    target_in_area = False 

    # 検出判定
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  #クラスId
        label_name = model.names[cls_id] #クラス名
        conf = box.conf[0].item()        #信頼度

        if label_name == TARGET_LABEL and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if area_x1 <= center_x <= area_x2 and area_y1 <= center_y <= area_y2:
                target_in_area = True
                break

    # 時間計測 & ステータス決定
    now = time.time()
    elapsed_time = 0.0

    if target_in_area:
        if in_area_start_time is None:
            in_area_start_time = now

        elapsed_time = now - in_area_start_time

        # 3秒以上滞在で status = 1、それ未満は 0
        current_status = 1 if elapsed_time >= REQUIRED_TIME else 0
    else:
        in_area_start_time = None
        current_status = 0

    #ステータスが切り替わった「瞬間」だけ DB に書き込む
    if current_status != prev_status:
        update_status_in_db(current_status)
        prev_status = current_status  # 最新状態に更新

    # --- 画面描画処理 ---
    annotated_frame = results[0].plot()

    if current_status == 1:
        color = (0, 255, 0)  # 緑 (3秒達成 & DB更新済み)
        status_text = f"STATUS: 1 (OK)"
    elif target_in_area:
        color = (0, 255, 255)  # 黄 (エリア内でカウント中)
        status_text = f"STATUS: 0 (Counting)"
    else:
        color = (0, 0, 255)  # 赤 (エリア外)
        status_text = "STATUS: 0 (No Target)"

    cv2.rectangle(annotated_frame, (area_x1, area_y1), (area_x2, area_y2), color, 3)
    cv2.putText(
        annotated_frame,
        f"{status_text} ({TARGET_LABEL} >= {CONF_THRESHOLD})",
        (area_x1, area_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.imshow("YOLO Vertical Area Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
if 'cursor' in locals():
    cursor.close()
if 'conn' in locals():
    conn.close()
    print("Oracle DBとの接続を切断しました。")
print("処理を終了しました。")