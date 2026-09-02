import cv2
import time
import oracledb
import threading
import os
from ultralytics import YOLO

# FFMPEGの低遅延・バッファ無効化設定
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;0"

# 別スレッドで最新フレームを取得するクラス
class ThreadedRTSPStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # フレーム読み取り専用のスレッドを起動
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                continue
            # 常にバッファを消費し、最新フレームを保持し続ける
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

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

# DBステータス更新関数
def update_status_in_db(status):
    try:
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
model = YOLO(r"C:\YOLO\YOLO\runs\detect\train-14\weights\best_openvino_model/")

SOURCE = "rtsp://ebara:Ebara1234@192.168.0.10/Src/MediaInput/stream_1"
TARGET_LABEL = "object"
AREA_LABEL = "area"
CONF_THRESHOLD = 0.9
AREA_CONF_THRESHOLD = 0.5
REQUIRED_TIME = 3.0

# カメラの別スレッド読み込みを開始
stream = ThreadedRTSPStream(SOURCE).start()

# ウィンドウの設定
cv2.namedWindow("YOLO Vertical Area Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Vertical Area Detection", 1024, 576) # 表示したい（横, 縦）サイズに調整

# タイマーと状態管理変数
in_area_start_time = None
current_status = 0
prev_status = 0  

LOST_TOLERANCE = 0.5  # 検知が外れても0.5秒間はタイマーを維持
last_seen_time = None

print("処理を開始します。停止するには画面上で 'q' キーを押してください。")

# メインループ推論・判定処理
while True:
    # 別スレッドから最新の1フレームだけを取得
    success, frame = stream.read()
    
    if not success or frame is None:
        # パケット落ちやデコードエラー時はそのフレームをスキップして続行
        time.sleep(0.01) # CPU負荷軽減のための微小ウェイト
        continue

    # 推論処理
    results = model.predict(
        source=frame,
        device="CPU",   # 低遅延重視ならCPU、CPU負担軽減ならGPU
        verbose=False
    )

    # デバッグ用（検知結果のリアルタイム確認）
    # for box in results[0].boxes:
    #     cls_id = int(box.cls[0].item())
    #     label = model.names[cls_id]
    #     conf = box.conf[0].item()
    #     print(f"[検知] Label: {label}, Conf: {conf:.2f}")
    
    # 検出判定
    area_box = None
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        label_name = model.names[cls_id]
        conf = box.conf[0].item()

        if label_name == AREA_LABEL and conf >= AREA_CONF_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_box = (int(x1), int(y1), int(x2), int(y2))
            break

    target_in_area = False 

    if area_box is not None:
        area_x1, area_y1, area_x2, area_y2 = area_box

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label_name = model.names[cls_id]
            conf = box.conf[0].item()

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
        last_seen_time = now
        if in_area_start_time is None:
            in_area_start_time = now
        elapsed_time = now - in_area_start_time
        current_status = 1 if elapsed_time >= REQUIRED_TIME else 0
    else:
    # 最後に検知してから 0.5 秒以内ならタイマーをリセットせずに保持
        if last_seen_time is not None and (now - last_seen_time) < LOST_TOLERANCE:
            if in_area_start_time is not None:
                elapsed_time = now - in_area_start_time
                current_status = 1 if elapsed_time >= REQUIRED_TIME else 0
        else:
        # 0.5秒を超えて見失った場合のみ完全リセット
            in_area_start_time = None
            last_seen_time = None
            current_status = 0

    # ステータスが切り替わった「瞬間」だけ DB に書き込む
    if current_status != prev_status:
        update_status_in_db(current_status)
        prev_status = current_status  # 最新状態に更新

    # 画面描画処理
    annotated_frame = results[0].plot()

    if area_box is not None:
        area_x1, area_y1, area_x2, area_y2 = area_box
        if current_status == 1:
            color = (0, 255, 0)  # 緑 (3秒達成 & DB更新済み)
            status_text = "STATUS: 1 (OK)"
        elif target_in_area:
            color = (0, 255, 255)  # 黄 (エリア内でカウント中)
            status_text = f"STATUS: 0 (Counting: {elapsed_time:.1f}s)"
        else:
            color = (0, 0, 255)  # 赤 (エリア内に対象なし)
            status_text = "STATUS: 0 (No Target)"

        # 検知したエリア枠を強調描画
        cv2.rectangle(annotated_frame, (area_x1, area_y1), (area_x2, area_y2), color, 3)
        cv2.putText(
            annotated_frame,
            f"{status_text} ({TARGET_LABEL} in {AREA_LABEL})",
            (area_x1+300, area_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    else:
        # エリア自体がカメラ映像内に見つからない場合
        cv2.putText(
            annotated_frame,
            f"Area '{AREA_LABEL}' Not Found",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("YOLO Vertical Area Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 後処理
stream.stop()  # 別スレッドのカメラ読み込みを停止
cv2.destroyAllWindows()

if 'cursor' in locals():
    cursor.close()
if 'conn' in locals():
    conn.close()
    print("Oracle DBとの接続を切断しました。")
print("処理を終了しました。")