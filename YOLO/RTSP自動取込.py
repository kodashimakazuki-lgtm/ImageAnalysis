import cv2
import datetime
import time
import os

def record_rtsp_stream(rtsp_url, save_dir, file_prefix, split_minutes):
    # 保存先ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"ディレクトリを作成しました: {save_dir}")

    # ストリームの開始
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("エラー: RTSPストリームを開けませんでした。")
        return

    # オリジナルの解像度とFPSを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30.0
    
    print(f"解析完了: {width}x{height}, {fps}FPS で録画を開始します。")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    last_split_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ストリーム中断。再接続を試みます...")
                time.sleep(5)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            now = datetime.datetime.now()

            # ファイル分割の判定 (初回または指定時間経過)
            if out is None or (now - last_split_time).total_seconds() >= split_minutes * 60:
                if out is not None:
                    out.release()

                # パスとファイル名の結合
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                filename = f"{file_prefix}_{timestamp}.mp4"
                save_path = os.path.join(save_dir, filename) # フルパスを作成
                
                out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                last_split_time = now
                print(f"録画中: {save_path}")

            out.write(frame)

    except KeyboardInterrupt:
        print("/nユーザーにより停止されました。")
    finally:
        if out is not None:
            out.release()
        cap.release()

if __name__ == "__main__":
    # --- 設定項目 ---
    RTSP_URL = "rtsp://ebara:Ebara1234@192.168.0.10/Src/MediaInput/stream_1"
    SAVE_DIRECTORY = r"C:\YOLO\YOLO\runs\detect" # ← ここに保存先のパスを指定（相対・絶対パス両対応）
    FILE_PREFIX = "Camera01"
    SPLIT_MIN = 1 
    # ----------------
    # ctrl+cで終了

    record_rtsp_stream(RTSP_URL, SAVE_DIRECTORY, FILE_PREFIX, SPLIT_MIN)