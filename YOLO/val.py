#精度表示用
import csv

# 学習結果が保存されているCSVファイル
csv_path = r"C:\YOLO\YOLO\runs\detect\train-2\results.csv"

try:
    with open(csv_path, mode="r", encoding="utf-8") as f:
        # CSVの各行を辞書形式で読み込む
        reader = csv.DictReader(f)
        
        # 列名（ヘッダー）の余計な空白を削除してクリーンアップ
        reader.fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
        
        # 最後の行（最終エポック）までループして取得
        last_row = None
        for row in reader:
            last_row = row

    if last_row:
        # 各列の値を取得（前後の空白を削除し、数値に変換）
        def get_float(key):
            val = last_row.get(key, "0")
            return float(val.strip()) if val else 0.0

        print("=== YOLO 最終学習結果（精度 & Loss同時出力） ===")
        # 1. 精度
        print(f"mAP50-95 (総合精度)     : {get_float('metrics/mAP50-95(B)'):.4f}")  #0.45 ～ 0.55
        print(f"mAP50 (位置ズレ許容精度): {get_float('metrics/mAP50(B)'):.4f}")      #0.65 ～ 0.80
        print(f"Precision (適合率)      : {get_float('metrics/precision(B)'):.4f}") #0.75 ～ 0.85
        print(f"Recall (再現率)         : {get_float('metrics/recall(B)'):.4f}")    #0.70 ～ 0.80
        print("-" * 45)
        # 2. 検証データ（val）に対する損失値（Loss）
        print(f"Box Loss (位置の損失)   : {get_float('val/box_loss'):.4f}") #0.5 〜 1.5
        print(f"Class Loss (分類の損失) : {get_float('val/cls_loss'):.4f}") #0.01 〜 0.5 以下
        print(f"DFL Loss (境界の損失)   : {get_float('val/dfl_loss'):.4f}") #0.8 〜 1.2 前後
        print("================================================")
    else:
        print(f"エラー: {csv_path} の中身が空です。")

except FileNotFoundError:
    print(f"エラー: {csv_path} が見つかりません。")
