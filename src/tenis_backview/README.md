# Package Structure

- `configs/`: 設定値
- `data/`: dataset.py / datamodule.py / type.py
- `generate_dataset/`: KaggleデータをYOLO学習データへ変換
- `inference/`: `yolo_inf.py`
- `models/`: 必要時に学習対象アーキテクチャを配置
- `scripts/`: CLIエントリ (`prepare_dataset.py`, `train.py`, `visualize.py`)
- `training/`: `lightning.py` に学習環境
- `visualization/`: 推論結果オーバーレイ
