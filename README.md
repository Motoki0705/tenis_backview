# tenis_backview

`gastonarielfrancois/tenis-backview` を使って YOLO をファインチューニングするプロジェクトです。

## 目的

- クラスは `player` と `ball` のみで学習
- `ball` はデータセットの中心点から擬似BBoxを作成
- `player` はデータセットの中心点に対し、YOLO人物検出BBoxの中から対応BBoxを選んで教師信号化

この手法で、人物全般ではなく「プレイヤー」に限定される方向へ学習させます。

## セットアップ

```bash
uv sync
```

## ディレクトリ構成

```text
src/tenis_backview/
├── configs/
├── data/
│   ├── dataset.py
│   ├── datamodule.py
│   └── type.py
├── generate_dataset/
├── inference/
│   └── yolo_inf.py
├── models/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── visualize.py
├── training/
│   └── lightning.py
└── visualization/
```

## 1. データセット生成（`data/tenis-backview`）

```python
import kagglehub

path = kagglehub.dataset_download("gastonarielfrancois/tenis-backview")
print("Path to dataset files:", path)
```

CLIで生成する場合:

```bash
uv run tenis-backview-prepare-dataset \
	--dataset-ref gastonarielfrancois/tenis-backview \
	--dataset-dir data/tenis-backview \
	--output-dir data/tenis-backview/yolo \
	--person-model checkpoints/player/model.pt
```

生成物:

- `data/tenis-backview/yolo/images/{train,val,test}`
- `data/tenis-backview/yolo/labels/{train,val,test}`
- `data/tenis-backview/yolo/dataset.yaml`

## 2. 学習

```bash
uv run tenis-backview \
	--prepare-dataset \
	--dataset-ref gastonarielfrancois/tenis-backview \
	--dataset-dir data/tenis-backview \
	--data-yolo-dir data/tenis-backview/yolo \
	--person-model checkpoints/player/model.pt \
	--base-model yolo11n.pt \
	--epochs 30 \
	--batch 16 \
	--imgsz 960 \
	--output-dir outputs/train \
	--run-name player_ball_finetune
```

重み出力例:

- `outputs/train/player_ball_finetune/weights/best.pt`

## 3. 可視化

```bash
uv run tenis-backview-visualize \
	--model outputs/train/player_ball_finetune/weights/best.pt \
	--video data/sample.mp4 \
	--output outputs/visualize/sample_overlay.mp4
```

## 補足

- `outputs/` と `checkpoints/` は Git 管理対象外です。
- `training/lightning.py` は Lightning を学習環境として利用し、実学習は Ultralytics YOLO の学習APIを呼び出します。
