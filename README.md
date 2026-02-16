# tenis_backview

`gastonarielfrancois/tenis-backview` を使って YOLO をファインチューニングするプロジェクトです。

## 目的

- `player` と `ball` を別々のモデルとして学習
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

Hydra設定を使って生成する場合:

```bash
uv run tenis-backview-prepare-dataset
```

上書き例:

```bash
uv run tenis-backview-prepare-dataset frame_stride=2
```

生成物:

- `data/tenis-backview/yolo/player/{images,labels,...}`
- `data/tenis-backview/yolo/ball/{images,labels,...}`
- `data/tenis-backview/yolo/player/dataset.yaml`
- `data/tenis-backview/yolo/ball/dataset.yaml`

## 2. 学習（別々に実行）

### プレイヤー学習

```bash
uv run tenis-backview --config-name train_player
```

### ボール学習（ベースモデル: `checkpoints/ball/model.pt`）

```bash
uv run tenis-backview --config-name train_ball
```

設定は以下で管理します。

- `src/tenis_backview/configs/train_player.yaml`
- `src/tenis_backview/configs/train_ball.yaml`

重み出力例:

- `outputs/train/player/test_run_player/weights/best.pt`
- `outputs/train/ball/test_run_ball/weights/best.pt`

## 3. 可視化

```bash
uv run tenis-backview-visualize
```

上書き例:

```bash
uv run tenis-backview-visualize model=outputs/train/ball/test_run_ball/weights/best.pt
```

## 補足

- `outputs/` と `checkpoints/` は Git 管理対象外です。
- `training/lightning.py` は Lightning を学習環境として利用し、実学習は Ultralytics YOLO の学習APIを呼び出します。
- 学習時にはバッチ損失をトレースし、`first_batch_loss` と `last_batch_loss` を出力します。
