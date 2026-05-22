# ML_LR_comparison

機器學習期末專題：動態學習率排程之效能對比與特徵視覺化分析。

以 **ResNet-18** 在 **Tiny-ImageNet (200 類)** 上訓練影像分類器，對比 5 種學習率排程的收斂行為與泛化能力，並以 **Grad-CAM** 視覺化各排程在訓練早 / 中 / 晚期的特徵聚焦演進。

## 排程對照組

| ID | Scheduler | 重點 |
| -- | --------- | ---- |
| `fixed` | Fixed LR | 對照組，全程不衰減 |
| `step` | StepLR | 每 N epoch 斷崖式衰減 |
| `cosine` | CosineAnnealingLR | 餘弦平滑退火 |
| `cosine_restart` | CosineAnnealingWarmRestarts | 帶重啟的餘弦退火 |
| `onecycle` | OneCycleLR | 先升後降的單週期 |

## 專案結構

```
.
├── src/
│   ├── data.py          # Tiny-ImageNet 下載 + DataLoader
│   ├── model.py         # ResNet-18 (200 類)
│   ├── schedulers.py    # 5 種 scheduler factory
│   ├── train.py         # 訓練 / 驗證 / checkpoint
│   ├── gradcam_viz.py   # Grad-CAM 對比圖
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   ├── run_experiment.py
│   └── run_all.py
├── configs/             # 5 個 YAML 實驗設定
├── notebooks/
│   └── colab_main.ipynb # Colab 入口
└── experiments/         # 訓練輸出 (gitignored)
```

## 快速開始 (Colab)

打開 `notebooks/colab_main.ipynb`，依序執行：

1. 掛載 Google Drive
2. `git clone` 本 repo
3. `pip install -r requirements.txt`
4. 下載 Tiny-ImageNet（首次約 3–5 分鐘）
5. 跑 5 組實驗（T4 GPU 上每組約 20–40 分鐘，視 epoch 數而定）
6. 產出 Grad-CAM 對比圖

## 本地執行（單一實驗）

```bash
python -m scripts.download_data --data-dir ./data
python -m scripts.run_experiment --config configs/cosine.yaml
```

跑完所有 5 組：

```bash
python -m scripts.run_all --profile t4    # T4 / V100 (預設參數)
python -m scripts.run_all --profile a100  # A100：大 batch + AMP + TF32 + 線性 LR scaling
```

`colab_main.ipynb` 會自動偵測 GPU 並選對應 profile，本地執行才需手動指定。

## 視覺化

訓練過程：
```bash
tensorboard --logdir experiments/
```

Grad-CAM 對比圖：
```bash
python -m src.gradcam_viz --experiments-dir experiments --out grad_cam_grid.png
```

## 設計細節

- 影像 resize 至 **224×224**（ResNet-18 最後一層 conv → 7×7 特徵圖，利於 Grad-CAM）
- 預設 optimizer: **AdamW**，weight decay 5e-4
- 各排程在 `configs/*.yaml` 中可獨立調整 base LR / epochs
- Checkpoints 在訓練前 / 中 / 晚期各存一份，供 Grad-CAM 對比使用
