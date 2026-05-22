# ML_LR_comparison

機器學習期末專題：動態學習率排程之效能對比與特徵視覺化分析。

以 **ResNet-18** 在 **Tiny-ImageNet (200 類)** 上訓練影像分類器，對比 5 種學習率排程的收斂行為與泛化能力，並以 **Grad-CAM** 視覺化各排程在訓練早 / 中 / 晚期的特徵聚焦演進。

## 進度與結果

| 階段 | 狀態 | 產出 |
|------|------|------|
| Phase 1 — Tiny-ImageNet 訓練 + 視覺化 | ✅ 完成 | [`results/tiny_imagenet/`](results/tiny_imagenet/) |
| Phase 2 — Imagewoof 對比實驗 | 🚧 規劃中 | — |
| 詳細進度報告 | ✅ 完成 | [`docs/REPORT.md`](docs/REPORT.md) |

**Tiny-ImageNet 主要結果**（ResNet-18 pretrained、20 epoch、A100、batch=384）：

| Scheduler | Best Val Acc | Final Val Acc | Final Train Acc |
|-----------|:------------:|:-------------:|:---------------:|
| `cosine_restart` | **70.57%** | 62.97% | 93.80% |
| `step` | 69.97% | **69.82%** | 98.03% |
| `cosine` | 69.40% | 69.32% | 99.92% |
| `onecycle` | 66.42% | 66.35% | 99.69% |
| `fixed` | 62.69% | 61.21% | 94.49% |

完整分析見 [`docs/REPORT.md`](docs/REPORT.md)。

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
│   ├── train.py         # 訓練 / 驗證 / checkpoint (AMP + TF32)
│   ├── profiles.py      # t4 / a100 GPU profile
│   ├── gradcam_viz.py   # Grad-CAM 對比圖
│   ├── plot_lr.py       # LR 曲線繪圖
│   ├── plot_curves.py   # train/val 曲線繪圖
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   ├── run_experiment.py
│   └── run_all.py
├── configs/             # 5 個 YAML 實驗設定
├── notebooks/
│   └── colab_main.ipynb # Colab 入口（GPU 自動偵測）
├── results/             # 已封存的訓練結果（commit 入庫）
│   └── tiny_imagenet/
│       ├── summary.json       # 5 組訓練的全部 epoch 級指標
│       ├── training_log.txt   # 完整 stdout 紀錄
│       ├── lr_curves.png
│       ├── curves.png
│       └── grad_cam_grid.png
├── docs/
│   └── REPORT.md        # 詳細進度報告
└── experiments/         # 當下訓練輸出 (gitignored，會寫到 Drive)
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
- Backbone 採 **ImageNet 預訓練權重** (`torchvision.models.ResNet18_Weights.IMAGENET1K_V1`) 作為起點
- 各排程在 `configs/*.yaml` 中共用同一 `base_lr` 確保對比公平
- Checkpoints 在訓練前 / 中 / 晚期各存一份，供 Grad-CAM 對比使用

## Base LR 選擇說明

所有 config 統一採用 `base_lr = 3e-4`（OneCycle 的 `max_lr = 3e-3`），原因如下：

1. **適配 fine-tuning，而非從零訓練。** 既然 backbone 已是 ImageNet 預訓練權重，目標
   是「微調」而非「重新學表徵」。文獻與業界經驗指出，AdamW fine-tuning 的合理區間
   為 `1e-4 ~ 5e-4`；過大的 LR (如 1e-3) 在前 1–2 epoch 就會破壞預訓練特徵，反而
   讓 val accuracy 倒退到接近從零訓練的水準，喪失使用 pretrained 的意義。
2. **避免 OneCycle 在 warmup 階段直接打爆預訓練特徵。** OneCycle 的峰值 LR 通常設
   為 base 的 10–30 倍 (本專題取 10 倍 → `max_lr = 3e-3`)。若 base 仍是 1e-3，峰值
   會衝到 1e-2，等同於從零訓練的學習率，與「微調」精神矛盾。
3. **保留排程之間的相對行為差異。** 5 種排程共用同一 `base_lr` 才能讓「衰減形狀」
   成為唯一變量：Fixed 全程平直、Step 兩次斷崖、Cosine 平滑下行、CosineRestart 帶
   重啟、OneCycle 先升後降。如果為了「衝高絕對精度」而對個別排程調整 LR，對比結
   論將失去說服力。
4. **與 A100 profile 的線性 scaling rule 相容。** A100 profile 把 batch 從 128 拉到
   384 (3×)，並按線性 scaling rule 把 `base_lr × 3 = 9e-4`，仍落在合理 fine-tuning
   區間內；若 base 是 1e-3，A100 上的有效 LR 會是 3e-3，明顯偏高。
