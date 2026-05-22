# 期末專題進度報告

**題目：** 動態學習率排程之效能對比與特徵視覺化分析
**模型：** ResNet-18（ImageNet 預訓練）
**資料集：** Tiny-ImageNet-200（首階段，已完成）；Imagewoof（次階段，規劃中）
**日期：** 2026-05-22

---

## 1. 摘要 (Executive Summary)

本階段在 Tiny-ImageNet-200 上對五種學習率排程進行了系統對比，使用 ImageNet 預訓練的 ResNet-18 作為起點、AdamW 優化器、20 epoch、A100 GPU、batch=384、AMP+TF32。**主要發現：**

1. 任何衰減策略皆顯著優於 Fixed LR（**62.7% → 66–71%**），證實 LR 排程在 fine-tuning 情境仍重要。
2. **CosineAnnealingWarmRestarts** 以 **best_val_acc = 70.57%** 居冠，但末次重啟造成 final_val_acc 跌至 62.97%，凸顯**「best vs final」評估方式選擇的重要性**。
3. **StepLR** 與 **CosineAnnealingLR** 並列第二（69.97% / 69.40%），其中 StepLR 的 train-val gap 最小（28.2%），泛化最穩定。
4. **OneCycleLR** 預設峰值 LR 對 fine-tuning 過猛（peak=9e-3），導致 epoch 2–5 val_acc 暴跌，最終僅 66.42%。
5. 普遍存在過擬合（train_acc 94–100% vs val_acc 62–71%），歸因於 Tiny-ImageNet 每類僅 ~500 張且增強策略保守。下一階段引入 Imagewoof 與更強增強驗證此假設。

---

## 2. 實作摘要

### 2.1 程式碼結構

```
src/
├── data.py           Tiny-ImageNet 下載 + ImageFolder 結構轉換 + 224×224 transforms
├── model.py          ResNet-18 (200 類 head 替換) + Grad-CAM target layer 抽取
├── schedulers.py     5 種 scheduler factory + step_granularity 標記
├── train.py          訓練迴圈 + AMP + TF32 + per-step LR 紀錄 + 階段 checkpoint
├── profiles.py       t4 / a100 profile (batch, workers, AMP, TF32, lr_scale)
├── gradcam_viz.py    5 排程 × 早/中/晚期 對比熱力圖
├── plot_lr.py        各排程 LR-vs-step 曲線
└── plot_curves.py    train/val loss 與 accuracy 曲線
configs/              5 個 YAML：fixed / step / cosine / cosine_restart / onecycle
scripts/              CLI 入口（download_data / run_experiment / run_all）
notebooks/            colab_main.ipynb（Colab 一鍵執行）
```

### 2.2 關鍵設計決策

| 決策 | 內容 | 原因 |
|------|------|------|
| 影像 resize 至 224×224 | Tiny-ImageNet 原始 64×64 強制放大 | ResNet-18 末層 conv 輸出 7×7 特徵圖，利於 Grad-CAM |
| 採用 ImageNet pretrained | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` | 20 epoch 預算下從零訓練僅能達 40–50%；fine-tuning 可達 62–71% |
| `base_lr = 3e-4` | 所有 config 統一 | AdamW fine-tuning 合理區間 `1e-4 ~ 5e-4`；過大會破壞預訓練特徵 |
| OneCycle `max_lr = 3e-3` | base × 10 | 維持 OneCycle 設計理念（10× 峰值）同時控制在 fine-tuning 邊界 |
| Capture epoch = {1, 10, 20} | 對應「早 / 中 / 晚期」 | Grad-CAM 演進可視化的關鍵時間點 |
| AdamW + weight_decay 5e-4 | 預設 optimizer | 對 BN/位置編碼較寬容；fine-tuning 場景常見選擇 |
| `cudnn.benchmark = True` | 啟用 | 固定輸入尺寸下挑選最快 kernel |
| A100 profile lr_scale = 3.0 | batch 從 128→384，依線性 scaling rule | 維持「等效 LR」一致 |

### 2.3 公平性保證

- 5 個排程**共用相同 base_lr、weight_decay、batch_size、image_size、augmentation**
- 唯一變量為「LR 隨時間的形狀」
- 同一 GPU profile 內所有排程使用同一 AMP / TF32 設定
- 同一 seed (42)，同一 pretrained 權重起點

---

## 3. 實驗環境

| 項目 | 內容 |
|------|------|
| GPU | NVIDIA A100-SXM4-40GB（Colab） |
| Profile | `a100`：batch=384, workers=8, AMP=True, TF32=True, lr_scale=3.0 |
| 有效 base_lr | 3e-4 × 3.0 = **9e-4** |
| 有效 OneCycle max_lr | 3e-3 × 3.0 = **9e-3** |
| Epochs | 20 |
| Steps per epoch | 260 |
| Total steps | 5,200 |
| 每組訓練時間 | ~24.5 分鐘 |
| 5 組總時間 | ~2 小時 |

---

## 4. 量化結果

### 4.1 主要指標

| Scheduler | Best Val Acc | Final Val Acc | Final Train Acc | Train-Val Gap | 排名 |
|-----------|:------------:|:-------------:|:---------------:|:-------------:|:----:|
| `cosine_restart` | **70.57%** ⭐ | 62.97% | 93.80% | 30.8% | 1 (best) |
| `step` | 69.97% | **69.82%** | 98.03% | 28.2% | 2 |
| `cosine` | 69.40% | 69.32% | 99.92% | 30.6% | 3 |
| `onecycle` | 66.42% | 66.35% | 99.69% | 33.3% | 4 |
| `fixed` | 62.69% | 61.21% | 94.49% | 33.3% | 5 (baseline) |

數據來源：`results/tiny_imagenet/summary.json`、`results/tiny_imagenet/training_log.txt`。

### 4.2 LR 曲線（5 種排程形狀對比）

![LR curves](../results/tiny_imagenet/lr_curves.png)

- **Fixed**（綠）：全程 9e-4 平直
- **Step**（紫）：epoch {6, 12, 18} 各 ×0.1 衰減，三次斷崖
- **Cosine**（藍）：平滑餘弦下降，末期 ~1e-5
- **Cosine Restart**（橘）：T_0=5 起步，warm restarts 三次（epoch ~5, 15, ...）
- **OneCycle**（紅）：先升至 peak 9e-3（~step 1500，epoch 6），再餘弦下降至 ~1e-7

### 4.3 訓練曲線

![Train/Val curves](../results/tiny_imagenet/curves.png)

關鍵觀察：

- **OneCycle val_loss 在 epoch 2–5 出現峰值**（>3.0），對應 LR 升至 7e-3+ 階段。為「LR 過大破壞預訓練特徵」的直接證據。
- **Cosine Restart 的 val_acc 曲線出現週期性波動**，每次 restart 後需要 2–3 epoch 才能再次收斂超越前次高點。
- **Fixed LR 的 val_loss 末期反升**（~2.2），其他排程末期 val_loss 皆 < 1.9。證實末期無 LR 衰減會在最佳解附近震盪。
- **Train acc 最終分佈**：cosine 99.92%、onecycle 99.69%、step 98.03%、fixed 94.49%、cosine_restart 93.80%。後兩者較低反映「LR 從未真正降到極小」，模型未完全擬合訓練集。

### 4.4 Grad-CAM 質化分析

![Grad-CAM grid](../results/tiny_imagenet/grad_cam_grid.png)

行：5 種排程（cosine / cosine_restart / fixed / onecycle / step）
列：input image、checkpoint @ epoch 1 / 10 / 20

觀察：

- **所有排程在 epoch 1 即已聚焦於主體**（紅色物件中央），這是 ImageNet pretrained 起點的功勞。
- **隨訓練進行注意力略微收緊**，但差異不如「從零訓練」場景明顯。
- **Fixed 在 epoch 10、20 注意力範圍較大且擴散**，視覺呈現出「未完全鞏固」的特徵 — 對應其量化結果最差的事實。
- **OneCycle 在 epoch 10 注意力反而短暫散開**，符合該階段 LR 仍處於高位（~7e-3）造成特徵震盪的現象。

---

## 5. 主要結論

### 5.1 量化層面

> **任何 LR 衰減策略都顯著優於 Fixed LR**（+3.7% ~ +7.9%），證實 LR 排程在 fine-tuning 情境同樣不可或缺。

> **「Best」vs「Final」評估策略對 Cosine Restart 影響最大**（70.57% vs 62.97%，差距 7.6%）。報告與部署中應明確採用 best checkpoint 評估或加入 early stopping，否則 Cosine Restart 的排名會與其真實能力嚴重不符。

> **StepLR 在 fine-tuning 場景與更現代的 Cosine 系列打成平手**，且 train-val gap 最小。簡單方法在合適場景仍具競爭力。

> **OneCycleLR 預設超參數（peak = base × 10）對 fine-tuning 過猛**。建議在 pretrained backbone fine-tuning 時降至 `peak = base × 3 ~ 5`。

### 5.2 質化層面

> Grad-CAM 顯示 **pretrained 起點已具備合理的物件級注意力**，後續 LR 排程主要影響「注意力的鞏固速度」與「中後期穩定性」，而非「注意力是否形成」。

> Fixed LR 的注意力擴散與 OneCycle 中期的注意力震盪皆可在熱力圖中直接觀察到，**質化與量化結果一致**。

---

## 6. 限制與後續工作

### 6.1 限制

1. **過擬合普遍** — 5 組 final train-val gap 在 28–33% 之間，主因：
   - Tiny-ImageNet 每類僅 ~500 張
   - 增強策略僅 RandomCrop + Flip + ColorJitter，未使用 MixUp/CutMix/RandAugment
   - 20 epoch 預算下 train_loss 已收斂至 < 0.1，模型容量足以記住訓練集
2. **僅單一 seed (42)** — 未做多 seed 平均，數值差異可能含 ±0.5–1.0% 隨機浮動。
3. **CosineRestart 與 OneCycle 的超參數未調** — 採用文獻常見預設，可能不是這個 dataset/model 組合的最佳值。

### 6.2 後續工作（已規劃）

1. **Imagewoof 對比實驗** — 引入 10 類細粒度狗品種資料集（每類 ~950 張，原生 224×224）：
   - **更多每類樣本** → 預期降低 train-val gap
   - **天然 224×224** → Grad-CAM 視覺化更精細
   - **細粒度分類** → 注意力差異更明顯（耳形、口鼻、毛色等局部特徵）
   - 5 種排程結論的**跨資料集 robustness 驗證**
2. **可選：增強策略 ablation** — 在 Tiny-ImageNet 上以 cosine 為例新增 MixUp/CutMix 對比，驗證「過擬合是主因，非排程不夠好」的假設。

### 6.3 不在本專題範圍但值得備註

- ViT / EfficientNet 等更大模型在小資料集 fine-tuning 下，scheduler 影響可能更大或更小，本研究未涵蓋。
- 學習率 + weight decay + label smoothing 等正則化的交互作用未做網格搜尋。

---

## 7. 檔案索引

| 路徑 | 內容 |
|------|------|
| `results/tiny_imagenet/summary.json` | 5 組訓練的全部 epoch 級指標 + 配置 + 計時 |
| `results/tiny_imagenet/training_log.txt` | 完整訓練 stdout 紀錄（Colab 原始輸出） |
| `results/tiny_imagenet/lr_curves.png` | 5 種 scheduler 的 LR-vs-step 曲線 |
| `results/tiny_imagenet/curves.png` | 5 組 train/val loss 與 accuracy 對比 |
| `results/tiny_imagenet/grad_cam_grid.png` | 5 排程 × 早/中/晚期 Grad-CAM 對比 |
| `notebooks/colab_main.ipynb` | Colab 一鍵執行入口（含本次跑出的 outputs） |
| `configs/*.yaml` | 5 個 scheduler 的訓練設定 |

---

*本報告對應 git commit hash 將寫入下一個 commit。後續 Imagewoof 階段完成後將更新本檔案、追加 `results/imagewoof/` 與「跨資料集對比」章節。*
