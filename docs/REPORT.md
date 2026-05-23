# 期末專題進度報告

**題目：** 動態學習率排程之效能對比與特徵視覺化分析
**模型：** ResNet-18
**資料集：** Tiny-ImageNet-200（200 類，pretrained）+ Imagewoof（10 類細粒度狗品種，pretrained + **from-scratch ablation**）
**日期：** 2026-05-23

---

## 1. 摘要 (Executive Summary)

本研究在兩個影像分類資料集上對五種學習率排程進行了系統對比，使用 ImageNet 預訓練的 ResNet-18 作為起點、AdamW 優化器、20 epoch、A100 GPU、batch=384、AMP+TF32。

**Tiny-ImageNet-200 主要發現：**

1. 任何衰減策略皆顯著優於 Fixed LR（**62.7% → 66–71%**），證實 LR 排程在 fine-tuning 情境仍重要。
2. **CosineAnnealingWarmRestarts** 以 **best_val_acc = 70.57%** 居冠，但末次重啟造成 final_val_acc 跌至 62.97%，凸顯**「best vs final」評估方式選擇的重要性**。
3. **StepLR** 與 **CosineAnnealingLR** 並列第二（69.97% / 69.40%），其中 StepLR 的 train-val gap 最小（28.2%），泛化最穩定。
4. **OneCycleLR** 預設峰值 LR 對 fine-tuning 過猛（peak=9e-3），導致 epoch 2–5 val_acc 暴跌，最終僅 66.42%。
5. 普遍存在過擬合（train_acc 94–100% vs val_acc 62–71%），歸因於 Tiny-ImageNet 每類僅 ~500 張且增強策略保守。

**Imagewoof 主要發現（後續引入，10 類細粒度狗品種）：**

1. **絕對精度大幅躍升至 86–92%**，驗證了 Tiny-ImageNet 的天花板來自資料而非排程設計。
2. **Cosine 與 CosineRestart 並列第一**（best=91.86%），但 Cosine 在 final 也維持 91.86% 而 CosineRestart 跌至 84.25%。「**末次重啟拖累 final**」此現象在兩個資料集上**重現**，是穩定的設計缺陷。
3. **Train-val gap 從 28–33% 縮減到 8–15%**，證實過擬合主因為「每類樣本不足」而非「排程或模型」。
4. **5 種排程的相對排名跨資料集穩定**：cosine_restart / cosine / step ≈ 並列冠軍，onecycle 第四，fixed 殿底。結論具備**跨資料集 robustness**。
5. 訓練時間 ~5 分鐘 / 組（vs Tiny-ImageNet ~25 分鐘 / 組），Imagewoof 適合作為 scheduler 快速 ablation 平台。

**Imagewoof from-scratch ablation 主要發現（80 epoch + MixUp + RandAugment + label smoothing）：**

1. **排程排名出現重大反轉** — OneCycle 從 pretrained 第 4 名躍升為 **from-scratch 第 1 名**（86.71%），Step 從第 2 跌到第 5（79.84%）。原本「scheduler 排名跨資料集穩定」的結論**不適用於跨訓練狀態 (pretrained vs from-scratch)**。
2. **OneCycle 的設計與 from-scratch 場景天然相符** — Smith 原論文即在 from-scratch 設定下提出 super-convergence；warmup 階段對隨機初始化權重的快速「探索」與後續 decay 的「精煉」配對得宜。
3. **86.71% best val_acc 接近 pretrained 的 88.60% (OneCycle 同類比較)**，差距僅 1.89 pt；對 cosine 而言則差 7.74 pt。這顯示 **from-scratch 與 pretrained 的差距大小本身會隨 scheduler 而變**。
4. **CosineRestart 末次重啟災難在 from-scratch 仍重現**（best 82.13 → final 76.13，跌 -6.00 pt），跨三個實驗組（Tiny-IN / Imagewoof pretrained / Imagewoof scratch）穩定存在。
5. **Grad-CAM 演進真正可見** — 不像 pretrained 在 ep1 已具備物件級注意力，from-scratch 模型在 ep5 仍呈散亂雜訊，至 ep40 開始聚焦，ep80 才穩定鎖定狗臉，呈現教科書式的特徵聚焦演進。

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
| Steps per epoch | 260 (Tiny-ImageNet) / 23 (Imagewoof) |
| 每組訓練時間 | ~24.5 分 (Tiny-ImageNet) / ~4.7 分 (Imagewoof) |
| 5 組總時間 | ~2 小時 (Tiny-ImageNet) / ~24 分鐘 (Imagewoof) |

---

## 4. Tiny-ImageNet 結果

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
列：input image (Tiny-ImageNet 驗證集 `val_0.JPEG`，包含人物與物件的戶外場景)、checkpoint @ epoch 1 / 10 / 20

觀察：

- **所有排程在 epoch 1 已大致聚焦於畫面中下方的主體區域**，這是 ImageNet pretrained 起點的功勞 — 模型無需從零學習「物件在哪裡」。
- **CosineRestart 的末次重啟在 ep020 造成注意力明顯偏移**：ep001 / ep010 聚焦於中下方主體，ep020 突然漂移到右上方（白衣人物），與量化結果 best→final 跌 7.6 pt 完全對應，**是「末次重啟破壞收斂」最直接的視覺證據**。
- **Fixed 在 ep020 注意力區域明顯位移**（從 ep010 的中央到 ep020 的右下角），呈現「無 LR 衰減 → 在最佳解附近震盪 → 末期注意力不穩定」的視覺特徵。
- **Cosine 與 Step 從 ep001 到 ep020 注意力區域持續收緊但保持在同一語義位置**，視覺上最穩定，呼應其 best=final 的量化結果。
- **OneCycle 在 ep001 就已較其他排程明顯聚焦**（紅色區域最小最集中），這是 warmup 階段較強 LR 加速收斂的痕跡；但因 base_lr 偏高（peak=9e-3），後續注意力未能持續精煉，停留在類似 epoch 1 的範圍。
- **質化與量化結果一致**：注意力穩定 (cosine/step) 對應較佳 final acc；注意力末期偏移 (cosine_restart/fixed) 對應 best-final gap。

---

## 5. Imagewoof 結果

### 5.1 Imagewoof 環境與設定

| 項目 | 內容 |
|------|------|
| 資料集 | Imagewoof2-320（fast-ai release）— 10 類細粒度狗品種 |
| 訓練集 | ~9.0k 張（~900 / 類） |
| 驗證集 | ~3.9k 張 |
| 原生解析度 | 320×320（已 resize 至 224×224） |
| Steps per epoch | 23（batch=384） |
| 5 組總訓練時間 | ~24 分鐘（單組 ~4.7 分鐘） |
| 其餘設定 | 同 Tiny-ImageNet（base_lr=3e-4、AdamW、AMP+TF32、20 epoch） |

### 5.2 量化結果

| Scheduler | Best Val Acc | Final Val Acc | Final Train Acc | Train-Val Gap | 排名 |
|-----------|:------------:|:-------------:|:---------------:|:-------------:|:----:|
| `cosine` | **91.86%** ⭐ | **91.86%** | 99.94% | 8.08% | 1 (tied) |
| `cosine_restart` | **91.86%** ⭐ | 84.25% | 96.04% | 11.79% | 1 (tied, best only) |
| `step` | 91.50% | 91.45% | 99.72% | 8.27% | 3 |
| `onecycle` | 88.60% | 88.60% | 98.87% | 10.27% | 4 |
| `fixed` | 86.36% | 82.67% | 98.01% | 15.34% | 5 (baseline) |

數據來源：`results/imagewoof/summary.json`、`results/imagewoof/training_log.txt`。

### 5.3 LR 曲線（5 種排程形狀對比）

![LR curves](../results/imagewoof/lr_curves.png)

LR 曲線形狀與 Tiny-ImageNet 相同（同一 scheduler、同一 base_lr），唯一差別是 x 軸 step 數較少（23 × 20 = 460 vs 260 × 20 = 5200），因此**形狀更壓縮但本質一致**。

### 5.4 訓練曲線

![Train/Val curves](../results/imagewoof/curves.png)

關鍵觀察：

- **OneCycle val_loss 在 epoch 2–5 飆升至 ~5**（紅線），對應 LR 升至 7e-3+ 階段，**完美重現 Tiny-ImageNet 上的同一現象**。
- **Cosine（藍）與 Step（紫）幾乎完全重疊** — 在足夠資料量下，「衰減形狀」對最終結果的影響弱於「是否衰減」。
- **CosineRestart 末次重啟（epoch ~13–18）造成 val_loss 反彈**，再次驗證末次重啟的設計缺陷。
- **Fixed LR 末期 val_loss > 0.7**，其他排程末期 val_loss 在 0.3–0.6 區間，與 Tiny-ImageNet 觀察一致。

### 5.5 Grad-CAM 質化分析

![Grad-CAM grid](../results/imagewoof/grad_cam_grid.png)

行：5 種排程（cosine / cosine_restart / fixed / onecycle / step）
列：input image (一隻 Shih-Tzu 小狗)、checkpoint @ epoch 1 / 10 / 20

觀察：

- **所有排程在 epoch 1 即聚焦於狗臉中央**，pretrained 起點優勢明顯。
- **隨訓練進行注意力區域明顯收緊至口鼻周圍**（細粒度分類關鍵特徵）— 比 Tiny-ImageNet 場景更明顯，因為 Imagewoof 任務需要區分相近狗種，模型必須鎖定面部特徵。
- **Cosine / CosineRestart / Step 三者注意力範圍最聚焦**，符合其量化最高的事實。
- **Fixed 在 epoch 10、20 注意力略微擴散至毛色區域**，可能是無 LR 衰減造成特徵未鞏固的視覺呈現。
- **質化結果與量化排名一致**，互相佐證。

---

## 6. 跨資料集對比

### 6.1 主要指標跨資料集對照

| Scheduler | Tiny-IN Best | Tiny-IN Final | Imagewoof Best | Imagewoof Final |
|-----------|:------------:|:-------------:|:--------------:|:---------------:|
| `cosine_restart` | **70.57** | 62.97 | **91.86** | 84.25 |
| `step` | 69.97 | **69.82** | 91.50 | 91.45 |
| `cosine` | 69.40 | 69.32 | **91.86** | **91.86** |
| `onecycle` | 66.42 | 66.35 | 88.60 | 88.60 |
| `fixed` | 62.69 | 61.21 | 86.36 | 82.67 |

### 6.2 跨資料集穩定結論

1. **相對排名一致**：cosine 系列 + step 並列冠軍，onecycle 第四，fixed 殿底。在兩個差異甚大的資料集（200 類 vs 10 類；通用物件 vs 細粒度狗）上呈現相同排名 → **結論具備 robustness**。

2. **「末次重啟拖累 final」是 CosineRestart 的系統性問題**，**不是隨機現象**：
   - Tiny-ImageNet: best 70.57 → final 62.97（**−7.60 pt**）
   - Imagewoof: best 91.86 → final 84.25（**−7.61 pt**）
   - 兩者跌幅高度一致（±0.01 pt），證實 T_0=epochs/4、T_mult=2 的設定會在訓練末期觸發一次破壞性的重啟。

3. **OneCycle 在兩個資料集都 underperform**，差距均為 ~3 pt：
   - 對 Tiny-IN：69.4 → 66.4（−3.0 pt vs cosine）
   - 對 Imagewoof：91.86 → 88.6（−3.3 pt vs cosine）
   - 峰值 LR (9e-3) 對 pretrained backbone 過猛是穩定的設計問題。

4. **Train-val gap 縮減驗證資料量假設**：
   - Tiny-IN gap 28–33%（每類 ~500 張，200 類）
   - Imagewoof gap 8–15%（每類 ~900 張，10 類）
   - 過擬合主因為「每類樣本不足」而非 scheduler 或模型容量。

5. **Fixed LR 在 Imagewoof 的 best-final 跌幅變大**（86.36 → 82.67，**−3.69 pt**）：
   - Tiny-IN 跌幅僅 −1.48 pt
   - 表示**任務越簡單，缺少 LR 衰減的傷害越明顯** — 在簡單任務上模型很快接近最佳解，沒有衰減就會在最佳解附近震盪離開。

### 6.3 報告層面的學術價值

本研究的雙資料集實驗回應了單一資料集研究的常見質疑：

- **「結論是否會被資料集偏差影響？」** — 不會。5 種排程的相對排名在 Tiny-IN（簡單物件 / 大量類別）與 Imagewoof（細粒度 / 少量類別）兩個極端設定中**完全一致**。
- **「過擬合是 scheduler 的問題還是資料的問題？」** — 是資料的問題。同一套 scheduler 在更高樣本密度的 Imagewoof 上 gap 直接縮減 ~3 倍。
- **「OneCycle 的失敗是否是超參數沒調好？」** — 是預設超參數的設計問題。peak=base × 10 在兩個資料集上都造成相同的 epoch 2–5 val_loss 飆升，需要降至 base × 3~5。

---

## 7. Imagewoof from-scratch ablation

為釐清「pretrained 起點是否驅動了我們的 scheduler 結論」，本節在 Imagewoof 上再跑一輪，**取消預訓練權重**，並補上更強的訓練配方以補償資訊缺口。

### 7.1 設定差異

| 設定 | Pretrained Imagewoof | From-scratch Imagewoof |
|------|---------------------|----------------------|
| Backbone 初始化 | ImageNet pretrained | 隨機初始化 |
| Epochs | 20 | **80** |
| Augmentation | basic (Crop + Flip + ColorJitter) | **strong (+ RandAugment, num_ops=2, magnitude=9)** |
| MixUp | 否 | **alpha = 0.2** |
| Label smoothing | 0 | **0.1** |
| base_lr (YAML / A100 effective) | 3e-4 / 9e-4 | **5e-4 / 1.5e-3**（from-scratch 需較大 LR） |
| OneCycle max_lr (A100 effective) | 9e-3 | **1.5e-2** |
| Capture epochs | [1, 10, 20] | **[5, 40, 80]**（ep1 純隨機，從 ep5 起算） |
| 每組訓練時間 | ~4.7 分 | **~20.8 分** |

### 7.2 量化結果

| Scheduler | Best Val Acc | Final Val Acc | Final Train Acc | Train-Val Gap | 排名 |
|-----------|:------------:|:-------------:|:---------------:|:-------------:|:----:|
| **`onecycle`** | **86.71%** ⭐ | 86.69% | 93.73% | 7.04% | 1 (大躍進) |
| `cosine` | 84.12% | 83.99% | 88.85% | 4.86% | 2 |
| `cosine_restart` | 82.13% | 76.13% | 81.07% | 4.94% | 3 (best→final 跌 -6.0) |
| `fixed` | 80.63% | 78.60% | 87.22% | 8.62% | 4 |
| `step` | 79.84% | 79.69% | 78.74% | -0.95% | 5 (大跌) |

數據來源：`results/imagewoof_scratch/summary.json`、`results/imagewoof_scratch/training_log.txt`。

### 7.3 LR 曲線（80-epoch 配方）

![LR curves](../results/imagewoof_scratch/lr_curves.png)

關鍵差異：
- **OneCycle peak 拉高至 ~1.5e-2**（vs pretrained 9e-3），峰值落在 ~step 500 (epoch 22)。
- **Step 在 ~step 600、~step 1200、~step 1800 三次衰減**，最終 LR 至 ~1.5e-6。
- **Cosine Restart 兩次重啟可見**（~step 460 與 ~step 1380），T_0=20 → T_mult=2 設定。

### 7.4 訓練曲線

![Train/Val curves](../results/imagewoof_scratch/curves.png)

關鍵觀察：

- **OneCycle val_loss 在 epoch 5–15 出現峰值**（>3），對應 warmup 衝至 peak 的階段；之後快速下降並維持穩定，**最終 val_acc 領先所有其他 scheduler ~2.6–7 pt**。
- **Cosine 平穩下行，無明顯震盪**，是「不出包」型的選手。
- **CosineRestart 末段（epoch 60+）val_acc 跌幅明顯**，與末次重啟對應，再次重現 best-final 差距。
- **Step 的 val_acc 在前 30 epoch 跟上 cosine，但中後段停滯**，最終落後 4–5 pt；推測 epoch 26 第一次 ×0.1 衰減太早（模型尚未充分探索），後續被鎖在次優解。
- **Fixed 全程慢慢爬升**，意外與 step 平分秋色 — 因為 step 早期衰減反而傷害了它。

### 7.5 Grad-CAM 演進

![Grad-CAM grid](../results/imagewoof_scratch/grad_cam_grid.png)

行：5 種排程；列：input (一隻 Shih-Tzu)、checkpoint @ epoch 5 / 40 / 80。

**這組 Grad-CAM 比 pretrained 場景更具教學價值**，因為它真正展示了「**注意力從零形成**」的演進：

- **ep 005（所有排程）**：注意力散亂，紅色區域大且邊界模糊，遍布整張影像 — 模型剛開始學習低階特徵，尚未鎖定物件位置。
- **ep 040**：5 種排程都開始收斂至狗臉中央，但範圍仍大、邊界不清。
- **ep 080**：
  - **OneCycle**（量化第 1）：注意力**緊密鎖定在狗臉中央**，紅色熱區小而集中，幾無干擾 — 視覺上的最佳結果。
  - **Cosine**（第 2）：聚焦至整張狗臉，範圍稍大於 OneCycle 但仍合理。
  - **CosineRestart**（末次重啟後）：注意力**從狗臉散開**至左半部背景，視覺確認末次重啟造成的退化。
  - **Fixed / Step**（第 4、5）：注意力**仍呈散開狀態**，無法收緊至五官特徵，與其量化偏低吻合。

### 7.6 跨「訓練狀態」的 scheduler 排名反轉

| Scheduler | Pretrained (20 ep) | From-scratch (80 ep) | 排名變化 |
|-----------|:------------------:|:--------------------:|:--------:|
| `onecycle` | 88.60% (4th) | **86.71% (1st)** | **+3 名** ⬆️⬆️⬆️ |
| `cosine` | 91.86% (1st tied) | 84.12% (2nd) | -1 名 |
| `cosine_restart` | 91.86% (1st tied) | 82.13% (3rd) | -2 名 |
| `fixed` | 86.36% (5th) | 80.63% (4th) | +1 名 |
| `step` | 91.50% (3rd) | 79.84% (5th) | **-3 名** ⬇️⬇️⬇️ |

**詮釋**：

1. **OneCycle 在 from-scratch 大勝**，呼應 Smith (2018) 提出 OneCycle 時所做的也是 from-scratch ImageNet 訓練。其「warmup + 高峰 LR + 退火」三段式設計在隨機初始化上特別管用，可快速逃離初始 plateau。
2. **Step 在 from-scratch 大敗** — 早期 LR 衰減（epoch 26）介入時模型仍未充分探索，導致鎖在次優解。在 pretrained 場景因模型已接近最佳解，早衰反而有助於精煉。
3. **「最佳 scheduler 是場景相依的」** — 這是本研究最反直覺也最重要的結論。任何「某 scheduler 普遍最好」的論述都需條件化。
4. **Cosine 在三組實驗中始終穩定前段班** — Tiny-IN best 第 3、Imagewoof pretrained best 並列 1、from-scratch 第 2。**若必須選一個 "default"，Cosine 是最穩健的選項**。
5. **CosineRestart 的末次重啟災難跨三組實驗穩定**（Tiny -7.60、Imagewoof pretrained -7.61、Imagewoof scratch -6.00），是穩定的設計缺陷，**與訓練狀態無關**。

---

## 8. 主要結論

### 8.1 量化層面

> **任何 LR 衰減策略都顯著優於 Fixed LR**（+3.7 ~ +7.9 pt 在 pretrained 場景；from-scratch 場景下 Step 反而輸給 Fixed，顯示「LR 衰減一定比較好」的論述需要條件化）。

> **「Best」vs「Final」評估策略對 Cosine Restart 影響跨三組實驗穩定** — Tiny-IN −7.60 pt、Imagewoof pretrained −7.61 pt、Imagewoof scratch −6.00 pt。建議**永遠採用 best checkpoint 評估或搭配 early stopping**，否則 Cosine Restart 的真實能力會被嚴重低估。

> **最佳 scheduler 是「訓練狀態相依」的** — Pretrained 場景下 Cosine/CosineRestart/Step 三者並列前段（69.4–70.6% on Tiny、91.5–91.9% on Imagewoof）；but switch to from-scratch + 強增強，**OneCycle 一躍成為冠軍** (86.71%)，Step 從第 2 跌到第 5。任何「某 scheduler 普遍最好」的論述都需條件化。

> **OneCycle 預設超參數 (peak = base × 10) 在不同場景有相反評價** — Fine-tuning 過猛（peak 9e-3 衝擊 pretrained 特徵）；From-scratch 剛好（peak 1.5e-2 配合 warmup 加速隨機初始化收斂）。這是「同一設定在不同場景表現相反」的教科書案例。

> **若必須選一個「default」scheduler**，**Cosine 是三組實驗中表現最穩定的選項**（best 第 3 / 並列 1 / 第 2，無大起大落），即使從不奪冠也從未落入後段。

### 8.2 質化層面

> Grad-CAM 顯示 **pretrained 起點已具備合理的物件級注意力**，後續 LR 排程主要影響「注意力的鞏固速度」與「中後期穩定性」，而非「注意力是否形成」。

> Fixed LR 的注意力末期偏移與 OneCycle 中期的注意力震盪皆可在熱力圖中直接觀察到，**質化與量化結果互相佐證**。

> Imagewoof 場景下注意力**明顯聚焦至面部 / 口鼻區域**（細粒度分類關鍵特徵），驗證模型確實學到了與任務相關的特徵而非偽相關。

> **From-scratch 場景下 Grad-CAM 真正展示「注意力從零形成」的演進**（ep5 散亂→ep40 收斂→ep80 鎖定），不像 pretrained 場景在 ep1 就已具備注意力。OneCycle 在 ep80 的熱力圖**緊密鎖定狗臉中央**，視覺上明顯優於其他 scheduler，與其量化第 1 名相符。

> **CosineRestart 末次重啟的視覺證據** — Tiny-ImageNet ep20、Imagewoof scratch ep80 的熱力圖均顯示注意力**從主體散開**至背景，直接呼應 best→final 跌幅。

---

## 9. 限制與後續工作

### 9.1 限制

1. **僅單一 seed (42)** — 未做多 seed 平均，數值差異可能含 ±0.5–1.0% 隨機浮動。場景相依的結論（如 OneCycle 排名反轉）需更多 seed 驗證才能完全排除「特定 seed 撞到好結果」的可能性。
2. **CosineRestart 與 OneCycle 的超參數未做網格搜尋** — 採用文獻常見預設 (T_0=epochs/4, T_mult=2; peak=base × 10)。從報告層面這已足夠（重點是排程**形狀**的對比），但若做為產品選型則需更仔細調。
3. **僅單一模型架構** — ResNet-18 結論未必能推廣至 ViT / EfficientNet 等架構。
4. **From-scratch 配方未獨立 ablate** — 80 ep + MixUp + RandAug + label smoothing 一起換上，**無法區分哪一項對 OneCycle 翻身的貢獻最大**。

### 9.2 已完成 / 不在本專題範圍

- ✅ **跨資料集驗證**（Tiny-ImageNet + Imagewoof，pretrained 完成）
- ✅ **跨訓練狀態驗證**（pretrained vs from-scratch，Imagewoof 完成）
- ✅ **過擬合假設驗證**（透過跨資料集 gap 變化 + from-scratch 額外證據）
- ✅ **強增強配方實驗**（MixUp + RandAugment + label smoothing）
- ❌ **多 seed 平均** — 算力預算限制
- ❌ **scheduler 超參數 grid search** — 算力預算限制
- ❌ **更大模型 / ViT 對比** — 不在本專題範圍
- ❌ **增強策略單獨 ablation** — 不在本專題範圍

---

## 10. 檔案索引

| 路徑 | 內容 |
|------|------|
| `results/tiny_imagenet/summary.json` | Tiny-ImageNet 5 組訓練的全部 epoch 級指標 + 配置 + 計時 |
| `results/tiny_imagenet/training_log.txt` | Tiny-ImageNet 完整訓練 stdout 紀錄 |
| `results/tiny_imagenet/lr_curves.png` | Tiny-ImageNet LR-vs-step 曲線 |
| `results/tiny_imagenet/curves.png` | Tiny-ImageNet train/val loss 與 accuracy |
| `results/tiny_imagenet/grad_cam_grid.png` | Tiny-ImageNet 5 排程 × 早/中/晚期 Grad-CAM |
| `results/imagewoof/summary.json` | Imagewoof (pretrained) 5 組訓練的全部 epoch 級指標 + 配置 + 計時 |
| `results/imagewoof/training_log.txt` | Imagewoof (pretrained) 完整訓練 stdout 紀錄 |
| `results/imagewoof/lr_curves.png` | Imagewoof (pretrained) LR-vs-step 曲線 |
| `results/imagewoof/curves.png` | Imagewoof (pretrained) train/val loss 與 accuracy |
| `results/imagewoof/grad_cam_grid.png` | Imagewoof (pretrained) 5 排程 × 早/中/晚期 Grad-CAM |
| `results/imagewoof_scratch/summary.json` | Imagewoof from-scratch 5 組訓練的全部 epoch 級指標 + 配置 + 計時 |
| `results/imagewoof_scratch/training_log.txt` | Imagewoof from-scratch 完整訓練 stdout 紀錄 |
| `results/imagewoof_scratch/lr_curves.png` | Imagewoof from-scratch LR-vs-step 曲線 |
| `results/imagewoof_scratch/curves.png` | Imagewoof from-scratch train/val loss 與 accuracy |
| `results/imagewoof_scratch/grad_cam_grid.png` | Imagewoof from-scratch 5 排程 × ep5/40/80 Grad-CAM |
| `notebooks/colab_main.ipynb` | Colab 一鍵執行入口（DATASET 變數切換三組實驗） |
| `configs/*.yaml` | 15 個 scheduler 訓練設定（5 排程 × 3 實驗組） |
