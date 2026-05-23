// Build presentation.pptx for the ML LR-comparison final project.
// Run: node docs/build_deck.js  (from repo root)
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const IMG = (p) => path.join(ROOT, p);

// ---------- Palette ("Ocean Gradient") ----------
const NAVY = "21295C";       // dark, title slides
const DEEP = "065A82";       // primary
const TEAL = "1C7293";       // secondary
const INK = "1E293B";        // body text
const SUB = "64748B";        // muted text
const BG = "FFFFFF";         // content slide background
const SOFT = "F1F5F9";       // soft card bg
const HILITE = "DBEAFE";     // highlight cell
const WARN = "DC2626";       // for negative deltas
const OK = "059669";         // for positive deltas
const WHITE = "FFFFFF";

// ---------- Helpers ----------
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" × 5.625"
pres.author = "ML LR-comparison project";
pres.title = "動態學習率排程之效能對比與特徵視覺化分析";

const SW = 10, SH = 5.625;

function dot(slide, x, y) {
  slide.addShape(pres.shapes.OVAL, {
    x, y, w: 0.12, h: 0.12, fill: { color: DEEP }, line: { color: DEEP }
  });
}

function pageTitle(slide, text, subtitle) {
  slide.addText(text, {
    x: 0.5, y: 0.3, w: 9, h: 0.55,
    fontSize: 26, fontFace: "Calibri", color: NAVY, bold: true,
    margin: 0, align: "left"
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 0.85, w: 9, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: SUB,
      margin: 0, align: "left"
    });
  }
}

function pageFooter(slide, n) {
  slide.addText(`${n} · 動態學習率排程之效能對比`, {
    x: 0.5, y: 5.30, w: 9, h: 0.25,
    fontSize: 9, fontFace: "Calibri", color: SUB, align: "right", margin: 0
  });
}

function contentSlide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}

// =========================================================
// Slide 1 — Title
// =========================================================
{
  const s = darkSlide();
  s.addText("動態學習率排程", {
    x: 0.6, y: 1.5, w: 8.8, h: 0.8, fontSize: 40, fontFace: "Calibri",
    color: WHITE, bold: true, margin: 0
  });
  s.addText("之效能對比與特徵視覺化分析", {
    x: 0.6, y: 2.25, w: 8.8, h: 0.7, fontSize: 32, fontFace: "Calibri",
    color: "CADCFC", margin: 0
  });
  // accent dot
  s.addShape(pres.shapes.OVAL, {
    x: 0.6, y: 3.30, w: 0.15, h: 0.15, fill: { color: "5EEAD4" }, line: { color: "5EEAD4" }
  });
  s.addText("ResNet-18 × Tiny-ImageNet & Imagewoof × 5 Schedulers × pretrained / from-scratch", {
    x: 0.85, y: 3.20, w: 8.5, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: "CADCFC", margin: 0
  });
  s.addText("機器學習期末專題  ·  2026/05", {
    x: 0.6, y: 4.7, w: 8.8, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: "CADCFC", margin: 0
  });
}

// =========================================================
// Slide 2 — Motivation
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "動機與研究問題", "Why study learning-rate schedulers, and what do we actually want to know?");

  // Left: 3 research questions stacked as cards
  const cardX = 0.5, cardW = 4.7, cardH = 1.05;
  const qs = [
    ["Q1", "LR 衰減策略真的重要嗎？", "vs 整段固定 LR 的對照"],
    ["Q2", "scheduler 排名穩定嗎？", "跨資料集（200 類 vs 10 類細粒度）"],
    ["Q3", "pretrained 與 from-scratch 場景下，最佳 scheduler 一致嗎？"],
  ];
  qs.forEach(([tag, h, sub], i) => {
    const y = 1.35 + i * 1.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y, w: cardW, h: cardH,
      fill: { color: SOFT }, line: { color: SOFT }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y, w: 0.08, h: cardH, fill: { color: DEEP }, line: { color: DEEP }
    });
    s.addText(tag, {
      x: cardX + 0.2, y: y + 0.10, w: 0.7, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
    });
    s.addText(h, {
      x: cardX + 0.85, y: y + 0.10, w: cardW - 1.0, h: 0.4,
      fontSize: 15, fontFace: "Calibri", color: INK, bold: true, margin: 0
    });
    if (sub) {
      s.addText(sub, {
        x: cardX + 0.85, y: y + 0.55, w: cardW - 1.0, h: 0.4,
        fontSize: 12, fontFace: "Calibri", color: SUB, margin: 0
      });
    }
  });

  // Right: takeaway card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.6, y: 1.35, w: 3.9, h: 3.4,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("本研究方法", {
    x: 5.8, y: 1.55, w: 3.6, h: 0.4,
    fontSize: 15, fontFace: "Calibri", color: "5EEAD4", bold: true, margin: 0
  });
  s.addText([
    { text: "三組對照實驗", options: { bullet: true, color: WHITE, fontSize: 14, bold: true, breakLine: true } },
    { text: "Tiny-ImageNet (200 類, pretrained)", options: { bullet: { indent: 14 }, color: "CADCFC", fontSize: 12, breakLine: true, indentLevel: 1 } },
    { text: "Imagewoof (10 類, pretrained)", options: { bullet: { indent: 14 }, color: "CADCFC", fontSize: 12, breakLine: true, indentLevel: 1 } },
    { text: "Imagewoof (10 類, from-scratch + 強增強)", options: { bullet: { indent: 14 }, color: "CADCFC", fontSize: 12, breakLine: true, indentLevel: 1 } },
    { text: "5 種 scheduler 同條件對比", options: { bullet: true, color: WHITE, fontSize: 14, bold: true, breakLine: true } },
    { text: "Grad-CAM 質化視覺化", options: { bullet: true, color: WHITE, fontSize: 14, bold: true } },
  ], {
    x: 5.8, y: 1.95, w: 3.6, h: 2.7, fontFace: "Calibri", margin: 0, paraSpaceAfter: 4
  });

  pageFooter(s, 2);
}

// =========================================================
// Slide 3 — 5 schedulers + LR curves
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "五種學習率排程", "形狀差異是本研究唯一變量 — 其他超參數全部對齊");

  // Left: table-style scheduler list
  const sched = [
    ["Fixed", "全程不衰減", SUB],
    ["StepLR", "每 N epoch 斷崖式衰減", DEEP],
    ["Cosine", "餘弦平滑退火", DEEP],
    ["Cosine Restart", "帶重啟的餘弦", DEEP],
    ["OneCycle", "先升後降（含 warmup）", DEEP],
  ];
  const ly = 1.35, lx = 0.5, rowH = 0.55;
  sched.forEach((row, i) => {
    const y = ly + i * rowH;
    if (i % 2 === 0) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: lx, y, w: 4.0, h: rowH, fill: { color: SOFT }, line: { color: SOFT }
      });
    }
    s.addShape(pres.shapes.OVAL, {
      x: lx + 0.1, y: y + 0.20, w: 0.15, h: 0.15,
      fill: { color: row[2] }, line: { color: row[2] }
    });
    s.addText(row[0], {
      x: lx + 0.35, y: y + 0.05, w: 1.5, h: 0.45,
      fontSize: 14, fontFace: "Calibri", color: INK, bold: true, margin: 0
    });
    s.addText(row[1], {
      x: lx + 1.8, y: y + 0.05, w: 2.5, h: 0.45,
      fontSize: 12, fontFace: "Calibri", color: SUB, margin: 0
    });
  });

  // Right: LR curve image (the imagewoof_scratch version shows widest range)
  s.addImage({
    path: IMG("results/imagewoof_scratch/lr_curves.png"),
    x: 4.7, y: 1.3, w: 4.9, h: 2.75
  });
  s.addText("實例：Imagewoof from-scratch (80 epoch) 的 5 種 LR 形狀", {
    x: 4.7, y: 4.10, w: 4.9, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: SUB, align: "center", margin: 0
  });

  // Bottom callout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.50, w: 9, h: 0.55, fill: { color: HILITE }, line: { color: HILITE }
  });
  s.addText("✦ 5 個 scheduler 共用相同 base_lr、weight_decay、batch、aug — 唯一差異是「LR 隨時間的形狀」。", {
    x: 0.65, y: 4.55, w: 8.8, h: 0.45,
    fontSize: 13, fontFace: "Calibri", color: NAVY, italic: true, margin: 0
  });

  pageFooter(s, 3);
}

// =========================================================
// Slide 4 — Experiment Setup & Fairness
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "實驗設定", "Three controlled groups, identical evaluation protocol");

  // Setup table
  const headers = ["實驗組", "資料集", "Backbone", "Epoch", "Augmentation", "MixUp / LS"];
  const rows = [
    ["Tiny-IN", "Tiny-ImageNet (200)", "ResNet-18 (pretrained)", "20", "basic", "—"],
    ["Imagewoof (pre)", "Imagewoof (10)", "ResNet-18 (pretrained)", "20", "basic", "—"],
    ["Imagewoof (scratch)", "Imagewoof (10)", "ResNet-18 (random)", "80", "+ RandAugment", "α=0.2 / 0.1"],
  ];
  const tbl = [
    headers.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DEEP }, align: "center", valign: "middle", fontSize: 12 } })),
    ...rows.map((r) => r.map((c) => ({ text: c, options: { color: INK, align: "center", valign: "middle", fontSize: 11, fontFace: "Calibri" } })))
  ];
  s.addTable(tbl, {
    x: 0.5, y: 1.30, w: 9, colW: [1.7, 2.0, 1.8, 0.7, 1.6, 1.2],
    rowH: 0.45, fontFace: "Calibri",
    border: { pt: 0.5, color: "E2E8F0" }
  });

  // Common setting card
  s.addText("共同設定（保證公平性）", {
    x: 0.5, y: 3.40, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
  });
  const fair = [
    "AdamW · weight_decay 5e-4 · seed 42",
    "A100 GPU · batch=384 · AMP + TF32",
    "image_size 224 × 224 (利於 Grad-CAM 7×7 特徵圖)",
    "每組 5 個 scheduler 共用同 base_lr、同 augmentation",
    "checkpoint 在早 / 中 / 晚期各存一份",
  ];
  fair.forEach((line, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.5 + col * 4.7;
    const y = 3.80 + row * 0.4;
    s.addShape(pres.shapes.OVAL, {
      x, y: y + 0.12, w: 0.10, h: 0.10, fill: { color: TEAL }, line: { color: TEAL }
    });
    s.addText(line, {
      x: x + 0.25, y, w: 4.4, h: 0.35,
      fontSize: 12, fontFace: "Calibri", color: INK, margin: 0
    });
  });

  pageFooter(s, 4);
}

// =========================================================
// Slide 5 — Phase 1: Tiny-ImageNet results
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "實驗一：Tiny-ImageNet (200 類)", "Pretrained ResNet-18 · 20 epoch · ~25 min/組");

  // Results table
  const headers = ["Scheduler", "Best Val", "Final Val", "Train-Val Gap"];
  const rows = [
    ["cosine_restart", "70.57%", "62.97%", "30.8%", true],
    ["step", "69.97%", "69.82%", "28.2%", false],
    ["cosine", "69.40%", "69.32%", "30.6%", false],
    ["onecycle", "66.42%", "66.35%", "33.3%", false],
    ["fixed", "62.69%", "61.21%", "33.3%", false],
  ];
  const tbl = [
    headers.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DEEP }, align: "center", valign: "middle", fontSize: 12 } })),
    ...rows.map((r) => [
      { text: r[0], options: { color: INK, fontSize: 11, fontFace: "Consolas", align: "left", valign: "middle" } },
      { text: r[1], options: { color: r[4] ? OK : INK, bold: r[4], fontSize: 12, align: "center", valign: "middle" } },
      { text: r[2], options: { color: INK, fontSize: 11, align: "center", valign: "middle" } },
      { text: r[3], options: { color: SUB, fontSize: 11, align: "center", valign: "middle" } },
    ])
  ];
  s.addTable(tbl, {
    x: 0.5, y: 1.30, w: 5.2, colW: [1.7, 1.1, 1.1, 1.3],
    rowH: 0.42, fontFace: "Calibri", border: { pt: 0.5, color: "E2E8F0" }
  });

  // Curves image right
  s.addImage({
    path: IMG("results/tiny_imagenet/curves.png"),
    x: 5.95, y: 1.30, w: 3.7, h: 2.50
  });

  // Bottom observations
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.20, w: 9, h: 0.9, fill: { color: SOFT }, line: { color: SOFT }
  });
  s.addText("關鍵觀察", {
    x: 0.65, y: 4.25, w: 2.0, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
  });
  s.addText([
    { text: "✦ ", options: { color: DEEP, fontSize: 12 } },
    { text: "任何衰減策略都顯著優於 Fixed (+3.7 ~ +7.9 pt)。", options: { color: INK, fontSize: 12, breakLine: true } },
    { text: "✦ ", options: { color: DEEP, fontSize: 12 } },
    { text: "CosineRestart 雖 best 領先，但 final 跌 7.6 pt → 末次重啟破壞收斂。", options: { color: INK, fontSize: 12, breakLine: true } },
    { text: "✦ ", options: { color: DEEP, fontSize: 12 } },
    { text: "Train-Val gap 28–33% → 過擬合明顯（每類僅 ~500 張）。", options: { color: INK, fontSize: 12 } },
  ], {
    x: 0.65, y: 4.55, w: 8.7, h: 0.55, fontFace: "Calibri", margin: 0
  });

  pageFooter(s, 5);
}

// =========================================================
// Slide 6 — Phase 2: Imagewoof pretrained
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "實驗二：Imagewoof (10 類細粒度狗品種)", "Pretrained ResNet-18 · 20 epoch · ~5 min/組");

  const headers = ["Scheduler", "Best Val", "Final Val", "Train-Val Gap"];
  const rows = [
    ["cosine", "91.86%", "91.86%", "8.08%", true],
    ["cosine_restart", "91.86%", "84.25%", "11.79%", true],
    ["step", "91.50%", "91.45%", "8.27%", false],
    ["onecycle", "88.60%", "88.60%", "10.27%", false],
    ["fixed", "86.36%", "82.67%", "15.34%", false],
  ];
  const tbl = [
    headers.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DEEP }, align: "center", valign: "middle", fontSize: 12 } })),
    ...rows.map((r) => [
      { text: r[0], options: { color: INK, fontSize: 11, fontFace: "Consolas", align: "left", valign: "middle" } },
      { text: r[1], options: { color: r[4] ? OK : INK, bold: r[4], fontSize: 12, align: "center", valign: "middle" } },
      { text: r[2], options: { color: INK, fontSize: 11, align: "center", valign: "middle" } },
      { text: r[3], options: { color: SUB, fontSize: 11, align: "center", valign: "middle" } },
    ])
  ];
  s.addTable(tbl, {
    x: 0.5, y: 1.30, w: 5.2, colW: [1.7, 1.1, 1.1, 1.3],
    rowH: 0.42, fontFace: "Calibri", border: { pt: 0.5, color: "E2E8F0" }
  });

  s.addImage({
    path: IMG("results/imagewoof/curves.png"),
    x: 5.95, y: 1.30, w: 3.7, h: 2.50
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.20, w: 9, h: 0.9, fill: { color: SOFT }, line: { color: SOFT }
  });
  s.addText("關鍵觀察", {
    x: 0.65, y: 4.25, w: 2.0, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
  });
  s.addText([
    { text: "✦ ", options: { color: DEEP, fontSize: 12 } },
    { text: "絕對精度躍升至 86–92%（pretrained 對小資料集的關鍵作用）。", options: { color: INK, fontSize: 12, breakLine: true } },
    { text: "✦ ", options: { color: DEEP, fontSize: 12 } },
    { text: "Train-Val gap 縮減至 8–15%，過擬合主因確認為「每類樣本不足」。", options: { color: INK, fontSize: 12, breakLine: true } },
    { text: "✦ ", options: { color: DEEP, fontSize: 12 } },
    { text: "CosineRestart 末次重啟災難重現：best 91.86 → final 84.25 (-7.61 pt)。", options: { color: INK, fontSize: 12 } },
  ], {
    x: 0.65, y: 4.55, w: 8.7, h: 0.55, fontFace: "Calibri", margin: 0
  });

  pageFooter(s, 6);
}

// =========================================================
// Slide 7 — Cross-dataset finding (pretrained)
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "跨資料集發現：排名穩定", "Tiny-ImageNet vs Imagewoof（兩者皆 pretrained）");

  // Comparison table
  const headers = ["Scheduler", "Tiny-IN", "Imagewoof", "兩者排名"];
  const rows = [
    ["cosine_restart", "70.57% (1)", "91.86% (1)", "並列冠軍"],
    ["step", "69.97% (2)", "91.50% (3)", "前段班"],
    ["cosine", "69.40% (3)", "91.86% (1)", "前段班"],
    ["onecycle", "66.42% (4)", "88.60% (4)", "穩定第 4"],
    ["fixed", "62.69% (5)", "86.36% (5)", "穩定殿底"],
  ];
  const tbl = [
    headers.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DEEP }, align: "center", valign: "middle", fontSize: 12 } })),
    ...rows.map((r) => [
      { text: r[0], options: { color: INK, fontSize: 12, fontFace: "Consolas", align: "left", valign: "middle" } },
      { text: r[1], options: { color: INK, fontSize: 12, align: "center", valign: "middle" } },
      { text: r[2], options: { color: INK, fontSize: 12, align: "center", valign: "middle" } },
      { text: r[3], options: { color: TEAL, bold: true, fontSize: 12, align: "center", valign: "middle" } },
    ])
  ];
  s.addTable(tbl, {
    x: 0.5, y: 1.30, w: 9.0, colW: [2.0, 2.2, 2.2, 2.6],
    rowH: 0.45, fontFace: "Calibri", border: { pt: 0.5, color: "E2E8F0" }
  });

  // Conclusion banner (taller to host the suspense line)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.05, w: 9, h: 1.15, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("初步結論", {
    x: 0.65, y: 4.12, w: 2, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: "5EEAD4", bold: true, margin: 0
  });
  s.addText("5 種排程的相對排名在 Tiny-ImageNet 與 Imagewoof 上「幾乎完全一致」 — 看起來結論具備跨資料集 robustness。", {
    x: 0.65, y: 4.42, w: 8.7, h: 0.45,
    fontSize: 14, fontFace: "Calibri", color: WHITE, margin: 0
  });
  s.addText("但下一頁，我們把 pretrained 拿掉 …", {
    x: 0.65, y: 4.83, w: 8.7, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: "FCA5A5", italic: true, margin: 0
  });

  pageFooter(s, 7);
}

// =========================================================
// Slide 8 — Phase 3: From-scratch setup
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "實驗三：Imagewoof from-scratch", "拿掉 ImageNet pretrained，補上強增強配方");

  // Diff table
  const headers = ["設定", "Pretrained Imagewoof", "From-scratch Imagewoof"];
  const rows = [
    ["Backbone 初始化", "ImageNet pretrained", "隨機初始化", true],
    ["Epochs", "20", "80", true],
    ["Augmentation", "basic", "+ RandAugment", true],
    ["MixUp", "—", "α = 0.2", true],
    ["Label smoothing", "0", "0.1", true],
    ["base_lr (A100 effective)", "9e-4", "1.5e-3", true],
    ["每組訓練時間", "~5 分鐘", "~21 分鐘", false],
  ];
  const tbl = [
    headers.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DEEP }, align: "center", valign: "middle", fontSize: 12 } })),
    ...rows.map((r) => [
      { text: r[0], options: { color: INK, fontSize: 11, align: "left", valign: "middle" } },
      { text: r[1], options: { color: SUB, fontSize: 11, align: "center", valign: "middle" } },
      { text: r[2], options: { color: r[3] ? DEEP : INK, bold: r[3], fontSize: 11, align: "center", valign: "middle" } },
    ])
  ];
  s.addTable(tbl, {
    x: 0.5, y: 1.30, w: 9, colW: [2.6, 3.2, 3.2],
    rowH: 0.36, fontFace: "Calibri", border: { pt: 0.5, color: "E2E8F0" }
  });

  // Motivation banner (taller — answer wraps to 2 lines)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.20, w: 9, h: 1.00, fill: { color: HILITE }, line: { color: HILITE }
  });
  s.addText("為什麼做這組？", {
    x: 0.65, y: 4.28, w: 2.5, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: NAVY, bold: true, margin: 0
  });
  s.addText("驗證「scheduler 排名」是否獨立於「是否有 ImageNet pretrained 起點」 — 這是 Tiny-IN + Imagewoof 雙保險還無法回答的問題。", {
    x: 0.65, y: 4.60, w: 8.7, h: 0.55,
    fontSize: 12, fontFace: "Calibri", color: INK, margin: 0
  });

  pageFooter(s, 8);
}

// =========================================================
// Slide 9 — RANKING FLIP (key slide)
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "🎯 關鍵發現：排名大反轉", "OneCycle 從第 4 躍居第 1；Step 從第 2 跌至第 5");

  // Cross-state comparison
  const headers = ["Scheduler", "Tiny-IN (pre)", "Imagewoof (pre)", "Imagewoof (scratch)", "變化"];
  const rows = [
    ["onecycle",     "66.42 (4)", "88.60 (4)", "86.71 (1) ⭐", "+3", OK],
    ["cosine",       "69.40 (3)", "91.86 (1)", "84.12 (2)",   "穩定", TEAL],
    ["cosine_restart","70.57 (1)","91.86 (1)", "82.13 (3)",   "−2", SUB],
    ["fixed",        "62.69 (5)", "86.36 (5)", "80.63 (4)",   "+1", SUB],
    ["step",         "69.97 (2)", "91.50 (3)", "79.84 (5)",   "−3", WARN],
  ];
  const tbl = [
    headers.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: DEEP }, align: "center", valign: "middle", fontSize: 12 } })),
    ...rows.map((r) => [
      { text: r[0], options: { color: INK, fontSize: 12, fontFace: "Consolas", align: "left", valign: "middle" } },
      { text: r[1], options: { color: INK, fontSize: 12, align: "center", valign: "middle" } },
      { text: r[2], options: { color: INK, fontSize: 12, align: "center", valign: "middle" } },
      { text: r[3], options: { color: INK, fontSize: 12, align: "center", valign: "middle" } },
      { text: r[4], options: { color: r[5], bold: true, fontSize: 13, align: "center", valign: "middle" } },
    ])
  ];
  s.addTable(tbl, {
    x: 0.5, y: 1.30, w: 9, colW: [2.0, 1.6, 1.8, 1.9, 1.7],
    rowH: 0.42, fontFace: "Calibri", border: { pt: 0.5, color: "E2E8F0" }
  });

  // Conclusion
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.10, w: 9, h: 1.05, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("修正後的結論", {
    x: 0.65, y: 4.18, w: 3, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: "5EEAD4", bold: true, margin: 0
  });
  s.addText([
    { text: "scheduler 排名「跨資料集穩定」但「跨訓練狀態 (pre / scratch) 重大反轉」。", options: { color: WHITE, fontSize: 14, bold: true, breakLine: true } },
    { text: "→ 任何「某 scheduler 普遍最好」的論述都需條件化。", options: { color: "CADCFC", fontSize: 12, italic: true } },
  ], {
    x: 0.65, y: 4.50, w: 8.7, h: 0.6, fontFace: "Calibri", margin: 0, paraSpaceAfter: 2
  });

  pageFooter(s, 9);
}

// =========================================================
// Slide 10 — Why OneCycle wins from scratch
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "為什麼 OneCycle 在 from-scratch 翻身？", "從 LR 曲線 + 訓練曲線 + Grad-CAM 三方互證");

  // Left: curves
  s.addImage({
    path: IMG("results/imagewoof_scratch/curves.png"),
    x: 0.5, y: 1.25, w: 4.7, h: 3.5
  });
  s.addText("Imagewoof scratch — train/val curves", {
    x: 0.5, y: 4.78, w: 4.7, h: 0.25,
    fontSize: 10, fontFace: "Calibri", color: SUB, align: "center", margin: 0
  });

  // Right: explanation cards
  const cards = [
    ["設計初衷", "Smith 2018 提出 OneCycle 時即在「從零訓練 ImageNet」設定下，主打 super-convergence。"],
    ["warmup 階段", "隨機初始化權重需要快速「探索」損失曲面；warmup 將 LR 拉升至 1.5e-2 完美達成此目的。"],
    ["fine-tuning 場景反例", "Pretrained 已接近最佳解；同一個峰值反而「打散預訓練特徵」，造成 epoch 2–5 val_loss 飆升。"],
  ];
  cards.forEach((c, i) => {
    const y = 1.25 + i * 1.20;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.4, y, w: 4.2, h: 1.05,
      fill: { color: SOFT }, line: { color: SOFT }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.4, y, w: 0.08, h: 1.05, fill: { color: DEEP }, line: { color: DEEP }
    });
    s.addText(c[0], {
      x: 5.6, y: y + 0.1, w: 3.9, h: 0.30,
      fontSize: 13, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
    });
    s.addText(c[1], {
      x: 5.6, y: y + 0.42, w: 3.9, h: 0.6,
      fontSize: 11, fontFace: "Calibri", color: INK, margin: 0
    });
  });

  pageFooter(s, 10);
}

// =========================================================
// Slide 11 — Grad-CAM evolution (qualitative)
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "質化證據：Grad-CAM 從零形成", "Imagewoof from-scratch · OneCycle 在 ep80 鎖定最緊");

  // The grad-cam grid is portrait-oriented (taller than wide). Fit by height.
  // image is ~485 x 590 px, ratio 0.82 (W/H). Use h=4.0, w = 4.0 * 0.82 = ~3.28
  const imgH = 4.0;
  const imgW = imgH * 0.82;
  s.addImage({
    path: IMG("results/imagewoof_scratch/grad_cam_grid.png"),
    x: 0.5, y: 1.10, w: imgW, h: imgH
  });

  // Right: callouts
  const cx = 0.5 + imgW + 0.3;
  const cw = 10 - cx - 0.5;
  s.addText("行：5 種 scheduler   ·   列：input、ep 5、ep 40、ep 80", {
    x: cx, y: 1.10, w: cw, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: SUB, italic: true, margin: 0
  });

  const points = [
    ["ep 005", "全部 scheduler 注意力散亂（剛開始訓練，隨機初始化）"],
    ["ep 040", "開始收斂至狗臉中央，邊界仍模糊"],
    ["ep 080", "OneCycle 緊密鎖定狗臉、Cosine 次之；CosineRestart 卻散開（末次重啟）；Fixed/Step 仍擴散"],
  ];
  points.forEach((p, i) => {
    const y = 1.50 + i * 1.10;
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y, w: cw, h: 1.0, fill: { color: SOFT }, line: { color: SOFT }
    });
    s.addText(p[0], {
      x: cx + 0.15, y: y + 0.10, w: 1.5, h: 0.35,
      fontSize: 13, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
    });
    s.addText(p[1], {
      x: cx + 0.15, y: y + 0.42, w: cw - 0.3, h: 0.55,
      fontSize: 11, fontFace: "Calibri", color: INK, margin: 0
    });
  });

  pageFooter(s, 11);
}

// =========================================================
// Slide 12 — CosineRestart's last-restart curse
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "穩定的設計缺陷：CosineRestart 末次重啟", "跨三組實驗的 best→final 跌幅高度一致");

  // Big stat callouts
  const stats = [
    ["Tiny-ImageNet", "−7.60 pt", "70.57 → 62.97"],
    ["Imagewoof (pre)", "−7.61 pt", "91.86 → 84.25"],
    ["Imagewoof (scratch)", "−6.00 pt", "82.13 → 76.13"],
  ];
  stats.forEach((st, i) => {
    const x = 0.5 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.30, w: 2.9, h: 2.0,
      fill: { color: SOFT }, line: { color: SOFT }
    });
    s.addText(st[0], {
      x: x + 0.15, y: 1.40, w: 2.6, h: 0.35,
      fontSize: 13, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
    });
    s.addText(st[1], {
      x: x + 0.15, y: 1.85, w: 2.6, h: 0.8,
      fontSize: 36, fontFace: "Calibri", color: WARN, bold: true, margin: 0
    });
    s.addText(st[2], {
      x: x + 0.15, y: 2.75, w: 2.6, h: 0.35,
      fontSize: 12, fontFace: "Consolas", color: SUB, margin: 0
    });
  });

  // Interpretation
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.55, w: 9, h: 1.60, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("詮釋", {
    x: 0.65, y: 3.65, w: 2, h: 0.30,
    fontSize: 13, fontFace: "Calibri", color: "5EEAD4", bold: true, margin: 0
  });
  s.addText([
    { text: "跌幅在三組實驗中 ±0.8 pt 之內 → 這不是隨機，是 ", options: { color: WHITE, fontSize: 13 } },
    { text: "T_0=epochs/4、T_mult=2 ", options: { color: "FBBF24", fontSize: 13, fontFace: "Consolas", bold: true } },
    { text: "設定的系統性缺陷。", options: { color: WHITE, fontSize: 13, breakLine: true } },
    { text: "末次重啟把 LR 重新拉回 ~base，破壞了已收斂的權重。", options: { color: "CADCFC", fontSize: 12, breakLine: true } },
    { text: " ", options: { color: WHITE, fontSize: 6, breakLine: true } },
    { text: "建議：", options: { color: "5EEAD4", fontSize: 13, bold: true } },
    { text: "永遠採用 best checkpoint 或 early stopping；勿用 final epoch 評估。", options: { color: WHITE, fontSize: 13 } },
  ], {
    x: 0.65, y: 3.95, w: 8.7, h: 1.15, fontFace: "Calibri", margin: 0, paraSpaceAfter: 2
  });

  pageFooter(s, 12);
}

// =========================================================
// Slide 13 — Key findings summary
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "三個跨實驗組穩定的核心結論", "Cross-experiment robust findings");

  const findings = [
    ["1", "排名跨資料集穩定 / 跨訓練狀態反轉",
     "Pretrained 場景 cosine 系列 + step 領先；From-scratch 場景 OneCycle 翻身奪冠。「最佳 scheduler」是場景相依的。"],
    ["2", "CosineRestart 末次重啟災難穩定存在",
     "三組實驗 best→final 跌 −6.0 / −7.6 / −7.6 pt — 設計缺陷，與訓練狀態無關。永遠用 best checkpoint 評估。"],
    ["3", "Cosine 是最穩健的「default」選擇",
     "三組實驗排名：3 / 並列 1 / 2 — 從未奪冠也從未落入後段，零驚喜的安全牌。"],
  ];
  findings.forEach((f, i) => {
    const y = 1.30 + i * 1.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9, h: 1.10, fill: { color: SOFT }, line: { color: SOFT }
    });
    // big number
    s.addShape(pres.shapes.OVAL, {
      x: 0.7, y: y + 0.25, w: 0.7, h: 0.7, fill: { color: NAVY }, line: { color: NAVY }
    });
    s.addText(f[0], {
      x: 0.7, y: y + 0.25, w: 0.7, h: 0.7,
      fontSize: 24, fontFace: "Calibri", color: "5EEAD4", bold: true,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(f[1], {
      x: 1.6, y: y + 0.15, w: 7.7, h: 0.40,
      fontSize: 15, fontFace: "Calibri", color: NAVY, bold: true, margin: 0
    });
    s.addText(f[2], {
      x: 1.6, y: y + 0.55, w: 7.7, h: 0.55,
      fontSize: 12, fontFace: "Calibri", color: INK, margin: 0
    });
  });

  pageFooter(s, 13);
}

// =========================================================
// Slide 14 — Limitations & future work
// =========================================================
{
  const s = contentSlide();
  pageTitle(s, "限制與未來工作", "Honest assessment of what we did not validate");

  // Two columns: limitations / future work
  s.addText("限制", {
    x: 0.5, y: 1.30, w: 4.5, h: 0.4,
    fontSize: 17, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
  });
  const lims = [
    "僅單一 seed (42)；排名反轉需更多 seed 驗證",
    "scheduler 超參數未做網格搜尋（採文獻預設）",
    "僅單一架構 (ResNet-18)；ViT/EfficientNet 未驗證",
    "from-scratch 配方 4 項一起換上，無法單獨歸因",
  ];
  s.addText(
    lims.map((l, i) => ({
      text: l,
      options: { bullet: true, color: INK, fontSize: 13, fontFace: "Calibri",
                 breakLine: i < lims.length - 1, paraSpaceAfter: 8 }
    })),
    { x: 0.5, y: 1.75, w: 4.5, h: 3.2, margin: 0, valign: "top" }
  );

  s.addText("未來工作", {
    x: 5.2, y: 1.30, w: 4.3, h: 0.4,
    fontSize: 17, fontFace: "Calibri", color: DEEP, bold: true, margin: 0
  });
  const future = [
    "多 seed (×3) 平均，排除單 seed 巧合",
    "From-scratch 配方分項 ablation",
    "scheduler × optimizer (SGD+M / AdamW) 對比",
    "更大模型 (ResNet-50 / ViT) 推廣性驗證",
    "MixUp / CutMix / RandAug 單獨對 gap 的影響",
  ];
  s.addText(
    future.map((l, i) => ({
      text: l,
      options: { bullet: true, color: INK, fontSize: 13, fontFace: "Calibri",
                 breakLine: i < future.length - 1, paraSpaceAfter: 8 }
    })),
    { x: 5.2, y: 1.75, w: 4.3, h: 3.2, margin: 0, valign: "top" }
  );

  pageFooter(s, 14);
}

// =========================================================
// Slide 15 — Thank you / Q&A
// =========================================================
{
  const s = darkSlide();
  s.addText("謝謝聆聽", {
    x: 0.5, y: 1.5, w: 9, h: 1.0,
    fontSize: 56, fontFace: "Calibri", color: WHITE, bold: true,
    align: "center", margin: 0
  });
  s.addText("Questions & Discussion", {
    x: 0.5, y: 2.55, w: 9, h: 0.5,
    fontSize: 22, fontFace: "Calibri", color: "5EEAD4",
    align: "center", margin: 0
  });
  s.addText("GitHub:  github.com/eric20041027/ML_LR_comparison", {
    x: 0.5, y: 4.6, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Consolas", color: "CADCFC",
    align: "center", margin: 0
  });
  s.addText("詳細報告：docs/REPORT.md   ·   結果封存：results/{tiny_imagenet, imagewoof, imagewoof_scratch}/", {
    x: 0.5, y: 5.00, w: 9, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: "94A3B8",
    align: "center", margin: 0
  });
}

// =========================================================
// Write file
// =========================================================
pres.writeFile({ fileName: path.join(__dirname, "presentation.pptx") })
    .then((p) => console.log("Wrote:", p));
