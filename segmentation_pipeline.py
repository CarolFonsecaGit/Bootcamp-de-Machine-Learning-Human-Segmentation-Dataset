#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  Segmentation Pipeline — Full Body TikTok Dataset
=============================================================================
  3 Methods:
    1. FastSAM  (YOLO-based fast SAM — literature)
    2. Otsu Thresholding               (literature)
    3. GrabCut + Morphological Refine   (own method)
=============================================================================
"""

import os, sys, csv, time, random, warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(r"d:\segmentation_full_body_tik_tok_2615_img")
IMG_DIR    = BASE_DIR / "images"
MASK_DIR   = BASE_DIR / "masks"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

N_SAM     = 15
N_CLASSIC = 80
N_VISUAL  = 5

SEED = 42
random.seed(SEED); np.random.seed(SEED)
METRIC_NAMES = ["IoU", "Dice", "Precision", "Recall", "Accuracy"]

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def progress(idx, total, label, extra=""):
    pct = (idx+1)/total*100
    bar_len = 30
    filled = int(bar_len * (idx+1) / total)
    bar = "#" * filled + "." * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  ({idx+1}/{total})  {label} {extra}       ", end="", flush=True)

def list_pairs():
    names = sorted(os.listdir(IMG_DIR))
    return [(IMG_DIR/n, MASK_DIR/n) for n in names if (MASK_DIR/n).exists()]

def load_bgr(p):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)

def load_mask(p):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
    return m.astype(np.uint8)

def keep_largest(b):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
    if n <= 1: return b
    return (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

def fill_holes(b):
    h, w = b.shape
    fl = b.copy()
    cv2.floodFill(fl, np.zeros((h+2, w+2), np.uint8), (0,0), 1)
    return b | (~fl.astype(bool)).astype(np.uint8)

def metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.sum(p & g); fp = np.sum(p & ~g)
    fn = np.sum(~p & g); tn = np.sum(~p & ~g)
    e = 1e-8
    return {"IoU": float(tp/(tp+fp+fn+e)), "Dice": float(2*tp/(2*tp+fp+fn+e)),
            "Precision": float(tp/(tp+fp+e)), "Recall": float(tp/(tp+fn+e)),
            "Accuracy": float((tp+tn)/(tp+tn+fp+fn+e))}

# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1 — FastSAM (YOLO-based, runs in seconds on CPU)
# ═══════════════════════════════════════════════════════════════════════════
def run_fastsam(pairs):
    print("\n  >> METHOD 1 - FastSAM (YOLO-based Segment Anything)")
    print("  " + "-"*50)
    from ultralytics import FastSAM
    model = FastSAM("FastSAM-s.pt")  # auto-downloads ~23 MB

    results, preds = [], {}
    total = len(pairs)
    t_start = time.time()
    for i, (ip, mp) in enumerate(pairs):
        gt = load_mask(mp); h, w = gt.shape
        try:
            res = model(str(ip), imgsz=512, conf=0.3, iou=0.7, verbose=False)
            pred = np.zeros((h, w), dtype=np.uint8)
            if res and res[0].masks is not None:
                masks = res[0].masks.data.cpu().numpy()
                best_a, best_m = 0, None
                for j in range(masks.shape[0]):
                    m = masks[j]
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    a = m.sum()
                    if a > best_a: best_a, best_m = a, m
                if best_m is not None:
                    pred = (best_m > 0.5).astype(np.uint8)
            met = metrics(pred, gt)
            results.append(met); preds[ip.name] = pred
            elapsed = time.time() - t_start
            eta = elapsed / (i+1) * (total - i - 1)
            progress(i, total, "FastSAM", f"IoU={met['IoU']:.3f}  ETA={eta:.0f}s")
        except Exception as e:
            progress(i, total, "FastSAM", f"SKIP: {e}")
    print()  # newline after progress bar
    return results, preds

# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2 — Otsu Thresholding
# ═══════════════════════════════════════════════════════════════════════════
def otsu_seg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, bn = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    bn = cv2.morphologyEx(bn, cv2.MORPH_CLOSE, k, iterations=2)
    bn = cv2.morphologyEx(bn, cv2.MORPH_OPEN, k, iterations=1)
    return keep_largest(bn)

def run_otsu(pairs):
    print("\n  >> METHOD 2 - Otsu Thresholding")
    print("  " + "-"*50)
    results, preds = [], {}
    total = len(pairs)
    t_start = time.time()
    for i, (ip, mp) in enumerate(pairs):
        img = load_bgr(ip); gt = load_mask(mp)
        pred = otsu_seg(img)
        met = metrics(pred, gt); results.append(met); preds[ip.name] = pred
        elapsed = time.time() - t_start
        eta = elapsed / (i+1) * (total - i - 1)
        if i % 5 == 0 or i == total-1:
            progress(i, total, "Otsu", f"IoU={met['IoU']:.3f}  ETA={eta:.0f}s")
    print()
    return results, preds

# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3 — GrabCut + Morphological Refinement (OWN METHOD)
# ═══════════════════════════════════════════════════════════════════════════
def grabcut_seg(img):
    h, w = img.shape[:2]
    mx, my = int(w*0.10), int(h*0.05)
    rect = (mx, my, w-2*mx, h-2*my)
    mg = np.zeros((h,w), np.uint8)
    cv2.grabCut(img, mg, rect, np.zeros((1,65),np.float64),
                np.zeros((1,65),np.float64), 5, cv2.GC_INIT_WITH_RECT)
    p = np.where((mg==1)|(mg==3), 1, 0).astype(np.uint8)
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    kl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    p = cv2.dilate(p, ks, iterations=2)
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, kl, iterations=3)
    p = cv2.erode(p, ks, iterations=1)
    return keep_largest(fill_holes(p))

def run_grabcut(pairs):
    print("\n  >> METHOD 3 - GrabCut + Morph Refinement (Own Method)")
    print("  " + "-"*50)
    results, preds = [], {}
    total = len(pairs)
    t_start = time.time()
    for i, (ip, mp) in enumerate(pairs):
        img = load_bgr(ip); gt = load_mask(mp)
        try:
            pred = grabcut_seg(img)
        except:
            pred = np.zeros(gt.shape, np.uint8)
        met = metrics(pred, gt); results.append(met); preds[ip.name] = pred
        elapsed = time.time() - t_start
        eta = elapsed / (i+1) * (total - i - 1)
        if i % 5 == 0 or i == total-1:
            progress(i, total, "GrabCut", f"IoU={met['IoU']:.3f}  ETA={eta:.0f}s")
    print()
    return results, preds

# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
def agg(results):
    return {m: {"mean": np.mean([r[m] for r in results]),
                "std":  np.std([r[m] for r in results])} for m in METRIC_NAMES}

def save_csv(aa, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method"]+[f"{m}_mean" for m in METRIC_NAMES]+[f"{m}_std" for m in METRIC_NAMES])
        for n, a in aa.items():
            w.writerow([n]+[f"{a[m]['mean']:.4f}" for m in METRIC_NAMES]
                         +[f"{a[m]['std']:.4f}" for m in METRIC_NAMES])
    print(f"  [OK] Saved: {path.name}")

def print_table(aa):
    print("\n" + "="*90)
    print("  COMPARATIVE RESULTS")
    print("="*90)
    print(f"  {'Method':<38}" + "".join(f"{m:>12}" for m in METRIC_NAMES))
    print("  "+"-"*87)
    for n, a in aa.items():
        r = f"  {n:<38}"
        for m in METRIC_NAMES: r += f"  {a[m]['mean']:.4f}±{a[m]['std']:.2f}"
        print(r)
    print("="*90)

def plot_bar(aa, path):
    methods = list(aa.keys()); x = np.arange(len(METRIC_NAMES)); w = 0.25
    colors = ["#2196F3","#FF9800","#4CAF50"]
    fig, ax = plt.subplots(figsize=(12,6))
    for i, mt in enumerate(methods):
        means = [aa[mt][m]["mean"] for m in METRIC_NAMES]
        stds = [aa[mt][m]["std"] for m in METRIC_NAMES]
        off = (i - len(methods)/2 + 0.5) * w
        bars = ax.bar(x+off, means, w, yerr=stds, label=mt, color=colors[i%3],
                      edgecolor="white", capsize=3)
        for b, v in zip(bars, means):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Score"); ax.set_title("Segmentation — Comparative Evaluation",
                                          fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(METRIC_NAMES, fontsize=11)
    ax.set_ylim(0,1.15); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(str(path), dpi=150); plt.close(fig)
    print(f"  [OK] Saved: {path.name}")

def plot_visual(pairs, ap, path):
    n = min(N_VISUAL, len(pairs)); sel = pairs[:n]
    fig, axes = plt.subplots(n, 5, figsize=(20, 4*n))
    if n == 1: axes = axes[np.newaxis, :]
    titles = ["Original","Ground Truth","FastSAM","Otsu","GrabCut+Morph"]
    keys = ["FastSAM","Otsu","GrabCut"]
    for r, (ip, mp) in enumerate(sel):
        img = cv2.cvtColor(load_bgr(ip), cv2.COLOR_BGR2RGB); gt = load_mask(mp)
        axes[r,0].imshow(img); axes[r,1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        for c, k in enumerate(keys):
            if k in ap and ip.name in ap[k]:
                axes[r,c+2].imshow(ap[k][ip.name], cmap="gray", vmin=0, vmax=1)
            else:
                axes[r,c+2].text(0.5,0.5,"N/A",transform=axes[r,c+2].transAxes,
                                 ha="center",va="center",fontsize=14,color="red")
        for c in range(5):
            axes[r,c].axis("off")
            if r==0: axes[r,c].set_title(titles[c], fontsize=13, fontweight="bold")
    fig.suptitle("Visual Comparison", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Saved: {path.name}")

def write_report(aa, timings, path):
    L = []
    L.append("="*80)
    L.append("  RELATÓRIO — Segmentação de Corpo Inteiro (TikTok Dataset)")
    L.append("="*80 + "\n")
    L.append("1. METODOLOGIA\n" + "-"*40)
    L.append("\n1.1 Dataset")
    L.append("   Nome: Full Body TikTok Segmentation (2615 imagens)")
    L.append("   Tipo: Segmentação binária (pessoa vs. fundo)")
    L.append("   Formato: PNG images + PNG masks (ground truth)\n")
    L.append("1.2 Métodos da Literatura\n")
    L.append("   Método 1 — FastSAM (Fast Segment Anything Model)")
    L.append("   Ref: Zhao et al., 'Fast Segment Anything', 2023")
    L.append("   - Baseado em YOLOv8-seg (detecção + segmentação)")
    L.append("   - 50x mais rápido que SAM original, sem perda severa de qualidade")
    L.append("   - Zero-shot: segmenta tudo, selecionamos a maior máscara\n")
    L.append("   Método 2 — Limiarização de Otsu")
    L.append("   Ref: Otsu, 'A Threshold Selection Method', IEEE 1979")
    L.append("   - Grayscale → Gaussian Blur → Otsu → Morfologia")
    L.append("   - Sem deep learning, totalmente clássico e rápido\n")
    L.append("1.3 Método Próprio\n")
    L.append("   Método 3 — GrabCut + Refinamento Morfológico")
    L.append("   - GrabCut (Rother et al., 2004) com retângulo central")
    L.append("   - Pós-processamento: dilate → close → erode → fill holes → largest CC")
    L.append("   - Pipeline personalizado para este domínio\n")
    L.append("1.4 Métricas: IoU, Dice, Precision, Recall, Accuracy\n")
    L.append("2. RESULTADOS\n" + "-"*40 + "\n")
    hdr = f"  {'Método':<38}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
    L.append(hdr); L.append("  "+"-"*87)
    for name, a in aa.items():
        r = f"  {name:<38}"
        for m in METRIC_NAMES: r += f"  {a[m]['mean']:.4f}±{a[m]['std']:.2f}"
        L.append(r)
    L.append("\n  Tempos:")
    for n2, t in timings.items(): L.append(f"    {n2}: {t:.1f}s")
    best = max(aa, key=lambda k: aa[k]["IoU"]["mean"])
    L.append(f"\n  Melhor IoU: {best} ({aa[best]['IoU']['mean']:.4f})\n")
    L.append("3. CONCLUSÕES\n" + "-"*40 + "\n")
    L.append("3.1 Compilado")
    L.append("   - Modelos de deep learning (FastSAM) geralmente superam métodos clássicos.")
    L.append("   - Otsu é rápido mas limitado a cenários de alto contraste.")
    L.append("   - GrabCut+Morph oferece boa relação velocidade/qualidade.\n")
    L.append("3.2 Contribuições")
    L.append("   - Comparação quantitativa de 3 abordagens distintas.")
    L.append("   - Método próprio com GrabCut + pipeline morfológico.\n")
    L.append("3.3 Trabalhos Futuros")
    L.append("   - Fine-tuning do SAM neste dataset.")
    L.append("   - U-Net ou DeepLabv3+ treinados supervisionadamente.")
    L.append("   - YOLO (detecção de pessoa) + SAM (prompt por bounding box).")
    L.append("   - SAM 2 para consistência temporal em vídeo.\n")
    L.append("="*80)
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"  [OK] Saved: {path.name}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  SEGMENTATION PIPELINE — Full Body TikTok")
    print("  Estimated total time: ~5-10 minutes")
    print("=" * 60)
    sys.stdout.flush()

    all_pairs = list_pairs()
    print(f"\n  Total image-mask pairs: {len(all_pairs)}")

    sam_pairs = random.sample(all_pairs, min(N_SAM, len(all_pairs)))
    classic_pairs = random.sample(all_pairs, min(N_CLASSIC, len(all_pairs)))
    visual_set = sam_pairs[:N_VISUAL]

    aa, timings, ap = {}, {}, {}

    # ── FastSAM ──────────────────────────────────────────────────────────
    print("\n  [STEP 1/3] Running FastSAM...")
    sys.stdout.flush()
    t0 = time.time()
    r1, p1 = run_fastsam(sam_pairs)
    timings["FastSAM"] = time.time() - t0
    if r1: aa["FastSAM"] = agg(r1); ap["FastSAM"] = p1
    print(f"  [OK] FastSAM done in {timings['FastSAM']:.1f}s")
    sys.stdout.flush()

    # ── Otsu ─────────────────────────────────────────────────────────────
    print("\n  [STEP 2/3] Running Otsu Thresholding...")
    sys.stdout.flush()
    t0 = time.time()
    r2, p2 = run_otsu(classic_pairs)
    timings["Otsu Thresholding"] = time.time() - t0
    aa["Otsu Thresholding"] = agg(r2); ap["Otsu"] = p2
    print(f"  [OK] Otsu done in {timings['Otsu Thresholding']:.1f}s")
    sys.stdout.flush()

    # ── GrabCut ──────────────────────────────────────────────────────────
    print("\n  [STEP 3/3] Running GrabCut + Morphological Refinement...")
    sys.stdout.flush()
    t0 = time.time()
    r3, p3 = run_grabcut(classic_pairs)
    timings["GrabCut+Morph (Own)"] = time.time() - t0
    aa["GrabCut+Morph (Own)"] = agg(r3); ap["GrabCut"] = p3
    print(f"  [OK] GrabCut done in {timings['GrabCut+Morph (Own)']:.1f}s")
    sys.stdout.flush()

    # ── Generate Outputs ─────────────────────────────────────────────────
    print("\n  [OUTPUT] Generating tables, charts and report...")
    sys.stdout.flush()
    print_table(aa)
    save_csv(aa, RESULT_DIR / "metrics_comparison.csv")
    plot_bar(aa, RESULT_DIR / "metrics_barplot.png")
    plot_visual(visual_set, ap, RESULT_DIR / "visual_comparison.png")
    write_report(aa, timings, RESULT_DIR / "report.txt")

    print("\n" + "="*60)
    print("  ALL DONE! Results saved in: results/")
    print("=" * 60 + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
