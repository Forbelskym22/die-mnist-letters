"""
Model Dispatch - Vizualizace natrénovaného modelu písmen 🚂

Tento skript "vypravuje" natrénovaný model - vizualizuje jeho architekturu,
naučené konvoluční filtry a feature maps.

Železniční metafora:
  marshall.py → příprava vlaku (konfigurace)
  train.py    → trénink/vlak jede
  dispatch.py → vypravení vlaku (vizualizace)

Spuštění:
    python dispatch.py                      # Auto-detekce nejnovějšího modelu
    python dispatch.py --model_dir PATH     # Specifický model

Automaticky zavoláno po train.py v rámci run.bat workflow.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import SimpleCNN z train.py
from train import SimpleCNN, LETTERS

# ============================================================================
# ARGUMENTY
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Vizualizace natrénovaného modelu")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Cesta ke složce s modelem (default: auto-detekce nejnovějšího)"
    )
    return parser.parse_args()

# ============================================================================
# AUTO-DETEKCE NEJNOVĚJŠÍHO MODELU
# ============================================================================

def find_latest_model():
    """Najde nejnovější model v ../shared/models/"""
    models_dir = Path("../shared/models")

    if not models_dir.exists():
        print(f"❌ Složka s modely neexistuje: {models_dir}")
        return None

    model_dirs = sorted(models_dir.glob("run_*"))

    if not model_dirs:
        print(f"❌ Žádné modely v {models_dir}")
        return None

    return model_dirs[-1]

# ============================================================================
# POMOCNÁ FUNKCE - CENTROVÁNÍ A Z-ORDER OKEN
# ============================================================================

def center_and_order_figures(figures):
    """
    Centruje matplotlib okna na obrazovku a nastaví správný Z-order.
    První figure v listu bude nahoře (viditelný jako první).
    """
    try:
        for fig in figures:
            # Získat Tk window handle z matplotlib figure
            manager = fig.canvas.manager
            window = manager.window

            # Vynutit kompletní update včetně vykreslení
            fig.canvas.draw()
            window.update()

            # Získat rozměry obrazovky a okna
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            window_width = window.winfo_width()
            window_height = window.winfo_height()

            # Vypočítat centrální pozici (střed okna uprostřed obrazovky)
            x = max(0, (screen_width - window_width) // 2)
            y = max(0, (screen_height - window_height) // 2)

            # Nastavit pozici
            window.geometry(f"+{x}+{y}")

        # Nastavit Z-order: poslední na zásobník = první viditelný
        # Proto procházíme v opačném pořadí
        for fig in reversed(figures):
            manager = fig.canvas.manager
            window = manager.window
            window.lift()
            window.focus_force()
    except Exception as e:
        # Pokud backend není TkAgg nebo jiný problém, ignorujeme
        print(f"⚠️  Nepodařilo se centrovat okna: {e}")

# ============================================================================
# HLAVNÍ PROGRAM
# ============================================================================

args = parse_args()

print("=" * 60)
print("🚂 Model Dispatch - Letter Visualization")
print("=" * 60)

# Auto-detekce nebo explicitní cesta
if args.model_dir is None:
    print("\nHledám nejnovější model...")
    args.model_dir = find_latest_model()
    if args.model_dir is None:
        print("\n❌ Žádný model k vizualizaci!")
        print("Nejdřív spusťte trénink: python train.py")
        sys.exit(1)
    print(f"✓ Nalezen: {args.model_dir.name}")
else:
    if not args.model_dir.exists():
        print(f"\n❌ Model složka neexistuje: {args.model_dir}")
        sys.exit(1)
    print(f"\nModel: {args.model_dir.name}")

# Try letter_cnn.pt first, then digit_cnn.pt for compatibility
model_path = args.model_dir / "letter_cnn.pt"
if not model_path.exists():
    model_path = args.model_dir / "digit_cnn.pt"
config_path = args.model_dir / "config.json"

# Kontrola existence souborů
if not model_path.exists():
    print(f"❌ Model soubor nenalezen: {model_path}")
    sys.exit(1)

if not config_path.exists():
    print(f"❌ Config soubor nenalezen: {config_path}")
    sys.exit(1)

# ============================================================================
# NAČTENÍ KONFIGURACE
# ============================================================================

print("\nNačítám konfiguraci...")
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

dropout = config.get("dropout", 0.2)
num_classes = config.get("num_classes", 26)
print(f"✓ Dropout: {dropout}")
print(f"✓ Classes: {num_classes}")

# ============================================================================
# NAČTENÍ MODELU
# ============================================================================

print(f"\nNačítám model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=num_classes, dropout=dropout)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"✓ Model načten (zařízení: {device})")

# ============================================================================
# TEXT SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("Model Architecture Summary")
print("=" * 60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: SimpleCNN (32×32 input)")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {total_params - trainable_params:,}")

print("\nArchitecture:")
print("  Conv2d(1→32, 3×3) + BN + ReLU + MaxPool → 16×16")
print("  Conv2d(32→64, 3×3) + BN + ReLU + MaxPool → 8×8")
print("  Conv2d(64→128, 3×3) + BN + ReLU + MaxPool → 4×4")
print("  Flatten + Linear(2048→128) + ReLU + Dropout")
print(f"  Linear(128→{num_classes})")

# ============================================================================
# VIZUALIZACE CONV FILTRŮ
# ============================================================================

print("\n" + "=" * 60)
print("Vizualizuji konvoluční filtry...")
print("=" * 60)

# Získání vah z jednotlivých vrstev
conv_layers = [layer for layer in model.features if isinstance(layer, torch.nn.Conv2d)]

# Layer 1: 32 filtrů
conv1_weights = conv_layers[0].weight.data.cpu().numpy()  # [32, 1, 3, 3]
fig1, axes1 = plt.subplots(4, 8, figsize=(12, 6))
fig1.suptitle('Conv Layer 1 - Learned Filters (32 filters, 3×3)', fontsize=14)

for i in range(32):
    ax = axes1[i // 8, i % 8]
    filter_img = conv1_weights[i, 0, :, :]
    ax.imshow(filter_img, cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i}', fontsize=8)
    ax.axis('off')

plt.tight_layout()

# Layer 2: 64 filtrů (průměr přes 32 input channels)
conv2_weights = conv_layers[1].weight.data.cpu().numpy()  # [64, 32, 3, 3]
conv2_weights_avg = conv2_weights.mean(axis=1)  # [64, 3, 3]

fig2, axes2 = plt.subplots(8, 8, figsize=(12, 12))
fig2.suptitle('Conv Layer 2 - Learned Filters (64 filters, 3×3, averaged)', fontsize=14)

for i in range(64):
    ax = axes2[i // 8, i % 8]
    filter_img = conv2_weights_avg[i, :, :]
    ax.imshow(filter_img, cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i}', fontsize=8)
    ax.axis('off')

plt.tight_layout()

# Layer 3: 128 filtrů (průměr přes 64 input channels)
conv3_weights = conv_layers[2].weight.data.cpu().numpy()  # [128, 64, 3, 3]
conv3_weights_avg = conv3_weights.mean(axis=1)  # [128, 3, 3]

fig3, axes3 = plt.subplots(8, 16, figsize=(16, 8))
fig3.suptitle('Conv Layer 3 - Learned Filters (128 filters, 3×3, averaged)', fontsize=14)

for i in range(128):
    ax = axes3[i // 16, i % 16]
    filter_img = conv3_weights_avg[i, :, :]
    ax.imshow(filter_img, cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i}', fontsize=7)
    ax.axis('off')

plt.tight_layout()

# ============================================================================
# FEATURE MAPS - UKÁZKOVÉ PÍSMENO
# ============================================================================

print("\nVytvářím feature maps pro ukázkové písmeno...")

# Načteme ukázkové písmeno z composed datasetu
data_dir = Path("../shared/data/composed")
test_dir = data_dir / "test"

# Pokud test split neexistuje, zkusíme train
if not test_dir.exists():
    test_dir = data_dir / "train"
    if not test_dir.exists():
        print("⚠️  Varování: Žádná testovací data k dispozici, přeskakuji feature maps")
        test_dir = None

if test_dir is not None:
    # Najdeme nějaké písmeno
    sample_found = False
    for letter in LETTERS:
        letter_dir = test_dir / letter
        if letter_dir.exists():
            letter_files = list(letter_dir.glob("*.bmp"))
            if letter_files:
                sample_path = letter_files[0]
                sample_label = letter
                sample_found = True
                break

    if sample_found:
        print(f"Ukázkové písmeno: {sample_label} ({sample_path.name})")

        # Načtení a preprocessing
        img = Image.open(sample_path).convert('L')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 32, 32]

        # Forward pass s uložením intermediate outputs
        with torch.no_grad():
            x = img_tensor
            # Po conv1
            feat1 = model.features[2](model.features[1](model.features[0](x)))  # Conv+BN+ReLU
            x = model.features[3](feat1)  # MaxPool

            # Po conv2
            feat2 = model.features[6](model.features[5](model.features[4](x)))  # Conv+BN+ReLU
            x = model.features[7](feat2)  # MaxPool

            # Po conv3
            feat3 = model.features[10](model.features[9](model.features[8](x)))  # Conv+BN+ReLU

        # Vizualizace
        fig4, axes4 = plt.subplots(4, 10, figsize=(15, 7))
        fig4.suptitle(f'Feature Maps - Sample letter: {sample_label}', fontsize=14)

        # Originál
        axes4[0, 0].imshow(img_tensor.cpu().squeeze(), cmap='gray')
        axes4[0, 0].set_title('Input', fontsize=10)
        axes4[0, 0].axis('off')
        for col in range(1, 10):
            axes4[0, col].axis('off')

        # Feature maps z conv1 (prvních 10)
        feat1_np = feat1.cpu().squeeze().numpy()  # [32, 32, 32]

        print("\nDiagnostika Conv1 Feature Maps (prvních 10):")
        for col in range(10):
            feat_map = feat1_np[col]
            fmin, fmax = feat_map.min(), feat_map.max()
            fmean, fstd = feat_map.mean(), feat_map.std()
            n_unique = len(np.unique(feat_map))
            n_zeros = np.sum(feat_map == 0)
            n_total = feat_map.size

            print(f"  C1-{col}: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}, std={fstd:.4f}, "
                  f"unique={n_unique}, zeros={n_zeros}/{n_total} ({100*n_zeros/n_total:.1f}%)")

            # Robustní vizualizace
            if fstd < 1e-6:  # Konstantní nebo skoro konstantní
                axes4[1, col].text(0.5, 0.5, 'Dead\nfilter', ha='center', va='center',
                                  transform=axes4[1, col].transAxes, fontsize=8)
                axes4[1, col].set_facecolor('#1a1a1a')
            else:
                # Percentilová normalizace (eliminuje outliers)
                vmin_robust = np.percentile(feat_map, 2)
                vmax_robust = np.percentile(feat_map, 98)

                # Ochrana proti min==max
                if vmax_robust - vmin_robust < 1e-6:
                    vmax_robust = vmin_robust + 1e-6

                axes4[1, col].imshow(feat_map, cmap='RdBu_r', vmin=vmin_robust, vmax=vmax_robust)

            axes4[1, col].set_title(f'C1-{col}', fontsize=8)
            axes4[1, col].axis('off')

        # Feature maps z conv2 (prvních 10)
        feat2_np = feat2.cpu().squeeze().numpy()  # [64, 16, 16]

        print("\nDiagnostika Conv2 Feature Maps (prvních 10):")
        for col in range(10):
            feat_map = feat2_np[col]
            fmin, fmax = feat_map.min(), feat_map.max()
            fmean, fstd = feat_map.mean(), feat_map.std()
            n_unique = len(np.unique(feat_map))
            n_zeros = np.sum(feat_map == 0)
            n_total = feat_map.size

            print(f"  C2-{col}: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}, std={fstd:.4f}, "
                  f"unique={n_unique}, zeros={n_zeros}/{n_total} ({100*n_zeros/n_total:.1f}%)")

            # Robustní vizualizace
            if fstd < 1e-6:  # Konstantní nebo skoro konstantní
                axes4[2, col].text(0.5, 0.5, 'Dead\nfilter', ha='center', va='center',
                                  transform=axes4[2, col].transAxes, fontsize=8)
                axes4[2, col].set_facecolor('#1a1a1a')
            else:
                # Percentilová normalizace (eliminuje outliers)
                vmin_robust = np.percentile(feat_map, 2)
                vmax_robust = np.percentile(feat_map, 98)

                # Ochrana proti min==max
                if vmax_robust - vmin_robust < 1e-6:
                    vmax_robust = vmin_robust + 1e-6

                axes4[2, col].imshow(feat_map, cmap='RdBu_r', vmin=vmin_robust, vmax=vmax_robust)

            axes4[2, col].set_title(f'C2-{col}', fontsize=8)
            axes4[2, col].axis('off')

        # Feature maps z conv3 (prvních 10)
        feat3_np = feat3.cpu().squeeze().numpy()  # [128, 8, 8]

        print("\nDiagnostika Conv3 Feature Maps (prvních 10):")
        for col in range(10):
            feat_map = feat3_np[col]
            fmin, fmax = feat_map.min(), feat_map.max()
            fmean, fstd = feat_map.mean(), feat_map.std()
            n_unique = len(np.unique(feat_map))
            n_zeros = np.sum(feat_map == 0)
            n_total = feat_map.size

            print(f"  C3-{col}: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}, std={fstd:.4f}, "
                  f"unique={n_unique}, zeros={n_zeros}/{n_total} ({100*n_zeros/n_total:.1f}%)")

            # Robustní vizualizace
            if fstd < 1e-6:  # Konstantní nebo skoro konstantní
                axes4[3, col].text(0.5, 0.5, 'Dead\nfilter', ha='center', va='center',
                                  transform=axes4[3, col].transAxes, fontsize=8)
                axes4[3, col].set_facecolor('#1a1a1a')
            else:
                # Percentilová normalizace (eliminuje outliers)
                vmin_robust = np.percentile(feat_map, 2)
                vmax_robust = np.percentile(feat_map, 98)

                # Ochrana proti min==max
                if vmax_robust - vmin_robust < 1e-6:
                    vmax_robust = vmin_robust + 1e-6

                axes4[3, col].imshow(feat_map, cmap='RdBu_r', vmin=vmin_robust, vmax=vmax_robust)

            axes4[3, col].set_title(f'C3-{col}', fontsize=8)
            axes4[3, col].axis('off')

        plt.tight_layout()

# ============================================================================
# ZOBRAZENÍ
# ============================================================================

print("\n" + "=" * 60)
print("✓ Vizualizace připravena!")
print("=" * 60)
print("\nZobrazuji okna:")
print("  1. Conv Layer 1 Filters (32 filters)")
print("  2. Conv Layer 2 Filters (64 filters)")
print("  3. Conv Layer 3 Filters (128 filters)")
if test_dir and sample_found:
    print("  4. Feature Maps (ukázkové písmeno)")
print("\nZavřete okna pro ukončení programu.")

# Shromáždit všechny figures pro centrování a správný Z-order
figures = [fig1, fig2, fig3]
if test_dir and sample_found:
    figures.append(fig4)

# Centrovat a nastavit správné pořadí
center_and_order_figures(figures)

plt.show()

print("\n🚂 Dispatch dokončen!")
