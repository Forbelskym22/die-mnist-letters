# LetterLearner

Trénovací část projektu pro vlastní alternativu datasetu MNIST s písmeny A-Z. Skript `train.py` načítá obrázky vytvořené nástroji LetterCollector i LetterComposer, trénuje na nich malou konvoluční síť a ukládá nejlepší model společně s historií učení.

## Jak začít

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nebo spusťte `run.bat`, který virtuální prostředí vytvoří, nainstaluje potřebné balíčky (PyTorch, Torchvision, NumPy, Pillow, tqdm, matplotlib) a ponechá je připravené pro další použití.

Spuštění `run.bat` bez parametrů nejprve otevře editor hyperparametrů `marshall.py`. Po jeho zavření (a případném uložení změn do `config.json`) se automaticky spustí `train.py` se stejnými argumenty, které byly předány dávce. Po dokončení tréninku se automaticky spustí vizualizace modelu (`dispatch.py`) a okno zůstane otevřené, dokud nepotvrdíte zprávu. Pokud spouštíte dávku z vlastního terminálu, přidejte jako první argument `--no-pause`.

## Konfigurace

Hyperparametry (počty epoch, velikost batch, tempo učení, dropout…) jsou v souboru `config.json`. Pokud chybí, `train.py` použije vestavěné výchozí hodnoty a soubor automaticky vytvoří. Každý parametr lze přepsat argumentem příkazové řádky (`--epochs`, `--learning_rate`, `--step_size`, `--dropout`, `--seed`, …).

Spuštěním `run.bat marshall` otevřete pouze editor konfigurace (bez následného tréninku); volbou `run.bat train ...` zase spustíte jen trénink a editor přeskočíte.

## Struktura dat

Data se očekávají ve složce `../shared/data/composed/` nebo v cestě zadané parametrem `--data_dir`. Lze použít buď jednoduchou strukturu:

```
data/
 ├─A/00001.bmp
 ├─B/00001.bmp
 └─… (až Z/)
```

nebo variantu se splitem (doporučeno):

```
composed/
 ├─train/A/…, B/…, … Z/…
 ├─val/A/…, B/…, … Z/…
 └─test/A/…, B/…, … Z/…
```

V druhém případě se poměry `--val_ratio` a `--test_ratio` ignorují a dataset se použije tak, jak je.

## Spuštění tréninku

```bash
run.bat --epochs 25
```

Pro čistě dávkové spuštění bez editoru použijte `run.bat train --epochs 25`.

Skript nahraje data, vytrénuje model a do složky `../shared/models/run_YYYYMMDD_HHMMSS[_NN]/` uloží:

- `config.json` – použitou konfiguraci (včetně parametrů z CLI a num_classes),
- `letter_cnn.pt` – váhy nejlepšího modelu dle validační chyby,
- `training_history.json` – průběh metrik po epochách,
- případně `test_metrics.json`, pokud byla konfigurace spuštěna i na testovací sadě.

Parametr `--use_cpu` vynutí běh na CPU; jinak se automaticky použije GPU, pokud je dostupné.

## Vizualizace modelu

Po úspěšném natrénování se automaticky spustí vizualizace modelu (`dispatch.py`), která zobrazí:

- **Architekturu** - textový summary s počtem parametrů
- **Konvoluční filtry** - naučené váhy všech tří vrstev (32, 64, 128 filtrů)
- **Feature maps** - průchod ukázkového písmene sítí po jednotlivých vrstvách

**Manuální spuštění vizualizace:**

```bash
python dispatch.py                              # Auto-detekce nejnovějšího modelu
python dispatch.py --model_dir PATH             # Specifický model
```

Vizualizace se automaticky volá v rámci `run.bat` workflow, ale můžete ji spustit samostatně kdykoliv později.

## Architektura sítě

SimpleCNN je malá konvoluční síť optimalizovaná pro 32×32 grayscale obrázky:

```
Input: 1×32×32 (grayscale písmeno)
    ↓
Conv2d(1→32, 3×3) + BatchNorm + ReLU + MaxPool → 32×16×16
    ↓
Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool → 64×8×8
    ↓
Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool → 128×4×4
    ↓
Flatten → 2048
    ↓
Linear(2048→128) + ReLU + Dropout
    ↓
Linear(128→26) → Output (A-Z)
```

## Rozdíly oproti DigitLearner

| Vlastnost | DigitLearner | LetterLearner |
|-----------|--------------|---------------|
| Počet tříd | 10 (0-9) | 26 (A-Z) |
| Název modelu | digit_cnn.pt | letter_cnn.pt |
| Dataset | DigitDataset | LetterDataset |
| Složky | 0-9 | A-Z |

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST (Digital Identification Exercise - MNIST), který používá centrální adresářovou strukturu:

**Vstupní data**: `../shared/data/composed/` (vytvořená pomocí LetterComposer)
- train/ - trénovací data (podsložky A-Z)
- val/ - validační data (podsložky A-Z)
- test/ - testovací data (podsložky A-Z)

**Výstupní modely**: `../shared/models/run_YYYYMMDD_HHMMSS/`
- config.json - konfigurace běhu (včetně num_classes)
- letter_cnn.pt - natrénované váhy
- training_history.json - průběh tréninku
- test_metrics.json - výsledky na testovací sadě

## Workflow

Typický workflow pro trénink modelu písmen:

1. **Sběr dat** - LetterCollector (ruční kreslení A-Z)
2. **Kompozice datasetu** - LetterComposer (rozdělení na train/val/test)
3. **Trénink** - LetterLearner (tento nástroj)
4. **Testování** - LetterTeaser (real-time inference)
