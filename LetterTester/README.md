# LetterTester

CLI nástroj pro kvantifikované testování natrénovaných modelů z LetterLearner. Vypočítá detailní metriky včetně confusion matrix, per-class accuracy, precision, recall a F1-score pro všech 26 písmen A-Z.

**Features:**

- 📊 **Grafická vizualizace výsledků** - Automaticky zobrazí interaktivní GUI s grafy
- 🎯 **Interaktivní výběr modelu** - Vyberte z seznamu dostupných modelů písmen
- 📁 **Volba testovacích dat** - Testujte na celém datasetu nebo jen test split
- 🖱️ **Klikací confusion matrix** - Klikněte na buňku a uvidíte příklady chyb
- 💾 **Export grafů** - Uložte vizualizace jako PNG
- 🔤 **26 tříd (A-Z)** - Plná podpora pro všechna písmena abecedy

## Instalace

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nejrychlejší je spustit `run.bat`, který všechno připraví automaticky.

## Spuštění

### Interaktivní režim (doporučeno)

Nejjednodušší použití s interaktivním výběrem:

```bash
run.bat
```

Skript se vás zeptá:

1. **Který model testovat?** - Zobrazí seznam všech modelů s metrikami
2. **Jaká data použít?** - Kompletní dataset (doporučeno) nebo jen test split

Po testování se automaticky otevře vizualizační okno s grafy.

### S explicitními parametry

```bash
run.bat --model_dir ../shared/models/run_YYYYMMDD_HHMMSS --data_dir ../shared/data/test
```

Nebo přímé spuštění Pythonu:

```bash
python main.py --model_dir ../shared/models/run_20251024_100000 --data_dir ../shared/data
```

## Příkazové řádkové parametry

### Základní parametry

- `--model_dir PATH` - Cesta ke složce s natrénovaným modelem (musí obsahovat `config.json` a `letter_cnn.pt`). Pokud není zadána, automaticky se použije nejnovější model z `../shared/models/`
- `--data_dir PATH` - Cesta k testovacím datům (struktura `A/`, `B/`, ..., `Z/` s BMP soubory). Default: interaktivní výběr

### Volitelné parametry

- `--batch_size N` - Velikost batche pro evaluaci (default: 64)
- `--use_cpu` - Vynutit použití CPU i když je GPU dostupné
- `--output PATH` - Cesta k výstupnímu JSON souboru (default: `../shared/tests/letter_test_results_TIMESTAMP.json`)

## Struktura dat

Testovací data očekávají stejnou strukturu jako LetterLearner:

```
test_data/
├── A/
│   ├── 0001.bmp
│   ├── 0002.bmp
│   └── ...
├── B/
│   ├── 0001.bmp
│   └── ...
...
└── Z/
    ├── 0001.bmp
    └── ...
```

## Interaktivní vizualizace

Po dokončení testování se automaticky otevře GUI okno s vizualizací výsledků.

### Co vizualizace zobrazuje

1. **Overall Score Panel**
   - Celková accuracy s barevným indikátorem (zelená >90%, žlutá 70-90%, červená <70%)
   - Average loss
   - Celkový počet testovacích vzorků
   - Počet tříd (26 písmen A-Z)

2. **Confusion Matrix Heatmap** (INTERAKTIVNÍ!)
   - Barevná mapa záměn 26×26 pro všechna písmena
   - **Klikněte na buňku** → Zobrazí se okno s 4-6 náhodnými příklady té konkrétní chyby
   - Vidíte skutečné obrázky, které model plete

3. **Per-Class Bar Charts**
   - Accuracy, Precision, Recall, F1-Score pro každé písmeno
   - Rychlý přehled slabých míst modelu

4. **Nejčastější chyby**
   - Top 8 nejčastějších záměn (např. O→Q, I→L, B→D)
   - Okamžitě vidíte, která písmena si model plete

5. **Export grafů**
   - Tlačítko pro uložení všech grafů jako PNG
   - Vhodné pro dokumentaci a prezentace

## Konzolový výstup

Kromě GUI vizualizace se výsledky vytisknou i do konzole:

```
==========================================================================================
VÝSLEDKY TESTOVÁNÍ
==========================================================================================

Overall Accuracy: 0.9234
Average Loss:     0.2456
Total Samples:    2600

Model:            run_20251218_140040
Data:             test
Device:           cuda

------------------------------------------------------------------------------------------
PER-CLASS METRICS
------------------------------------------------------------------------------------------
Class   Accuracy  Precision     Recall   F1-Score    Samples
------------------------------------------------------------------------------------------
    A     0.9500     0.9400     0.9500     0.9450        100
    B     0.9200     0.9100     0.9200     0.9149        100
    C     0.9300     0.9350     0.9300     0.9325        100
...
    Z     0.9100     0.9000     0.9100     0.9050        100
```

## Výsledky testování

Výsledky se ukládají do JSON souboru s kompletními informacemi:

```json
{
  "timestamp": "20251218_143025",
  "overall_accuracy": 0.9234,
  "average_loss": 0.2456,
  "total_samples": 2600,
  "num_classes": 26,
  "letters": ["A", "B", ..., "Z"],
  "per_class_metrics": {
    "A": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1_score": 0.945},
    ...
  },
  "confusion_matrix": [[...], ...],
  "predictions": [...]
}
```

## Typické záměny písmen

Model může mít problémy s podobnými písmeny:

- **O vs Q** - kruhový tvar
- **I vs L** - vertikální čáry
- **B vs D** - podobná křivka
- **M vs W** - zrcadlové tvary
- **C vs G** - částečné kruhy
- **P vs R** - podobný horní díl
- **V vs U** - dolní zakončení

Vizualizace vám pomůže identifikovat tyto problémy a případně nasbírat více trénovacích dat pro problematická písmena.

## Porovnání s DigitTester

| Feature | DigitTester | LetterTester |
|---------|-------------|--------------|
| Počet tříd | 10 (0-9) | 26 (A-Z) |
| Confusion matrix | 10×10 | 26×26 |
| Model soubor | digit_cnn.pt | letter_cnn.pt |
| Výstupní JSON | test_results_*.json | letter_test_results_*.json |
