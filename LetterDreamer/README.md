# LetterDreamer

Generátor syntetických písmen A-Z pro trénink neuronové sítě. Vytváří 32x32 grayscale BMP obrázky pomocí různých systémových fontů s náhodnými transformacemi.

## Co to dělá?

LetterDreamer generuje syntetická trénovací data pro model rozpoznávání písmen:

1. **Hledá systémové fonty** - Arial, Verdana, Times, Calibri, Comic Sans, atd.
2. **Renderuje písmena** - Každé písmeno A-Z ve více variantách
3. **Aplikuje augmentace**:
   - Rotace (±15°)
   - Změna velikosti (0.7x - 1.3x)
   - Posun v rámci obrázku
   - Variace tloušťky (normal/bold/thin)
   - Gaussian šum
   - Rozmazání
4. **Konzistentní pozadí** - Vždy černé pozadí s bílými písmeny (vysoký kontrast)
5. **Vysoká viditelnost** - Intenzita písmen 200-255 pro jasné zobrazení
6. **Ukládá ve struktuře** pro LetterLearner

## Použití

### Základní generování
```batch
cd LetterDreamer
run.bat
```

### S parametry
```batch
run.bat --samples 1000 --split --clean
```

### Všechny parametry
| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--output_dir` | `../shared/data/synthetic` | Výstupní složka |
| `--samples` | 500 | Počet vzorků na třídu |
| `--noise` | 18.0 | Sigma Gaussova šumu |
| `--seed` | 1234 | Seed pro reprodukovatelnost |
| `--clean` | false | Smazat existující data |
| `--split` | false | Rozdělit na train/val/test |
| `--val_split` | 0.2 | Poměr validačních dat |
| `--test_split` | 0.1 | Poměr testovacích dat |
| `--blur` | 0.8 | Max radius rozmazání |
| `--lowercase` | false | Zahrnout malá písmena (30% šance) |

## Čištění dat

Pro odstranění syntetických dat použijte cleanup skript:

```batch
:: Interaktivní režim
python cleanup.py

:: Smazat synthetic/ složku
python cleanup.py --synthetic

:: Smazat train/val/test složky
python cleanup.py --composed

:: Smazat vše bez potvrzení
python cleanup.py --all --force

:: Zobrazit pouze statistiky
python cleanup.py --stats
```

## Příklady

### Malý dataset pro testování
```batch
run.bat --samples 100 --split
```

### Velký dataset pro produkční trénink
```batch
run.bat --samples 2000 --split --clean --lowercase
```

### Dataset bez augmentací
```batch
run.bat --samples 500 --noise 5 --blur 0
```

## Výstupní struktura

### Bez `--split`:
```
synthetic/
├── A/0001.bmp, 0002.bmp, ...
├── B/0001.bmp, 0002.bmp, ...
└── ... Z/
```

### S `--split`:
```
synthetic/
├── train/A/, B/, ... Z/
├── val/A/, B/, ... Z/
└── test/A/, B/, ... Z/
```

## Kombinace s ručně sbíranými daty

Pro nejlepší výsledky zkombinujte syntetická data s ručně nasbíranými:

1. **Sbírejte ručně** pomocí LetterCollector
2. **Generujte synteticky** pomocí LetterDreamer
3. **Kombinujte** pomocí LetterComposer
4. **Trénujte** pomocí LetterLearner

```batch
:: 1. Generovat syntetická data
cd LetterDreamer
run.bat --samples 500 --split --output_dir ../shared/data/synthetic

:: 2. Kombinovat (v LetterComposer vyberte obě složky)
cd ../LetterComposer
run.bat

:: 3. Trénovat
cd ../LetterLearner
run.bat train
```

## Podporované fonty

LetterDreamer automaticky hledá tyto fonty ve Windows:
- **Sans-serif**: Arial, Verdana, Tahoma, Calibri, Segoe UI
- **Serif**: Times New Roman, Georgia, Garamond
- **Monospace**: Consolas, Courier New
- **Handwriting**: Comic Sans, Segoe Script

Pokud žádný font není nalezen, použije se fallback.

## Technické detaily

- **Výstupní formát**: 32×32 grayscale BMP
- **Počet tříd**: 26 (A-Z)
- **Max souborů**: 9999 na třídu (kvůli 4-digit naming)
- **Deduplikace**: MD5 hash pro unikátnost obrázků
