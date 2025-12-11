# LetterTeaser

**Real-time klasifikátor ručně kreslených písmen A-Z**

## Co to dělá?

LetterTeaser je interaktivní nástroj pro testování natrénovaných modelů rozpoznávání písmen v reálném čase. Umožňuje vám "poškádlit" (tease) model tím, že kreslíte písmena rukou a okamžitě vidíte, jak model interpretuje vaši kresbu.

### Hlavní funkce

1. **Načtení modelu** - Načte natrénovaný PyTorch model ze složky obsahující `config.json` a `letter_cnn.pt`

2. **Kreslící plátno** - 256×256 pixelů velký canvas, kde kreslíte písmena myší

3. **Real-time inference** - Každých 50ms se spouští inference a aktualizuje zobrazení pravděpodobností

4. **Vizualizace pravděpodobností** - Dva sloupce s 26 písmeny (A-M a N-Z), každé s procentuální pravděpodobností:
   - Modře zvýrazněné písmeno s nejvyšší pravděpodobností
   - Šedé pruhy pro ostatní písmena
   - Procenta zobrazena vedle každého pruhu

## Jak to funguje?

```
┌─────────────────────────────────────────────────────────────────┐
│                        Letter Teaser                            │
├─────────────────────────┬───────────────────────────────────────┤
│   [Načíst model...]     │   Model: model_2024_12 (500 vzorků)   │
├─────────────────────────┼───────────────────────────────────────┤
│                         │   A: ████████████       45.2%         │
│  ┌─────────────────┐    │   B: ███                12.1%         │
│  │                 │    │   C: ██                  8.3%         │
│  │     Canvas      │    │   ...                                 │
│  │    256×256      │    │   M: █                   2.1%         │
│  │                 │    │                                       │
│  │   (kresba)      │    │   N: ██                  7.5%         │
│  │                 │    │   O: ███                10.2%         │
│  └─────────────────┘    │   ...                                 │
│                         │   Z: █                   1.8%         │
│      [Smazat]           │                                       │
└─────────────────────────┴───────────────────────────────────────┘
```

### Proces inference

1. **Preprocessing kresby**:
   - Zmenšení z 256×256 na 32×32 pixelů (LANCZOS interpolace)
   - Normalizace hodnot na rozsah 0-1
   - Převod na PyTorch tensor [1, 1, 32, 32]

2. **Model prediction**:
   - Forward pass přes SimpleCNN síť
   - Softmax pro převod logitů na pravděpodobnosti
   - Výběr třídy s nejvyšší pravděpodobností

3. **Vizualizace**:
   - Aktualizace všech 26 pruhů s aktuálními pravděpodobnostmi
   - Zvýraznění nejvyšší hodnoty modrou barvou

## Použití

### Spuštění
```batch
cd LetterTeaser
run.bat
```

### Ovládání
1. Klikněte na **"Načíst model..."** a vyberte složku s natrénovaným modelem
2. Kreslete písmena myší na černém canvasu
3. Sledujte v reálném čase, jak model rozpoznává vaši kresbu
4. Klikněte na **"Smazat"** pro vyčištění canvasu

## Požadavky na model

Model musí být ve složce obsahující:
- `config.json` - konfigurace s parametry `dropout` a `num_classes` (musí být 26)
- `letter_cnn.pt` - váhy natrénované sítě SimpleCNN

### Struktura config.json
```json
{
    "dropout": 0.2,
    "num_classes": 26,
    "epochs": 30,
    "batch_size": 32
}
```

## Technické detaily

- **Inference rate**: 20 FPS (každých 50ms)
- **Input size**: 32×32 grayscale
- **Output**: 26 tříd (A-Z)
- **Framework**: PyTorch + Tkinter
- **Architektura**: SimpleCNN (3 konvoluční vrstvy + 2 plně propojené)

## Tipy pro použití

1. **Kreslte ve středu** - Model je trénován na centrovaných písmenech
2. **Používejte celý canvas** - Velká písmena fungují lépe
3. **Jednoduché tahy** - Podobně jako při sběru dat v LetterCollector
4. **Testujte různé styly** - Zjistíte, kde má model problémy

## Struktura souborů

```
LetterTeaser/
├── main.py           # Hlavní aplikace
├── requirements.txt  # Python závislosti
├── run.bat          # Spouštěcí skript
└── README.md        # Tato dokumentace
```

## Závislosti

- `torch>=2.0.0` - PyTorch pro inference
- `pillow>=9.0.0` - Zpracování obrázků
- `numpy>=1.21.0` - Numerické operace
