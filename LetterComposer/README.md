# LetterComposer

GUI aplikace pro **slučování a rozdělování datasetů** z různých zdrojů (adresářů nebo ZIP archivů) do standardní struktury train/val/test.

## Co dělá LetterComposer?

LetterComposer je nástroj pro přípravu dat před trénováním modelu. Umožňuje:

1. **Sloučit data z více zdrojů** - můžete zkombinovat data nasbíraná na různých počítačích nebo od různých lidí
2. **Rozdělit dataset** na trénovací, validační a testovací část (train/val/test split)
3. **Stratifikované rozdělení** - každé písmeno A-Z je rozděleno nezávisle, takže všechny splity mají stejné rozložení tříd
4. **Reprodukovatelnost** - použitím stejného seedu dostanete vždy stejné rozdělení

## Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ LetterCollector │────▶│ LetterComposer  │────▶│  LetterLearner  │
│   (sběr dat)    │     │ (sloučení+split)│     │   (trénování)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
   collected/              composed/
   ├── A/                  ├── train/
   ├── B/                  │   ├── A/
   ...                     │   ├── B/
   └── Z/                  │   └── ...
                           ├── val/
                           │   └── ...
                           └── test/
                               └── ...
```

## Funkce

- **Přidat adresář** - načte data z lokální složky s písmeny A-Z
- **Přidat ZIP** - rozbalí a načte data z ZIP archivu
- **Náhled vzorků** - zobrazí náhodné vzorky z každého zdroje pro kontrolu kvality
- **Sliders** - nastavení poměru train/val/test (výchozí 70/20/10%)
- **Seed** - pro reprodukovatelné rozdělení

## Instalace

```bash
cd LetterComposer
run.bat
```

Skript automaticky vytvoří virtuální prostředí a nainstaluje závislosti.

## Vstupní struktura

Očekává adresáře s podložkami pojmenovanými A-Z:

```
collected/
├── A/
│   ├── 0001.bmp
│   └── ...
├── B/
│   └── ...
...
└── Z/
    └── ...
```

## Výstupní struktura

```
composed/
├── train/
│   ├── A/
│   │   ├── 010001.bmp   (prefix 01 = zdroj 1, 0001 = původní číslo)
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Pojmenování souborů

Výstupní soubory mají formát `SSOOOO.bmp`:
- `SS` - dvouciferné ID zdroje (01-99)
- `OOOO` - čtyřciferné původní číslo souboru

To umožňuje sledovat původ každého vzorku a kombinovat data z až 99 různých zdrojů.

---

<sub>Dokumentace vygenerována AI asistentem - prosinec 2025</sub>
