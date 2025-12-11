# LetterCollector

Desktopová aplikace v Tkinteru pro sběr ručně psaných písmen (A-Z) v rozlišení 32×32 bodů. Plátno má tmavé pozadí a světlé tahy, uložené BMP soubory jsou kompatibilní s nástroji LetterComposer i LetterLearner.

## Automatické vyvažování datasetu

LetterCollector inteligentně vyvažuje dataset během sběru - **méně zastoupená písmena mají vyšší pravděpodobnost výběru**. Aplikace používá inverzní vážení (`weight = 1 / (count + 1)`), takže písmena s menším počtem vzorků se objevují častěji. To zajišťuje balancovaný dataset i při malém počtu vzorků, než převládne zákon velkých čísel.

## Instalace

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

Pohodlnější je spustit `run.bat`, který virtuální prostředí i závislosti připraví automaticky.

## Spuštění

```bash
run.bat
```

Otevře se okno s kreslicím plátnem. Levým tlačítkem myši kreslete písmeno, `Escape` vymaže plátno a `Enter` uloží aktuální kresbu do `shared/data/collected/<písmeno>/xxxx.bmp`.

Krátký stisk mezerníku přepne pohled na statistiku nasbíraných písmen, podržení mezerníku statistiku zobrazí pouze dočasně.

## Struktura výstupu

Data se ukládají do centrálního úložiště:

```
shared/data/collected/
├── A/
│   ├── 0001.bmp
│   ├── 0002.bmp
│   └── ...
├── B/
│   └── ...
...
└── Z/
    └── ...
```

Každý BMP soubor má 32×32 pixelů v odstínech šedi. Soubory lze přímo použít v LetterComposer nebo LetterLearner.

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST-Letters (Digital Identification Exercise - MNIST for Letters), který používá centrální adresářovou strukturu pro snadnou správu dat a modelů. Viz hlavní README pro kompletní workflow.

## Tipy

- Kompletní reset je možné provést smazáním obsahu složky `../shared/data/collected/`.
- Šířku kreslicího tahu a další parametry lze upravit v horní části souboru `main.py`.

---

<sub>Dokumentace vygenerována AI asistentem - prosinec 2025</sub>
