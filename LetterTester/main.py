"""
LetterTester - Kvantifikované testování natrénovaných modelů písmen.

CLI nástroj pro evaluaci modelů z LetterLearner na testovacích datech.
Vypočítá detailní metriky včetně confusion matrix a per-class accuracy.

Features:
- Interaktivní výběr modelu ze seznamu dostupných modelů
- Interaktivní výběr testovacích dat (kompletní dataset / test split / train+val)
- Automatická vizualizace výsledků s interaktivní confusion matrix
- Export grafů jako PNG
- Podpora 26 tříd (A-Z)

Spuštění:
    python main.py                    # Interaktivní výběr
    python main.py --model_dir PATH   # Explicitní model
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# 26 tříd písmen A-Z
LETTERS = [chr(ord('A') + i) for i in range(26)]
NUM_CLASSES = 26


class SimpleCNN(nn.Module):
    """Malá konvoluční síť vhodná pro 32x32 písmena."""

    def __init__(self, num_classes: int = 26, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class LetterDataset(Dataset):
    """Dataset, který čte BMP soubory ze struktury složek A-Z."""

    def __init__(self, root_dirs: List[Path], verify_size: bool = True) -> None:
        if isinstance(root_dirs, Path):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.verify_size = verify_size
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for root_dir in self.root_dirs:
            if not root_dir.exists():
                raise FileNotFoundError(f"Složka s daty '{root_dir}' neexistuje.")

            for label_idx, letter in enumerate(LETTERS):
                class_dir = root_dir / letter
                if not class_dir.exists():
                    continue
                for img_path in sorted(class_dir.glob("*.bmp")):
                    self.samples.append((img_path, label_idx))

        if not self.samples:
            raise ValueError(f"V zadaných složkách nejsou žádná .bmp data.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")

        if self.verify_size and img.size != (32, 32):
            raise ValueError(f"Obrázek {img_path} má velikost {img.size}, očekává se (32, 32).")

        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor, label, str(img_path)


def load_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Načte model a konfiguraci ze složky."""
    config_path = model_dir / "config.json"
    model_path = model_dir / "letter_cnn.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurace nenalezena: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model nenalezen: {model_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    dropout = config.get("dropout", 0.2)
    num_classes = config.get("num_classes", NUM_CLASSES)

    model = SimpleCNN(num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, config


def evaluate_with_confusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> Tuple[float, float, np.ndarray, List[Dict[str, object]]]:
    """Evaluuje model a vrátí loss, accuracy, confusion matrix a predictions."""
    model.eval()
    running_loss = 0.0
    total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            _, predicted = torch.max(outputs, 1)
            for path, true_label, pred_label in zip(paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true_label][pred_label] += 1
                predictions.append({
                    "path": path,
                    "true": int(true_label),
                    "pred": int(pred_label)
                })

    average_loss = running_loss / total if total else 0.0
    accuracy = confusion.trace() / total if total else 0.0

    return average_loss, accuracy, confusion, predictions


def compute_per_class_metrics(confusion: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Vypočítá per-class precision, recall, f1-score."""
    num_classes = confusion.shape[0]
    metrics = {}

    for cls_idx in range(num_classes):
        letter = LETTERS[cls_idx]
        tp = confusion[cls_idx, cls_idx]
        fp = confusion[:, cls_idx].sum() - tp
        fn = confusion[cls_idx, :].sum() - tp
        tn = confusion.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / confusion[cls_idx, :].sum() if confusion[cls_idx, :].sum() > 0 else 0.0

        metrics[letter] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "accuracy": float(accuracy),
            "samples": int(confusion[cls_idx, :].sum()),
        }

    return metrics


def save_results(results: Dict, output_path: Path) -> None:
    """Uloží výsledky do JSON souboru."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nVýsledky uloženy do: {output_path}")


def print_results(results: Dict) -> None:
    """Vytiskne výsledky do konzole."""
    print("\n" + "=" * 90)
    print("VÝSLEDKY TESTOVÁNÍ")
    print("=" * 90)
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Average Loss:     {results['average_loss']:.4f}")
    print(f"Total Samples:    {results['total_samples']}")
    print(f"\nModel:            {results['model_dir']}")
    print(f"Data:             {results['data_dir']}")
    print(f"Device:           {results['device']}")

    print("\n" + "-" * 90)
    print("PER-CLASS METRICS")
    print("-" * 90)
    print(f"{'Class':>5} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Samples':>10}")
    print("-" * 90)

    per_class = results["per_class_metrics"]
    for letter in LETTERS:
        if letter in per_class:
            m = per_class[letter]
            print(
                f"{letter:>5} "
                f"{m['accuracy']:>10.4f} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f} "
                f"{m['samples']:>10}"
            )

    print("\n" + "-" * 90)
    print("CONFUSION MATRIX (zkráceno - pro detaily viz vizualizace)")
    print("-" * 90)
    print("Řádky = skutečné třídy, Sloupce = predikované třídy")
    print()

    confusion = np.array(results["confusion_matrix"])
    num_classes = confusion.shape[0]

    # Header - A-Z
    print("     ", end="")
    for letter in LETTERS:
        print(f"{letter:>4}", end="")
    print()
    print("     " + "-" * (4 * num_classes))

    # Rows
    for i in range(num_classes):
        print(f"{LETTERS[i]:>4} |", end="")
        for j in range(num_classes):
            val = confusion[i, j]
            if val == 0:
                print("   .", end="")
            elif val < 10:
                print(f"  {val:1d}", end=" ")
            elif val < 100:
                print(f" {val:2d}", end=" ")
            else:
                print(f"{val:3d}", end=" ")
        print()

    print("=" * 90 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LetterTester - Kvantifikované testování modelů z LetterLearner"
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Cesta ke složce s modelem (default: nejnovější v ../shared/models/)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("..") / "shared" / "data" / "test",
        help="Cesta k testovacím datům (default: interaktivní výběr)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Velikost batche pro evaluaci (default: 64)",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Vynutit použití CPU i když je GPU dostupné",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Cesta k výstupnímu JSON souboru (default: ../shared/tests/test_results_TIMESTAMP.json)",
    )

    return parser.parse_args()


def get_model_info(model_dir: Path) -> Dict:
    """Načte informace o modelu z config.json a test_metrics.json."""
    info = {
        "name": model_dir.name,
        "path": model_dir,
        "epochs": "?",
        "accuracy": "?",
        "test_acc": None,
    }

    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                info["epochs"] = config.get("epochs", "?")
        except Exception:
            pass

    # Load test metrics if available
    test_metrics_path = model_dir / "test_metrics.json"
    if test_metrics_path.exists():
        try:
            with test_metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
                info["test_acc"] = metrics.get("test_accuracy")
        except Exception:
            pass

    # Load training history for final accuracy
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as f:
                history = json.load(f)
                if history:
                    last_epoch = history[-1]
                    info["accuracy"] = last_epoch.get("val_acc", last_epoch.get("train_acc", "?"))
        except Exception:
            pass

    return info


def detect_data_structure(data_dir: Path) -> Dict:
    """Detekuje strukturu datasetu a vrátí informace o splits."""
    info = {
        "has_splits": False,
        "splits": {},
        "total": 0,
    }

    if not data_dir.exists():
        return info

    # Check if it has train/val/test structure
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if train_dir.exists() and train_dir.is_dir():
        info["has_splits"] = True

        # Count samples in each split
        for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            if split_dir.exists() and split_dir.is_dir():
                count = 0
                for letter in LETTERS:
                    letter_dir = split_dir / letter
                    if letter_dir.exists():
                        count += len(list(letter_dir.glob("*.bmp")))
                info["splits"][split_name] = count
                info["total"] += count

    else:
        # Flat structure (just A-Z directories)
        count = 0
        for letter in LETTERS:
            letter_dir = data_dir / letter
            if letter_dir.exists():
                count += len(list(letter_dir.glob("*.bmp")))
        info["total"] = count

    return info


def create_test_dataset(selected_path: Path) -> LetterDataset:
    """Vytvoří dataset na základě vybrané cesty."""
    if selected_path.name == "train+val":
        # Special case: combine train and val
        base_dir = selected_path.parent
        dirs_to_load = []
        for split in ["train", "val"]:
            split_dir = base_dir / split
            if split_dir.exists():
                dirs_to_load.append(split_dir)
        return LetterDataset(dirs_to_load)

    # Check if it's a directory with train/val/test subdirs (user selected "all")
    if (selected_path / "train").exists():
        # Load all splits
        dirs_to_load = []
        for split in ["train", "val", "test"]:
            split_dir = selected_path / split
            if split_dir.exists():
                dirs_to_load.append(split_dir)
        return LetterDataset(dirs_to_load)

    # Otherwise, just load the selected directory
    return LetterDataset([selected_path])


def select_test_data_interactive(base_data_dir: Path) -> Optional[Path]:
    """Interaktivní výběr testovacích dat."""
    if not base_data_dir.exists():
        print(f"Chyba: Složka '{base_data_dir}' neexistuje")
        return None

    structure = detect_data_structure(base_data_dir)

    if not structure["has_splits"]:
        # Flat structure - just use it directly
        if structure["total"] == 0:
            print(f"Chyba: V '{base_data_dir}' nejsou žádná data")
            return None
        print(f"\nPoužívám kompletní dataset ({structure['total']} vzorků)")
        return base_data_dir

    # Has train/val/test splits - offer choices
    splits = structure["splits"]

    print("\n" + "=" * 80)
    print("VÝBĚR TESTOVACÍCH DAT")
    print("=" * 80)
    print(f"Dostupné splits v {base_data_dir.name}/:")

    split_info = []
    for split_name in ["train", "val", "test"]:
        count = splits.get(split_name, 0)
        if count > 0:
            split_info.append(f"{split_name}: {count} vzorků")

    print("  " + "  |  ".join(split_info))
    print("-" * 80)

    # Build options
    options = []
    option_paths = []

    # Option 1: Complete dataset
    options.append(f"1. Kompletní dataset ({structure['total']} vzorků) [DOPORUČENO]")
    options.append("   → Testuje na všech nasbíraných datech")
    option_paths.append("all")

    # Option 2: Only test split (if exists)
    if "test" in splits and splits["test"] > 0:
        options.append(f"\n2. Pouze test split ({splits['test']} vzorků)")
        options.append("   → Reprodukce výsledků z tréninku")
        option_paths.append("test")

    # Option 3: Train + Val (if exists)
    train_val_count = splits.get("train", 0) + splits.get("val", 0)
    if train_val_count > 0:
        options.append(f"\n3. Train + Val splits ({train_val_count} vzorků)")
        options.append("   → Data, která model viděl během učení")
        option_paths.append("train+val")

    for opt in options:
        print(opt)

    print("=" * 80)
    print()

    # Get user choice
    while True:
        try:
            choice = input(f"Vyberte [1-{len(option_paths)}] (Enter = kompletní dataset): ").strip()

            if choice == "":
                # Default: complete dataset
                choice = "1"

            choice_num = int(choice)
            if 1 <= choice_num <= len(option_paths):
                selected_option = option_paths[choice_num - 1]

                if selected_option == "all":
                    print(f"\nVybrána všechna data ({structure['total']} vzorků)")
                    return base_data_dir
                elif selected_option == "test":
                    test_dir = base_data_dir / "test"
                    print(f"\nVybrán test split ({splits['test']} vzorků)")
                    return test_dir
                elif selected_option == "train+val":
                    # For train+val, we need to combine them
                    # Return base_data_dir with a marker
                    print(f"\nVybrány train+val splits ({train_val_count} vzorků)")
                    return base_data_dir / "train+val"  # Special marker
            else:
                print(f"Chyba: Zadejte číslo mezi 1 a {len(option_paths)}")

        except ValueError:
            print("Chyba: Zadejte platné číslo nebo Enter")
        except KeyboardInterrupt:
            print("\n\nTestování zrušeno.")
            return None


def select_model_interactive(models_dir: Path) -> Optional[Path]:
    """Interaktivní výběr modelu ze seznamu."""
    if not models_dir.exists():
        print("Chyba: Složka ../shared/models/ neexistuje")
        print("Spusťte nejprve LetterLearner nebo zadejte --model_dir explicitně")
        return None

    # Find model directories that contain letter_cnn.pt
    run_dirs = sorted(
        [p for p in models_dir.glob("run_*") if p.is_dir() and (p / "letter_cnn.pt").exists()],
        key=lambda p: p.name,
        reverse=True
    )

    if not run_dirs:
        print("Chyba: Žádné modely písmen nenalezeny v ../shared/models/")
        print("Spusťte nejprve LetterLearner nebo zadejte --model_dir explicitně")
        return None

    # Show available models
    print("\n" + "=" * 80)
    print("DOSTUPNÉ MODELY PÍSMEN")
    print("=" * 80)
    print(f"{'#':<4} {'Model':<25} {'Epochs':<8} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 80)

    model_infos = []
    for idx, run_dir in enumerate(run_dirs, 1):
        info = get_model_info(run_dir)
        model_infos.append(info)

        acc_str = f"{info['accuracy']:.4f}" if isinstance(info['accuracy'], float) else str(info['accuracy'])
        test_acc_str = f"{info['test_acc']:.4f}" if info['test_acc'] is not None else "-"

        print(f"{idx:<4} {info['name']:<25} {info['epochs']:<8} {acc_str:<10} {test_acc_str:<10}")

    print("=" * 80)
    print()

    # Get user choice
    while True:
        try:
            choice = input(f"Vyberte model [1-{len(run_dirs)}] (Enter = nejnovější): ").strip()

            if choice == "":
                # Default: newest model
                return run_dirs[0]

            choice_num = int(choice)
            if 1 <= choice_num <= len(run_dirs):
                selected = run_dirs[choice_num - 1]
                print(f"\nVybrán model: {selected.name}")
                return selected
            else:
                print(f"Chyba: Zadejte číslo mezi 1 a {len(run_dirs)}")

        except ValueError:
            print("Chyba: Zadejte platné číslo nebo Enter")
        except KeyboardInterrupt:
            print("\n\nTestování zrušeno.")
            return None


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("🔤 LetterTester - Testování modelu písmen A-Z")
    print("=" * 60)

    # Select model if not specified
    if args.model_dir is None:
        models_dir = Path("..") / "shared" / "models"
        args.model_dir = select_model_interactive(models_dir)

        if args.model_dir is None:
            return

    # Determine device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Používám zařízení: {device}")

    # Load model
    print(f"Načítám model z: {args.model_dir}")
    model, config = load_model(args.model_dir, device)
    num_classes = config.get("num_classes", NUM_CLASSES)

    # Select test data if not specified via CLI
    if args.data_dir == Path("..") / "shared" / "data" / "test":
        # Default path - offer interactive selection
        base_data_dir = Path("..") / "shared" / "data"
        selected_data_path = select_test_data_interactive(base_data_dir)

        if selected_data_path is None:
            return
    else:
        # User specified via CLI - use directly
        selected_data_path = args.data_dir
        print(f"\nPoužívám data z: {selected_data_path}")

    # Load test data
    print(f"\nNačítám testovací data...")
    test_dataset = create_test_dataset(selected_data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Nalezeno {len(test_dataset)} testovacích vzorků")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    print("\nProvádím evaluaci...")
    avg_loss, accuracy, confusion, predictions = evaluate_with_confusion(
        model, test_loader, criterion, device, num_classes
    )

    # Compute per-class metrics
    per_class_metrics = compute_per_class_metrics(confusion)

    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "model_dir": str(args.model_dir.absolute()),
        "data_dir": str(args.data_dir.absolute()),
        "device": str(device),
        "batch_size": args.batch_size,
        "overall_accuracy": float(accuracy),
        "average_loss": float(avg_loss),
        "total_samples": len(test_dataset),
        "num_classes": num_classes,
        "letters": LETTERS,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions,
        "model_config": config,
    }

    # Print results
    print_results(results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = Path("../shared/tests") / f"letter_test_results_{timestamp}.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_results(results, output_path)

    # Launch visualization
    try:
        from visualize import ResultsVisualizer
        print("\nSpouštím vizualizaci...")
        viz = ResultsVisualizer(output_path)
        viz.show()
        # Vizualizace úspěšně dokončena, skript automaticky skončí
    except ImportError as e:
        print(f"\nVizualizace není dostupná: {e}")
        print("Pro vizualizaci nainstalujte: pip install matplotlib")
        input("\nStiskněte Enter pro ukončení...")
    except Exception as e:
        print(f"\nChyba při spuštění vizualizace: {e}")
        input("\nStiskněte Enter pro ukončení...")


if __name__ == "__main__":
    main()
