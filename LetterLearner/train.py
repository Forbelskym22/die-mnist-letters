"""
Jednoduchý trénovací skript pro neuronovou síť nad vlastní verzí datasetu písmen.

Spuštění:
    python train.py --data_dir ../LetterCollector/letters --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

DEFAULT_CONFIG_PATH = Path("config.json")
DEFAULT_MODELS_DIR = Path("..") / "shared" / "models"
DEFAULT_DATA_DIR = Path("..") / "shared" / "data"

# Letters A-Z for classification
LETTERS = [chr(ord('A') + i) for i in range(26)]
NUM_CLASSES = 26

@dataclass
class EpochStats:
    """Pomocná struktura pro ukládání průběžných výsledků."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

@dataclass
class HyperParams:
    """Konfigurace tréninku načítaná z konfigu nebo CLI."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    step_size: int = 10
    lr_gamma: float = 0.5
    dropout: float = 0.2
    seed: int = 42

    def update_from_dict(self, data: Dict[str, object]) -> None:
        for field in fields(self):
            if field.name in data:
                setattr(self, field.name, data[field.name])

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def as_dict(self) -> Dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def validate(self) -> None:
        if not 0.0 <= self.val_ratio <= 0.5:
            raise ValueError("val_ratio ocekavam v intervalu <0.0, 0.5>.")
        if not 0.0 <= self.test_ratio <= 0.5:
            raise ValueError("test_ratio ocekavam v intervalu <0.0, 0.5>.")
        if self.val_ratio + self.test_ratio >= 0.8:
            raise ValueError("Na trenink musi zustat alespon 20 % dat (upravte pomery val/test).")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout ocekavam v intervalu <0.0, 1.0>.")
        if self.step_size <= 0:
            raise ValueError("step_size musi byt kladne cislo.")
        if self.lr_gamma <= 0.0 or self.lr_gamma > 1.0:
            raise ValueError("lr_gamma ocekavam v rozsahu (0.0, 1.0].")
        if self.epochs <= 0:
            raise ValueError("epochs musi byt kladne cislo.")
        if self.batch_size <= 0:
            raise ValueError("batch_size musi byt kladne cislo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate musi byt kladne cislo.")

class LetterDataset(Dataset):
    """Dataset, který čte BMP soubory ze struktury složek A-Z."""

    def __init__(self, root_dir: Path, verify_size: bool = True) -> None:
        self.root_dir = root_dir
        self.verify_size = verify_size
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Složka s daty '{self.root_dir}' neexistuje.")

        for label_dir in sorted(self.root_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            dir_name = label_dir.name.upper()
            
            # Support both letter names (A-Z) and numeric indices (0-25)
            if len(dir_name) == 1 and dir_name in LETTERS:
                label = LETTERS.index(dir_name)
            else:
                try:
                    label = int(label_dir.name)
                    if label < 0 or label >= NUM_CLASSES:
                        continue  # Skip invalid numeric labels
                except ValueError:
                    continue  # Skip non-letter, non-numeric directories

            for image_path in sorted(label_dir.glob("*.bmp")):
                self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError("Nenašel jsem žádné BMP soubory. Zkontrolujte cestu k datasetu.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        # BMP načtu přes PIL a okamžitě překlopím do odstínů šedi.
        image = Image.open(image_path).convert("L")

        if self.verify_size and image.size != (32, 32):
            raise ValueError(
                f"Soubor {image_path} má rozměr {image.size}, očekáváno 32x32 px."
            )

        # Z obrázku udělám pole hodnot v rozsahu 0-1 a přidám kanál [1, 32, 32].
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)

        return tensor, torch.tensor(label, dtype=torch.long)

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

def set_seed(seed: int) -> None:
    """Nastaví seed pro reprodukovatelnost."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_loaders(
    dataset: LetterDataset,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, DataLoader]:
    """Rozdělí dataset na train/val/test a připraví DataLoadery."""
    total_len = len(dataset)
    if total_len < 3:
        raise ValueError("Dataset je příliš malý na rozdělení.")

    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    if train_len <= 0:
        raise ValueError("Zkontrolujte poměry pro validaci/test – na trénink nezbyla data.")

    generator = torch.Generator().manual_seed(seed)
    parts = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    loaders = {
        "train": DataLoader(parts[0], batch_size=batch_size, shuffle=True, num_workers=0),
    }
    if val_len > 0:
        loaders["val"] = DataLoader(parts[1], batch_size=batch_size, shuffle=False, num_workers=0)
    if test_len > 0:
        loaders["test"] = DataLoader(parts[2], batch_size=batch_size, shuffle=False, num_workers=0)
    return loaders


def create_loaders_from_subdirectories(
    base_dir: Path,
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Vytvoří DataLoadery přímo ze složek train/val/test."""
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dir = base_dir / split
        if split_dir.exists() and split_dir.is_dir():
            subset = LetterDataset(split_dir)
            if len(subset) == 0:
                continue
            loaders[split] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=0,
            )
    if "train" not in loaders:
        raise ValueError(f"Očekávám složku '{base_dir / 'train'}' s daty.")
    return loaders

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Provede jeden trenovaci epoch vcetne sberu statistik."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += inputs.size(0)

        avg_loss = running_loss / total if total else 0.0
        avg_acc = correct / total if total else 0.0
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

    progress.close()

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: Optional[str] = None,
) -> Tuple[float, float]:
    """Spocita chybu a presnost na validacnich nebo testovacich datech."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = None
    iterator = loader
    if desc:
        progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        iterator = progress

    with torch.no_grad():
        for inputs, targets in iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += inputs.size(0)

            if progress is not None:
                avg_loss = running_loss / total if total else 0.0
                avg_acc = correct / total if total else 0.0
                progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

    if progress is not None:
        progress.close()

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy

def save_history(history: List[EpochStats], output_dir: Path) -> None:
    """Uloží historii trénování do JSONu."""
    history_path = output_dir / "training_history.json"
    serialized = [asdict(entry) for entry in history]
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=False)

def load_hyperparams(config_path: Path, args: argparse.Namespace) -> Tuple[HyperParams, Path, bool]:
    """Nacti hyperparametry z konfigu a aplikuj CLI overrides."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    hp = HyperParams()
    loaded = False
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("Konfiguracni JSON musi obsahovat objekt (dict).")
            hp.update_from_dict(data)
            loaded = True
    elif config_path != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(f"Konfiguracni soubor {config_path} neexistuje.")

    overrides = {field.name: getattr(args, field.name, None) for field in fields(HyperParams)}
    hp.apply_overrides(overrides)
    hp.validate()
    return hp, config_path, loaded

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trains the letter classifier on data from LetterCollector."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON hyperparameter file.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with subfolders A-Z containing BMP images (default `data`).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory for all runs (default `models`).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Override validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=None, help="Override test split ratio.")
    parser.add_argument("--step_size", type=int, default=None, help="Override StepLR step_size.")
    parser.add_argument("--lr_gamma", type=float, default=None, help="Override StepLR gamma.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout probability.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    hyperparams, config_path, config_loaded = load_hyperparams(args.config, args)

    set_seed(hyperparams.seed)

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    config_state = "loaded" if config_loaded else "defaults used"
    config_status = f"{config_path} ({config_state})"

    base_dir: Path = args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    config_output_path = run_dir / "config.json"
    with config_output_path.open("w", encoding="utf-8") as handle:
        json.dump(hyperparams.as_dict(), handle, indent=2)

    tqdm.write("")
    tqdm.write("+++ LetterLearner training setup +++")
    tqdm.write("")
    setup_rows = [
        ("Config file", config_status, "--config"),
        ("Run directory", run_dir, "(auto)"),
        ("Data directory", args.data_dir, "--data_dir"),
        ("Models root", base_dir, "--output_dir"),
        ("Hardware target", device, "--use_cpu  force CPU"),
        ("Epochs / batch size", f"{hyperparams.epochs} / {hyperparams.batch_size}", "--epochs / --batch_size"),
        ("Learning rate", f"{hyperparams.learning_rate:g}", "--learning_rate"),
        ("Val/Test split", f"{hyperparams.val_ratio:.2f} / {hyperparams.test_ratio:.2f}", "--val_ratio / --test_ratio"),
        ("Dropout", f"{hyperparams.dropout:.2f}", "--dropout"),
        ("LR step/gamma", f"{hyperparams.step_size} / {hyperparams.lr_gamma}", "--step_size / --lr_gamma"),
        ("Seed", hyperparams.seed, "--seed"),
    ]

    data_root = args.data_dir
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory '{data_root}' not found. Provide --data_dir or generate a dataset with LetterComposer.")

    use_directory_split = (data_root / "train").exists() and (data_root / "train").is_dir()
    if use_directory_split:
        loaders = create_loaders_from_subdirectories(data_root, hyperparams.batch_size)
        data_layout_desc = "pre-split directories (train/val/test)"
    else:
        dataset = LetterDataset(data_root)
        loaders = create_loaders(
            dataset,
            batch_size=hyperparams.batch_size,
            val_ratio=hyperparams.val_ratio,
            test_ratio=hyperparams.test_ratio,
            seed=hyperparams.seed,
        )
        data_layout_desc = f"random split (val={hyperparams.val_ratio:.2f}, test={hyperparams.test_ratio:.2f})"

    if use_directory_split:
        setup_rows = [
            (label, "from data split" if label == "Val/Test split" else value,
             "(auto)" if label == "Val/Test split" else flag)
            for label, value, flag in setup_rows
        ]

    label_set = set()
    for loader in loaders.values():
        subset = loader.dataset
        if hasattr(subset, 'samples'):
            label_set.update(label for _, label in subset.samples)
    if not label_set:
        raise ValueError(f"V datové složce '{data_root}' chybí podsložky A-Z s obrázky.")
    if min(label_set) < 0:
        raise ValueError('Očekávám nezáporné číselné názvy tříd (0-25 nebo A-Z).')
    num_classes = max(label_set) + 1
    data_summary = {split: len(loader.dataset) for split, loader in loaders.items()}
    train_loader = loaders.get("train")
    if train_loader is None:
        raise ValueError("Nebyla nalezena žádná trénovací data. Zkontrolujte složku s daty.")
    setup_rows.append(("Data layout", data_layout_desc, "(auto)"))
    for split_name in ("train", "val", "test"):
        if split_name in data_summary:
            setup_rows.append((f"{split_name.capitalize()} samples", data_summary[split_name], "(auto)"))
    setup_rows.append(("Classes", f"{num_classes} (A-Z)" if num_classes == 26 else num_classes, "(auto)"))
    for label, value, flag in setup_rows:
        tqdm.write(f"{label:<20} : {str(value):<25} {flag}")
    tqdm.write("")
    tqdm.write("Progress bar shows processed mini-batches plus running loss/accuracy averages.")
    tqdm.write("")
    tqdm.write("[train]    : batches from the training split.")
    tqdm.write("[val]      : batches from the validation split.")
    tqdm.write("")
    tqdm.write("it/s       : iterations (mini-batches) processed per second.")
    tqdm.write("loss       : instantaneous cross-entropy over the current mini-batch.")
    tqdm.write("acc        : instantaneous accuracy (fraction correct) over the current mini-batch.")
    tqdm.write("")
    tqdm.write("train_loss : mean cross-entropy over processed training samples.")
    tqdm.write("train_acc  : fraction of correct predictions over processed training samples.")
    tqdm.write("val_loss   : mean cross-entropy over processed validation samples.")
    tqdm.write("val_acc    : fraction of correct predictions over processed validation samples.")
    tqdm.write("")

    model = SimpleCNN(num_classes=num_classes, dropout=hyperparams.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    scheduler = StepLR(optimizer, step_size=hyperparams.step_size, gamma=hyperparams.lr_gamma)

    output_dir: Path = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history: List[EpochStats] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")

    for epoch in range(1, hyperparams.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=hyperparams.epochs,
        )

        if "val" in loaders:
            val_desc = f"Epoch {epoch:02d}/{hyperparams.epochs:02d} [val]"
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device, desc=val_desc)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        scheduler.step()

        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )

        tqdm.write(
            f"Epoch {epoch:02d}/{hyperparams.epochs} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        # Pokud jsme neměli validační data, uložíme poslední stav.
        best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    model_path = output_dir / "letter_cnn.pt"
    torch.save(best_state, model_path)
    save_history(history, output_dir)

    # Also save num_classes in config for LetterTeaser compatibility
    config_with_classes = hyperparams.as_dict()
    config_with_classes['num_classes'] = num_classes
    with config_output_path.open("w", encoding="utf-8") as handle:
        json.dump(config_with_classes, handle, indent=2)

    tqdm.write(f"Natrénovaný model uložen do: {model_path}")

    tqdm.write(f"Artefakty uložené v: {output_dir}")

    test_results: Optional[Tuple[float, float]] = None

    if "test" in loaders and len(loaders["test"].dataset) > 0:
        # Načteme uložené váhy do modelu na aktuálním zařízení a spočítáme výsledky na testu.
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device, desc="Testing")
        tqdm.write(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.3f}")
        test_results = (test_loss, test_acc)

        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "device": str(device),
            "epochs": hyperparams.epochs,
            "batch_size": hyperparams.batch_size,
            "learning_rate": hyperparams.learning_rate,
            "val_ratio": hyperparams.val_ratio,
            "test_ratio": hyperparams.test_ratio,
            "step_size": hyperparams.step_size,
            "lr_gamma": hyperparams.lr_gamma,
            "dropout": hyperparams.dropout,
            "seed": hyperparams.seed,
            "num_classes": num_classes,
        }
        metrics_path = output_dir / "test_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        tqdm.write(f"Souhrnné metriky z testu uloženy do: {metrics_path}")

    valid_val_history = [stats for stats in history if not math.isnan(stats.val_loss)]
    if valid_val_history:
        best_epoch_stats = min(valid_val_history, key=lambda stats: stats.val_loss)
    else:
        best_epoch_stats = max(history, key=lambda stats: stats.train_acc)

    final_stats = history[-1]

    summary_lines = [
        "",
        "===== Training summary =====",
        f"Output directory : {output_dir}",
        f"Epochs total     : {hyperparams.epochs}",
        f"Best epoch       : {best_epoch_stats.epoch:02d} "
        f"(train_loss {best_epoch_stats.train_loss:.4f}, train_acc {best_epoch_stats.train_acc:.3f}, "
        f"val_loss {best_epoch_stats.val_loss:.4f}, val_acc {best_epoch_stats.val_acc:.3f})",
        f"Final epoch      : {final_stats.epoch:02d} "
        f"(train_loss {final_stats.train_loss:.4f}, train_acc {final_stats.train_acc:.3f}, "
        f"val_loss {final_stats.val_loss:.4f}, val_acc {final_stats.val_acc:.3f})",
        f"Model path       : {model_path}",
    ]

    if test_results is not None:
        t_loss, t_acc = test_results
        summary_lines.append(f"Test results     : loss {t_loss:.4f}, acc {t_acc:.3f}")

    summary_lines.append("=============================")

    for line in summary_lines:
        tqdm.write(line)


if __name__ == "__main__":
    main()
