#!/usr/bin/env python3
"""
LetterDreamer Cleanup Script
============================
Removes generated synthetic letter data from the shared/data directories.

Usage:
    python cleanup.py                    # Interactive mode
    python cleanup.py --synthetic        # Remove only synthetic/ folder
    python cleanup.py --composed         # Remove only train/val/test synthetic data
    python cleanup.py --all              # Remove all synthetic data from all locations
    python cleanup.py --force            # Skip confirmation prompts
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

# Letters A-Z
LETTERS = [chr(ord('A') + i) for i in range(26)]

# Default paths relative to script
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATA_DIR = SCRIPT_DIR.parent / "shared" / "data"
SYNTHETIC_DIR = SHARED_DATA_DIR / "synthetic"
TRAIN_DIR = SHARED_DATA_DIR / "train"
VAL_DIR = SHARED_DATA_DIR / "val"
TEST_DIR = SHARED_DATA_DIR / "test"


def count_files(directory: Path) -> int:
    """Count BMP files in a directory recursively."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*.bmp"))


def get_stats() -> dict:
    """Get statistics about existing synthetic data."""
    stats = {
        "synthetic": count_files(SYNTHETIC_DIR),
        "train": count_files(TRAIN_DIR),
        "val": count_files(VAL_DIR),
        "test": count_files(TEST_DIR),
    }
    stats["composed_total"] = stats["train"] + stats["val"] + stats["test"]
    stats["total"] = stats["synthetic"] + stats["composed_total"]
    return stats


def remove_bmp_files(directory: Path, verbose: bool = True) -> Tuple[int, int]:
    """Remove all BMP files from directory and its subdirectories.
    
    Returns:
        Tuple of (files_removed, errors)
    """
    if not directory.exists():
        return 0, 0
    
    removed = 0
    errors = 0
    
    for bmp_file in directory.rglob("*.bmp"):
        try:
            bmp_file.unlink()
            removed += 1
            if verbose and removed % 100 == 0:
                print(f"  Odstraněno {removed} souborů...", end="\r")
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ⚠️  Chyba při mazání {bmp_file}: {e}")
    
    if verbose and removed > 0:
        print(f"  Odstraněno {removed} souborů.       ")
    
    return removed, errors


def remove_empty_dirs(directory: Path) -> int:
    """Remove empty subdirectories."""
    if not directory.exists():
        return 0
    
    removed = 0
    # Sort by depth (deepest first) to handle nested empty dirs
    subdirs = sorted(
        [d for d in directory.rglob("*") if d.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True
    )
    
    for subdir in subdirs:
        try:
            if subdir.exists() and not any(subdir.iterdir()):
                subdir.rmdir()
                removed += 1
        except Exception:
            pass
    
    return removed


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    while True:
        response = input(f"{message} [a/n]: ").strip().lower()
        if response in ("a", "ano", "y", "yes"):
            return True
        if response in ("n", "ne", "no"):
            return False
        print("Zadejte 'a' pro ano nebo 'n' pro ne.")


def print_stats(stats: dict) -> None:
    """Print statistics about synthetic data."""
    print("\n📊 Statistiky syntetických dat:")
    print(f"   synthetic/      : {stats['synthetic']:,} souborů")
    print(f"   train/          : {stats['train']:,} souborů")
    print(f"   val/            : {stats['val']:,} souborů")
    print(f"   test/           : {stats['test']:,} souborů")
    print(f"   ─────────────────────────")
    print(f"   Celkem          : {stats['total']:,} souborů")


def clean_synthetic(force: bool = False) -> bool:
    """Remove files from synthetic/ directory."""
    count = count_files(SYNTHETIC_DIR)
    if count == 0:
        print("ℹ️  Složka synthetic/ je prázdná.")
        return True
    
    print(f"\n🗑️  Mažu {count:,} souborů ze složky synthetic/...")
    
    if not force and not confirm_action("Pokračovat?"):
        print("Zrušeno.")
        return False
    
    removed, errors = remove_bmp_files(SYNTHETIC_DIR)
    remove_empty_dirs(SYNTHETIC_DIR)
    
    print(f"✓ Odstraněno {removed:,} souborů" + (f", {errors} chyb" if errors else ""))
    return True


def clean_composed(force: bool = False) -> bool:
    """Remove files from train/val/test directories."""
    train_count = count_files(TRAIN_DIR)
    val_count = count_files(VAL_DIR)
    test_count = count_files(TEST_DIR)
    total = train_count + val_count + test_count
    
    if total == 0:
        print("ℹ️  Složky train/val/test jsou prázdné.")
        return True
    
    print(f"\n🗑️  Mažu data ze složek train/val/test:")
    print(f"   train: {train_count:,}, val: {val_count:,}, test: {test_count:,}")
    print(f"   Celkem: {total:,} souborů")
    
    if not force and not confirm_action("Pokračovat?"):
        print("Zrušeno.")
        return False
    
    total_removed = 0
    total_errors = 0
    
    for name, directory in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        dir_count = count_files(directory)
        if dir_count > 0:
            print(f"  📁 {name}/")
            removed, errors = remove_bmp_files(directory)
            total_removed += removed
            total_errors += errors
    
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        remove_empty_dirs(directory)
    
    print(f"✓ Odstraněno {total_removed:,} souborů" + (f", {total_errors} chyb" if total_errors else ""))
    return True


def clean_all(force: bool = False) -> bool:
    """Remove all synthetic data."""
    stats = get_stats()
    
    if stats["total"] == 0:
        print("ℹ️  Žádná syntetická data k odstranění.")
        return True
    
    print_stats(stats)
    print(f"\n⚠️  Tato akce odstraní všechna syntetická data ({stats['total']:,} souborů)!")
    
    if not force and not confirm_action("Opravdu smazat vše?"):
        print("Zrušeno.")
        return False
    
    success = True
    if stats["synthetic"] > 0:
        print(f"\n📁 synthetic/")
        removed, _ = remove_bmp_files(SYNTHETIC_DIR)
        remove_empty_dirs(SYNTHETIC_DIR)
    
    for name, directory in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        dir_count = count_files(directory)
        if dir_count > 0:
            print(f"📁 {name}/")
            removed, _ = remove_bmp_files(directory)
            remove_empty_dirs(directory)
    
    final_stats = get_stats()
    print(f"\n✓ Hotovo! Zbývá {final_stats['total']:,} souborů.")
    return success


def interactive_mode() -> None:
    """Interactive cleanup mode."""
    print("=" * 60)
    print("🧹 LetterDreamer Cleanup - Odstranění syntetických dat")
    print("=" * 60)
    
    stats = get_stats()
    print_stats(stats)
    
    if stats["total"] == 0:
        print("\n✓ Žádná data k odstranění.")
        return
    
    print("\nVolby:")
    print("  [1] Odstranit pouze synthetic/ složku")
    print("  [2] Odstranit pouze train/val/test složky")
    print("  [3] Odstranit vše")
    print("  [4] Zrušit")
    
    while True:
        choice = input("\nVyberte akci [1-4]: ").strip()
        if choice == "1":
            clean_synthetic()
            break
        elif choice == "2":
            clean_composed()
            break
        elif choice == "3":
            clean_all()
            break
        elif choice == "4":
            print("Zrušeno.")
            break
        else:
            print("Neplatná volba. Zadejte číslo 1-4.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Odstranění syntetických dat LetterDreameru.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady:
  python cleanup.py                    # Interaktivní režim
  python cleanup.py --synthetic        # Smazat synthetic/ složku
  python cleanup.py --composed         # Smazat train/val/test složky
  python cleanup.py --all --force      # Smazat vše bez potvrzení
"""
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--synthetic", "-s", action="store_true",
        help="Odstranit pouze složku synthetic/"
    )
    group.add_argument(
        "--composed", "-c", action="store_true",
        help="Odstranit pouze složky train/val/test"
    )
    group.add_argument(
        "--all", "-a", action="store_true",
        help="Odstranit všechna syntetická data"
    )
    
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Přeskočit potvrzovací výzvy"
    )
    
    parser.add_argument(
        "--stats", action="store_true",
        help="Zobrazit pouze statistiky, nic nemazat"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.stats:
        print("=" * 60)
        print("🧹 LetterDreamer Cleanup - Statistiky")
        print("=" * 60)
        stats = get_stats()
        print_stats(stats)
        return
    
    if args.synthetic:
        clean_synthetic(force=args.force)
    elif args.composed:
        clean_composed(force=args.force)
    elif args.all:
        clean_all(force=args.force)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
