"""Rename dataset files to avoid Hugging Face archive filename restrictions.

The Beautiful-Motifs short motif files use names like
``Short Beautiful Motif #00020.mid``.  Hugging Face rejects `#` inside archive
member names, so this script rewrites file names to a conservative ASCII form:
``Short_Beautiful_Motif_00020.mid``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    stem = Path(name).stem.replace("#", "")
    suffix = Path(name).suffix
    stem = SAFE_CHARS.sub("_", stem).strip("._-")
    stem = re.sub(r"_+", "_", stem)
    return f"{stem}{suffix.lower()}"


def sanitize_dataset_filenames(data_dir: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    renames: list[tuple[Path, Path]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        new_name = safe_name(path.name)
        if new_name == path.name:
            continue
        target = path.with_name(new_name)
        if target.exists() and target.resolve() != path.resolve():
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        renames.append((path, target))

    for source, target in renames:
        print(f"{source.relative_to(data_dir)} -> {target.relative_to(data_dir)}")
        if not dry_run:
            source.rename(target)
    return renames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    renames = sanitize_dataset_filenames(args.data_dir, dry_run=args.dry_run)
    action = "would rename" if args.dry_run else "renamed"
    print(f"{action} {len(renames)} files")


if __name__ == "__main__":
    main()
