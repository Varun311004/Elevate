from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET = ROOT / "dataset"

DEFAULT_OUTPUT = ROOT / "dataset_clean"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Existing five classes.
EXISTING_CLASSES = {
    "angry": "angry",
    "confused": "confused",
    "happy": "happy",
    "neutral": "neutral",
    "surprised": "surprise",
}

# Original Bored/Focused source.
# IMPORTANT:
# Change this path if your original folder is elsewhere.
ORIGINAL_BORED_FOCUSED = (
    ROOT / "Bored_vs_Focused_Training_Data"
)

BORED_SOURCE_FOLDER = "Bored"
FOCUSED_SOURCE_FOLDER = "Focused"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Missing directory:\n{directory}"
        )

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def unique_destination(
    destination: Path,
) -> Path:
    """
    Prevent filename collisions when two source datasets
    contain the same filename.
    """

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    counter = 2

    while True:
        candidate = (
            destination.parent
            / f"{stem}_{counter}{suffix}"
        )

        if not candidate.exists():
            return candidate

        counter += 1


def copy_class(
    source: Path,
    destination: Path,
) -> list[Path]:

    files = image_files(source)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    for source_file in files:

        target = unique_destination(
            destination / source_file.name
        )

        shutil.copy2(
            source_file,
            target,
        )

        copied.append(target)

    return copied


# ============================================================
# DATASET BUILD
# ============================================================

def build_clean_dataset(
    output_dir: Path,
) -> dict:

    if output_dir.exists():
        raise RuntimeError(
            f"\nOutput directory already exists:\n"
            f"{output_dir}\n\n"
            f"For safety, this script will NOT overwrite it.\n"
            f"Delete/rename it manually if you want to rebuild."
        )

    print()
    print("=" * 70)
    print("ELEVATE EMOTION DATASET CLEANUP")
    print("=" * 70)

    print(
        f"Existing dataset : {SOURCE_DATASET}"
    )

    print(
        f"Original B/F data: "
        f"{ORIGINAL_BORED_FOCUSED}"
    )

    print(
        f"Output dataset   : {output_dir}"
    )

    print()

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    report = {
        "source_dataset": str(
            SOURCE_DATASET
        ),
        "original_bored_focused": str(
            ORIGINAL_BORED_FOCUSED
        ),
        "output_dataset": str(
            output_dir
        ),
        "classes": {},
        "cross_class_duplicates": [],
    }

    # --------------------------------------------------------
    # 1. Copy existing five classes
    # --------------------------------------------------------

    print("=" * 70)
    print("COPYING EXISTING DATASETS")
    print("=" * 70)

    for target_class, source_folder in (
        EXISTING_CLASSES.items()
    ):

        source = (
            SOURCE_DATASET
            / source_folder
        )

        destination = (
            output_dir
            / target_class
        )

        print(
            f"\n{target_class}:"
        )

        print(
            f"  source: {source}"
        )

        files = copy_class(
            source,
            destination,
        )

        print(
            f"  copied: {len(files):,}"
        )

        report["classes"][target_class] = {
            "source": str(source),
            "count": len(files),
            "type": "existing_dataset",
        }

    # --------------------------------------------------------
    # 2. Copy ORIGINAL Bored
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COPYING ORIGINAL BORED DATA")
    print("=" * 70)

    bored_source = (
        ORIGINAL_BORED_FOCUSED
        / BORED_SOURCE_FOLDER
    )

    bored_destination = (
        output_dir
        / "bored"
    )

    bored_files = copy_class(
        bored_source,
        bored_destination,
    )

    print(
        f"Original Bored images copied: "
        f"{len(bored_files):,}"
    )

    report["classes"]["bored"] = {
        "source": str(bored_source),
        "count": len(bored_files),
        "type": "original",
    }

    # --------------------------------------------------------
    # 3. Copy ORIGINAL Focused
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COPYING ORIGINAL FOCUSED DATA")
    print("=" * 70)

    focused_source = (
        ORIGINAL_BORED_FOCUSED
        / FOCUSED_SOURCE_FOLDER
    )

    focused_destination = (
        output_dir
        / "focused"
    )

    focused_files = copy_class(
        focused_source,
        focused_destination,
    )

    print(
        f"Original Focused images copied: "
        f"{len(focused_files):,}"
    )

    report["classes"]["focused"] = {
        "source": str(focused_source),
        "count": len(focused_files),
        "type": "original",
    }

    return report


# ============================================================
# DUPLICATE AUDIT
# ============================================================

def audit_cross_class_duplicates(
    output_dir: Path,
    report: dict,
) -> None:

    print()
    print("=" * 70)
    print("CROSS-CLASS EXACT DUPLICATE AUDIT")
    print("=" * 70)

    hash_to_files: dict[
        str,
        list[tuple[str, Path]]
    ] = defaultdict(list)

    class_dirs = sorted(
        p
        for p in output_dir.iterdir()
        if p.is_dir()
    )

    total_files = 0

    for class_dir in class_dirs:

        files = image_files(
            class_dir
        )

        print(
            f"Hashing {class_dir.name:>10}: "
            f"{len(files):,}"
        )

        for path in files:

            digest = sha256_file(path)

            hash_to_files[digest].append(
                (
                    class_dir.name,
                    path,
                )
            )

            total_files += 1

    conflicts = []

    for digest, entries in hash_to_files.items():

        classes = sorted(
            {
                class_name
                for class_name, _ in entries
            }
        )

        if len(classes) <= 1:
            continue

        conflicts.append(
            {
                "sha256": digest,
                "classes": classes,
                "files": [
                    {
                        "class": class_name,
                        "path": str(path),
                    }
                    for class_name, path in entries
                ],
            }
        )

    report[
        "cross_class_duplicates"
    ] = conflicts

    print()
    print(
        f"Total clean-dataset images: "
        f"{total_files:,}"
    )

    print(
        f"Cross-class duplicate groups: "
        f"{len(conflicts):,}"
    )

    if conflicts:

        print()
        print(
            "WARNING: Cross-class duplicates were found."
        )

        print(
            "They have NOT been deleted."
        )

        print(
            "They require review before training."
        )

        print()

        for index, conflict in enumerate(
            conflicts[:30],
            start=1,
        ):

            print(
                f"[{index}] "
                f"{' <-> '.join(conflict['classes'])}"
            )

            for item in conflict["files"]:
                print(
                    f"    {item['class']}: "
                    f"{item['path']}"
                )

        if len(conflicts) > 30:
            print()
            print(
                f"... plus "
                f"{len(conflicts) - 30:,} more groups."
            )

    else:

        print()
        print(
            "GOOD: No exact cross-class duplicates found."
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    output_dir: Path,
    report: dict,
) -> None:

    print()
    print("=" * 70)
    print("CLEAN DATASET SUMMARY")
    print("=" * 70)

    total = 0

    for class_name in [
        "angry",
        "bored",
        "focused",
        "confused",
        "happy",
        "neutral",
        "surprised",
    ]:

        count = (
            report["classes"]
            [class_name]
            ["count"]
        )

        total += count

        print(
            f"{class_name:>10}: "
            f"{count:>6,}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':>10}: "
        f"{total:>6,}"
    )

    print()
    print(
        f"Cross-class duplicate groups: "
        f"{len(report['cross_class_duplicates']):,}"
    )

    print()
    print(
        f"Dataset created at:\n"
        f"{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global ORIGINAL_BORED_FOCUSED

    parser = argparse.ArgumentParser(
        description=(
            "Build a clean Elevate emotion dataset "
            "using original Bored/Focused images."
        )
    )

    parser.add_argument(
        "--original-bf",
        type=Path,
        default=ORIGINAL_BORED_FOCUSED,
        help=(
            "Path to Bored_vs_Focused_Training_Data"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output directory for the clean dataset"
        ),
    )

    args = parser.parse_args()

    ORIGINAL_BORED_FOCUSED = (
        args.original_bf.resolve()
    )

    output_dir = (
        args.output.resolve()
    )

    report = build_clean_dataset(
        output_dir
    )

    audit_cross_class_duplicates(
        output_dir,
        report,
    )

    report_path = (
        output_dir
        / "dataset_cleanup_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(
        output_dir,
        report,
    )

    print()
    print("=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)

    print(
        f"Report:\n{report_path}"
    )

    if report["cross_class_duplicates"]:

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Cross-class duplicates were found."
        )

        print(
            "DO NOT train yet."
        )

        print(
            "Review the report first."
        )

    else:

        print()
        print(
            "No cross-class exact duplicates were found."
        )

        print(
            "The dataset is ready for the next audit stage."
        )


if __name__ == "__main__":
    main()