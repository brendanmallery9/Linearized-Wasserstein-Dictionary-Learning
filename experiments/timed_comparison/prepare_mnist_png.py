from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from torchvision import datasets


def parse_digits(values: list[str] | None) -> list[int]:
    if not values:
        return list(range(10))
    digits: list[int] = []
    for value in values:
        for part in value.replace(",", " ").split():
            digit = int(part)
            if digit < 0 or digit > 9:
                raise ValueError(f"MNIST digit must be in [0, 9], got {digit}")
            digits.append(digit)
    return sorted(set(digits))


def prepare_mnist_png(
    output_dir: Path,
    *,
    mnist_root: Path,
    max_per_digit: int,
    digits: list[int] | None = None,
    force: bool = False,
) -> dict:
    digits = digits or list(range(10))
    image_dir = output_dir / "all"
    if image_dir.exists() and any(image_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"{image_dir} already has files; pass --force to replace it"
            )
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    mnist = datasets.MNIST(root=str(mnist_root), train=True, download=True)
    counts = {digit: 0 for digit in digits}
    selected: list[dict] = []

    for dataset_index, (image, label) in enumerate(mnist):
        label = int(label)
        if label not in counts:
            continue
        if counts[label] >= max_per_digit:
            continue

        sample_index = counts[label]
        filename = f"digit{label}_sample{sample_index:05d}_mnist{dataset_index:05d}.png"
        image.save(image_dir / filename)
        selected.append(
            {
                "digit": label,
                "sample_index": sample_index,
                "mnist_train_index": dataset_index,
                "filename": filename,
            }
        )
        counts[label] += 1

        if all(counts[digit] >= max_per_digit for digit in digits):
            break

    missing = {digit: max_per_digit - count for digit, count in counts.items() if count < max_per_digit}
    if missing:
        raise RuntimeError(f"Could not collect requested MNIST images: {missing}")

    metadata = {
        "format": "mnist_png_for_heitz_wdl",
        "version": 1,
        "image_dir": str(image_dir),
        "mnist_root": str(mnist_root),
        "digits": digits,
        "max_per_digit": max_per_digit,
        "counts": counts,
        "num_images": len(selected),
        "selected": selected,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MNIST train images as PNGs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, default=Path("mnist_raw"))
    parser.add_argument("--max-per-digit", type=int, default=10)
    parser.add_argument("--digits", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metadata = prepare_mnist_png(
        args.output_dir,
        mnist_root=args.mnist_root,
        max_per_digit=args.max_per_digit,
        digits=parse_digits(args.digits),
        force=args.force,
    )
    print(f"Wrote {metadata['num_images']} PNGs to {metadata['image_dir']}")


if __name__ == "__main__":
    main()

