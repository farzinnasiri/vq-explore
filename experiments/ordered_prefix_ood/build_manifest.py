import argparse
import csv
import hashlib
import random
import subprocess
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(class_dir):
    return sorted(
        path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def stable_sample(paths, count, seed, key):
    key_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big"
    )
    return sorted(random.Random(key_seed).sample(paths, count))


def all_image_rows(dataset, root):
    relative_paths = subprocess.check_output(
        [
            "find",
            str(root),
            "-mindepth",
            "2",
            "-maxdepth",
            "2",
            "-type",
            "f",
            "-printf",
            "%P\\n",
        ],
        text=True,
    ).splitlines()
    return [
        {"dataset": dataset, "wnid": Path(path).parts[0], "image": path}
        for path in sorted(relative_paths)
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet", type=Path, required=True)
    parser.add_argument("--imagenet-r", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=50)
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_images:
        rows = all_image_rows("imagenet", args.imagenet)
        rows.extend(all_image_rows("imagenet_r", args.imagenet_r))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "wnid", "image"])
            writer.writeheader()
            writer.writerows(rows)
        class_counts = {
            dataset: len({row["wnid"] for row in rows if row["dataset"] == dataset})
            for dataset in ("imagenet", "imagenet_r")
        }
        print(
            f"Wrote {len(rows)} rows from all images: "
            f"{class_counts['imagenet']} ImageNet classes and "
            f"{class_counts['imagenet_r']} ImageNet-R classes"
        )
        return

    imagenet_classes = {
        path.name: image_paths(path) for path in args.imagenet.iterdir() if path.is_dir()
    }
    imagenet_r_classes = {
        path.name: image_paths(path) for path in args.imagenet_r.iterdir() if path.is_dir()
    }
    eligible = []
    for wnid in sorted(imagenet_classes.keys() & imagenet_r_classes.keys()):
        if (
            len(imagenet_classes[wnid]) >= args.images_per_class
            and len(imagenet_r_classes[wnid]) >= args.images_per_class
        ):
            eligible.append(wnid)

    selected_classes = stable_sample(eligible, args.classes, args.seed, "classes")
    rows = []
    for dataset, root, class_images in (
        ("imagenet", args.imagenet, imagenet_classes),
        ("imagenet_r", args.imagenet_r, imagenet_r_classes),
    ):
        for wnid in selected_classes:
            selected_images = stable_sample(
                class_images[wnid],
                args.images_per_class,
                args.seed,
                f"{dataset}:{wnid}",
            )
            for path in selected_images:
                rows.append(
                    {
                        "dataset": dataset,
                        "wnid": wnid,
                        "image": str(path.relative_to(root)),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "wnid", "image"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} rows: {len(selected_classes)} classes x "
        f"{args.images_per_class} images x 2 datasets"
    )


if __name__ == "__main__":
    main()
