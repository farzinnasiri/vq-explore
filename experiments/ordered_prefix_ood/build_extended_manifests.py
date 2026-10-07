import argparse
import csv
import hashlib
import json
import random
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


def write_manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "wnid", "image"])
        writer.writeheader()
        writer.writerows(rows)


def build_sketch(args):
    with args.reference_manifest.open(newline="") as handle:
        wnids = sorted(
            {
                row["wnid"]
                for row in csv.DictReader(handle)
                if row["dataset"] == "imagenet"
            }
        )

    rows = []
    for wnid in wnids:
        images = image_paths(args.imagenet_sketch / wnid)
        for path in stable_sample(
            images, args.sketch_images_per_class, args.seed, f"sketch:{wnid}"
        ):
            rows.append(
                {
                    "dataset": "imagenet_sketch",
                    "wnid": wnid,
                    "image": str(path.relative_to(args.imagenet_sketch)),
                }
            )
    write_manifest(args.sketch_output, rows)
    print(f"Wrote {len(rows)} ImageNet-Sketch rows across {len(wnids)} classes")


def objectnet_to_wnids(mapping_root, imagenet_root):
    objectnet_mapping = json.loads(
        (mapping_root / "objectnet_to_imagenet_1k.json").read_text()
    )
    pytorch_to_id = json.loads(
        (mapping_root / "pytorch_to_imagenet_2012_id.json").read_text()
    )
    id_to_pytorch = {int(value): int(key) for key, value in pytorch_to_id.items()}
    labels = (mapping_root / "imagenet_to_label_2012_v2").read_text().splitlines()
    pytorch_to_wnid = {
        index: path.name
        for index, path in enumerate(sorted(path for path in imagenet_root.iterdir() if path.is_dir()))
    }

    result = {}
    for objectnet_label, imagenet_labels in objectnet_mapping.items():
        wnids = []
        for label in imagenet_labels.split("; "):
            imagenet_id = labels.index(label) + 1
            wnids.append(pytorch_to_wnid[id_to_pytorch[imagenet_id]])
        result[objectnet_label] = sorted(set(wnids))
    return result


def build_objectnet(args):
    mapping_root = args.objectnet / "mappings"
    folder_to_label = json.loads(
        (mapping_root / "folder_to_objectnet_label.json").read_text()
    )
    label_to_folder = {label: folder for folder, label in folder_to_label.items()}
    label_to_wnids = objectnet_to_wnids(mapping_root, args.imagenet)

    rows = []
    for label in sorted(label_to_wnids):
        category = label_to_folder[label]
        objectnet_images = image_paths(args.objectnet / "images" / category)
        imagenet_images = []
        for wnid in label_to_wnids[label]:
            imagenet_images.extend(image_paths(args.imagenet / wnid))

        for dataset, root, images in (
            ("imagenet_objectnet", args.imagenet, imagenet_images),
            ("objectnet", args.objectnet / "images", objectnet_images),
        ):
            selected = stable_sample(
                images,
                args.objectnet_images_per_class,
                args.seed,
                f"{dataset}:{category}",
            )
            for path in selected:
                rows.append(
                    {
                        "dataset": dataset,
                        "wnid": category,
                        "image": str(path.relative_to(root)),
                    }
                )

    write_manifest(args.objectnet_output, rows)
    print(
        f"Wrote {len(rows)} ObjectNet-pair rows across "
        f"{len(label_to_wnids)} categories"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--imagenet", type=Path, required=True)
    parser.add_argument("--imagenet-sketch", type=Path, required=True)
    parser.add_argument("--objectnet", type=Path, required=True)
    parser.add_argument("--sketch-output", type=Path, required=True)
    parser.add_argument("--objectnet-output", type=Path, required=True)
    parser.add_argument("--sketch-images-per-class", type=int, default=25)
    parser.add_argument("--objectnet-images-per-class", type=int, default=44)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main():
    args = parse_args()
    build_sketch(args)
    build_objectnet(args)


if __name__ == "__main__":
    main()
