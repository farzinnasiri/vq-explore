#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path,
                   help="ImageNetV2 folder with class dirs 000..999")
    p.add_argument("dst", type=Path,
                   help="Output folder (will contain n######## dirs)")
    p.add_argument("--index", type=Path, default=Path("imagenet_class_index.json"),
                   help="imagenet_class_index.json (idx -> [wnid, name])")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlinking")
    args = p.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    with open(args.index, "r") as f:
        idx_to_wnid = {int(k): v[0] for k, v in json.load(f).items()}

    for d in sorted(args.src.iterdir()):
        if not d.is_dir():
            continue
        idx = int(d.name)          # "000" -> 0
        wnid = idx_to_wnid[idx]
        out = args.dst / wnid
        out.mkdir(parents=True, exist_ok=True)

        for img in d.iterdir():
            if not img.is_file():
                continue
            target = out / img.name
            if target.exists():
                continue
            if args.copy:
                target.write_bytes(img.read_bytes())
            else:
                os.symlink(img.resolve(), target)

    print("Done:", args.dst.resolve())

if __name__ == "__main__":
    main()
