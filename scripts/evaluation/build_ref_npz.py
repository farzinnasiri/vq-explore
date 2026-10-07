import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.append("/app")
from dataset.augmentation import center_crop_arr

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--image-size", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    transforms.ToTensor(),
])

dataset = ImageFolder(args.data_root, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=False,
    drop_last=False,
)

batches = []
for x, _ in tqdm(loader, desc=Path(args.out).name):
    x = x.permute(0, 2, 3, 1).numpy()
    x = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    batches.append(x)

arr = np.concatenate(batches, axis=0)
np.savez(args.out, arr_0=arr)
print(f"Saved {args.out} with shape={arr.shape}", flush=True)
