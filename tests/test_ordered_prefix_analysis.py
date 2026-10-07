"""Synthetic checks of manifest selection and the saved analysis conventions."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import numpy as np
except ImportError:
    np = None


class ManifestTests(unittest.TestCase):
    def test_repeatable_small_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("id", "ood"):
                for cls in ("n00000001", "n00000002", "n00000003"):
                    folder = root / dataset / cls
                    folder.mkdir(parents=True)
                    for image in range(4):
                        (folder / f"{image}.JPEG").touch()
            command = [sys.executable, str(ROOT / "experiments/ordered_prefix_ood/build_manifest.py"),
                       "--imagenet", str(root / "id"), "--imagenet-r", str(root / "ood"),
                       "--classes", "2", "--images-per-class", "2", "--seed", "7"]
            first, second = root / "first.csv", root / "second.csv"
            for path in (first, second):
                subprocess.run(command + ["--output", str(path)], check=True, capture_output=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r["wnid"] for r in rows}), 2)


@unittest.skipIf(np is None, "Install analysis dependencies for numeric checks")
class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("summary", ROOT / "experiments/ordered_prefix_ood/summarize_results.py")
        cls.summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.summary)

    def test_metric_sign_and_budget_endpoints(self):
        self.assertAlmostEqual(self.summary.metric_loss(.5, .2, "lpips"), .3)
        self.assertAlmostEqual(self.summary.metric_loss(20, 25, "psnr"), 5)
        curve = {(prefix, "lpips"): (256 - prefix) / 256 for prefix in self.summary.PREFIXES}
        self.assertEqual(self.summary.required_budget(curve, "lpips", 0), 256)
        self.assertEqual(self.summary.required_budget(curve, "lpips", 1), 8)

    def test_complete_synthetic_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for model in ("alpha", "beta"):
                for domain in ("imagenet", "imagenet_r"):
                    run = root / model / domain
                    run.mkdir(parents=True)
                    (run / "metadata.json").write_text(json.dumps({"model": model, "dataset": domain}))
                    path = run / "per_image_metrics.csv"
                    inputs.append(str(path))
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["model", "dataset", "wnid", "image", "prefix_tokens", "lpips", "psnr", "ssim"])
                        writer.writeheader()
                        for cls in ("class1", "class2"):
                            for prefix in self.summary.PREFIXES:
                                loss = (256 - prefix) / 256 * (.1 if domain == "imagenet" else .2)
                                writer.writerow(dict(model=model, dataset=domain, wnid=cls, image=f"{cls}/image.JPEG",
                                                     prefix_tokens=prefix, lpips=.2 + loss, psnr=30 - 10 * loss, ssim=.9 - loss))
                    curves = self.summary.class_curves(self.summary.load_rows(path))
                    self.assertEqual(len(curves), 2)
                    self.assertEqual(curves["class1"][(256, "lpips")], 0)
            output = root / "analysis/summary.json"
            subprocess.run([sys.executable, str(ROOT / "experiments/ordered_prefix_ood/summarize_results.py"),
                            *inputs, "--output", str(output), "--bootstrap-samples", "100"],
                           check=True, capture_output=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["primary_prefix"], 32)
            self.assertEqual(set(result["models"]), {"alpha", "beta"})


if __name__ == "__main__":
    unittest.main()
