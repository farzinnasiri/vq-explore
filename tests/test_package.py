"""Lightweight checks requiring only the Python standard library."""
import ast
import csv
import importlib.util
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare", ROOT / "tools/prepare_notebooks.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PackageTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for directory in ("scripts", "tools", "experiments", "tests"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=path):
                    ast.parse(path.read_text(), filename=str(path))

    def test_notebooks_are_clean(self):
        notebooks = list((ROOT / "notebooks").rglob("*.ipynb"))
        self.assertEqual(len(notebooks), 13)
        for path in notebooks:
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs"), [])
                    self.assertIsNone(cell.get("execution_count"))

    def test_path_preparation_preserves_calculation(self):
        cell = {"cell_type": "code", "metadata": {}, "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": "old"}],
                "source": ['root = Path("/exp/data/example")\n',
                           'other = Path(os.environ.get("CH08_METADATA_BASE", "/mnt/jupyter"))\n',
                           'result = (values - full).mean()\n']}
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.ipynb", Path(directory) / "target.ipynb"
            source.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
            prepare.prepare(source, target)
            copied = json.loads(target.read_text())["cells"][-1]
            text = "".join(copied["source"])
            self.assertIn('VQ_JUPYTER_ROOT / "data/example"', text)
            self.assertIn('str(VQ_JUPYTER_ROOT)', text)
            self.assertIn('result = (values - full).mean()', text)
            self.assertEqual(json.loads(source.read_text())["cells"][0]["execution_count"], 1)

    def test_primary_manifest_and_shards(self):
        directory = ROOT / "experiments/ordered_prefix_ood/manifests"
        def keys(path):
            with path.open() as handle:
                return [(r["dataset"], r["wnid"], r["image"]) for r in csv.DictReader(handle)]
        primary = keys(directory / "matched_10k.csv")
        self.assertEqual(len(primary), 20000)
        self.assertEqual(len(set(primary)), 20000)
        counts = Counter((d, c) for d, c, _ in primary)
        self.assertEqual(len(counts), 400)
        self.assertEqual(set(counts.values()), {50})
        parts = keys(directory / "matched_10k_part1.csv") + keys(directory / "matched_10k_part2.csv")
        self.assertEqual(set(primary), set(parts))
        self.assertEqual(len(parts), len(primary))
        shards = [row for path in (directory / "part2_shards").glob("*.csv") for row in keys(path)]
        self.assertEqual(set(shards), set(keys(directory / "matched_10k_part2.csv")))
        self.assertEqual(len(shards), 10000)

    def test_no_heavy_model_payloads(self):
        prohibited = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".npy", ".npz", ".zip", ".tar"}
        for directory in ("scripts", "tools", "configs", "experiments", "notebooks", "results", "reports"):
            for path in (ROOT / directory).rglob("*"):
                if path.is_file():
                    self.assertNotIn(path.suffix, prohibited)
                    self.assertLess(path.stat().st_size, 10 * 1024 * 1024, str(path))

    def test_report_has_local_images(self):
        path = ROOT / "reports/ordered-prefix/ordered-prefix-ood.html"
        text = path.read_text()
        for source in re.findall(r'<img[^>]+src="([^"]+)"', text):
            self.assertFalse(source.startswith("http"))
            if not source.startswith("data:"):
                self.assertTrue((path.parent / source).is_file())


if __name__ == "__main__":
    unittest.main()
