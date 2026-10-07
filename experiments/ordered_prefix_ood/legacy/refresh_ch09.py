"""Validate and combine FlexTok exports, then refresh the server notebook inputs."""
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path('/megaverse/storage/nasiri/ordered-prefix-ood-full')
NB = Path('/megaverse/storage/nasiri/jupyter/vq-explore/notebooks')
RAW = NB / 'chapter9_outputs/data/raw'
PREFIXES = {8, 16, 32, 64, 96, 128, 160, 208, 256}
FIELDS = ['model', 'preprocessing', 'lpips_backbone', 'prefixes', 'seed',
          'decoder_timesteps', 'guidance_scale', 'perform_norm_guidance',
          'attention_backend', 'decoder_noise']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    backup = NB / 'chapter9_outputs/backups' / stamp
    backup.mkdir(parents=True)
    shutil.copytree(RAW, backup / 'raw')
    notebook = NB / 'ch09_ordered_prefix_distribution_shift.ipynb'
    shutil.copy2(notebook, backup / notebook.name)
    for name in ('derived',):
        shutil.copytree(NB / 'chapter9_outputs/data' / name, backup / name)
    shutil.copytree(NB / 'chapter9_outputs/figures', backup / 'figures')
    merged_dir = ROOT / 'outputs/flextok_matched10k_complete'
    all_manifest = list(csv.DictReader((ROOT / 'manifests/matched_10k.csv').open()))
    prepared = []
    for dataset in ('imagenet', 'imagenet_r'):
        rows, sources, seen = [], [], set()
        reference = None
        groups = [('part1', 'flextok_matched10k_part1', 24)] + [
            ('gpu' + str(g), 'flextok_matched10k_part2_shards/gpu' + str(g), 24 if g in (1, 2) else 1)
            for g in range(4)]
        for group, folder, batch in groups:
            directory = ROOT / 'outputs' / folder / dataset
            path = directory / 'per_image_metrics.csv'
            meta = json.loads((directory / 'metadata.json').read_text())
            if reference is None:
                reference = meta
            assert all(meta[k] == reference[k] for k in FIELDS), (group, 'settings mismatch')
            source_rows = list(csv.DictReader(path.open()))
            for row in source_rows:
                key = (row['image'], int(row['prefix_tokens']))
                assert key not in seen, (dataset, 'duplicate', key)
                seen.add(key)
                assert row['dataset'] == dataset and row['model'] == reference['model']
                assert all(math.isfinite(float(row[k])) for k in ('lpips', 'psnr', 'ssim', 'decode_seconds'))
                assert float(row['lpips']) >= 0 and -1.001 <= float(row['ssim']) <= 1.001
                row.update(source_group=group, source_batch_size=batch)
                rows.append(row)
            sources.append(dict(group=group, csv=str(path), sha256=digest(path),
                                metadata=meta, launch_batch_size=batch, rows=len(source_rows)))
        expected = {(r['image'], p) for r in all_manifest if r['dataset'] == dataset for p in PREFIXES}
        assert seen == expected, (dataset, 'coverage mismatch')
        expected_classes = {r['image']: r['wnid'] for r in all_manifest if r['dataset'] == dataset}
        assert all(r['wnid'] == expected_classes[r['image']] for r in rows)
        images = {(r['wnid'], r['image']) for r in rows}
        counts = Counter(wnid for wnid, image in images)
        assert len(images) == 10000 and len(counts) == 200 and set(counts.values()) == {50}
        meta = {k: reference[k] for k in FIELDS}
        meta.update(dataset=dataset, n_images=10000, n_classes=200, sources=sources,
                    manifest_sha256=digest(ROOT / 'manifests/matched_10k.csv'),
                    provenance_note='Batch sizes and hardware vary by source; decoder noise is batch seeded.')
        prepared.append((dataset, rows, meta))
    for dataset, rows, meta in prepared:
        dest = merged_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r['wnid'], r['image'], int(r['prefix_tokens'])))
        with (dest / 'per_image_metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (dest / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        for suffix in ('per_image_metrics.csv', 'metadata.json'):
            shutil.copy2(dest / suffix, RAW / ('flextok_' + dataset + '_' + suffix))
        print(dataset, len(rows), 'validated rows', flush=True)

    n = json.loads(notebook.read_text())
    replacements = {
        'FlexTok currently contains a completed matched subset of 5,000 images per dataset from 100 shared classes.':
        'FlexTok contains 10,000 images per dataset: 50 images from each of all 200 shared classes. The merged exports retain source group and batch size, and metadata lists source paths and SHA-256 hashes.',
        'The class sets are not identical across models: One-D-Piece covers all 200 ImageNet-R classes, while the completed FlexTok snapshot covers 100. Cross-model magnitudes are therefore descriptive, not a controlled model ranking.':
        'Both models now cover the same 200 shared classes. FlexTok uses 50 ImageNet-R images per class; One-D-Piece uses all available ImageNet-R images. Class weighting is equal, but image sampling and decoder mechanics still differ between models.',
        '- FlexTok currently covers 100 shared classes and One-D-Piece covers 200. Their effect sizes cannot yet be treated as a controlled head-to-head comparison.':
        '- Both models cover 200 shared classes. ImageNet-R sample counts differ between models. FlexTok source groups also differ in batch size and GPU architecture; its noise is batch seeded. Source-group comparisons below are descriptive because classes differ between groups.',
        '- The next defensible extension is replication on another class-matchable shift and completion of the same 200-class scope for both models.':
        '- Both models now cover 200 shared classes. Further work should match ImageNet-R images across models and replicate on another shift.',
    }
    for cell in n['cells']:
        text = ''.join(cell['source'])
        for old, new in replacements.items():
            text = text.replace(old, new)
        if 'frame = load_run(model, dataset)' in text:
            text = text.replace('frame = load_run(model, dataset)', '''frame = load_run(model, dataset)
        assert frame.groupby("image")["prefix_tokens"].apply(lambda x: set(x) == set(PREFIX_ORDER)).all()
        assert np.isfinite(frame[METRICS + ["decode_seconds"]].to_numpy()).all()''')
        if 'scope_df = pd.DataFrame(scope_rows)' in text:
            text = text.replace('scope_df = pd.DataFrame(scope_rows)', '''assert len(shared_wnids["FlexTok"]) == 200
assert shared_wnids["FlexTok"] == shared_wnids["One-D-Piece"]
scope_df = pd.DataFrame(scope_rows)
save_table(scope_df, "ch09_comparison_scope.csv")''')
        cell['source'] = text.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    extra = '''# Source groups use different classes, so this is a sensitivity description, not a hardware test.
source_frames = []
for dataset in DATASET_ORDER:
    f = pd.read_csv(RAW_DIR / RUN_FILES[("FlexTok", dataset)])
    source_frames.append(f[["wnid", "source_group", "source_batch_size"]].drop_duplicates())
source_classes = pd.concat(source_frames).drop_duplicates()
assert not source_classes.duplicated("wnid").any()
source_delta = class_delta[class_delta.model_label == "FlexTok"].merge(source_classes, on="wnid", validate="many_to_one")
source_summary = source_delta.groupby(["source_group", "source_batch_size", "prefix_tokens"], as_index=False).agg(
    classes=("wnid", "nunique"), lpips_delta=("lpips_delta", "mean"),
    psnr_delta=("psnr_delta", "mean"), ssim_delta=("ssim_delta", "mean"))
save_table(source_summary, "ch09_source_group_sensitivity.csv")
display(source_summary[source_summary.prefix_tokens == 32])
'''
    if not any('## 14. Source Group Sensitivity' in ''.join(c['source']) for c in n['cells']):
        n['cells'].append(dict(cell_type='markdown', metadata={}, source=['## 14. Source Group Sensitivity\n', 'The first half and four new shards retain separate provenance. Different classes and random draws prevent attributing differences here to hardware or batch size alone.\n']))
        n['cells'].append(dict(cell_type='code', metadata={}, source=extra.splitlines(keepends=True), execution_count=None, outputs=[]))
    notebook.write_text(json.dumps(n, indent=1) + '\n')
    print('Backup:', backup)
    print('Updated:', notebook)


if __name__ == '__main__':
    main()
