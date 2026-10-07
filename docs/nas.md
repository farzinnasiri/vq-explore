# NAS handover

Entry point on Elysium: `/megaverse/storage/nasiri/vq-tokenizer-properties/`.

| Folder | Purpose |
| --- | --- |
| `code/2026-10-07/` | Published lightweight package snapshot |
| `original_notebooks/2026-10-07/` | Executed original notebook copies before Git output clearing |
| `thesis_assets/2026-10-07/` | Thesis figure/table source assets used by the two plotting/supporting-results scripts |
| `datasets/` | Links to prepared dataset directories, including shared lab storage |
| `checkpoints/` | Links to grid-tokenizer weights and ordered-prefix HF cache |
| `results/` | Links to raw exports, locality caches, ordered-prefix run families and notebook outputs |

The `datasets`, `checkpoints` and `results` entries are a **navigation index**, not duplicate copies. Targets were checked on Elysium. Shared dataset links still require lab filesystem permissions. Existing directories were not moved or replaced; changing/removing a link does not delete its target.

The machine-readable layout is `configs/nas-catalog.json`. Inspect or safely recreate the index:

```bash
python tools/organize_nas.py
python tools/organize_nas.py --apply
```

Without `--apply` it only checks. Conflicting destinations are refused. Paths can be adjusted in a local catalog copy for another machine. Do not assume symlinks survive a bulk archive export: include their targets explicitly if handing the data to someone outside the lab.

## Which saved data matters

- Thesis RQ1/RQ5: code counts and positional counts in `results/code_usage`; reconstruction arrays, references and metrics in `results/reconstruction`.
- RQ2: `results/global_noise_{llamagen,vqgan}`.
- RQ3: `results/encoder_locality`, including the consolidated latest LlamaGen H1 export.
- RQ4: `results/decoder_locality`, which keeps raw runs, relation files and analysis caches separate.
- Ordered-prefix primary: `results/ordered_prefix_full`, plus `results/notebook_outputs/chapter9_outputs/data/raw` used by the final notebook. Primary matching uses the saved 10k manifest, not all One-D-Piece source rows.
- Earlier pilot/confirmatory/granular/Sketch/ObjectNet extensions: corresponding separate result links. They are historical evidence, not interchangeable primary replications.

Known input issues are retained because these are the actual thesis exports: ObjectNet's identification border was not cropped, and the OrganAMNIST export omits 1,643 training images. Correct these for a new experiment, but do not silently replace inputs when reanalysing the thesis results.

Heavy model source checkouts remain in their original separate directories. The published package links to exact upstream Git revisions instead of redistributing them. Cache files, datasets and checkpoint files are ignored by Git.
