# Setup

Commands below assume the repository root. NAS access is needed for the saved data; cloning this repository does not download it.

## Analysis environment

Use Python 3.11 in a separate environment. This is a supported reconstruction recipe, not a claim that every historical run used identical package versions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/analysis-requirements.txt
export VQ_NAS_ROOT=/megaverse/storage/nasiri
python tools/prepare_workspace.py --nas-root "$VQ_NAS_ROOT" --workspace work/analysis
python tools/run_notebook.py work/analysis/notebooks/ch09_ordered_prefix_distribution_shift.ipynb
```

PyTorch/LPIPS are needed by analyses that recompute pixel distances. CPU analysis is possible but can be slow. Install an appropriate CUDA build instead for GPU work. Notebook outputs go into the working copy, not the original NAS archive. `VQ_JUPYTER_ROOT` overrides the historical `NAS_ROOT/jupyter` input tree when mounted elsewhere.

## External model source

```bash
python tools/fetch_upstreams.py llamagen vqgan flextok one_d_piece guided_diffusion --dry-run
python tools/fetch_upstreams.py llamagen vqgan flextok one_d_piece guided_diffusion
```

Revisions and source URLs are in `configs/upstreams.lock.json`. Checkouts are ignored under `third_party/`; do not add them to Git. The tool refuses to overwrite a different or dirty existing checkout. LlamaGen/VQGAN use the recorded forks because the exporters depend on their token-index interfaces. The forks are dependencies, not copied into this project.

Use **separate environments** for the old VQGAN code, LlamaGen, FlexTok and One-D-Piece. Their upstream dependencies conflict; installing all four into the analysis environment is not supported. Follow the recorded forks' Docker/installation recipes at the locked revision, rather than the original papers' generic environments.

| Grid runtime | Recorded fork Docker recipe |
| --- | --- |
| VQGAN | Python 3.8, PyTorch `1.13.1+cu117`, torchvision `0.14.1+cu117`, `docker/requirements.txt` (Lightning 1.0.8, OmegaConf 2.0.0) |
| LlamaGen | PyTorch 2.1.2, torchvision 0.16.2, cu121 wheel index; `docker/Dockerfile` and `docker/requirements.txt` |

VQGAN's original `environment.yaml` pins the older PyTorch 1.7 stack; the experiment fork's Dockerfile explicitly replaces that with 1.13.1. Do not silently substitute the old training environment. The fork Docker build helpers supply lab user/group arguments; inspect `docker/build.sh` before using them on another host. Checkpoints are loaded only from trusted project files.

The ordered-prefix Dockerfiles are copies of the build recipes used for the experiments, with external source as their build context:

```bash
docker build -f "$PWD/experiments/ordered_prefix_ood/Dockerfile.flextok" -t nasiri/flextok-prefix-pilot:latest third_party/flextok
docker build -f "$PWD/experiments/ordered_prefix_ood/Dockerfile.one_d_piece" -t nasiri/one-d-piece-prefix-pilot:latest third_party/one_d_piece
```

They pin PyTorch/CUDA and One-D-Piece's transformers version, but not every transitive dependency. The existing NAS server images are the historical runtime, not a fully reconstructable container digest archive. Build recipes were inspected, not rebuilt during packaging.

`environments/recorded-{flextok,one_d_piece}-runtime.json` records installed package versions and Python from those existing images, inspected without GPU access or inference. Use these when resolving dependency drift during a rebuild; they are an observed inventory, not a portable requirements lock (some system/editable packages cannot be installed from PyPI). Recorded image IDs start `8a78f8e3c600` and `a9e61cedeae2`, respectively.

## OpenAI evaluator environment

Use a separate **Linux Python 3.8** environment for the historical TensorFlow evaluator. The saved evaluator Docker recipe used these core versions; this small requirements file extracts its CPU evaluation dependencies without copying the evaluator implementation:

```bash
python3.8 -m venv /path/to/evaluator-env
source /path/to/evaluator-env/bin/activate
python -m pip install -r environments/evaluation-requirements.txt
```

The evaluator downloads its Inception graph if not already present in its working/output directory. For offline evaluation, copy the archived evaluator cache from the NAS first. Optional `eval_fidelity.py` also needs its own PyTorch/torchvision and torch-fidelity installation; it is not used to regenerate the thesis OpenAI metric tables.

## Checkpoints

| System | Recorded checkpoint |
| --- | --- |
| LlamaGen VQ-16 | `NAS_ROOT/llamagen/checkpoints/vq_ds16_c2i.pt` |
| VQGAN f16 / 16384 | `NAS_ROOT/vq-gan/checkpoints/vqgan_imagenet_f16_16384/{last.ckpt,model.yaml}` |
| FlexTok d12-d12 | HF `EPFL-VILAB/flextok_d12_d12_in1k`, snapshot `fe24b9794f32cc5c002781d25dd81a7768bed71c` |
| One-D-Piece S-256 | HF `turing-motors/One-D-Piece-S-256`, snapshot `250e95dcb0b29fcfeef4396d1788a9dbd18ca455` |

The main ordered-prefix runners pin these observed HF revisions. Existing cached weights are indexed on the NAS. Historical/pilot scripts retain their original behavior and are not the recommended new-run entry point.

## Dataset layouts

Grid-tokenizer exporters expect `ImageFolder`-style class directories. Use the already prepared paths in `configs/nas-catalog.json`. For ImageNet-V2, the saved WNID layout was made with:

```bash
python scripts/datasets/make_imagenetv2_wnid_layout.py ORIGINAL_NUMERIC_DIR NEW_WNID_DIR --index imagenet_class_index.json
```

Obtain `imagenet_class_index.json` with the original dataset export; the helper does not fetch it. Ordered-prefix manifests contain relative paths and WNIDs/category IDs; dataset roots are supplied at runtime. See the ordered-prefix README for ObjectNet's separate mapped-category control.
