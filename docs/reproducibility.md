# Reproducing the thesis experiments

EditPano is training-free: every experiment runs frozen, publicly available
models. All protocols below read scenes in the layout of
[data_format.md](data_format.md); the dataset itself is not distributed with
this repository.

> On shared machines cap CPU threads (`export OMP_NUM_THREADS=4`); oversubscribed
> BLAS threads make CPU metrics 10–20× slower. On offline compute nodes set
> `HF_HUB_OFFLINE=1` after caching weights (`editpano doctor` lists what is cached).

## Environment

Install with the recorded versions when bit-exact reproduction matters:

```bash
pip install -c requirements/thesis-env.txt -e ".[edit,grounding,eval]"
pip install "diffusers @ git+https://github.com/huggingface/diffusers@e5aa719241f9b74d6700be3320a777799bfab70a"
pip install "git+https://github.com/openai/CLIP.git"
```

Newer OpenCV/Pillow releases change a small number of ROI-border pixels (with
OpenCV 5.0 about 0.2 % of pixels by at most a few grey levels); the golden tests
then skip rather than fail, because they are defined for the recorded versions.

## Frozen operating point

| Component | Setting |
| --- | --- |
| Editor | `Qwen/Qwen-Image-Edit-2511`, seed 0, 40 steps, true-CFG 4.0, guidance 1.0, negative prompt `" "`, bf16 on CUDA |
| Mask reference | cyan-tinted second reference image + fixed instruction suffix |
| Patch | 512 × 512, FoV = clip(1.5 · max(span_λ·cos φ, span_φ), 30°, 150°) |
| Patch constraint | Gaussian ρ = 3 px |
| Composite | inward chamfer-DT smoothstep ramp, r = 8 px |
| Gate | τ = 0.7 (explicit detections clamped up to 0.7) |
| Tiling | 60° tiles, 20 % overlap, ≤ 6 tiles |
| Multi-instance | 110° group budget, containment gate 0.8, ≤ 6 groups |

`configs/editpano.yaml` spells out exactly these values; `editpano config`
prints the resolved configuration.

## Protocols

```bash
DATA=/path/to/data/test          # 31 test scenes
P=/path/to/prompts/test.json     # one instruction per scene
```

**Reference-ROI comparison and baselines** (thesis Tables 3.3, Fig. 3.4–3.5)

```bash
ARMS="whole_erp erp_crop whole_then_paste editpano editpano_tiled cubemap"
editpano bench run   --data-root $DATA --prompts $P --out-dir runs/ref --arms $ARMS
editpano bench score --data-root $DATA --prompts $P --out-dir runs/ref --arms $ARMS --region-clip
editpano bench report --data-root $DATA --prompts $P --out-dir runs/ref --reference editpano
```

SD-Inpaint on the ERP box: `--arms sd_inpaint_roi --config configs/sd_inpaint.yaml`
(English `tgt` field). Large runs: `scripts/slurm/benchmark.sbatch` shards cases
over a SLURM array; outputs are signature-cached, so jobs are resumable.

**Edit-type generalisation** — a prompts file with an `edit_type` per row and
`clip_text: null` for removals; filter with `--edit-types removal addition`.

**Second frozen editor** — `--config configs/sd_inpaint.yaml --prompt-key tgt`;
keep a `clip_text` column to score Region-CLIP against the original instruction.

**Predicted localisation** — arm `editpano_detected` (frozen
`IDEA-Research/grounding-dino-tiny`, query = `object_label`). Metrics are always
referenced to the reference ROI, so detector misses are penalised.

**Repeated targets** — arm `editpano_multi`, scored with `--instance-support`.

**Parameter sweeps** — one output directory per value, e.g.
`--set composite.feather_px=16`, `--set projection.fov_margin=2.0`,
`--set projection.patch_size=768`.

**Reprojection cycle consistency** (Table 3.4, editor-free, CPU)

```bash
editpano diagnose cycle --data-root $DATA --prompts $P --patches runs/ref/editpano
```

`--patches` points at the `<case_id>/local_patch.png` edited patches saved by the
`editpano` arm (omit it to
round-trip the source patch).

**Longitude-origin (seam) study** (Table 3.9, Fig. 3.10–3.13)

```bash
editpano seam build    --data-root $DATA --prompts $P --out-dir seam/dataset   # 31 × {center, near, cross}
editpano seam validate --manifest seam/dataset/manifest.json --expected-scenes 31
editpano seam run      --manifest seam/dataset/manifest.json --out-dir seam/runs    # QA gate re-checked
editpano seam score    --manifest seam/dataset/manifest.json --out-dir seam/runs --region-clip
editpano seam report   --out-dir seam/runs
```

Add `--tiled` for the wide-ROI extension and `--seeds 0 1 2` for seed
stability; `seam run` refuses to start if the QA gate fails.

## Verification of this implementation

The package was checked against the original research implementation and its
cached thesis artefacts (CPU unless noted):

| Check | Result |
| --- | --- |
| Replaying the 31 cached thesis patches through the new reprojection vs the cached thesis panoramas | **31/31 bit-identical** |
| Full pipeline with a deterministic editor, new vs original code, 31 real scenes | **31/31 bit-identical** |
| Seam strategies (identity/current/split/direct-circular/canonical + tiled), 12 cases | **84/84 bit-identical** |
| Replaying all 93 cached thesis canonical-seam patches | **93/93 bit-identical** |
| Multi-instance and tiled editing, 4 scenes | **8/8 bit-identical** |
| Rebuilding the 93-case seam benchmark | manifest and pixels **93/93 identical**, QA gate passes |
| BG-PSNR / coverage recomputed on cached outputs | 78.48 vs 19.51 dB, 0.718 vs 0.722 — identical to the thesis |
| Cross-YE-LPIPS on cached seam outputs | identical to the thesis (max difference 8e-17) |
| Cycle consistency on cached patches | 37.28 dB analytic vs 26.96 dB scatter, 48.8 % holes |

`tests/unit/test_golden.py` pins this behaviour: it compares SHA-256 digests of
outputs produced by the original code on synthetic inputs (exact comparison in
the recorded NumPy/OpenCV/Pillow versions).
