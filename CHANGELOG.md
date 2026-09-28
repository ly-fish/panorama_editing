# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-28

First release of the refactored research code as the installable `editpano` package.

### Added
- Pure geometric core (`editpano.geometry`): adaptive gnomonic views, dense analytic
  inverse reprojection, inward-feathered / Gaussian / Poisson compositing, circular (S¹)
  ROIs with canonical longitude shifts, narrow-FoV tiling, sphere rotation.
- Operators: single-ROI `LocalEditOperator`, seam strategies with circular
  canonicalisation (`SeamStrategyRunner`), grouped multi-instance editing, and the
  `EditPano` facade with confidence gating and branch routing.
- Editor port with a plugin registry (`qwen`, `sd_inpaint`, deterministic `fake:*`
  editors, `editpano.editors` entry points) and a prompt-policy hook.
- Frozen Grounding DINO localisation and Grounded-SAM reference-ROI loading.
- DiT360 text-to-panorama generation as a thin `FluxPipeline` subclass.
- Evaluation: BG-PSNR, coverage, Region-CLIP, cycle consistency, YE-LPIPS, DS; paired
  sign-flip tests, bootstrap CIs, Cohen's d_z, Holm; reference-ROI benchmark with nine
  arms and signature-cached, shardable runs; seam benchmark builder with QA gate and the
  full equivariance study.
- `editpano` CLI, typed configuration with YAML and dotted overrides, SLURM templates,
  CI, pre-commit, 86 tests including bit-exact golden tests against the thesis code.

### Changed (relative to the research code)
- Model pipelines are used from upstream `diffusers` instead of vendored copies.
- The structured-prompt (AESG) suffix and the keyword-heuristic localiser were removed
  (ablated as non-load-bearing / never reaching the local branch); the prompt-policy hook
  and explicit `gate.fallback` replace them.
- The abandoned LoRA / flow-training code paths are not part of the package.
