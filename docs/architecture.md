# Architecture

EditPano is organised as a small hexagonal (ports-and-adapters) system: a pure
geometric core, operators that compose it with an abstract editor port, and
adapters for concrete models, localisers and experiments at the edges.

```
                       ┌──────────────────────────── editpano.cli ─────────────────────────────┐
                       │  edit · generate · bench · seam · diagnose · editors · config · doctor │
                       └───────────────┬────────────────────────────────┬──────────────────────┘
                                       │                                │
              ┌────────────────────────▼─────────┐        ┌─────────────▼──────────────────────┐
              │ pipeline.EditPano (facade)        │        │ evaluation                          │
              │  ROI resolution → gate → routing  │◄───────│  datasets · arms · benchmark        │
              └──┬──────────────┬──────────────┬──┘        │  seam_benchmark · seam_study        │
                 │              │              │           │  metrics/* · stats                  │
     ┌───────────▼───┐  ┌───────▼────────┐  ┌──▼──────────────┐└───────────────────────────────────┘
     │ LocalEdit-    │  │ SeamStrategy-  │  │ MultiInstance-  │
     │ Operator  (L) │◄─│ Runner (F_can, │  │ Editor          │
     │               │  │ split, tiles)  │  │ (groups + gate) │
     └──┬─────────┬──┘  └────────────────┘  └─────────────────┘
        │         │
        │   ┌─────▼──────────────── port ───────────┐     ┌────────── localization ───────────┐
        │   │ editors.PatchEditor (Protocol)         │     │ annotations (reference ROI)        │
        │   │  qwen · sd_inpaint · fake:* · plugins  │     │ GroundingDinoDetector (frozen)     │
        │   └────────────────────────────────────────┘     └────────────────────────────────────┘
   ┌────▼──────────────────── geometry (numpy + OpenCV only) ─────────────────────────────┐
   │ camera (views, adaptive FoV) · projection (gnomonic + dense inverse) · compositing    │
   │ masks · circular (S¹ ROI, canonical shift) · tiling · rotation                        │
   └───────────────────────────────────────────────────────────────────────────────────────┘
```

## Layers and dependency rule

| Layer | Package | May import | Heavy deps |
| --- | --- | --- | --- |
| Geometry | `editpano.geometry` | numpy, OpenCV, Pillow | none |
| Domain | `editpano.types`, `editpano.config`, `editpano.errors` | geometry | none |
| Editor port | `editpano.editors.base` | domain | none |
| Operators | `editpano.pipeline` | geometry, domain, editor port | none |
| Adapters | `editpano.editors.{qwen,sd_inpaint}`, `editpano.localization.grounding`, `editpano.generation` | anything above | torch / diffusers / transformers, imported lazily |
| Evaluation | `editpano.evaluation` | everything above | scikit-image, scipy, LPIPS, CLIP, imported lazily |
| Interface | `editpano.cli` | everything | – |

Dependencies point inwards only. `import editpano` never imports torch; the
whole geometric core, all operators and both experiment pipelines run and are
tested on a CPU with deterministic fake editors.

## The operator algebra

* **`LocalEditOperator` – ``L``** (thesis Eq. 3.1): adaptive view → gnomonic
  patch → patch support → frozen edit → patch constraint → dense analytic
  inverse → inward composite. One editor call.
* **`SeamStrategyRunner.canonical` – ``F_can = T₋ᵣ ∘ L ∘ Tᵣ``**: roll image and
  support so the circular arc centre is at ``u = W/2``, apply ``L`` once, roll
  back. With a deterministic editor this is *exactly* equivariant to integer
  longitude shifts, which the tests check bit-for-bit.
* **Tiling**: ``I⁽ᵏ⁾ = L(I⁽ᵏ⁻¹⁾, t, b_k)`` over a grid of narrow-FoV sub-boxes;
  composes with ``F_can`` (canonical-tiled).
* **`MultiInstanceEditor`**: greedy FoV-budgeted grouping + containment gate,
  one ``L`` per group with the union of tight instance boxes as support.
* **`EditPano`** routes a request to one of the above (or to whole-ERP editing
  when no ROI passes the confidence gate) and returns an `EditResult` with every
  patch, the view parameters and JSON provenance (config fingerprint, editor
  identity and snapshot revision, timings).

## Invariants and where they are enforced

| Invariant | Mechanism | Test |
| --- | --- | --- |
| Pixels outside the support are bit-identical (`inward`) | ramp lives inside Ω, `out[~Ω] = I[~Ω]` | `test_compositing.py`, `test_operators.py` |
| Analytic reprojection is hole-free | dense inverse over the ERP grid | `test_projection.py` |
| Seam-position equivariance of `F_can` | half-up rounding, integer rolls, periodic sampler | `test_operators.py::test_canonical_strategy_is_exactly_longitude_equivariant` |
| Behaviour identical to the evaluated thesis implementation | numerics ported operation-for-operation | `test_golden.py` (SHA-256 of legacy outputs) |
| Experiment caches are never stale | run signature = config + editor + revision + arm + version | `test_pipelines.py` |
| Seam benchmark integrity | QA gate (bit-exact inverse roll, invariants) | `test_pipelines.py::test_seam_benchmark_build_and_qa` |

## Extension points

* **New editor** — implement `info` and `edit(EditRequest) -> PIL.Image`, then
  `register_editor("my-editor")(MyEditor)` or expose it under the
  `editpano.editors` entry-point group of your own package.
* **Prompt policy** — `EditPano(prompt_policy=fn)` maps the instruction ``t`` to
  the effective prompt ``t'`` (e.g. to append structured scene constraints).
* **Detector** — any object with `detect_roi(image, phrase)` and
  `detect_all(image, phrase)` returning `Detection`s can be passed as
  `EditPano(detector=...)`.
* **Benchmark arm** — add an `Arm` to `editpano.evaluation.arms.ARMS`.

## Design decisions

* **Upstream model code, not vendored copies.** The research code carried
  copies of the diffusers Qwen-Image-Edit (≈900 lines) and FLUX (≈1000 lines)
  pipelines. The Qwen copy differed from upstream only by a token-fusion hook
  that was disabled in every evaluated configuration, so the adapter now calls
  `diffusers.QwenImageEditPlusPipeline` directly. DiT360's only change to FLUX —
  circular padding of the latent token grid — is a 30-line subclass.
* **No keyword heuristic localiser.** The legacy low-confidence keyword crop
  could never pass the confidence gate, so it only ever caused a whole-ERP
  fallback; that fallback is now explicit (`gate.fallback: whole | raise`).
* **Structured-prompt (AESG) suffix removed.** Its component ablation was null
  (not load-bearing); the `prompt_policy` hook keeps the extension point.
* **Frozen configuration objects** with strict validation and dotted overrides,
  so every run is described by one hashable value.
