# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,edit,eval for the model backends
make check                       # ruff + mypy + pytest (CPU only, no weights)
```

* Geometry changes must keep `tests/unit/test_golden.py` green: those digests pin
  behaviour to the implementation evaluated in the thesis. If a change is
  intentional, regenerate the digests and explain why in the changelog.
* New editors implement the `PatchEditor` protocol (`info`, `edit`) and must be
  deterministic given `EditRequest.seed`.
* Keep heavy imports (torch, diffusers, transformers, lpips, clip) inside the
  functions that need them; `import editpano` must stay torch-free.
* Never commit data, model weights, experiment outputs or credentials
  (see `.gitignore`).
