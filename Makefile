PY ?= python

.PHONY: install dev lint format typecheck test test-fast check clean

install:            ## runtime install with editing backends
	$(PY) -m pip install -e ".[edit,grounding,eval]"

dev:                ## development install (tests, lint, types)
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

typecheck:
	$(PY) -m mypy

test:               ## full CPU test suite (no model weights needed)
	$(PY) -m pytest

test-fast:          ## skip tests that need torch/LPIPS
	$(PY) -m pytest -m "not requires_torch"

check: lint typecheck test   ## what CI runs

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov src/*.egg-info
