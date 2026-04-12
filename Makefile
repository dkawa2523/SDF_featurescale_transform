PYTHON ?= python

.PHONY: install-dev lint typecheck test check clean

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy wafergeo

test:
	$(PYTHON) -m pytest -q

check: lint typecheck test

clean:
	$(PYTHON) -c "import pathlib, shutil; root=pathlib.Path('.'); names={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','outputs'}; [shutil.rmtree(p, ignore_errors=True) for p in root.rglob('*') if p.name in names]; [shutil.rmtree(root / n, ignore_errors=True) for n in names]"
