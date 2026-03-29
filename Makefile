PY311 ?= python3.11

.PHONY: install-dev lint typecheck test check

install-dev:
	$(PY311) -m pip install -e '.[dev]'

lint:
	$(PY311) -m ruff check .

typecheck:
	$(PY311) -m mypy wafergeo

test:
	$(PY311) -m pytest -q

check: lint typecheck test
