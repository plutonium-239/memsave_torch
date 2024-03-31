.DEFAULT: help

help:
	@echo "install"
	@echo "			Install as an editable package"
	@echo "install-test"
	@echo "			Install as an editable package + test"
	@echo "ruff"
	@echo "			Run ruff formatter and linter"
	@echo "test"
	@echo "			Run pytest (tests directory)"
	@echo "test-full"
	@echo "			Run pytest with expensive tests (tests directory)"

.PHONY: install

install:
	@pip install -e .

.PHONY: install-test

install-test:
	@pip install -e .[test]

.PHONY: ruff

ruff:
	@ruff format
	@ruff check --fix --show-fixes

.PHONY: test test-full

test:
	@pytest -vx --cov=memsave_torch test -m quick

test-full:
	@pytest -vx --cov=memsave_torch test
