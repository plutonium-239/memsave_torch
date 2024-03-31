.DEFAULT: help

help:
	@echo "ruff"
	@echo "			Run ruff formatter and linter"
	@echo "test"
	@echo "			Run pytest (tests directory)"

.PHONY: ruff

ruff:
	@ruff format
	@ruff check --fix --show-fixes

.PHONY: test test-full

test:
	@pytest -vx --cov=memsave_torch test -m quick

test-full:
	@pytest -vx --cov=memsave_torch test
