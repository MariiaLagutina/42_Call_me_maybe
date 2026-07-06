UV_CACHE_DIR := $(HOME)/goinfre/uv
HF_HOME := $(HOME)/goinfre/huggingface
TRANSFORMERS_CACHE := $(HOME)/goinfre/huggingface

export UV_CACHE_DIR
export HF_HOME
export TRANSFORMERS_CACHE

.PHONY: install run debug lint lint-strict clean prepare-cache space test

prepare-cache:
	mkdir -p $(HOME)/goinfre/uv
	mkdir -p $(HOME)/goinfre/huggingface
	mkdir -p $(HOME)/goinfre/call_me_maybe_venv
	mkdir -p $(HOME)/.cache
	rm -rf $(HOME)/.cache/uv
	rm -rf $(HOME)/.cache/huggingface
	ln -sfn $(HOME)/goinfre/uv $(HOME)/.cache/uv
	ln -sfn $(HOME)/goinfre/huggingface $(HOME)/.cache/huggingface
	ln -sfn $(HOME)/goinfre/call_me_maybe_venv .venv

install: prepare-cache
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

test:
	uv run pytest -q

space:
	df -h ~
	du -h -d 1 ~ 2>/dev/null | sort -hr | head -20