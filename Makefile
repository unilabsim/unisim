# Common development and packaging commands for UniSim.

.PHONY: sync
sync:
	uv sync --locked --extra mujoco

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: test
test:
	uv run pytest -q

.PHONY: test-no-sync
test-no-sync:
	uv run --no-sync pytest -q

.PHONY: check
check: lint test

.PHONY: package
package:
	uv build --out-dir dist

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist
