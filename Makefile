.PHONY: setup test check

setup:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest -q

check:
	python -m compileall -q engine strategies rl scripts tests
	python -m pytest -q
