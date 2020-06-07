test:
	pytest -s ./tests

test_no_jit:
	env NUMBA_DISABLE_JIT=1 pytest -s ./tests

coverage:
	env NUMBA_DISABLE_JIT=1 pytest -s tests --cov=embiggen --cov-report=html && open htmlcov/index.html